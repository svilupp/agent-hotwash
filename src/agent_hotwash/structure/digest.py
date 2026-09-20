"""Capped JeV digest with pre-computed deterministic facts (§5.4)."""

from __future__ import annotations

import json
from collections import Counter
from typing import TYPE_CHECKING, Any

from agent_hotwash.canonical import events_in_range
from agent_hotwash.config import Config
from agent_hotwash.events import Event, EventKind, Session
from agent_hotwash.structure.facts import (
    EDIT_OPS,
    READ_OPS,
    command_of,
    episode_facts,
    is_test_like,
    norm_sig,
    paths_of,
    session_events_before,
)

if TYPE_CHECKING:
    from agent_hotwash.structure.episodes import Episode
    from agent_hotwash.structure.tasks import Task

DIGEST_SCHEMA_VERSION = 3
_MESSAGE_KINDS = {
    EventKind.user_msg: "user",
    EventKind.assistant_msg: "assistant",
    EventKind.thinking: "thinking",
}
_AGENT_KINDS = frozenset({"agent.spawn", "agent.wait", "agent.message"})
_HEAD_TAIL_MARKER = "\n…\n"
_OP_CAP = 12
_HEAD_TAIL_CAP = 400
_DIFF_CAP = 1200
_CHARS_PER_TOKEN = 4


def _head_tail(text: str | None, cap: int) -> str:
    """Keep the head and tail of a long string, capped at ``cap`` chars."""
    if not text:
        return ""
    if len(text) <= cap:
        return text
    keep = max(1, (cap - len(_HEAD_TAIL_MARKER)) // 2)
    return text[:keep] + _HEAD_TAIL_MARKER + text[-keep:]


def _est_tokens(obj: Any) -> int:
    blob = json.dumps(obj, ensure_ascii=False, separators=(",", ":"), default=str)
    return max(1, len(blob) // _CHARS_PER_TOKEN)


def _command_of(event: Event) -> str:
    """Command text for display; falls back to the tool name for bare tool calls."""
    return command_of(event) or event.tool_name or ""


def _task_events(session: Session, task: Task) -> list[Event]:
    events: list[Event] = []
    for turn in task.turns:
        events.extend(events_in_range(session, turn.event_start, turn.event_end))
    if events:
        return events
    return session.events


def _observed(events: list[Event]) -> dict[str, int]:
    files: set[str] = set()
    edits = reads = tests_run = tests_failed = subagents = 0
    for ev in events:
        files.update(paths_of(ev))
        if ev.op_kind in EDIT_OPS:
            edits += 1
        if ev.op_kind in READ_OPS:
            reads += 1
        if is_test_like(ev) and ev.kind is EventKind.tool_result:
            tests_run += 1
            if ev.ok is False or (ev.exit_code not in (0, None)):
                tests_failed += 1
        if ev.op_kind in {"agent.spawn", "agent.wait", "agent.message"}:
            subagents += 1
    return {
        "files": len(files),
        "edits": edits,
        "reads": reads,
        "tests_run": tests_run,
        "tests_failed": tests_failed,
        "subagents": subagents,
    }


def _repeat_meta(event: Event, history: list[Event]) -> tuple[int, int | None]:
    sig = norm_sig(event)
    matches = [ev for ev in history if norm_sig(ev) == sig and ev.kind is event.kind]
    if not matches:
        return 0, None
    last = matches[-1]
    ago = None
    if event.idx >= 0 and last.idx >= 0:
        ago = event.idx - last.idx
    return len(matches), ago


def _path_edit_count(path: str, history: list[Event]) -> int:
    return sum(1 for ev in history if ev.op_kind in EDIT_OPS and path in paths_of(ev))


def _path_previously_read(path: str, history: list[Event]) -> bool:
    return any(ev.op_kind in READ_OPS and path in paths_of(ev) for ev in history)


def _prev_same_cmd_exit(event: Event, history: list[Event]) -> int | None:
    cmd = _command_of(event)
    if not cmd:
        return None
    for ev in reversed(history):
        if _command_of(ev) == cmd and ev.exit_code is not None:
            return ev.exit_code
        if _command_of(ev) == cmd and ev.kind is EventKind.tool_result:
            return ev.exit_code
    return None


def _op_entry(
    event: Event,
    *,
    history: list[Event],
    episode_events: list[Event],
    head_cap: int,
    diff_cap: int,
    seen_edit: bool,
    changed_paths: set[str],
) -> dict[str, Any]:
    kind = event.op_kind or event.tool_name or "other"
    paths = paths_of(event)
    repeat_count, last_seen = _repeat_meta(event, history)
    entry: dict[str, Any] = {"kind": kind}
    cmd = _command_of(event)
    if cmd:
        entry["cmd"] = _head_tail(cmd, head_cap)
    if event.exit_code is not None:
        entry["exit"] = event.exit_code
    if event.output:
        entry["out_head"] = _head_tail(event.output, head_cap)[:head_cap]
        entry["out_tail"] = event.output[-head_cap:] if len(event.output) > head_cap else event.output
        if len(event.output) <= head_cap:
            entry["out_tail"] = _head_tail(event.output, head_cap)
    else:
        entry["out_head"] = ""
        entry["out_tail"] = ""
    if event.output_tokens_original is not None:
        entry["out_tokens"] = event.output_tokens_original
    entry["repeat_count"] = repeat_count
    if last_seen is not None:
        entry["last_seen_calls_ago"] = last_seen
    entry["paths"] = list(paths)
    if paths:
        diffs = [art.diff_head for art in event.artifacts if art.diff_head]
        if diffs:
            entry["diff_head"] = _head_tail("\n".join(diffs), diff_cap)
        entry["path_previously_read"] = any(_path_previously_read(p, history) for p in paths)
        entry["path_edit_count"] = max((_path_edit_count(p, history) for p in paths), default=0)
    if is_test_like(event):
        entry["class"] = "test"
        entry["after_edit"] = seen_edit or any(e.op_kind in EDIT_OPS for e in episode_events if e.idx <= event.idx)
        entry["targets_changed_paths"] = bool(changed_paths.intersection(paths)) if paths else False
        prev_exit = _prev_same_cmd_exit(event, history)
        if prev_exit is not None:
            entry["prev_same_cmd_exit"] = prev_exit
    return entry


def _message_entry(event: Event, cap: int) -> dict[str, Any] | None:
    """One JeV-addressable message: ``messages[i].kind`` / ``.text`` / ``.phase``."""
    kind = _MESSAGE_KINDS.get(event.kind)
    if kind is None or not event.text:
        return None
    return {"kind": kind, "text": _head_tail(event.text, cap), "phase": event.phase or ""}


def _op_family(kind: str, class_: object) -> str:
    """Map a canonical op kind onto a coarse family for ``episode.counts``."""
    if class_ == "test":
        return "run"
    if kind in READ_OPS:
        return "read"
    if kind in EDIT_OPS:
        return "edit"
    if kind in _AGENT_KINDS:
        return "agent"
    if kind.startswith("cmd.") or kind.startswith("mcp."):
        return "run"
    return "other"


def _majority(counter: Counter[str]) -> str | None:
    if not counter:
        return None
    ranked = counter.most_common(2)
    if len(ranked) == 1 or ranked[0][1] > ranked[1][1]:
        return ranked[0][0]
    return None


def _episode_counts(op_entries: list[dict[str, Any]], messages: list[dict[str, Any]]) -> dict[str, Any]:
    """Pre-counted episode evidence so JeV does not tally ``ops[]`` itself."""
    by_kind: Counter[str] = Counter(str(e.get("kind") or "other") for e in op_entries)
    by_family: Counter[str] = Counter(_op_family(str(e.get("kind") or "other"), e.get("class")) for e in op_entries)
    return {
        "n_ops": len(op_entries),
        "by_kind": dict(by_kind),
        "majority_kind": _majority(by_kind),
        "by_family": dict(by_family),
        "majority_family": _majority(by_family),
        "n_failed": sum(1 for e in op_entries if e.get("exit") not in (0, None)),
        "n_tests": sum(1 for e in op_entries if e.get("class") == "test"),
        "n_final_answer": sum(1 for m in messages if m.get("phase") == "final_answer"),
        "test_after_edit": any(e.get("class") == "test" and e.get("after_edit") for e in op_entries),
    }


def _last_ops_summary(events: list[Event], n: int = 4) -> list[str]:
    out: list[str] = []
    for ev in events:
        if ev.kind is EventKind.tool_call or (ev.op_kind and ev.kind is EventKind.tool_result):
            cmd = _command_of(ev)
            bit = ev.op_kind or ev.tool_name or "op"
            if cmd:
                bit = f"{bit} {cmd}"
            if ev.exit_code is not None:
                bit += f" (exit {ev.exit_code})"
            out.append(_head_tail(bit, 120))
    return out[-n:]


def _trim_strings(obj: Any, cap: int) -> Any:
    if isinstance(obj, str) and len(obj) > cap:
        return _head_tail(obj, cap)
    if isinstance(obj, dict):
        return {k: _trim_strings(v, cap) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_trim_strings(v, cap) for v in obj]
    return obj


def _shrink(digest: dict[str, Any], budget: int) -> dict[str, Any]:
    """Hard-fail toward a smaller cap until the digest fits ``budget`` tokens."""
    if _est_tokens(digest) <= budget:
        return digest
    cap = _HEAD_TAIL_CAP
    for _ in range(16):
        digest = _trim_strings(digest, cap)
        ep = digest.get("episode")
        if isinstance(ep, dict):
            ops = ep.get("ops")
            if isinstance(ops, list) and len(ops) > 1:
                ep["omitted_ops"] = int(ep.get("omitted_ops") or 0) + (len(ops) - 1)
                ep["ops"] = ops[:1]
            ep["messages"] = []
        if _est_tokens(digest) <= budget:
            return digest
        cap = max(8, cap // 2)
    return {
        "task": {"request": "", "amendments": [], "deliverables": [], "status": "open", "observed": {}},
        "episode": {"ops": [], "omitted_ops": 0, "facts": {}, "messages": [], "counts": {}, "usage": {}},
        "digest_schema_version": DIGEST_SCHEMA_VERSION,
    }


def build_digest(task: Task, episode: Episode, session: Session, config: Config) -> dict[str, Any]:
    """Build the capped digest JeV sees. Whole digest ≤ ``state_token_budget`` tokens."""
    budget = config.structure.episodes.state_token_budget
    events = events_in_range(session, episode.event_start, episode.event_end)
    history = session_events_before(session, episode.event_start)
    task_events = _task_events(session, task)

    # Same builder segment_episodes used; anything already persisted on the atom
    # (``position``, hand-set facts) wins so digest and atom always agree.
    facts: dict[str, Any] = {**episode_facts(session, episode, task), **episode.facts}
    facts.pop("op_outcomes", None)  # per-op outcomes are already visible in ``ops[].exit``

    op_events = [
        ev for ev in events if ev.kind is EventKind.tool_call or (ev.op_kind and ev.kind is EventKind.tool_result)
    ]
    # Prefer the call side; skip a result when the call is already present.
    results_by_id = {ev.call_id: ev for ev in op_events if ev.kind is EventKind.tool_result and ev.call_id}
    ordered: list[Event] = []
    seen_ids: set[str] = set()
    for ev in events:
        if ev.kind is EventKind.tool_call:
            merged = ev
            result = results_by_id.get(ev.call_id or "")
            if result is not None:
                merged = ev.model_copy(
                    update={
                        "exit_code": result.exit_code if result.exit_code is not None else ev.exit_code,
                        "ok": result.ok if result.ok is not None else ev.ok,
                        "output": result.output or ev.output,
                        "output_tokens_original": result.output_tokens_original or ev.output_tokens_original,
                        "error_text": result.error_text or ev.error_text,
                    }
                )
                if ev.call_id:
                    seen_ids.add(ev.call_id)
            ordered.append(merged)
        elif ev.kind is EventKind.tool_result and ev.call_id and ev.call_id not in seen_ids:
            ordered.append(ev)
            seen_ids.add(ev.call_id)

    changed_paths = {p for ev in events if ev.op_kind in EDIT_OPS for p in paths_of(ev)}
    seen_edit = any(ev.op_kind in EDIT_OPS for ev in history)
    op_entries: list[dict[str, Any]] = []
    running_hist = list(history)
    for ev in ordered:
        op_entries.append(
            _op_entry(
                ev,
                history=running_hist,
                episode_events=events,
                head_cap=_HEAD_TAIL_CAP,
                diff_cap=_DIFF_CAP,
                seen_edit=seen_edit,
                changed_paths=changed_paths,
            )
        )
        running_hist.append(ev)
        if ev.op_kind in EDIT_OPS:
            seen_edit = True

    raw_messages = [m for ev in events if (m := _message_entry(ev, _HEAD_TAIL_CAP)) is not None]
    omitted_messages = max(0, len(raw_messages) - 6)
    messages = raw_messages[-6:]
    counts = _episode_counts(op_entries, raw_messages)

    omitted = 0
    omitted_kinds: Counter[str] = Counter()
    if len(op_entries) > _OP_CAP:
        omitted_list = op_entries[_OP_CAP:]
        omitted = len(omitted_list)
        omitted_kinds = Counter(str(e.get("kind") or "other") for e in omitted_list)
        op_entries = op_entries[:_OP_CAP]

    preceding_events = history[-12:]
    last_message = ""
    for ev in reversed(preceding_events):
        if ev.kind is EventKind.assistant_msg and ev.text:
            last_message = ev.text
            break
        if ev.kind is EventKind.user_msg and ev.text:
            last_message = ev.text
            break

    turn = next((t for t in session.turns if t.turn_id == episode.turn_id), None)
    effort = None
    if turn is not None:
        effort = turn.model_config_active.reasoning_effort

    usage = episode.usage
    digest: dict[str, Any] = {
        "task": {
            "request": _head_tail(task.ledger.request, _HEAD_TAIL_CAP),
            "amendments": [_head_tail(a, _HEAD_TAIL_CAP) for a in task.ledger.amendments],
            "deliverables": list(task.ledger.deliverables),
            "status": task.ledger.status,
            "observed": _observed(task_events),
        },
        "episode": {
            "position": facts.get("position", ""),
            "trigger": episode.trigger,
            "prior_outcome": (
                "failed_test"
                if facts.get("error_kinds") and "test_failure" in facts["error_kinds"]
                else ("error" if facts.get("error_kinds") else None)
            ),
            "preceding": {
                "last_message": _head_tail(last_message, _HEAD_TAIL_CAP),
                "last_ops": _last_ops_summary(preceding_events),
            },
            "ops": op_entries,
            "omitted_ops": omitted,
            "omitted_kind_counts": dict(omitted_kinds),
            "counts": counts,
            "facts": facts,
            "messages": messages,
            "omitted_messages": omitted_messages,
            "model": {"effort": effort},
            "usage": {
                "output": usage.output,
                "reasoning": usage.reasoning_output,
                "input": usage.input,
                "cache_read": usage.cache_read,
            },
        },
        "digest_schema_version": DIGEST_SCHEMA_VERSION,
    }
    return _shrink(digest, budget)


__all__ = ["DIGEST_SCHEMA_VERSION", "build_digest"]
