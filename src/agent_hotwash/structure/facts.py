"""Deterministic per-episode facts shared by segmentation, digest and diagnostics.

One builder, :func:`episode_facts`, computes every flag the digest shows JeV
(§5.4 ``episode.facts``) and every key the diagnostics read back off
``Episode.facts`` (repeats, re-reads, verification, impediments, …). Both
``segment_episodes`` (which persists them on the atom) and ``build_digest``
call it, so the two views can never disagree.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from agent_hotwash.canonical import events_in_range
from agent_hotwash.events import Event, EventKind, Session
from agent_hotwash.primitives.commands import classify_command
from agent_hotwash.primitives.errors import classify_impediment

if TYPE_CHECKING:
    from agent_hotwash.structure.episodes import Episode
    from agent_hotwash.structure.tasks import Task

READ_OPS = frozenset({"cmd.read", "cmd.search", "cmd.list"})
EDIT_OPS = frozenset({"file.edit", "file.write", "file.delete"})  # every artifact mutation

SUCCESS_RE = re.compile(
    r"(all (?:checks|tests) passed|should work now|fixed\.|that works|completed successfully)",
    re.IGNORECASE,
)
VERIFY_RE = re.compile(r"(\d+\s+passed|\bpytest\b|\bjest\b|tests? run)", re.IGNORECASE)


def command_of(event: Event) -> str:
    """Best-effort command string for an op event."""
    args = event.tool_args or {}
    for key in ("command", "cmd"):
        val = args.get(key)
        if isinstance(val, str) and val.strip():
            return val
        if isinstance(val, list):
            return " ".join(str(x) for x in val)
    if event.classifications:
        parts = [c.cmd for c in event.classifications if c.cmd]
        if parts:
            return " ".join(parts)
    return event.text or ""


def is_test_like(event: Event) -> bool:
    """True for build/test commands or test-runner tool names."""
    cmd = command_of(event)
    if cmd and classify_command(cmd) == "build_test":
        return True
    blob = f"{event.tool_name or ''} {event.op_kind or ''} {cmd}".lower()
    return any(tok in blob for tok in ("pytest", "jest", "vitest"))


def norm_sig(event: Event) -> str:
    """Normalised op signature used for repeat / failure-signature counting."""
    if event.tool_norm_args:
        return event.tool_norm_args
    return f"{event.op_kind or event.tool_name or ''}|{command_of(event)}"


def paths_of(event: Event) -> list[str]:
    paths = [art.path for art in event.artifacts if art.path]
    if event.path and event.path not in paths:
        paths.append(event.path)
    return paths


def is_failed(event: Event) -> bool:
    return event.ok is False or (event.exit_code not in (0, None) and event.kind is EventKind.tool_result)


def session_events_before(session: Session, end_idx: int) -> list[Event]:
    return [ev for ev in session.events if (ev.idx if ev.idx >= 0 else 0) < end_idx]


def excerpt(pattern: re.Pattern[str], texts: list[str]) -> str | None:
    for text in texts:
        if not text:
            continue
        m = pattern.search(text)
        if m:
            start = max(0, m.start() - 20)
            end = min(len(text), m.end() + 40)
            return text[start:end].strip()
    return None


def _failure_signature_repeats(events: list[Event], history: list[Event]) -> int:
    failed = [norm_sig(ev) for ev in events if is_failed(ev)]
    if not failed:
        return 0
    hist_fail = [norm_sig(ev) for ev in history if is_failed(ev)]
    return sum(hist_fail.count(sig) for sig in set(failed))


def _paths_reread_unchanged(history: list[Event], events: list[Event]) -> int:
    """Reads *in this episode* of a path already read and not edited since."""
    edited: set[str] = set()
    last_read: set[str] = set()

    def _step(ev: Event, count: bool) -> int:
        if ev.kind is not EventKind.tool_call:
            return 0
        paths = paths_of(ev)
        if ev.op_kind in EDIT_OPS:
            edited.update(paths)
            return 0
        if ev.op_kind not in {"cmd.read", "cmd.search"}:
            return 0
        hits = 0
        for p in paths:
            if count and p in last_read and p not in edited:
                hits += 1
            last_read.add(p)
            edited.discard(p)
        return hits

    for ev in history:
        _step(ev, count=False)
    return sum(_step(ev, count=True) for ev in events)


def _repeated_ops(events: list[Event], history: list[Event]) -> int:
    """Ops in the episode whose signature already occurred (history or earlier in episode)."""
    seen = {norm_sig(ev) for ev in history if ev.kind is EventKind.tool_call}
    repeats = 0
    for ev in events:
        if ev.kind is not EventKind.tool_call:
            continue
        sig = norm_sig(ev)
        if sig in seen:
            repeats += 1
        seen.add(sig)
    return repeats


def _edits_without_subsequent_test(events: list[Event]) -> int:
    pending = 0
    for ev in events:
        if ev.kind is EventKind.tool_call and ev.op_kind in EDIT_OPS:
            pending += 1
        if is_test_like(ev):
            pending = 0
    return pending


def _error_kinds(events: list[Event]) -> list[str]:
    kinds: list[str] = []
    for ev in events:
        if is_failed(ev):
            cat = ev.error_category or ("test_failure" if is_test_like(ev) else "other")
            if cat not in kinds:
                kinds.append(cat)
    return kinds


def _env_impediment(events: list[Event]) -> str | None:
    """Impediment kind from a failed op, not incidental wording in success text."""
    for ev in events:
        if not is_failed(ev) and not ev.error_text:
            continue
        kind = classify_impediment(ev.error_text or ev.output or ev.text)
        if kind:
            return kind
    return None


def _op_outcomes(events: list[Event]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for ev in events:
        if ev.kind is not EventKind.tool_result:
            continue
        ok: bool | None = ev.ok
        if ok is None and ev.exit_code is not None:
            ok = ev.exit_code == 0
        if ok is None:
            continue
        out.append({"class": "test" if is_test_like(ev) else (ev.op_kind or ev.tool_name or "other"), "ok": ok})
    return out


def episode_facts(session: Session, episode: Episode, task: Task | None = None) -> dict[str, Any]:
    """Deterministic facts for one atom. Keys are stable; diagnostics read them.

    ``task`` (when given) contributes ``artifact_overlap`` and the last final
    answer for the success/verification excerpts.
    """
    events = events_in_range(session, episode.event_start, episode.event_end)
    history = session_events_before(session, episode.event_start)
    texts = [ev.text or "" for ev in events if ev.kind is EventKind.assistant_msg]
    if task is not None and task.ledger.last_answer:
        texts.append(task.ledger.last_answer)
    outputs = [ev.output or "" for ev in events]

    calls = [ev for ev in events if ev.kind is EventKind.tool_call]
    distinct_read = {p for ev in calls if ev.op_kind in {"cmd.read", "cmd.search"} for p in paths_of(ev)}
    overlap: list[str] = []
    if task is not None:
        deliverable_set = set(task.ledger.deliverables) | set(task.ledger.artifacts)
        overlap = sorted(deliverable_set.intersection(set(episode.artifacts)))

    error_kinds = _error_kinds(events)
    outcomes = _op_outcomes(events)
    tests = [row for row in outcomes if row["class"] == "test"]
    declares = excerpt(SUCCESS_RE, texts)
    cites = excerpt(VERIFY_RE, texts)
    verification = excerpt(VERIFY_RE, outputs + texts)
    verified_by_test = any(row["ok"] for row in tests)

    return {
        "failure_signature_repeats": _failure_signature_repeats(events, history),
        "repeat_count": _repeated_ops(events, history),
        "distinct_paths_read": len(distinct_read),
        "paths_reread_unchanged": _paths_reread_unchanged(history, events),
        "edits": sum(1 for ev in calls if ev.op_kind in EDIT_OPS),
        "artifact_change": any(ev.op_kind in EDIT_OPS for ev in calls),
        "edits_without_subsequent_test": _edits_without_subsequent_test(events),
        "tests_run": len(tests),
        "tests_failed": sum(1 for row in tests if not row["ok"]),
        "outstanding_failure": bool(outcomes) and outcomes[-1]["ok"] is False,
        "error_kinds": error_kinds,
        "env_impediment": _env_impediment(events),
        "after_compaction": any(ev.kind is EventKind.compaction for ev in history),
        "declares_success": declares is not None,
        "declares_success_excerpt": declares,
        "cites_verification": cites is not None,
        "verification": verification is not None or verified_by_test,
        "verification_excerpt": verification,
        "artifact_overlap": overlap,
        "op_outcomes": outcomes,
    }


__all__ = [
    "EDIT_OPS",
    "READ_OPS",
    "SUCCESS_RE",
    "VERIFY_RE",
    "command_of",
    "episode_facts",
    "excerpt",
    "is_failed",
    "is_test_like",
    "norm_sig",
    "paths_of",
    "session_events_before",
]
