"""code-bench run-dir parser (format A) + the three stdout decoders.

A code-bench *run dir* (``runs/<exp>/<instance>/<run_id>/``) holds one
``stdout.jsonl`` plus harness metadata: ``run.json`` (whose ``harness`` field is
the authoritative agent kind), ``metrics.json`` (token/cost totals), and
``verification.json`` (ground-truth ``resolved``). The stdout stream is decoded
by one of three format-specific decoders selected on ``harness``:

- **codex** — ``thread.started`` / ``item.started|completed`` / ``turn.completed``
  (single cumulative usage). No per-event timestamps.
- **claude** — stream-json ``system`` / ``assistant`` / ``user`` / ``result``;
  content blocks ``text`` / ``thinking`` / ``tool_use`` / ``tool_result``;
  ``parent_tool_use_id`` links inline subagents. Per-assistant usage.
- **pi** — chatty ``session`` / ``*_start`` / ``*_update`` / ``*_end``;
  ``isError`` is top-level on ``tool_execution_end``; per-``message_end`` usage.

Where stream usage is genuinely all-zero (rare), totals are backfilled from
``metrics.json`` and the trace is flagged ``usage_reliable=False``.

See ``docs/research/format_findings.md`` for the verified field shapes.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from agent_hotwash.events import (
    AgentKind,
    Event,
    EventKind,
    Provenance,
    Session,
    Trace,
    Usage,
)
from agent_hotwash.sources._common import (
    build_session,
    flatten_text,
    iter_jsonl,
    total_stream_tokens,
    truncate,
)

if TYPE_CHECKING:
    from pathlib import Path

_HARNESS_AGENT = {
    "codex": AgentKind.codex,
    "claude": AgentKind.claude,
    "pi": AgentKind.pi,
}


# ---------------------------------------------------------------------------
# codex stdout decoder
# ---------------------------------------------------------------------------


def decode_codex_stdout(records: list[dict[str, Any]]) -> list[Event]:
    """Decode a code-bench codex ``stdout.jsonl`` to raw events (no timestamps)."""
    events: list[Event] = []
    for r in records:
        rtype = r.get("type")
        if rtype == "item.started":
            item = r.get("item", {})
            if item.get("type") in ("command_execution", "file_change"):
                events.append(_codex_tool_call(item))
        elif rtype == "item.completed":
            item = r.get("item", {})
            itype = item.get("type")
            if itype == "agent_message":
                events.append(Event(kind=EventKind.assistant_msg, text=item.get("text"), raw_type=itype))
            elif itype in ("command_execution", "file_change"):
                events.append(_codex_tool_result(item))
        elif rtype == "turn.completed":
            events.append(_codex_usage_event(r.get("usage", {})))
    return events


def _codex_tool_call(item: dict[str, Any]) -> Event:
    itype = item["type"]
    if itype == "file_change":
        changes = item.get("changes", [])
        path = changes[0].get("path") if changes and isinstance(changes[0], dict) else None
        return Event(
            kind=EventKind.tool_call,
            tool_name="file_change",
            call_id=item.get("id"),
            tool_args={"changes": changes, "path": path} if path else {"changes": changes},
            raw_type=itype,
        )
    return Event(
        kind=EventKind.tool_call,
        tool_name="command_execution",
        call_id=item.get("id"),
        tool_args={"command": item.get("command", "")},
        raw_type=itype,
    )


def _codex_tool_result(item: dict[str, Any]) -> Event:
    exit_code = item.get("exit_code")
    status = item.get("status")
    output = item.get("aggregated_output")
    ok = not (status == "failed" or (exit_code not in (0, None)))
    return Event(
        kind=EventKind.tool_result,
        call_id=item.get("id"),
        tool_name=item["type"],
        ok=ok,
        exit_code=exit_code if isinstance(exit_code, int) else None,
        output=truncate(output),
        error_text=truncate(output) if not ok else None,
        raw_type=item["type"],
    )


def _codex_usage_event(usage: dict[str, Any]) -> Event:
    return Event(
        kind=EventKind.meta,
        raw_type="turn.completed",
        usage=Usage(
            input=usage.get("input_tokens"),
            output=usage.get("output_tokens"),
            cache_read=usage.get("cached_input_tokens"),
            cumulative=True,
        ),
    )


# ---------------------------------------------------------------------------
# claude stdout decoder (stream-json)
# ---------------------------------------------------------------------------


def _claude_events(records: list[dict[str, Any]]) -> list[tuple[str | None, Event]]:
    """Decode claude stream-json into (parent_tool_use_id, Event) pairs.

    ``parent_tool_use_id`` (a ``toolu_`` id) tags events belonging to an inline
    subagent so the caller can split them into linked child Sessions.

    One API response is emitted as several jsonl lines (one per content block),
    each repeating the SAME ``message.id`` and ``message.usage``. Usage is
    attached only to the first block of each unique ``message.id`` so summation
    bills each request once (not once per block). See ``_build_sessions`` for the
    separate output-token reconciliation these streams also need.
    """
    out: list[tuple[str | None, Event]] = []
    seen_msg_ids: set[str] = set()
    for r in records:
        rtype = r.get("type")
        parent = r.get("parent_tool_use_id")
        if rtype == "assistant":
            msg = r.get("message", {})
            msg_id = msg.get("id")
            first_of_message = msg_id is None or msg_id not in seen_msg_ids
            if msg_id is not None:
                seen_msg_ids.add(msg_id)
            usage = _claude_usage(msg.get("usage")) if first_of_message else None
            used = False
            for block in msg.get("content", []) or []:
                if not isinstance(block, dict):
                    continue
                ev = _claude_block(block)
                if ev is None:
                    continue
                if not used:
                    ev.usage = usage
                    used = True
                out.append((parent, ev))
        elif rtype == "user":
            out.extend((parent, ev) for ev in _claude_user(r))
    return out


def _claude_result_line(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The final ``result`` record, which carries ``usage`` / ``modelUsage`` /
    ``total_cost_usd`` for the whole run (format_findings.md format 2)."""
    for r in reversed(records):
        if r.get("type") == "result":
            return r
    return None


def _claude_block(block: dict[str, Any]) -> Event | None:
    btype = block.get("type")
    if btype == "text":
        return Event(kind=EventKind.assistant_msg, text=block.get("text"), raw_type="text")
    if btype == "thinking":
        return Event(kind=EventKind.thinking, text=block.get("thinking") or block.get("text"), raw_type="thinking")
    if btype == "tool_use":
        return Event(
            kind=EventKind.tool_call,
            tool_name=block.get("name"),
            call_id=block.get("id"),
            tool_args=block.get("input") or {},
            raw_type="tool_use",
        )
    return None


def _claude_user(r: dict[str, Any]) -> list[Event]:
    from agent_hotwash.sources._common import parse_ts

    ts = parse_ts(r.get("timestamp"))
    content = r.get("message", {}).get("content")
    events: list[Event] = []
    if isinstance(content, str):
        if content.strip():
            events.append(Event(kind=EventKind.user_msg, text=content, ts=ts))
        return events
    if not isinstance(content, list):
        return events
    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "tool_result":
            text = flatten_text(block.get("content"))
            is_error = bool(block.get("is_error"))
            events.append(
                Event(
                    kind=EventKind.tool_result,
                    call_id=block.get("tool_use_id"),
                    ts=ts,
                    ok=not is_error,
                    output=truncate(text),
                    error_text=truncate(text) if is_error else None,
                    raw_type="tool_result",
                )
            )
        elif block.get("type") == "text":
            events.append(Event(kind=EventKind.user_msg, text=block.get("text"), ts=ts))
    return events


def _claude_usage(usage: dict[str, Any] | None) -> Usage | None:
    if not usage:
        return None
    return Usage(
        input=usage.get("input_tokens"),
        output=usage.get("output_tokens"),
        cache_read=usage.get("cache_read_input_tokens"),
        cache_write=usage.get("cache_creation_input_tokens"),
    )


# ---------------------------------------------------------------------------
# pi stdout decoder
# ---------------------------------------------------------------------------


def decode_pi_stdout(records: list[dict[str, Any]]) -> list[Event]:
    """Decode a code-bench pi ``stdout.jsonl`` — ignore chatty ``*_update`` deltas."""
    from agent_hotwash.sources._common import parse_ts

    events: list[Event] = []
    for r in records:
        rtype = r.get("type")
        if rtype == "tool_execution_start":
            events.append(
                Event(
                    kind=EventKind.tool_call,
                    tool_name=r.get("toolName"),
                    call_id=r.get("toolCallId"),
                    tool_args=r.get("args") or {},
                    raw_type=rtype,
                )
            )
        elif rtype == "tool_execution_end":
            result = r.get("result") or {}
            text = flatten_text(result.get("content"))
            is_error = bool(r.get("isError"))  # SURPRISE: top-level flag
            events.append(
                Event(
                    kind=EventKind.tool_result,
                    call_id=r.get("toolCallId"),
                    tool_name=r.get("toolName"),
                    ok=not is_error,
                    output=truncate(text),
                    error_text=truncate(text) if is_error else None,
                    raw_type=rtype,
                )
            )
        elif rtype == "message_end":
            msg = r.get("message") or {}
            role = msg.get("role")
            ts = parse_ts(msg.get("timestamp"))
            usage = _pi_usage(msg.get("usage"))
            if role == "assistant":
                events.append(
                    Event(
                        kind=EventKind.assistant_msg,
                        text=flatten_text(msg.get("content")),
                        ts=ts,
                        usage=usage,
                        raw_type=rtype,
                    )
                )
            elif role == "user":
                events.append(
                    Event(kind=EventKind.user_msg, text=flatten_text(msg.get("content")), ts=ts, raw_type=rtype)
                )
    return events


def _pi_usage(usage: dict[str, Any] | None) -> Usage | None:
    if not usage:
        return None
    return Usage(
        input=usage.get("input"),
        output=usage.get("output"),
        cache_read=usage.get("cacheRead"),
        cache_write=usage.get("cacheWrite"),
    )


# ---------------------------------------------------------------------------
# run-dir assembly
# ---------------------------------------------------------------------------


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _stdout_path(run_dir: Path) -> Path | None:
    """Locate the attempt's stdout stream, falling back to the live variant."""
    primary = run_dir / "stdout.jsonl"
    if primary.exists():
        return primary
    for candidate in sorted(run_dir.glob("traces/*/stdout.live.jsonl")):
        return candidate
    live = run_dir / "stdout.live.jsonl"
    return live if live.exists() else None


def is_run_dir(path: Path) -> bool:
    """A code-bench run dir has a ``run.json`` and some stdout stream."""
    return (path / "run.json").is_file() and _stdout_path(path) is not None


def load_run_dir(run_dir: Path) -> Trace | None:
    """Parse one code-bench run dir into a :class:`Trace` (``None`` if unreadable)."""
    run = _read_json(run_dir / "run.json")
    harness = run.get("harness")
    agent = _HARNESS_AGENT.get(harness or "", AgentKind.unknown)
    stdout = _stdout_path(run_dir)
    if stdout is None:
        return None

    metrics = _read_json(run_dir / "metrics.json")
    verification = _read_json(run_dir / "verification.json")
    model = run.get("model") or metrics.get("model")

    records = list(iter_jsonl(stdout))
    files = [
        p for p in (run_dir / "run.json", stdout, run_dir / "metrics.json", run_dir / "verification.json") if p.exists()
    ]
    notes: list[str] = []

    root, subagents = _build_sessions(agent, records, run, model)

    # Backfill usage from metrics.json when the stream carries none (glm pi edge).
    if total_stream_tokens(root.events) == 0:
        if _backfill_usage(root, metrics):
            notes.append("usage absent in stream; backfilled totals from metrics.json")
            root = root.model_copy(update={"usage_reliable": False})
        else:
            notes.append("stream usage all-zero and no metrics.json backfill available")

    resolved = verification.get("resolved") if "resolved" in verification else None
    harness_meta: dict[str, Any] = {"run": run, "metrics": metrics, "verification": verification}
    # Lift claude's stream-final `total_cost_usd` so cost provenance is authoritative
    # even independent of metrics.json (see analytics._provenance_cost).
    if agent is AgentKind.claude:
        result = _claude_result_line(records)
        cost = result.get("total_cost_usd") if result else None
        if isinstance(cost, (int, float)) and not isinstance(cost, bool):
            harness_meta["total_cost_usd"] = float(cost)
    provenance = Provenance(
        source_format="codebench",
        detector_confidence="high" if agent is not AgentKind.unknown else "low",
        root_path=run_dir,
        files=files,
        harness_meta=harness_meta,
        notes=notes,
    )
    return Trace(
        trace_id=run.get("run_id") or run_dir.name,
        agent=agent,
        model=model,
        experiment=run.get("experiment"),
        instance_id=run.get("instance_id"),
        root=root,
        subagents=subagents,
        provenance=provenance,
        resolved=resolved if isinstance(resolved, bool) else None,
    )


def _build_sessions(
    agent: AgentKind, records: list[dict[str, Any]], run: dict[str, Any], model: str | None
) -> tuple[Session, list[Session]]:
    session_id = run.get("session_id") or run.get("run_id") or "codebench"
    if agent is AgentKind.claude:
        pairs = _claude_events(records)
        root_events = [ev for parent, ev in pairs if not parent]
        by_parent: dict[str, list[Event]] = {}
        for parent, ev in pairs:
            if parent:
                by_parent.setdefault(parent, []).append(ev)
        # Token reconciliation against the run-final `result` line.
        #   * No per-message usage at all -> adopt the result line's usage wholesale
        #     so totals stay real (avoids metrics.json backfill / usage_reliable=False).
        #   * Per-message usage present -> its input / cache_read / cache_write are
        #     accurate, but stream-json `output_tokens` are streaming placeholders
        #     (they undercount output ~40x). Take the authoritative output total
        #     from the result line instead. See format_findings.md format 2.
        result = _claude_result_line(records)
        result_usage = _claude_usage(result.get("usage")) if result else None
        if not any(ev.usage for ev in root_events):
            if result_usage is not None:
                root_events.append(Event(kind=EventKind.meta, usage=result_usage, raw_type="result"))
        elif result_usage is not None and result_usage.output is not None:
            _reconcile_claude_output(root_events, result_usage.output)
        root = build_session(root_events, agent, session_id=session_id, model=model)
        subagents = [
            build_session(evs, agent, session_id=f"{session_id}:{tid}", model=model, parent_session_id=session_id)
            for tid, evs in by_parent.items()
        ]
        return root, subagents
    if agent is AgentKind.codex:
        raw = decode_codex_stdout(records)
    elif agent is AgentKind.pi:
        raw = decode_pi_stdout(records)
    else:
        raw = []
    return build_session(raw, agent, session_id=session_id, model=model), []


def _reconcile_claude_output(events: list[Event], result_output: int) -> None:
    """Replace stream-json per-message ``output_tokens`` placeholders with the
    authoritative run-total output from the ``result`` line.

    Per-message ``input`` / ``cache_read`` / ``cache_write`` are accurate and kept
    (so per-turn context-window metrics stay meaningful); only the placeholder
    ``output`` is zeroed, and the true total is carried on one appended meta event
    so summed tokens/cost match actual billing.
    """
    for ev in events:
        if ev.usage is not None:
            ev.usage.output = None
    events.append(Event(kind=EventKind.meta, usage=Usage(output=result_output), raw_type="result"))


def _backfill_usage(session: Session, metrics: dict[str, Any]) -> bool:
    tokens = metrics.get("tokens") if isinstance(metrics.get("tokens"), dict) else None
    if not tokens:
        return False
    usage = Usage(
        input=tokens.get("input"),
        output=tokens.get("output"),
        cache_read=tokens.get("cache_read"),
        cache_write=tokens.get("cache_write"),
    )
    if not any((usage.input, usage.output, usage.cache_read, usage.cache_write)):
        return False
    # Attach the backfilled total to the last event so it stays summable.
    if session.events:
        session.events[-1].usage = usage
    return True
