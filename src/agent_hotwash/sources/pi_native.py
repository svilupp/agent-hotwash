"""Native pi session parser (format D).

Layout: ``~/.pi/agent/sessions/<project-slug>/<ISO-ts>_<uuid>.jsonl``, where the
project slug is a mangled cwd (e.g. ``--Users-jan-Documents-GitHub--``) and each
file is one session. Records are flat ``{type, ...}`` objects (no ``payload``
envelope, unlike codex; no ``uuid``/``parentUuid`` DAG, unlike claude):

- ``session`` — first record; ``id`` is the session id, plus ``version``/``cwd``.
- ``model_change`` — ``provider`` + ``modelId`` (the active model).
- ``thinking_level_change`` — ignored (meta).
- ``message`` — wraps a ``message`` object keyed by ``role``:
  - ``user``: ``content`` blocks (``{type:text,text}``), epoch-ms ``timestamp``.
  - ``assistant``: ``content`` blocks (``thinking`` / ``toolCall`` / ``text``),
    per-response ``usage`` (``input``/``output``/``cacheRead``/``cacheWrite``),
    ``model``/``provider``, epoch-ms ``timestamp``.
  - ``toolResult``: ``toolCallId`` + ``toolName``, ``content`` blocks, and a
    top-level ``isError`` flag (mirrors code-bench pi's ``tool_execution_end``).

This is a close cousin of the code-bench pi stdout stream; the usage shape is
identical, so ``_pi_usage`` is reused from :mod:`agent_hotwash.sources.codebench`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from agent_hotwash.canonical import build_turns, declared_row, observe_capabilities
from agent_hotwash.events import (
    AgentKind,
    CapLevel,
    Event,
    EventKind,
    Provenance,
    Trace,
)
from agent_hotwash.sources._common import (
    build_session,
    flatten_text,
    iter_jsonl,
    parse_ts,
    truncate,
)
from agent_hotwash.sources.codebench import _pi_usage

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

_DECLARED = declared_row(
    per_call_usage=True,
    per_turn_model=True,
    reasoning_effort=False,
    reasoning_text=True,
    reasoning_tokens=False,
    timestamps=True,
    op_timing=False,
    parsed_commands=False,
    file_diffs=False,
    full_tool_output=CapLevel.partial,
    output_size_original=False,
    thread_linkage=False,
    compaction_summaries=False,
    final_answer_marker=False,
    context_window=False,
)


def decode_pi_native(records: list[dict[str, Any]]) -> tuple[list[Event], str | None, str | None]:
    """Decode native pi session records into (events, session_id, model)."""
    events: list[Event] = []
    session_id: str | None = None
    model: str | None = None

    for r in records:
        rtype = r.get("type")
        if rtype == "session":
            session_id = r.get("id") or session_id
        elif rtype == "model_change":
            model = r.get("modelId") or model
            events.append(
                Event(
                    kind=EventKind.meta,
                    ts=parse_ts(r.get("timestamp")),
                    raw_type="model_change",
                    tool_args={"model": r.get("modelId"), "provider": r.get("provider")},
                )
            )
        elif rtype == "message":
            msg = r.get("message")
            if isinstance(msg, dict):
                events.extend(_message(msg))
                model = model or msg.get("model")
    return events, session_id, model


def _message(msg: dict[str, Any]) -> list[Event]:
    role = msg.get("role")
    ts = parse_ts(msg.get("timestamp"))
    if role == "user":
        return [Event(kind=EventKind.user_msg, text=flatten_text(msg.get("content")) or None, ts=ts, raw_type="user")]
    if role == "toolResult":
        content = flatten_text(msg.get("content"))
        is_error = bool(msg.get("isError"))
        return [
            Event(
                kind=EventKind.tool_result,
                call_id=msg.get("toolCallId"),
                tool_name=msg.get("toolName"),
                ts=ts,
                ok=not is_error,
                output=truncate(content),
                error_text=truncate(content) if is_error else None,
                raw_type="toolResult",
            )
        ]
    if role == "assistant":
        return _assistant(msg, ts)
    return []


def _assistant(msg: dict[str, Any], ts: Any) -> list[Event]:
    """Split an assistant message's content blocks into thinking / tool_call /
    assistant_msg events. Per-response usage is attached to the first event so
    summation bills the response once."""
    usage = _pi_usage(msg.get("usage"))
    events: list[Event] = []
    for block in msg.get("content", []) or []:
        if not isinstance(block, dict):
            continue
        btype = block.get("type")
        if btype == "thinking":
            events.append(
                Event(kind=EventKind.thinking, text=block.get("thinking") or None, ts=ts, raw_type="thinking")
            )
        elif btype == "toolCall":
            events.append(
                Event(
                    kind=EventKind.tool_call,
                    tool_name=block.get("name"),
                    call_id=block.get("id"),
                    tool_args=block.get("arguments") or {},
                    ts=ts,
                    raw_type="toolCall",
                )
            )
        elif btype == "text":
            events.append(Event(kind=EventKind.assistant_msg, text=block.get("text") or None, ts=ts, raw_type="text"))
    if usage is not None and events:
        events[0].usage = usage
    return events


def load_session_file(path: Path) -> Trace:
    """Parse one native pi session file into a Trace."""
    records = list(iter_jsonl(path))
    events, session_id, model = decode_pi_native(records)
    session_id = session_id or path.stem
    root = build_session(events, AgentKind.pi, session_id=session_id, model=model)
    root.turns = build_turns(root)
    root.capabilities = observe_capabilities(root, _DECLARED)
    provenance = Provenance(
        source_format="pi_native",
        detector_confidence="high",
        root_path=path,
        files=[path],
        notes=[],
    )
    return Trace(
        trace_id=session_id,
        agent=AgentKind.pi,
        model=model,
        root=root,
        provenance=provenance,
        capabilities=root.capabilities,
    )


def iter_sessions_tree(root: Path) -> Iterator[Trace]:
    """Yield a Trace per native pi session file under a sessions tree/project dir."""
    for path in sorted(root.rglob("*.jsonl")):
        if looks_like_native_pi(path):
            yield load_session_file(path)


def looks_like_native_pi(path: Path) -> bool:
    """True when the first record is a native-pi ``session`` header.

    The signature is a flat ``{type:"session"}`` record carrying ``version`` and
    ``cwd`` and no ``payload`` envelope — this distinguishes it from a codex
    ``session_meta`` (which nests a ``payload``) and a claude session file (whose
    records carry ``uuid``/``parentUuid`` and different ``type`` values).
    """
    if not path.is_file() or path.suffix != ".jsonl":
        return False
    for r in iter_jsonl(path):
        return r.get("type") == "session" and "version" in r and "cwd" in r and "payload" not in r
    return False
