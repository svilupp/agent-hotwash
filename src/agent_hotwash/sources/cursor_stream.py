"""Cursor Agent stream-json, including the stdout logs written by Cloop."""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Any, TypedDict

from agent_hotwash.canonical import build_turns, declared_row, observe_capabilities
from agent_hotwash.events import (
    AgentKind,
    CapLevel,
    Event,
    EventKind,
    Provenance,
    RoleHint,
    SourceRef,
    Trace,
    TurnStatus,
    Usage,
)
from agent_hotwash.sources._common import (
    build_session,
    flatten_text,
    iter_jsonl,
    parse_ts,
    record_source,
    source_artifacts,
    source_diagnostics,
    truncate,
)
from agent_hotwash.sources._cursor import (
    attach_file_change,
    integer_field,
    object_field,
    string_field,
    tool_call,
    tool_result,
    user_prompt,
)

if TYPE_CHECKING:
    from datetime import datetime
    from pathlib import Path

_INIT_FIELDS = {"apiKeySource", "cwd", "model", "permissionMode"}
_USAGE_FIELDS = {
    "input": "inputTokens",
    "output": "outputTokens",
    "cache_read": "cacheReadTokens",
    "cache_write": "cacheWriteTokens",
    "reasoning_output": "reasoningTokens",
}
_DECLARED = declared_row(
    per_turn_model=True,
    reasoning_text=CapLevel.partial,
    reasoning_tokens=CapLevel.partial,
    timestamps=CapLevel.partial,
    op_timing=True,
    file_diffs=CapLevel.partial,
    full_tool_output=CapLevel.partial,
    final_answer_marker=True,
)


class _EventFields(TypedDict):
    source: SourceRef
    ts: datetime | None
    raw_type: str | None
    response_id: str | None


def looks_like_stream(path: Path) -> bool:
    if not path.is_file() or path.suffix not in {".jsonl", ".ndjson"}:
        return False
    with path.open("rb") as stream:
        for line in stream:
            try:
                record = object_field(json.loads(line))
            except ValueError:
                continue
            kind, subtype = record.get("type"), record.get("subtype")
            if kind == "system" and subtype == "init":
                # Claude's print stream shares these four keys but also has its own version field.
                return bool(
                    string_field(record.get("session_id"))
                    and record.keys() >= _INIT_FIELDS
                    and "claude_code_version" not in record
                )
            if not string_field(record.get("session_id")):
                continue
            if kind == "tool_call" and subtype in ("started", "completed") and _tool_payload(record):
                return True
            if kind == "thinking" and subtype in ("delta", "completed") and "timestamp_ms" in record:
                return True
            if kind == "result" and "inputTokens" in object_field(record.get("usage")):
                return True
    return False


def _tool_payload(record: dict[str, Any]) -> tuple[str, dict[str, Any]] | None:
    tool = object_field(record.get("tool_call"))
    for key, value in tool.items():
        if key.endswith("ToolCall") and isinstance(value, dict):
            return key.removesuffix("ToolCall"), value
    function = object_field(tool.get("function"))
    if name := string_field(function.get("name")):
        args = function.get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except ValueError:
                args = {"arguments": args}
        return name, {"args": args, "result": function.get("result")}
    return None


def _invocation_anchor(record: dict[str, Any]) -> str | None:
    for field in ("call_id", "model_call_id", "request_id"):
        if identifier := string_field(record.get(field)):
            return f"{field}:{identifier}"
    return None


def stream_identity(path: Path, size: int) -> tuple[str | None, str | None]:
    """Find the chat ID and an invocation-specific anchor in an export prefix."""
    session_id = None
    with path.open("rb") as stream:
        while size:
            line = stream.readline(size)
            if not line:
                break
            size -= len(line)
            try:
                record = object_field(json.loads(line))
            except ValueError:
                continue
            session_id = session_id or string_field(record.get("session_id"))
            if session_id and (anchor := _invocation_anchor(record)):
                return session_id, anchor
    return session_id, None


def _append_tool(
    record: dict[str, Any],
    fields: _EventFields,
    cwd: str | None,
    calls: dict[str, Event],
    events: list[Event],
) -> None:
    payload = _tool_payload(record)
    if payload is None:
        return
    name, data = payload
    subtype = record.get("subtype")
    wrapper = object_field(record.get("tool_call"))
    args = object_field(data.get("args")).copy()
    argument_id = args.pop("toolCallId", None)
    call_id = (
        string_field(record.get("call_id")) or string_field(wrapper.get("toolCallId")) or string_field(argument_id)
    )
    start = parse_ts(wrapper.get("startedAtMs"))
    end = parse_ts(wrapper.get("completedAtMs"))
    call = calls.get(call_id) if call_id else None
    if (subtype == "started" or (subtype == "completed" and call_id and (args or start))) and call is None:
        call_fields = fields.copy()
        call_fields["ts"] = start or (fields["ts"] if subtype == "started" else None)
        call = tool_call(name, args, cwd, call_id=call_id, **call_fields)
        events.append(call)
        if call_id:
            calls[call_id] = call
    if subtype == "completed":
        result_fields = fields.copy()
        result_time = end or fields["ts"]
        result_fields["ts"] = result_time
        if call is not None and call.ts and result_time and result_time >= call.ts:
            call.ts_end = result_time
        payload = data.get("result")
        if payload is None:
            payload = record.get("result")
        outcome = object_field(payload)
        result = tool_result(
            name, outcome, fallback=payload if not isinstance(payload, dict) else None, call_id=call_id, **result_fields
        )
        if call is not None:
            attach_file_change(call, result, outcome, cwd)
        events.append(result)


def _terminal_usage(record: dict[str, Any]) -> Usage | None:
    raw = object_field(record.get("usage"))
    counts = {field: integer_field(raw.get(key)) for field, key in _USAGE_FIELDS.items()}
    supplied = {key: count for key, count in counts.items() if count is not None and count >= 0}
    return Usage.model_validate(supplied) if supplied else None


def _assistant_turn(events: list[Event]) -> tuple[list[Event], bool]:
    messages = []
    saw_tool = False
    for event in reversed(events):
        if event.kind is EventKind.session_end or (
            event.kind is EventKind.user_msg and event.role_hint != RoleHint.injected
        ):
            break
        if not messages and event.kind in (EventKind.tool_call, EventKind.tool_result):
            saw_tool = True
        if event.kind is EventKind.assistant_msg:
            messages.append(event)
    return messages[::-1], saw_tool


def _unseen_text(text: str, messages: list[Event]) -> str:
    if not messages:
        return text
    combined = "".join(event.text or "" for event in messages)
    if text == messages[-1].text:
        return ""
    return text.removeprefix(combined)


def _extend_message(event: Event, suffix: str, fields: _EventFields) -> None:
    event.text = (event.text or "") + suffix
    if event.source is not None and event.source != fields["source"]:
        event.alternate_sources.append(event.source)
    event.source = fields["source"]


def _final_answer(events: list[Event], result: str | None, fields: _EventFields) -> None:
    messages, saw_tool = _assistant_turn(events)
    last = messages[-1] if messages and not saw_tool else None
    unseen = _unseen_text(result, messages) if result else ""
    if unseen and result:
        if last is not None and unseen != result:
            _extend_message(last, unseen, fields)
        else:
            last = Event(kind=EventKind.assistant_msg, text=unseen, **fields)
            events.append(last)
    if last is not None:
        last.phase = "final_answer"


def _decode(
    records: list[dict[str, Any]], cwd: str | None, trace_id: str, model: str | None
) -> tuple[list[Event], dict[str, Any] | None]:
    events: list[Event] = []
    calls: dict[str, Event] = {}
    pieces: list[str] = []
    pending: Event | None = None
    pending_assistant: Event | None = None
    partial_output = False
    terminal = None

    def flush() -> None:
        nonlocal pending, pending_assistant
        if pending is not None:
            pending.text = "".join(pieces)
            events.append(pending)
            if pending.kind is EventKind.assistant_msg:
                pending_assistant = pending
            pending = None
            pieces.clear()

    for index, record in enumerate(records):
        source = record_source(record, index)
        source = source.model_copy(update={"item_id": f"stream:{trace_id}:{source.line_number or index + 1}"})
        kind, subtype = record.get("type"), record.get("subtype")
        timestamp_ms = parse_ts(record.get("timestamp_ms"))
        timestamp = timestamp_ms or parse_ts(record.get("timestamp"))
        fields: _EventFields = {
            "source": source,
            "ts": timestamp,
            "raw_type": string_field(kind),
            "response_id": string_field(record.get("model_call_id")),
        }
        text = flatten_text(object_field(record.get("message")).get("content"))
        if kind == "thinking" and subtype == "delta":
            message_kind, delta = EventKind.thinking, string_field(record.get("text"))
        elif kind == "assistant" and timestamp_ms and not fields["response_id"]:
            message_kind = EventKind.assistant_msg
            delta = text or None
            partial_output = True
        else:
            message_kind, delta = None, None
        if message_kind is not None:
            if not delta:
                continue
            if pending is None or pending.kind is not message_kind:
                flush()
                pending = Event(kind=message_kind, ts_end=timestamp, **fields)
            elif timestamp is not None:
                pending.ts_end = timestamp
            pieces.append(delta)
            continue
        if kind not in ("thinking", "user", "assistant", "system", "tool_call", "result"):
            continue
        if kind == "thinking" and subtype == "completed" and pending is not None and pending.kind is EventKind.thinking:
            pending.ts_end = timestamp or pending.ts_end
        flush()

        if kind == "user" and text:
            prompt, role_hint = user_prompt(text)
            events.append(
                Event(
                    kind=EventKind.user_msg,
                    text=prompt,
                    role_hint=role_hint,
                    turn_id=f"{trace_id}:line{source.line_number or index + 1}",
                    **fields,
                )
            )
            if role_hint is not RoleHint.injected:
                pending_assistant = None
                partial_output = False
        elif kind == "assistant" and text:
            if partial_output:
                if pending_assistant is not None and text.startswith(pending_assistant.text or ""):
                    suffix = text.removeprefix(pending_assistant.text or "")
                    if suffix:
                        _extend_message(pending_assistant, suffix, fields)
                    pending_assistant.response_id = fields["response_id"] or pending_assistant.response_id
                    continue
                messages, _ = _assistant_turn(events)
                text = _unseen_text(text, messages)
            if text:
                events.append(Event(kind=EventKind.assistant_msg, text=text, **fields))
        elif kind == "system" and subtype == "init":
            events.append(Event(kind=EventKind.session_start, **fields))
        elif kind == "tool_call":
            _append_tool(record, fields, cwd, calls, events)
            pending_assistant = None
        elif kind == "result":
            terminal = record
            ok = record.get("is_error") is not True and subtype in (None, "success")
            result = string_field(record.get("result"))
            if ok:
                _final_answer(events, result, fields)
            events.append(
                Event(
                    kind=EventKind.session_end,
                    ok=ok,
                    error_text=truncate(result) if not ok else None,
                    invocation_usage=_terminal_usage(record),
                    usage_model=model,
                    **fields,
                )
            )
    flush()
    return events, terminal


def load_stream(path: Path, *, records: list[dict[str, Any]] | None = None, fallback_model: str | None = None) -> Trace:
    if records is None:
        records = list(iter_jsonl(path))
    artifacts = source_artifacts(records)
    if not artifacts:
        raise ValueError("Cursor stream has no JSON records")
    session_ids = {value for record in records if (value := string_field(record.get("session_id")))}
    if len(session_ids) > 1:
        raise ValueError("Cursor stream contains more than one session id")
    if (
        sum(r.get("type") == "result" for r in records) > 1
        or sum(r.get("type") == "system" and r.get("subtype") == "init" for r in records) > 1
    ):
        raise ValueError("Cursor stream contains more than one invocation")
    native_id = next(iter(session_ids), None)
    init = next((r for r in records if r.get("type") == "system" and r.get("subtype") == "init"), {})
    cwd = string_field(init.get("cwd"))
    model = string_field(init.get("model")) or fallback_model
    sha = artifacts[0]["source_sha256"]
    request_id = next(
        (
            value
            for r in reversed(records)
            if r.get("type") == "result" and (value := string_field(r.get("request_id")))
        ),
        None,
    )
    anchor = next((value for r in records if (value := _invocation_anchor(r))), None)
    identity = f"request_id:{request_id}" if request_id else anchor or f"artifact:{sha}"
    invocation_id = hashlib.sha256(identity.encode()).hexdigest()[:20]
    trace_id = f"{native_id or path.stem}:{invocation_id}"
    events, terminal = _decode(records, cwd, trace_id, model)
    root = build_session(
        events,
        AgentKind.cursor,
        session_id=native_id or path.stem,
        model=model,
        usage_reliable=any(ev.invocation_usage is not None for ev in events),
    )
    root.native_session_id_known = native_id is not None
    root.identity_basis = "native_session_id" if native_id else "source_filename_fallback"
    root.source_artifacts = artifacts
    root.source_diagnostics = source_diagnostics(records)
    root.turns = build_turns(root)
    for turn in root.turns:
        for event in root.events[turn.event_start : turn.event_end + 1]:
            event.turn_id = turn.turn_id
        last = root.events[turn.event_end]
        if last.kind is EventKind.session_end:
            turn.source_end = last.source
            turn.status = TurnStatus.completed if last.ok else TurnStatus.aborted
    root.capabilities = observe_capabilities(root, _DECLARED)
    if terminal is not None:
        notes = ["Cursor CLI reports token usage at invocation scope."]
    else:
        notes = ["Cursor CLI stream ended without a terminal result."]
    return Trace(
        trace_id=trace_id,
        agent=AgentKind.cursor,
        model=model,
        root=root,
        capabilities=root.capabilities,
        provenance=Provenance(
            source_format="cursor_stream",
            detector_confidence="high",
            root_path=path,
            files=[path],
            harness_meta={
                "cwd": cwd,
                "request_id": request_id,
                "duration_ms": (terminal or {}).get("duration_ms"),
            },
            notes=notes,
        ),
    )
