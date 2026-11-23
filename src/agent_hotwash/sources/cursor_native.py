"""Cursor Agent CLI chat stores at ~/.cursor/chats/<workspace>/<chat>/store.db."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections import Counter
from contextlib import closing
from typing import TYPE_CHECKING, Any

from agent_hotwash.canonical import build_turns, declared_row, observe_capabilities
from agent_hotwash.events import (
    AgentKind,
    CapLevel,
    Event,
    EventKind,
    Provenance,
    RoleHint,
    Session,
    SourceRef,
    ThreadLink,
    ThreadLinkKind,
    Trace,
    TurnStatus,
)
from agent_hotwash.sources._common import build_session, flatten_text, parse_ts
from agent_hotwash.sources._cursor import (
    attach_file_change,
    object_field,
    string_field,
    tool_call,
    tool_result,
    user_prompt,
)
from agent_hotwash.sources.cursor_checkpoint import (
    CheckpointTiming,
    Fields,
    blob_ids,
    bytes_fields,
    integer,
    protobuf,
    read_blob,
    read_timing,
    text,
    text_field,
)

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

_DECLARED = declared_row(
    per_turn_model=CapLevel.partial,
    reasoning_text=CapLevel.partial,
    timestamps=CapLevel.partial,
    op_timing=CapLevel.partial,
    file_diffs=CapLevel.partial,
    full_tool_output=CapLevel.partial,
    compaction_summaries=CapLevel.partial,
    final_answer_marker=CapLevel.partial,
    thread_linkage=CapLevel.partial,
    context_window=CapLevel.partial,
)


def _connect(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)


def _metadata(db: sqlite3.Connection) -> dict[str, Any]:
    row = db.execute("SELECT value FROM meta WHERE key = '0'").fetchone()
    if row is None or not isinstance(row[0], str):
        return {}
    try:
        return object_field(json.loads(bytes.fromhex(row[0])))
    except ValueError:
        return {}


def store_identity(path: Path) -> tuple[str, str] | None:
    if path.name != "store.db" or not path.is_file():
        return None
    try:
        with closing(_connect(path)) as db:
            meta = _metadata(db)
    except sqlite3.DatabaseError:
        return None
    session_id = string_field(meta.get("agentId"))
    root_id = meta.get("latestRootBlobId")
    return (session_id, root_id) if session_id and isinstance(root_id, str) else None


def _sidecar(path: Path) -> dict[str, Any]:
    try:
        return object_field(json.loads(path.read_bytes()))
    except (OSError, ValueError):
        return {}


def _cursor_options(message: dict[str, Any]) -> dict[str, Any]:
    return object_field(object_field(message.get("providerOptions")).get("cursor"))


def _message_refs(
    db: sqlite3.Connection, fields: Fields, diagnostics: list[dict[str, Any]]
) -> Iterator[tuple[str, str | None]]:
    """Yield message ids and compactions on the current history, including archived prefixes."""
    archive_ids = dict.fromkeys([*blob_ids(fields, 11), *blob_ids(fields, 13)])
    archives = []
    summaries = set()
    for archive_id in archive_ids:
        data = read_blob(db, archive_id)
        if data is None:
            diagnostics.append({"status": "missing_cursor_archive", "item_id": archive_id})
            continue
        try:
            archive = protobuf(data)
        except ValueError:
            diagnostics.append({"status": "unreadable_cursor_archive", "item_id": archive_id})
            continue
        archives.append((archive_id, archive))
        summaries.update(blob_ids(archive, 4))
    for archive_id, archive in archives:
        for message_id in blob_ids(archive, 1):
            if message_id not in summaries:
                yield message_id, None
        yield archive_id, text_field(archive, 2) or ""
    for message_id in blob_ids(fields, 1):
        if message_id not in summaries:
            yield message_id, None


def _tool_outcome(
    db: sqlite3.Connection, message_result: dict[str, Any], block: dict[str, Any], diagnostics: list[dict[str, Any]]
) -> tuple[dict[str, Any], Any, bool | None]:
    structured = object_field(_cursor_options(block).get("highLevelToolCallResult")) or message_result
    outcome = object_field(structured.get("output"))
    is_error = structured.get("isError")
    if not isinstance(is_error, bool):
        is_error = block.get("isError")
    fallback = block.get("result", block.get("output"))
    if fallback is None:
        fallback = block.get("experimental_content")
    # Read results can point to their captured text in another blob.
    success = object_field(outcome.get("success"))
    blob_id = string_field(success.get("contentBlobId"))
    if blob_id:
        captured = text(read_blob(db, blob_id))
        if captured is None:
            diagnostics.append({"status": "unreadable_cursor_tool_output", "item_id": blob_id})
        else:
            outcome = {**outcome, "success": {**success, "content": captured}}
    return outcome, fallback, is_error if isinstance(is_error, bool) else None


def _decode_messages(
    db: sqlite3.Connection,
    refs: Iterator[tuple[str, str | None]],
    origin: SourceRef,
    cwd: str | None,
    timing: CheckpointTiming,
    diagnostics: list[dict[str, Any]],
) -> Iterator[Event]:
    occurrences: Counter[str] = Counter()
    call_rows: dict[str, Event] = {}
    active_model, request_id = None, None
    for index, (blob_id, summary) in enumerate(refs, 1):
        occurrences[blob_id] += 1
        item_id = blob_id if occurrences[blob_id] == 1 else f"{blob_id}:{occurrences[blob_id]}"
        source = origin.model_copy(update={"record_index": index, "item_id": item_id})
        if summary is not None:
            yield Event(kind=EventKind.compaction, source=source, text=summary or None)
            continue
        data = read_blob(db, blob_id)
        try:
            message = object_field(json.loads(data)) if data is not None else {}
        except ValueError:
            message = {}
        if not message:
            diagnostics.append({"status": "unreadable_cursor_message", "item_id": blob_id})
            continue
        role, content = message.get("role"), message.get("content")
        options = _cursor_options(message)
        if role == "user":
            body = flatten_text(content)
            if not body:
                continue
            if options.get("isSummary") is True:
                yield Event(kind=EventKind.compaction, source=source, text=body)
                continue
            native_request = string_field(options.get("requestId"))
            prompt, role_hint = user_prompt(body, has_request=native_request is not None)
            if role_hint is not RoleHint.injected:
                request_id = native_request
            yield Event(
                kind=EventKind.user_msg,
                text=prompt,
                role_hint=role_hint,
                source=source,
                span_id=native_request,
                ts=timing.prompts.get(native_request) if native_request else None,
            )
        elif role == "assistant" and isinstance(content, str):
            if (model := string_field(options.get("modelName"))) and model != active_model:
                active_model = model
                yield Event(kind=EventKind.meta, raw_type="model_change", tool_args={"model": model}, source=source)
            if content:
                start, end = timing.message(request_id, EventKind.assistant_msg, content)
                yield Event(kind=EventKind.assistant_msg, text=content, source=source, ts=start, ts_end=end)
        elif role in ("assistant", "tool") and isinstance(content, list):
            count = sum(object_field(entry).get("type") == "tool-result" for entry in content)
            message_result = object_field(options.get("highLevelToolCallResult")) if count == 1 else {}
            for ordinal, entry in enumerate(content):
                block = object_field(entry)
                block_type = block.get("type")
                ref = source.model_copy(update={"block_ordinal": ordinal})
                model = string_field(_cursor_options(block).get("modelName")) or string_field(options.get("modelName"))
                if role == "assistant" and model and model != active_model:
                    active_model = model
                    yield Event(kind=EventKind.meta, raw_type="model_change", tool_args={"model": model}, source=ref)
                body = string_field(block.get("text"))
                if block_type == "text" and role == "assistant" and body:
                    start, end = timing.message(request_id, EventKind.assistant_msg, body)
                    yield Event(kind=EventKind.assistant_msg, text=body, source=ref, ts=start, ts_end=end)
                elif block_type in ("reasoning", "redacted-reasoning"):
                    start, end = timing.message(request_id, EventKind.thinking, body or "")
                    yield Event(
                        kind=EventKind.thinking,
                        text=body,
                        reasoning_encrypted=block_type == "redacted-reasoning",
                        source=ref,
                        ts=start,
                        ts_end=end,
                    )
                elif block_type == "tool-call" and (name := string_field(block.get("toolName"))):
                    call_id = string_field(block.get("toolCallId"))
                    args = object_field(block.get("args", block.get("input")))
                    start, end = timing.tools.get(call_id or "", (None, None))
                    call = tool_call(name, args, cwd, call_id=call_id, source=ref, ts=start, ts_end=end)
                    if call_id:
                        call_rows[call_id] = call
                    yield call
                elif block_type == "tool-result":
                    name = string_field(block.get("toolName"))
                    call_id = string_field(block.get("toolCallId"))
                    outcome, fallback, is_error = _tool_outcome(db, message_result, block, diagnostics)
                    _, end = timing.tools.get(call_id or "", (None, None))
                    result = tool_result(
                        name, outcome, fallback=fallback, is_error=is_error, call_id=call_id, source=ref, ts=end
                    )
                    if call := call_rows.get(call_id or ""):
                        attach_file_change(call, result, outcome, cwd)
                    yield result


def _finish_native_turns(session: Session, fields: Fields, timing: CheckpointTiming) -> None:
    pending = {text(value) for value in bytes_fields(fields, 4)}
    for index, turn in enumerate(session.turns):
        rows = session.events[turn.event_start : turn.event_end + 1]
        results = {ev.call_id for ev in rows if ev.kind is EventKind.tool_result and ev.call_id}
        missing = any(ev.kind is EventKind.tool_call and ev.call_id not in results for ev in rows)
        request = session.events[turn.event_start].span_id or ""
        reply = timing.final_replies.get(request)
        answer = next(
            (ev for ev in reversed(rows) if reply and ev.kind is EventKind.assistant_msg and ev.text == reply[0]),
            None,
        )
        # UI steps establish the final visible response even when its JSON block precedes reasoning.
        if not missing and reply and answer is not None and (index < len(session.turns) - 1 or not pending):
            answer.phase = "final_answer"
            turn.status = TurnStatus.completed
            turn.final_message = answer.text
            turn.ts_end = reply[1]
            turn.source_end = answer.source
        elif missing or index == len(session.turns) - 1:
            turn.status = TurnStatus.open
        for event in rows:
            event.turn_id = turn.turn_id
    try:
        tokens = next(bytes_fields(fields, 5), None)
        capacity = integer(protobuf(tokens), 2) if tokens is not None else None
    except ValueError:
        capacity = None
    if session.turns and capacity and capacity > 0:
        session.turns[-1].context_window_tokens = capacity


def _read_session(
    db: sqlite3.Connection,
    fields: Fields,
    origin: SourceRef,
    session_id: str,
    cwd: str | None,
    artifact: dict[str, Any],
    *,
    created: Any = None,
    fallback_model: str | None = None,
    parent: str | None = None,
) -> Session | None:
    diagnostics: list[dict[str, Any]] = []
    timing = read_timing(db, fields, diagnostics)
    started = integer(fields, 26)
    timestamp = (parse_ts(started) if started else None) or parse_ts(created)
    events = [
        Event(kind=EventKind.session_start, ts=timestamp, source=origin),
        *_decode_messages(db, _message_refs(db, fields, diagnostics), origin, cwd, timing, diagnostics),
    ]
    visible = [
        event for event in events[1:] if event.kind is not EventKind.meta and event.role_hint is not RoleHint.injected
    ]
    if not visible:
        if diagnostics:
            raise ValueError("Cursor checkpoint has no readable conversation evidence")
        return None
    if parent and (prompt := next((ev for ev in visible if ev.kind is EventKind.user_msg), None)):
        prompt.role_hint = RoleHint.delegation
    for event in visible:
        if event.kind is EventKind.user_msg and event.span_id:
            event.turn_id = f"{session_id}:request:{event.span_id}"
    session = build_session(
        events, AgentKind.cursor, session_id=session_id, parent_session_id=parent, usage_reliable=False
    )
    session.native_session_id_known = True
    session.identity_basis = "native_session_id"
    session.source_artifacts = [artifact.copy()]
    session.source_diagnostics = [{"source_path": origin.path, **diagnostic} for diagnostic in diagnostics]
    session.turns = build_turns(session)
    _finish_native_turns(session, fields, timing)
    session.model = next((ev.tool_args["model"] for ev in events if ev.raw_type == "model_change"), fallback_model)
    session.capabilities = observe_capabilities(session, _DECLARED)
    known_models = sum(bool(turn.model_config_active.model) for turn in session.turns)
    if known_models < len(session.turns):
        session.capabilities.observed.per_turn_model = CapLevel.partial if known_models else CapLevel.false
    return session


def _subagent_states(
    db: sqlite3.Connection, fields: Fields, diagnostics: list[dict[str, Any]]
) -> Iterator[tuple[str, Fields, str]]:
    for field_number in (16, 31):
        for entry in bytes_fields(fields, field_number):
            child_id = None
            try:
                state = protobuf(entry)
                child_id = text_field(state, 1)
                value = next(bytes_fields(state, 2), None)
                if child_id and value is not None:
                    if field_number == 31:
                        value = read_blob(db, value.hex())
                    if value is not None:
                        yield child_id, protobuf(value), f"checkpoint.subagent_state.{field_number}"
                        continue
            except ValueError:
                pass
            diagnostics.append({"status": "unreadable_cursor_subagent", "item_id": child_id})


def _read_children(
    db: sqlite3.Connection, fields: Fields, root: Session, origin: SourceRef, cwd: str | None
) -> tuple[list[Session], list[ThreadLink]]:
    children: list[Session] = []
    links: list[ThreadLink] = []
    seen = {root.session_id}
    pending = [(root.session_id, fields)]
    while pending:
        parent_id, checkpoint = pending.pop()
        for child_id, state, evidence in _subagent_states(db, checkpoint, root.source_diagnostics):
            if child_id in seen:
                continue
            branch = next(bytes_fields(state, 1), None)
            if branch is None:
                root.source_diagnostics.append({"status": "missing_cursor_subagent_history", "item_id": child_id})
                continue
            try:
                child_fields = protobuf(branch)
                child = _read_session(
                    db,
                    child_fields,
                    origin,
                    child_id,
                    cwd,
                    root.source_artifacts[0],
                    created=integer(state, 2),
                    fallback_model=text_field(state, 5),
                    parent=parent_id,
                )
            except ValueError:
                child = None
            if child is None:
                root.source_diagnostics.append({"status": "unreadable_cursor_subagent", "item_id": child_id})
                continue
            seen.add(child_id)
            children.append(child)
            links.append(
                ThreadLink(child_id=child_id, parent_id=parent_id, kind=ThreadLinkKind.spawn, evidence=[evidence])
            )
            pending.append((child_id, child_fields))
    return children, links


def load_store(path: Path) -> Trace | None:
    path = path.resolve()
    sidecar_path = path.with_name("meta.json")
    sidecar = _sidecar(sidecar_path)
    cwd = string_field(sidecar.get("cwd"))
    try:
        with closing(_connect(path)) as db:
            # Pin one snapshot while Cursor may be saving new checkpoints to the WAL.
            db.execute("BEGIN")
            meta = _metadata(db)
            root_id = string_field(meta.get("latestRootBlobId"))
            session_id = string_field(meta.get("agentId"))
            if not root_id or not session_id:
                return None
            checkpoint = read_blob(db, root_id)
            if checkpoint is None:
                raise ValueError("Cursor store references a missing checkpoint")
            fields = protobuf(checkpoint)
            sha = hashlib.sha256(checkpoint).hexdigest()
            origin = SourceRef(record_index=0, path=str(path), source_sha256=sha, item_id=root_id)
            artifact = {
                "source_path": str(path),
                "source_sha256": sha,
                "hash_basis": "cursor_conversation_checkpoint",
                "bytes": path.stat().st_size,
            }
            parent = string_field(object_field(meta.get("subagentInfo")).get("parentAgentId"))
            root = _read_session(
                db,
                fields,
                origin,
                session_id,
                cwd,
                artifact,
                created=meta.get("createdAt"),
                fallback_model=string_field(meta.get("lastUsedModel")),
                parent=parent,
            )
            if root is None:
                return None
            children, links = _read_children(db, fields, root, origin, cwd)
    except sqlite3.DatabaseError as exc:
        raise ValueError(f"Unable to read Cursor chat database: {exc}") from exc
    notes = ["Cursor checkpoints include timing when saved UI steps match message or tool IDs."]
    if root.source_diagnostics:
        notes.append("Some Cursor checkpoint references could not be read.")
    incomplete_linkage = bool(parent or any("subagent" in d["status"] for d in root.source_diagnostics))
    if incomplete_linkage:
        root.degraded.append("thread_linkage")
        root.capabilities.observed.thread_linkage = CapLevel.partial
    elif links:
        root.capabilities.observed.thread_linkage = CapLevel.true
    return Trace(
        trace_id=session_id,
        agent=AgentKind.cursor,
        model=root.model,
        root=root,
        subagents=children,
        links=links,
        capabilities=root.capabilities,
        provenance=Provenance(
            source_format="cursor_native",
            detector_confidence="high",
            root_path=path,
            files=[path, sidecar_path] if sidecar else [path],
            harness_meta={"cwd": cwd},
            thread_linkage="partial" if incomplete_linkage else "full" if links else None,
            notes=notes,
        ),
    )
