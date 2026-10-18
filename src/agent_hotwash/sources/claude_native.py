"""Native Claude Code session parser (format B).

Layout: ``~/.claude/projects/<slug>/<session>.jsonl`` plus, for any session that
spawned subagents, ``<session>/subagents/agent-*.jsonl`` with a sibling
``agent-*.meta.json`` (``{agentType, spawnMode, description, toolUseId,
spawnDepth}``; ``toolUseId`` links the child to the parent ``tool_use``).

Records are appended chronologically. Assistant records are split one content
block per line and grouped by ``requestId``; user records (grouped by
``promptId``) carry ``tool_result`` blocks plus a structured top-level
``toolUseResult``. Every assistant/user record carries an ISO-8601 timestamp.

See ``docs/research/format_findings.md`` for the verified field shapes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from agent_hotwash.events import (
    AgentKind,
    Event,
    EventKind,
    Provenance,
    RoleHint,
    Session,
    Trace,
    Usage,
)
from agent_hotwash.sources._common import (
    build_session,
    diagnostic_excerpt,
    flatten_text,
    iter_jsonl,
    output_metadata,
    parse_ts,
    record_source,
    source_artifacts,
    source_diagnostics,
    truncate,
)

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

# Auxiliary line types tolerated (ignored) alongside assistant/user/system.
_AUX_TYPES = {
    "attachment",
    "file-history-snapshot",
    "last-prompt",
    "mode",
    "queue-operation",
    "bridge-session",
    "ai-title",
    "summary",
}


def _decode_records(records: list[dict[str, Any]]) -> list[Event]:
    events: list[Event] = []
    # One API response is split one content block per jsonl line, every line
    # repeating the same ``requestId`` and ``message.usage``. Attach usage only to
    # the first line of each requestId so summed tokens bill each request once
    # (native lines carry the response's real, final usage — unlike code-bench
    # stream-json, so no output reconciliation is needed here).
    seen_request_ids: set[str] = set()
    for record_index, r in enumerate(records):
        start = len(events)
        rtype = r.get("type")
        if rtype == "assistant":
            request_id = r.get("requestId")
            first_of_request = request_id is None or request_id not in seen_request_ids
            if request_id is not None:
                seen_request_ids.add(request_id)
            events.extend(_assistant(r, with_usage=first_of_request))
        elif rtype == "user":
            events.extend(_user(r))
        elif rtype == "system" and r.get("subtype") == "compact_boundary":
            events.append(
                Event(kind=EventKind.compaction, ts=parse_ts(r.get("timestamp")), raw_type="compact_boundary")
            )
        for ordinal, ev in enumerate(events[start:]):
            ev.source = record_source(r, record_index, ordinal, r.get("uuid"))
            if rtype == "user" and (r.get("isMeta") or r.get("isCompactSummary")):
                ev.role_hint = RoleHint.injected
            if rtype == "assistant":
                ev.response_id = r.get("requestId") or r.get("message", {}).get("id")
                if ev.usage is not None:
                    ev.usage_model = r.get("message", {}).get("model")
    return events


def _assistant(r: dict[str, Any], *, with_usage: bool = True) -> list[Event]:
    msg = r.get("message", {})
    ts = parse_ts(r.get("timestamp"))
    usage = _usage(msg.get("usage")) if with_usage else None
    uuid = r.get("uuid")
    parent = r.get("parentUuid")
    out: list[Event] = []
    for block in msg.get("content", []) or []:
        if not isinstance(block, dict):
            continue
        btype = block.get("type")
        if btype == "text":
            ev = Event(kind=EventKind.assistant_msg, text=block.get("text"), ts=ts, raw_type="text")
        elif btype == "thinking":
            ev = Event(
                kind=EventKind.thinking, text=block.get("thinking") or block.get("text"), ts=ts, raw_type="thinking"
            )
        elif btype == "tool_use":
            ev = Event(
                kind=EventKind.tool_call,
                tool_name=block.get("name"),
                call_id=block.get("id"),
                tool_args=block.get("input") or {},
                ts=ts,
                raw_type="tool_use",
            )
        else:
            continue
        ev.span_id = uuid
        ev.parent_span_id = parent
        out.append(ev)
    # Attach per-response usage to the first emitted event of the requestId group.
    if out and usage is not None:
        out[0].usage = usage
    return out


def _user(r: dict[str, Any]) -> list[Event]:
    # ``isMeta`` user records are system-injected context (local-command
    # caveats, cross-session agent-messages, hook output), not real prompts.
    # They only ever carry string content -- never tool_result blocks -- so
    # dropping them keeps user-turn counts and user text honest.
    if r.get("isMeta"):
        return []
    ts = parse_ts(r.get("timestamp"))
    uuid = r.get("uuid")
    parent = r.get("parentUuid")
    tool_use_result = r.get("toolUseResult")
    content = r.get("message", {}).get("content")
    out: list[Event] = []
    if isinstance(content, str):
        if content.strip():
            ev = Event(kind=EventKind.user_msg, text=content, ts=ts)
            ev.span_id, ev.parent_span_id = uuid, parent
            out.append(ev)
        return out
    if not isinstance(content, list):
        return out
    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "tool_result":
            text = flatten_text(block.get("content"))
            is_error = bool(block.get("is_error"))
            exit_code = _exit_from_tool_use_result(tool_use_result)
            ev = Event(
                kind=EventKind.tool_result,
                call_id=block.get("tool_use_id"),
                ts=ts,
                ok=not is_error,
                exit_code=exit_code,
                output=truncate(text),
                error_text=truncate(text) if is_error else None,
                diagnostic_excerpt=diagnostic_excerpt(text) if is_error else None,
                **output_metadata(text),
                raw_type="tool_result",
            )
            ev.span_id, ev.parent_span_id = uuid, parent
            out.append(ev)
        elif block.get("type") == "text":
            ev = Event(kind=EventKind.user_msg, text=block.get("text"), ts=ts)
            ev.span_id, ev.parent_span_id = uuid, parent
            out.append(ev)
    return out


def _exit_from_tool_use_result(tur: Any) -> int | None:
    if isinstance(tur, dict) and isinstance(tur.get("exitCode"), int):
        return tur["exitCode"]
    return None


def _usage(usage: dict[str, Any] | None) -> Usage | None:
    if not usage:
        return None
    return Usage(
        input=usage.get("input_tokens"),
        output=usage.get("output_tokens"),
        cache_read=usage.get("cache_read_input_tokens"),
        cache_write=usage.get("cache_creation_input_tokens"),
    )


def _model_of(records: list[dict[str, Any]]) -> str | None:
    for r in records:
        if r.get("type") == "assistant":
            model = r.get("message", {}).get("model")
            if model:
                return model
    return None


def _load_subagents(session_file: Path, session_id: str, model: str | None) -> list[Session]:
    """Load ``<session>/subagents/agent-*.jsonl`` as linked child Sessions."""
    sub_dir = session_file.with_suffix("") / "subagents"
    if not sub_dir.is_dir():
        return []
    sessions: list[Session] = []
    for jsonl in sorted(sub_dir.glob("agent-*.jsonl")):
        meta_path = jsonl.with_name(jsonl.stem + ".meta.json")
        meta: dict[str, Any] = {}
        if meta_path.is_file():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8", errors="replace"))
            except (OSError, json.JSONDecodeError):
                meta = {}
        records = list(iter_jsonl(jsonl))
        events = _decode_records(records)
        sub_id = meta.get("agentId") or meta.get("toolUseId") or jsonl.stem
        session = build_session(
            events,
            AgentKind.claude,
            session_id=f"{session_id}:{sub_id}",
            model=model or _model_of(records),
            parent_session_id=session_id,
        )
        session.native_session_id_known = True
        session.identity_basis = "native_child_id"
        session.source_artifacts = source_artifacts(records)
        session.source_diagnostics = source_diagnostics(records)
        sessions.append(session)
    return sessions


def load_session_file(session_file: Path, *, alternate_session_files: list[Path] | None = None) -> Trace:
    """Parse one native Claude session file (+ its subagents) into a Trace."""
    records = list(iter_jsonl(session_file))
    events = _decode_records(records)
    session_id = next(
        (sid for r in records if isinstance(sid := r.get("sessionId"), str) and sid),
        session_file.stem,
    )
    model = _model_of(records)
    root = build_session(events, AgentKind.claude, session_id=session_id, model=model)
    root.native_session_id_known = any(isinstance(r.get("sessionId"), str) and r["sessionId"] for r in records)
    root.identity_basis = "native_session_id" if root.native_session_id_known else "source_filename_fallback"
    root.source_artifacts = source_artifacts(records)
    root.source_diagnostics = source_diagnostics(records)
    children: dict[str, tuple[Session, bytes]] = {}
    for parent_file in [session_file, *(alternate_session_files or [])]:
        for child in _load_subagents(parent_file, session_id, model):
            raw = Path(child.source_artifacts[0]["source_path"]).read_bytes() if child.source_artifacts else b""
            previous = children.get(child.session_id)
            if previous is None:
                children[child.session_id] = (child, raw)
                continue
            existing, existing_raw = previous
            if len(raw) > len(existing_raw):
                selected, selected_raw, alternate, alternate_raw = child, raw, existing, existing_raw
            else:
                selected, selected_raw, alternate, alternate_raw = existing, existing_raw, child, raw
            if not selected_raw.startswith(alternate_raw) or (
                selected_raw != alternate_raw and alternate_raw and not alternate_raw.endswith(b"\n")
            ):
                raise ValueError(f"Conflicting Claude child exports for {child.session_id}")
            selected.source_artifacts.extend(alternate.source_artifacts)
            selected.source_diagnostics.extend(alternate.source_diagnostics)
            for event, alternate_event in zip(selected.events, alternate.events, strict=False):
                if alternate_event.source is not None:
                    event.alternate_sources.append(alternate_event.source)
                event.alternate_sources.extend(alternate_event.alternate_sources)
            children[child.session_id] = (selected, selected_raw)
    subagents = [child for child, _raw in children.values()]

    provenance = Provenance(
        source_format="claude_native",
        detector_confidence="high",
        root_path=session_file,
        files=[session_file, *(Path(a["source_path"]) for child in subagents for a in child.source_artifacts)],
        notes=[],
    )
    return Trace(
        trace_id=session_id,
        agent=AgentKind.claude,
        model=model,
        root=root,
        subagents=subagents,
        provenance=provenance,
    )


def iter_project_dir(project_dir: Path) -> Iterator[Trace]:
    """Yield one Trace per top-level ``<session>.jsonl`` in a project dir."""
    for session_file in sorted(project_dir.glob("*.jsonl")):
        yield load_session_file(session_file)


def looks_like_native_claude(path: Path) -> bool:
    """True when a ``.jsonl`` file's records carry the uuid/parentUuid shape."""
    if not path.is_file() or path.suffix != ".jsonl":
        return False
    for r in iter_jsonl(path):
        if r.get("type") in ("assistant", "user") and "uuid" in r and "parentUuid" in r:
            return True
        if r.get("type") in ("assistant", "user"):
            return False
    return False
