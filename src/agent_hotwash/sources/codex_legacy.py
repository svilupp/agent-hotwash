"""Legacy native Codex decoder (CLI 0.142: ``event_msg.user_message`` /
``function_call name=exec_command``). Kept byte-stable for the 0.142 fixture;
also the fallback when a newer file carries no ``item_completed`` at all.
"""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING, Any

from agent_hotwash.events import Event, EventKind, Usage
from agent_hotwash.sources._common import (
    diagnostic_excerpt,
    flatten_text,
    output_metadata,
    parse_ts,
    record_source,
    truncate,
)
from agent_hotwash.sources.codex_items import _EXIT_RE, _parse_args, _patch_paths, _session_meta_fields

if TYPE_CHECKING:
    from collections.abc import Iterable

# ---------------------------------------------------------------------------
# legacy 0.142 path (kept for fixture parity)
# ---------------------------------------------------------------------------


def _exit_from_output_legacy(output: str) -> int | None:
    m = _EXIT_RE.search(output)
    return int(m.group(1)) if m else None


def _response_item(payload: dict[str, Any], ts: Any) -> Event | None:
    ptype = payload.get("type")
    if ptype == "message" and payload.get("role") in ("user", "assistant"):
        return Event(
            kind=EventKind.user_msg if payload["role"] == "user" else EventKind.assistant_msg,
            text=flatten_text(payload.get("content")) or None,
            ts=ts,
            raw_type="message",
        )
    if ptype == "reasoning":
        text = "\n".join(filter(None, (flatten_text(payload.get("summary")), flatten_text(payload.get("content")))))
        return Event(
            kind=EventKind.thinking,
            text=text or None,
            reasoning_encrypted=bool(payload.get("encrypted_content")),
            ts=ts,
            raw_type="reasoning",
        )
    if ptype in ("function_call", "custom_tool_call"):
        name = payload.get("name")
        args = _parse_args(payload)
        path: str | None = None
        if name == "apply_patch":
            body = args.get("raw") if isinstance(args.get("raw"), str) else None
            paths = _patch_paths(body) if body else []
            if paths:
                path = paths[0]
                args = {**args, "path": path, "paths": paths}
        return Event(
            kind=EventKind.tool_call,
            tool_name=name,
            call_id=payload.get("call_id"),
            tool_args=args,
            path=path,
            ts=ts,
            raw_type=ptype,
        )
    if ptype in ("function_call_output", "custom_tool_call_output"):
        output = payload.get("output")
        text = output if isinstance(output, str) else flatten_text(output)
        exit_code = _exit_from_output_legacy(text or "")
        ok = exit_code == 0 if exit_code is not None else "Error:" not in (text or "")
        return Event(
            kind=EventKind.tool_result,
            call_id=payload.get("call_id"),
            ts=ts,
            ok=ok,
            exit_code=exit_code,
            output=truncate(text),
            error_text=truncate(text) if not ok else None,
            diagnostic_excerpt=diagnostic_excerpt(text) if not ok else None,
            **output_metadata(text),
            raw_type=ptype,
        )
    if ptype == "web_search_call":
        return Event(
            kind=EventKind.tool_call, tool_name="web_search", call_id=payload.get("call_id"), ts=ts, raw_type=ptype
        )
    if ptype == "agent_message":
        return Event(
            kind=EventKind.assistant_msg,
            text=flatten_text(payload.get("content")) or payload.get("text"),
            ts=ts,
            raw_type=ptype,
        )
    return None


def _event_msg(payload: dict[str, Any], ts: Any) -> list[Event]:
    ptype = payload.get("type")
    if ptype == "user_message":
        return [Event(kind=EventKind.user_msg, text=payload.get("message"), ts=ts, raw_type=ptype)]
    if ptype == "agent_message":
        return [Event(kind=EventKind.assistant_msg, text=payload.get("message"), ts=ts, raw_type=ptype)]
    if ptype == "token_count":
        info = payload.get("info") or {}
        total = info.get("total_token_usage") or {}
        if total:
            input_tokens = total.get("input_tokens")
            cached = total.get("cached_input_tokens")
            uncached = input_tokens - cached if input_tokens is not None and cached is not None else input_tokens
            return [
                Event(
                    kind=EventKind.meta,
                    ts=ts,
                    raw_type="token_count",
                    usage=Usage(
                        input=uncached,
                        output=total.get("output_tokens"),
                        cache_read=cached,
                        reasoning_output=total.get("reasoning_output_tokens"),
                        cumulative=True,
                    ),
                )
            ]
        return []
    if ptype == "patch_apply_end":
        return [Event(kind=EventKind.meta, ts=ts, raw_type=ptype)]
    if ptype == "context_compacted":
        return [Event(kind=EventKind.compaction, ts=ts, raw_type=ptype)]
    return []


def _decode_legacy(records: Iterable[dict[str, Any]]) -> tuple[list[Event], str | None, str | None, dict[str, Any]]:
    records = list(records)
    mirrored = Counter(
        (r["payload"].get("type"), r["payload"].get("message"))
        for r in records
        if r.get("type") == "event_msg"
        and isinstance(r.get("payload"), dict)
        and r["payload"].get("type") in ("user_message", "agent_message")
        and isinstance(r["payload"].get("message"), str)
    )
    events: list[Event] = []
    session_id: str | None = None
    model: str | None = None
    meta: dict[str, Any] = {}
    for record_index, r in enumerate(records):
        start = len(events)
        rtype = r.get("type")
        payload = r.get("payload")
        if not isinstance(payload, dict):
            continue
        ts = parse_ts(r.get("timestamp"))
        if rtype == "session_meta":
            session_id = payload.get("id") or session_id
            model = model or payload.get("model")
            meta = _session_meta_fields(payload)
        elif rtype == "turn_context":
            model = model or payload.get("model")
            session_id = session_id or payload.get("session_id")
        elif rtype == "response_item":
            ev = _response_item(payload, ts)
            if ev is not None:
                form = "user_message" if ev.kind is EventKind.user_msg else "agent_message"
                mirror = (form, ev.text)
                if payload.get("type") == "message" and mirrored[mirror]:
                    mirrored[mirror] -= 1
                else:
                    events.append(ev)
        elif rtype == "event_msg":
            events.extend(_event_msg(payload, ts))
        elif rtype == "compacted":
            events.append(Event(kind=EventKind.compaction, ts=ts, raw_type="compacted"))
        for ordinal, event in enumerate(events[start:]):
            event.source = record_source(r, record_index, ordinal, payload.get("id"))
    return events, session_id, model, meta
