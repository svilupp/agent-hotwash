"""Native Codex rollout parser (format C).

Layout: ``~/.codex/sessions/YYYY/MM/DD/rollout-*.jsonl``. Every line is a
``{timestamp, type, payload}`` envelope with a top-level ISO-8601 timestamp.
Line types: ``session_meta`` (first; session id key is ``id``, not
``session_id``), ``turn_context`` (carries ``model``), ``response_item``,
``event_msg``, ``compacted``.

Messages are taken from ``event_msg`` (``user_message`` / ``agent_message``) to
avoid double-counting the ``response_item`` ``message`` echoes; reasoning and
tool calls come from ``response_item``. Tool output is an unstructured string —
the exit code is recovered by parsing ``Process exited with code N``.

See ``docs/research/format_findings.md`` for the verified field shapes.
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any

from agent_hotwash.events import (
    AgentKind,
    Event,
    EventKind,
    Provenance,
    Trace,
    Usage,
)
from agent_hotwash.sources._common import (
    build_session,
    flatten_text,
    iter_jsonl,
    parse_ts,
    truncate,
)

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

_EXIT_RE = re.compile(r"(?:Process |Command )?exited with code\s+(\d+)")


def _exit_from_output(output: str) -> int | None:
    m = _EXIT_RE.search(output)
    return int(m.group(1)) if m else None


def decode_codex_native(records: list[dict[str, Any]]) -> tuple[list[Event], str | None, str | None]:
    """Decode rollout records into (events, session_id, model)."""
    events: list[Event] = []
    session_id: str | None = None
    model: str | None = None

    for r in records:
        rtype = r.get("type")
        payload = r.get("payload")
        if not isinstance(payload, dict):
            continue
        ts = parse_ts(r.get("timestamp"))

        if rtype == "session_meta":
            session_id = payload.get("id") or session_id
            model = model or payload.get("model")
        elif rtype == "turn_context":
            model = model or payload.get("model")
            session_id = session_id or payload.get("session_id")
        elif rtype == "response_item":
            ev = _response_item(payload, ts)
            if ev is not None:
                events.append(ev)
        elif rtype == "event_msg":
            events.extend(_event_msg(payload, ts))
        elif rtype == "compacted":
            events.append(Event(kind=EventKind.compaction, ts=ts, raw_type="compacted"))
    return events, session_id, model


def _response_item(payload: dict[str, Any], ts: Any) -> Event | None:
    ptype = payload.get("type")
    if ptype == "reasoning":
        text = flatten_text(payload.get("summary")) or flatten_text(payload.get("content"))
        return Event(kind=EventKind.thinking, text=text or None, ts=ts, raw_type="reasoning")
    if ptype in ("function_call", "custom_tool_call"):
        return Event(
            kind=EventKind.tool_call,
            tool_name=payload.get("name"),
            call_id=payload.get("call_id"),
            tool_args=_parse_args(payload),
            ts=ts,
            raw_type=ptype,
        )
    if ptype in ("function_call_output", "custom_tool_call_output"):
        output = payload.get("output")
        text = output if isinstance(output, str) else flatten_text(output)
        exit_code = _exit_from_output(text or "")
        # Trust an explicit exit code when the output carries one; only fall back
        # to the "Error:" substring heuristic when no exit code was parsed.
        ok = exit_code == 0 if exit_code is not None else "Error:" not in (text or "")
        return Event(
            kind=EventKind.tool_result,
            call_id=payload.get("call_id"),
            ts=ts,
            ok=ok,
            exit_code=exit_code,
            output=truncate(text),
            error_text=truncate(text) if not ok else None,
            raw_type=ptype,
        )
    if ptype == "web_search_call":
        return Event(
            kind=EventKind.tool_call, tool_name="web_search", call_id=payload.get("call_id"), ts=ts, raw_type=ptype
        )
    return None


def _parse_args(payload: dict[str, Any]) -> dict[str, Any]:
    args = payload.get("arguments") or payload.get("input")
    if isinstance(args, str):
        try:
            parsed = json.loads(args)
        except json.JSONDecodeError:
            return {"raw": args}
        return parsed if isinstance(parsed, dict) else {"value": parsed}
    return args if isinstance(args, dict) else {}


def _event_msg(payload: dict[str, Any], ts: Any) -> list[Event]:
    ptype = payload.get("type")
    if ptype == "user_message":
        return [Event(kind=EventKind.user_msg, text=payload.get("message"), ts=ts, raw_type=ptype)]
    if ptype == "agent_message":
        return [Event(kind=EventKind.assistant_msg, text=payload.get("message"), ts=ts, raw_type=ptype)]
    if ptype == "token_count":
        # Verified against a real rollout (121 token_count events): summing the
        # per-event `last_token_usage` slightly over-counts (15,940,062 vs the
        # final `total_token_usage.total_tokens` of 15,904,536), so we take the
        # cumulative `total_token_usage` and let `de_cumulate` derive exact,
        # summable per-event deltas. Codex `input_tokens` is inclusive of
        # `cached_input_tokens`, so cached is subtracted out of `input` to avoid
        # double-counting cache in the token total (30.5M -> 15.9M).
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
                        cumulative=True,
                    ),
                )
            ]
        return []
    if ptype == "patch_apply_end":
        # The apply_patch tool already surfaces as a custom_tool_call +
        # custom_tool_call_output pair; keep this as a meta event so the patch
        # result is not double-counted.
        return [Event(kind=EventKind.meta, ts=ts, raw_type=ptype)]
    if ptype == "context_compacted":
        return [Event(kind=EventKind.compaction, ts=ts, raw_type=ptype)]
    return []


def load_rollout(path: Path) -> Trace:
    """Parse one native Codex rollout file into a Trace."""
    records = list(iter_jsonl(path))
    events, session_id, model = decode_codex_native(records)
    session_id = session_id or path.stem
    root = build_session(events, AgentKind.codex, session_id=session_id, model=model)
    provenance = Provenance(
        source_format="codex_native",
        detector_confidence="high",
        root_path=path,
        files=[path],
        notes=[],
    )
    return Trace(
        trace_id=session_id,
        agent=AgentKind.codex,
        model=model,
        root=root,
        provenance=provenance,
    )


def iter_sessions_tree(root: Path) -> Iterator[Trace]:
    """Yield a Trace per ``rollout-*.jsonl`` under a codex sessions tree/date dir."""
    for path in sorted(root.rglob("rollout-*.jsonl")):
        yield load_rollout(path)


def looks_like_native_codex(path: Path) -> bool:
    """True when the first record is a ``{timestamp,type,payload}`` session_meta."""
    if not path.is_file() or path.suffix != ".jsonl":
        return False
    for r in iter_jsonl(path):
        return r.get("type") == "session_meta" and isinstance(r.get("payload"), dict)
    return False
