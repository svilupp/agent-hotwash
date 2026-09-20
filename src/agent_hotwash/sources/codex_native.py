"""Native Codex rollout parser (format C).

Targets both the legacy 0.142 surface (``event_msg.user_message`` /
``function_call name=exec_command``) and the 0.150-0.155 surface
(``item_completed``-first, ``token_usage_record``, wrapper ``exec`` scripts).

Decision C1: one decoder, ``item_completed``-first; the legacy path is used
only when no ``item_completed`` record exists (or ``cli_version < 0.150``).
"""

from __future__ import annotations

import itertools
import re
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from agent_hotwash.canonical import declared_row, observe_capabilities
from agent_hotwash.events import (
    AgentKind,
    CapLevel,
    Event,
    EventKind,
    ModelCall,
    ModelConfig,
    Provenance,
    RoleHint,
    SourceRef,
    Trace,
    Turn,
    TurnStatus,
    Usage,
    UserInput,
)
from agent_hotwash.sources._common import (
    build_session,
    flatten_text,
    iter_jsonl,
    parse_ts,
    tool_category_of_op,
    truncate,
    truncate_head_tail,
)
from agent_hotwash.sources.codex_items import (
    _AGENT_FN,
    _DELEGATION_TAG,
    _flatten_output,
    _inter_agent_message,
    _item_to_events,
    _parse_args,
    _role_hint,
    _session_meta_fields,
    _source_thread_id,
)
from agent_hotwash.sources.codex_legacy import _decode_legacy

if TYPE_CHECKING:
    from collections.abc import Iterable
    from datetime import datetime
    from pathlib import Path

_NO_ITEMS_NOTE = "no item_completed; decoder should have used legacy"
# A ``compacted`` record and its ``ContextCompaction`` item describe one event
# (observed 5 ms apart on real rollouts). Pair them only when both close in the
# event stream *and* in time, so two genuine compactions are never collapsed.
_COMPACTION_PAIR_GAP = 12
_COMPACTION_PAIR_MAX_S = 60.0

_DECLARED_V2 = declared_row(
    per_call_usage=True,
    per_turn_model=True,
    reasoning_effort=True,
    reasoning_text=False,
    reasoning_tokens=True,
    timestamps=True,
    op_timing=True,
    parsed_commands=True,
    file_diffs=True,
    full_tool_output=CapLevel.partial,
    output_size_original=True,
    thread_linkage=True,
    compaction_summaries=CapLevel.partial,
    final_answer_marker=True,
    context_window=True,
)

_DECLARED_LEGACY = declared_row(
    per_call_usage=CapLevel.partial,
    per_turn_model=True,
    reasoning_effort=False,
    reasoning_text=True,
    reasoning_tokens=CapLevel.partial,
    timestamps=True,
    op_timing=False,
    parsed_commands=False,
    file_diffs=False,
    full_tool_output=CapLevel.partial,
    output_size_original=False,
    thread_linkage=False,
    compaction_summaries=CapLevel.partial,
    final_answer_marker=False,
    context_window=True,
)


def _parse_cli_version(raw: str | None) -> tuple[int, int, int]:
    if not raw:
        return (0, 0, 0)
    m = re.search(r"(\d+)\.(\d+)\.(\d+)", raw)
    if not m:
        return (0, 0, 0)
    return (int(m.group(1)), int(m.group(2)), int(m.group(3)))


def _usage_from_record(usage: dict[str, Any]) -> Usage:
    input_tokens = usage.get("input_tokens")
    cached = usage.get("cached_input_tokens") or 0
    uncached = input_tokens - cached if input_tokens is not None else None
    return Usage(
        input=uncached,
        output=usage.get("output_tokens"),
        cache_read=cached if input_tokens is not None else usage.get("cached_input_tokens"),
        cache_write=usage.get("cache_write_input_tokens"),
        reasoning_output=usage.get("reasoning_output_tokens"),
        cumulative=False,
    )


def _source(
    record_index: int,
    payload: dict[str, Any],
    item_id: str | None = None,
    *,
    envelope_ordinal: int | None = None,
) -> SourceRef:
    ordinal: int | None = envelope_ordinal if isinstance(envelope_ordinal, int) else payload.get("ordinal")
    if not isinstance(ordinal, int):
        ordinal = None
    return SourceRef(record_index=record_index, ordinal=ordinal, item_id=item_id)


def _model_config_from_turn_context(payload: dict[str, Any], provider: str | None) -> ModelConfig:
    collab = payload.get("collaboration_mode")
    mode = None
    if isinstance(collab, dict):
        mode = collab.get("mode")
    elif isinstance(collab, str):
        mode = collab
    return ModelConfig(
        provider=provider,
        model=payload.get("model"),
        reasoning_effort=payload.get("effort") or payload.get("reasoning_effort"),
        collaboration_mode=mode,
        service_tier=payload.get("service_tier"),
    )


def _decode_v2(
    records: Iterable[dict[str, Any]],
) -> tuple[list[Event], str | None, str | None, dict[str, Any], list[Turn], list[str]]:
    events: list[Event] = []
    notes: list[str] = []
    session_id: str | None = None
    model: str | None = None
    provider: str | None = None
    meta: dict[str, Any] = {}
    replay_start: int | None = None
    seen_second_meta = False
    open_group: str | None = None
    seen_usage_record = False
    seen_item_completed = False

    turns: dict[str, Turn] = {}
    turn_order: list[str] = []
    current_call_events: list[Event] = []
    current_call_start_idx = 0
    # agent function_call events by call_id, so the matching CollabAgentToolCall
    # item enriches instead of duplicating them
    agent_calls: dict[str, Event] = {}
    # index of the last ``compacted`` record's event (for ContextCompaction dedupe)
    last_compacted_idx: int | None = None
    last_compacted_ts: datetime | None = None
    # (event index, event, usage) from token_count records — used only when the
    # file has no token_usage_record at all
    token_count_usage: list[tuple[int, Event, Usage]] = []
    usage_by_response: dict[str, Usage] = {}
    first_thread_usage: dict[str, int] | None = None
    last_thread_usage: dict[str, int] | None = None
    first_usage_delta: Usage | None = None

    def _ensure_turn(turn_id: str | None, ts: Any, src: SourceRef | None = None) -> Turn | None:
        if not turn_id:
            return None
        if turn_id not in turns:
            turns[turn_id] = Turn(
                turn_id=turn_id,
                session_id=session_id or "",
                source_start=src,
                event_start=len(events),
                ts_start=ts,
                status=TurnStatus.open,
            )
            turn_order.append(turn_id)
        t = turns[turn_id]
        t.event_end = len(events)
        t.ts_end = ts or t.ts_end
        return t

    def _close_call(response_id: str | None, usage: Usage | None, turn_id: str | None, ts: Any) -> None:
        nonlocal current_call_events, current_call_start_idx
        if not current_call_events and usage is None:
            current_call_start_idx = len(events)
            return
        for ev in current_call_events:
            ev.response_id = response_id
            if turn_id:
                ev.turn_id = ev.turn_id or turn_id
        t = turns.get(turn_id) if turn_id else None
        if t is not None:
            t.model_calls.append(
                ModelCall(
                    response_id=response_id,
                    turn_id=turn_id,
                    event_start=current_call_start_idx,
                    event_end=max(current_call_start_idx, len(events) - 1),
                    usage=usage,
                    ts_start=current_call_events[0].ts if current_call_events else ts,
                    ts_end=ts,
                )
            )
        current_call_events = []
        current_call_start_idx = len(events)

    replay_until_own_thread = False

    def _in_replay(ordinal: int, payload: dict[str, Any]) -> bool:
        """Spawned children replay the parent's history before their own records.

        Only ``subagent_history_start_ordinal`` (or a second ``session_meta``)
        marks such a prefix. ``history_base.end_ordinal_exclusive`` on forks is
        an ordinal *in the parent's file* — fork files restart at 0 and carry no
        replayed records — so it must never be used as a cutoff here.
        """
        nonlocal replay_until_own_thread
        if replay_start is None and not seen_second_meta and not replay_until_own_thread:
            return False
        if replay_start is not None:
            return ordinal < replay_start
        # Second parent session_meta without an ordinal gate: skip until this
        # session's own thread_id appears on an item or usage record.
        if seen_second_meta or replay_until_own_thread:
            tid = payload.get("thread_id")
            if session_id and tid == session_id:
                replay_until_own_thread = False
                return False
            return True
        return False

    for record_index, r in enumerate(records):
        rtype = r.get("type")
        payload = r.get("payload")
        if not isinstance(payload, dict):
            # tolerate envelope-less item? skip
            if not isinstance(r, dict):
                continue
            payload = r
        ts = parse_ts(r.get("timestamp") or payload.get("timestamp"))
        # ``ordinal`` lives on the record envelope (0.15x); fall back to position.
        ordinal = r.get("ordinal")
        if not isinstance(ordinal, int):
            ordinal = record_index
        src = _source(record_index, payload, envelope_ordinal=ordinal)

        if rtype == "session_meta":
            if session_id is None:
                session_id = payload.get("id")
                meta = _session_meta_fields(payload)
                provider = payload.get("model_provider")
                start = payload.get("subagent_history_start_ordinal")
                if isinstance(start, int):
                    replay_start = start
            else:
                seen_second_meta = True
                if replay_start is None:
                    replay_until_own_thread = True
                notes.append("replay prefix: second session_meta")
            continue

        if _in_replay(ordinal, payload):
            # retain a capped compaction summary only (as meta, so it is not
            # counted as one of *this* thread's compactions); no turns/ops/spend
            if rtype == "compacted":
                msg = payload.get("message")
                events.append(
                    Event(
                        kind=EventKind.meta,
                        text=truncate(msg if isinstance(msg, str) else None, 400),
                        ts=ts,
                        source=src,
                        raw_type="compacted_replay",
                    )
                )
            continue

        if rtype == "turn_context":
            turn_id = payload.get("turn_id")
            cfg = _model_config_from_turn_context(payload, provider)
            model = model or cfg.model
            t = _ensure_turn(turn_id, ts, src)
            if t is not None:
                t.model_config_revisions.append(cfg)
                t.model_config_active = cfg
                cw = payload.get("model_context_window") or payload.get("context_window")
                if isinstance(cw, int):
                    t.context_window_tokens = cw
            events.append(Event(kind=EventKind.meta, ts=ts, source=src, turn_id=turn_id, raw_type="turn_context"))
            continue

        if rtype == "token_usage_record":
            thread_id = payload.get("thread_id")
            if session_id and thread_id and thread_id != session_id:
                notes.append("dropped usage record with foreign thread_id")
                continue
            seen_usage_record = True
            rid = payload.get("response_id")
            usage_raw = payload.get("usage") or {}
            usage = _usage_from_record(usage_raw) if usage_raw else None
            if rid and usage is not None:
                usage_by_response[rid] = usage
            turn_id = payload.get("turn_id")
            _ensure_turn(turn_id, ts, src)
            tu = payload.get("thread_token_usage")
            if isinstance(tu, dict):
                if first_thread_usage is None:
                    first_thread_usage = tu
                    first_usage_delta = usage
                last_thread_usage = tu
            ev = Event(
                kind=EventKind.meta,
                ts=ts,
                source=src,
                turn_id=turn_id,
                response_id=rid,
                usage=usage,
                raw_type="token_usage_record",
            )
            events.append(ev)
            _close_call(rid, usage, turn_id, ts)
            continue

        if rtype == "event_msg":
            etype = payload.get("type")
            turn_id = payload.get("turn_id")
            if etype == "task_started":
                t = _ensure_turn(turn_id, ts, src)
                cw = payload.get("model_context_window")
                if t is not None and isinstance(cw, int):
                    t.context_window_tokens = cw
                events.append(Event(kind=EventKind.meta, ts=ts, source=src, turn_id=turn_id, raw_type="task_started"))
                continue
            if etype == "task_complete":
                t = _ensure_turn(turn_id, ts, src)
                if t is not None:
                    t.status = TurnStatus.completed
                    t.final_message = payload.get("last_agent_message")
                    t.ts_end = ts
                events.append(Event(kind=EventKind.meta, ts=ts, source=src, turn_id=turn_id, raw_type="task_complete"))
                continue
            if etype == "turn_aborted":
                t = _ensure_turn(turn_id, ts, src)
                if t is not None:
                    t.status = TurnStatus.aborted
                    t.ts_end = ts
                events.append(Event(kind=EventKind.meta, ts=ts, source=src, turn_id=turn_id, raw_type="turn_aborted"))
                continue
            if etype == "thread_settings_applied":
                events.append(
                    Event(kind=EventKind.meta, ts=ts, source=src, turn_id=turn_id, raw_type="thread_settings_applied")
                )
                continue
            if etype == "token_count":
                # Billing uses token_usage_record. Keep a marker; remember the
                # per-response ``last_token_usage`` so files that never carry
                # usage records (0.150-alpha) can still be priced (see below).
                ev = Event(kind=EventKind.meta, ts=ts, source=src, turn_id=turn_id, raw_type="token_count")
                events.append(ev)
                info = payload.get("info")
                last = info.get("last_token_usage") if isinstance(info, dict) else None
                if isinstance(last, dict) and last:
                    token_count_usage.append((len(events) - 1, ev, _usage_from_record(last)))
                continue
            if etype == "item_completed":
                seen_item_completed = True
                item = payload.get("item") or {}
                if not isinstance(item, dict):
                    continue
                item_ts = parse_ts(payload.get("started_at_ms")) or ts
                item_src = _source(record_index, payload, item.get("id"), envelope_ordinal=ordinal)
                itype = item.get("type")
                if itype == "CollabAgentToolCall" and item.get("id") in agent_calls:
                    # Same call already emitted from its ``function_call``; just
                    # enrich it with the item's resolved receivers.
                    existing = agent_calls[item["id"]]
                    existing.tool_args = {
                        **(existing.tool_args or {}),
                        "sender_thread_id": item.get("sender_thread_id"),
                        "receiver_thread_ids": item.get("receiver_thread_ids"),
                    }
                    continue
                if itype == "ContextCompaction":
                    # The ``compacted`` record (which carries the summary) and this
                    # item describe the same compaction; count it once.
                    near_idx = (
                        last_compacted_idx is not None and len(events) - last_compacted_idx <= _COMPACTION_PAIR_GAP
                    )
                    # Compare *envelope* times: the item's started_at_ms is when
                    # the compaction began, minutes before the record is written.
                    near_ts = (
                        last_compacted_ts is None
                        or ts is None
                        or abs((ts - last_compacted_ts).total_seconds()) <= _COMPACTION_PAIR_MAX_S
                    )
                    if near_idx and near_ts:
                        last_compacted_idx = None
                        last_compacted_ts = None
                        events.append(
                            Event(kind=EventKind.meta, ts=item_ts, source=item_src, turn_id=turn_id, raw_type=itype)
                        )
                        continue
                emitted = _item_to_events(item, item_ts, item_src, turn_id or payload.get("turn_id"), open_group)
                t = _ensure_turn(turn_id or payload.get("turn_id"), item_ts, item_src)
                if t is not None and itype == "UserMessage":
                    text = flatten_text(item.get("content"))
                    hint = _role_hint(text)
                    if hint in (RoleHint.user, RoleHint.delegation) and t.user_input.kind == "none":
                        t.user_input = UserInput(
                            text=text or "",
                            kind=hint.value,
                            source_thread_id=_source_thread_id(text or "") if hint is RoleHint.delegation else None,
                        )
                    if hint is RoleHint.delegation and text:
                        sid = _source_thread_id(text)
                        if sid:
                            t.user_input.source_thread_id = sid
                if t is not None and itype == "FunctionCallOutput" and item.get("name") == "create_thread":
                    # agent_created_thread children learn their parent from the
                    # <codex_delegation> block echoed in this output.
                    sid = _source_thread_id(_flatten_output(item.get("output")))
                    if sid and sid != session_id:
                        if t.user_input.kind == "none":
                            t.user_input = UserInput(text="", kind="delegation", source_thread_id=sid)
                        elif t.user_input.source_thread_id is None:
                            t.user_input.source_thread_id = sid
                if t is not None and itype == "AgentMessage" and item.get("phase") == "final_answer":
                    t.final_message = flatten_text(item.get("content")) or t.final_message
                if t is not None and itype == "ContextCompaction":
                    t.compactions += 1
                for ev in emitted:
                    ev.group_id = ev.group_id or open_group
                    events.append(ev)
                    current_call_events.append(ev)
                continue
            continue

        if rtype == "response_item":
            ptype = payload.get("type")
            if ptype == "custom_tool_call" and payload.get("name") == "exec":
                if open_group is not None:
                    notes.append("malformed wrapper nesting: close on next open")
                open_group = payload.get("call_id") or payload.get("id")
                inp = payload.get("input")
                events.append(
                    Event(
                        kind=EventKind.meta,
                        text=truncate(_flatten_output(inp), 400),
                        call_id=open_group,
                        group_id=open_group,
                        ts=ts,
                        source=src,
                        raw_type="custom_tool_call",
                    )
                )
                continue
            if ptype == "custom_tool_call_output":
                cid = payload.get("call_id")
                events.append(
                    Event(
                        kind=EventKind.meta,
                        output=truncate_head_tail(_flatten_output(payload.get("output")), 400),
                        call_id=cid,
                        group_id=cid,
                        ts=ts,
                        source=src,
                        raw_type="custom_tool_call_output",
                    )
                )
                if cid == open_group or open_group is None:
                    open_group = None
                continue
            if ptype == "function_call":
                name = payload.get("name") or ""
                op = _AGENT_FN.get(name)
                if op:
                    ev = Event(
                        kind=EventKind.tool_call,
                        tool_name=op,
                        tool_category=tool_category_of_op(op),
                        op_kind=op,
                        tool_args=_parse_args(payload),
                        call_id=payload.get("call_id"),
                        ts=ts,
                        source=src,
                        group_id=open_group,
                        raw_type="function_call",
                    )
                    events.append(ev)
                    current_call_events.append(ev)
                    if isinstance(ev.call_id, str):
                        agent_calls[ev.call_id] = ev
                continue
            if ptype == "agent_message":
                if isinstance(payload.get("author"), str) and isinstance(payload.get("recipient"), str):
                    ev, is_task = _inter_agent_message(payload, ts, src, open_group, meta.get("agent_path"))
                    events.append(ev)
                    current_call_events.append(ev)
                    if is_task:
                        pt = payload.get("internal_chat_message_metadata_passthrough")
                        tid = pt.get("turn_id") if isinstance(pt, dict) else None
                        t = _ensure_turn(tid, ts, src)
                        if t is not None and t.user_input.kind == "none":
                            t.user_input = UserInput(
                                text=ev.text or "",
                                kind="delegation",
                                source_thread_id=meta.get("parent_thread_id"),
                            )
                elif not seen_item_completed:
                    ev = Event(
                        kind=EventKind.assistant_msg,
                        text=flatten_text(payload.get("content")) or payload.get("text"),
                        ts=ts,
                        source=src,
                        group_id=open_group,
                        raw_type="agent_message",
                    )
                    events.append(ev)
                    current_call_events.append(ev)
                continue
            continue

        if rtype == "compacted":
            hist = payload.get("replacement_history")
            n_hist = len(hist) if isinstance(hist, list) else 0
            last = None
            if isinstance(hist, list) and hist:
                last_entry = hist[-1]
                if isinstance(last_entry, dict):
                    last = last_entry.get("text") or last_entry.get("message")
                elif isinstance(last_entry, str):
                    last = last_entry
            ev = Event(
                kind=EventKind.compaction,
                text=truncate(last if isinstance(last, str) else payload.get("message"), 400),
                tool_args={"replacement_count": n_hist},
                ts=ts,
                source=src,
                raw_type="compacted",
            )
            last_compacted_idx = len(events)
            last_compacted_ts = ts
            events.append(ev)
            current_call_events.append(ev)
            if turn_order:
                turns[turn_order[-1]].compactions += 1
            continue

        if rtype in ("world_state", "inter_agent_communication_metadata"):
            continue

    # unterminated trailing call
    if current_call_events:
        last_tid = current_call_events[-1].turn_id
        _close_call(None, None, last_tid, current_call_events[-1].ts)

    # Files without any token_usage_record (0.150-alpha): fall back to the
    # per-response ``token_count.last_token_usage`` so spend is not None.
    # Compaction responses report 0 there, so this is a lower bound.
    if not seen_usage_record and token_count_usage:
        notes.append("usage from token_count (no token_usage_record)")
        for t in turns.values():
            t.model_calls = []  # only the usage-less trailing close exists here
        prev_end = -1
        for idx, ev, usage in token_count_usage:
            ev.usage = usage
            t = turns.get(ev.turn_id or "") if ev.turn_id else None
            if t is None and turn_order:
                t = turns[turn_order[-1]]
            if t is not None:
                t.model_calls.append(
                    ModelCall(
                        response_id=None,
                        turn_id=t.turn_id,
                        event_start=prev_end + 1,
                        event_end=idx,
                        usage=usage,
                        ts_start=events[prev_end + 1].ts if prev_end + 1 < len(events) else ev.ts,
                        ts_end=ev.ts,
                    )
                )
            prev_end = idx

    # Turn event ranges: derive from the events that carry each turn_id so the
    # last event of an open/terminal turn is included (``_ensure_turn`` runs
    # before the item's events are appended).
    span: dict[str, tuple[int, int]] = {}
    for idx, ev in enumerate(events):
        if ev.turn_id and ev.turn_id in turns:
            lo, hi = span.get(ev.turn_id, (idx, idx))
            span[ev.turn_id] = (min(lo, idx), max(hi, idx))
    for tid, (lo, hi) in span.items():
        t = turns[tid]
        t.event_start = min(t.event_start, lo)
        t.event_end = max(t.event_end, hi)

    # usage identity
    if first_thread_usage and last_thread_usage and first_usage_delta is not None:

        def _field(d: dict[str, int], key: str) -> int:
            return int(d.get(key) or 0)

        baseline = {
            "input_tokens": _field(first_thread_usage, "input_tokens")
            - (first_usage_delta.input or 0)
            - (first_usage_delta.cache_read or 0),
            "output_tokens": _field(first_thread_usage, "output_tokens") - (first_usage_delta.output or 0),
        }
        summed_in = sum((u.input or 0) + (u.cache_read or 0) for u in usage_by_response.values())
        summed_out = sum(u.output or 0 for u in usage_by_response.values())
        expect_in = _field(last_thread_usage, "input_tokens") - baseline["input_tokens"]
        expect_out = _field(last_thread_usage, "output_tokens") - baseline["output_tokens"]
        if summed_in != expect_in or summed_out != expect_out:
            notes.append("usage_identity")

    # reasoning subset identity
    for u in usage_by_response.values():
        if u.reasoning_output is not None and u.output is not None and u.reasoning_output > u.output:
            notes.append("usage_identity")
            break

    # stamp session_id on turns created before we knew it
    for t in turns.values():
        if not t.session_id and session_id:
            t.session_id = session_id
        if t.model_config_active.model:
            model = model or t.model_config_active.model

    if not seen_item_completed:
        notes.append(_NO_ITEMS_NOTE)

    return events, session_id, model, meta, [turns[k] for k in turn_order], notes


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------


def decode_codex_native(records: list[dict[str, Any]]) -> tuple[list[Event], str | None, str | None]:
    """Decode rollout records into (events, session_id, model).

    Legacy signature kept for existing tests. Prefer ``load_rollout``.
    """
    events, session_id, model, _meta, _turns, _notes = decode_codex_native_full(records)
    return events, session_id, model


def _mode_of(record: dict[str, Any]) -> str | None:
    """``"v2"``/``"legacy"`` once a record settles the surface, else ``None``."""
    payload = record.get("payload")
    if not isinstance(payload, dict):
        return None
    if record.get("type") == "session_meta":
        version = _parse_cli_version(payload.get("cli_version"))
        if version >= (0, 150):
            return "v2"
        # A missing/unparseable version must not commit to legacy: wait for
        # the first item_completed (v2) or the end of the stream (legacy).
        return "legacy" if version > (0, 0, 0) else None
    if record.get("type") == "event_msg" and payload.get("type") == "item_completed":
        return "v2"
    return None


def decode_codex_native_full(
    records: Iterable[dict[str, Any]],
) -> tuple[list[Event], str | None, str | None, dict[str, Any], list[Turn], list[str]]:
    """Full decode: events + session meta + turns + notes.

    Streams ``records`` once. Chooses v2 when the first ``session_meta`` reports
    ``cli_version >= 0.150`` *or* an ``item_completed`` is observed; otherwise
    the 0.142 path (byte-stable on the existing fixture). A header with no
    parseable version stays undecided until an ``item_completed`` appears
    (v2) or the stream ends (legacy). When a ≥0.150 file turns out to carry no
    ``item_completed`` at all and ``records`` is re-iterable (a ``Sequence``),
    it is re-read with the legacy decoder (C1); for one-shot iterators the
    caller must re-read (``load_rollout`` does) — the ``notes`` say so.
    """
    it = iter(records)
    prefix: list[dict[str, Any]] = []
    mode: str | None = None
    for r in it:
        prefix.append(r)
        mode = _mode_of(r)
        if mode is not None:
            break
    stream: Iterable[dict[str, Any]] = itertools.chain(prefix, it)

    if mode != "v2":
        events, session_id, model, meta = _decode_legacy(stream)
        return events, session_id, model, meta, [], []
    events, session_id, model, meta, turns, notes = _decode_v2(stream)
    if _NO_ITEMS_NOTE in notes and isinstance(records, Sequence):
        events, session_id, model, meta = _decode_legacy(records)
        return events, session_id, model, meta, [], ["fell back to legacy: no item_completed"]
    return events, session_id, model, meta, turns, notes


def load_rollout(path: Path) -> Trace:
    """Parse one native Codex rollout file into a Trace."""
    events, session_id, model, meta, turns, notes = decode_codex_native_full(iter_jsonl(path))
    if _NO_ITEMS_NOTE in notes:
        events, session_id, model, meta, turns, notes = decode_codex_native_full(list(iter_jsonl(path)))
    session_id = session_id or path.stem
    root = build_session(events, AgentKind.codex, session_id=session_id, model=model)
    root.turns = turns
    root.harness_version = meta.get("cli_version")
    root.thread_source = meta.get("thread_source")
    if meta.get("parent_thread_id"):
        root.parent_session_id = meta["parent_thread_id"]
    if meta.get("subagent_history_start_ordinal") is not None:
        root.replay_prefix = SourceRef(record_index=int(meta["subagent_history_start_ordinal"]))
    declared = _DECLARED_V2 if _parse_cli_version(meta.get("cli_version")) >= (0, 150) else _DECLARED_LEGACY
    root.capabilities = observe_capabilities(root, declared)
    if "usage_identity" in notes:
        root.degraded.append("usage_identity")
    provenance = Provenance(
        source_format="codex_native",
        detector_confidence="high",
        root_path=path,
        files=[path],
        notes=list(notes),
        harness_version=meta.get("cli_version"),
        harness_meta={
            k: meta[k]
            for k in (
                "cli_version",
                "thread_source",
                "parent_thread_id",
                "forked_from_id",
                "forked_from_ordinal_exclusive",
                "history_base",
                "subagent_history_start_ordinal",
                "agent_nickname",
                "agent_path",
                "depth",
                "cwd",
                "git",
                "originator",
            )
            if meta.get(k) is not None
        },
    )
    return Trace(
        trace_id=session_id,
        agent=AgentKind.codex,
        model=model,
        root=root,
        provenance=provenance,
        capabilities=root.capabilities,
    )


# How far into a file the directory index looks for delegation evidence when
# the session_meta itself names no parent (``agent_created_thread`` children
# carry the parent only inside the first ``<codex_delegation>`` wrapper).
_INDEX_SCAN_RECORDS = 2000


def index_session_meta(path: Path) -> dict[str, Any] | None:
    """Read the linkage-relevant head of a rollout for the directory index.

    Returns the first ``session_meta`` fields plus ``path`` and, when the meta
    names no parent, ``delegation_source_thread_id`` from the first
    ``<codex_delegation>`` wrapper seen within the first records.
    """
    fields: dict[str, Any] | None = None
    for i, rec in enumerate(iter_jsonl(path)):
        if fields is None:
            if rec.get("type") != "session_meta" or not isinstance(rec.get("payload"), dict):
                return None
            fields = _session_meta_fields(rec["payload"])
            fields["path"] = path
            if fields.get("parent_thread_id") or fields.get("forked_from_id"):
                return fields
            continue
        if i > _INDEX_SCAN_RECORDS or rec.get("type") == "token_usage_record":
            break
        text = _delegation_text(rec)
        if text is not None:
            fields["delegation_source_thread_id"] = _source_thread_id(text)
            break
    return fields


_CREATE_THREAD_FN = "create_thread"


def _delegation_text(rec: dict[str, Any]) -> str | None:
    """The ``<codex_delegation>`` wrapper that names this thread's *creator*.

    Only the ``create_thread`` tool output (or a delegation ``UserMessage``)
    qualifies: ``send_message_to_thread`` outputs also embed a
    ``<source_thread_id>`` but it names the *sender*, not a parent.
    """
    payload = rec.get("payload")
    if not isinstance(payload, dict):
        return None
    text: str | None = None
    rtype = rec.get("type")
    if rtype == "response_item" and payload.get("type") == "function_call_output":
        if payload.get("name") == _CREATE_THREAD_FN:
            text = _flatten_output(payload.get("output"))
    elif rtype == "event_msg" and payload.get("type") == "item_completed":
        item = payload.get("item")
        if isinstance(item, dict):
            if item.get("type") == "FunctionCallOutput" and item.get("name") == _CREATE_THREAD_FN:
                text = _flatten_output(item.get("output"))
            elif item.get("type") == "UserMessage":
                text = flatten_text(item.get("content"))
    if text and _DELEGATION_TAG in text:
        return text
    return None


def looks_like_native_codex(path: Path) -> bool:
    """True when the first record is a ``{timestamp,type,payload}`` session_meta."""
    if not path.is_file() or path.suffix != ".jsonl":
        return False
    for r in iter_jsonl(path):
        return r.get("type") == "session_meta" and isinstance(r.get("payload"), dict)
    return False
