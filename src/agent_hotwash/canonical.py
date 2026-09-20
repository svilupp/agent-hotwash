"""Harness-blind canonical builders: turns, model calls, capabilities, windows.

This module must not import ``sources.*`` or mention ``AgentKind`` /
``source_format``. Parsers call these helpers after they have emitted Events.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from agent_hotwash.events import (
    CAPABILITY_FIELDS,
    Capabilities,
    CapabilitySet,
    CapLevel,
    EventKind,
    ModelCall,
    ModelConfig,
    RoleHint,
    Turn,
    TurnStatus,
    UserInput,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from agent_hotwash.events import Event, Session


def logical_events(session: Session) -> list[Event]:
    """Events that detectors and windows may see — ``meta`` (wrappers, token
    fallbacks, settings) is excluded so synthetic pairing cannot shift counts.
    """
    return [ev for ev in session.events if ev.kind is not EventKind.meta]


def events_in_range(session: Session, start: int, end: int) -> list[Event]:
    """Events whose ``idx`` falls in the inclusive ``[start, end]`` range.

    ``build_session`` assigns ``idx == list position``, so this is a slice; the
    scan fallback covers hand-built sessions whose events were never indexed.
    """
    events = session.events
    if not events or end < start:
        return []
    lo, hi = max(0, start), min(len(events) - 1, end)
    if hi < lo:
        return []
    window = events[lo : hi + 1]
    if all(ev.idx == i for i, ev in zip(range(lo, hi + 1), window, strict=True)):
        return window
    return [ev for ev in events if start <= ev.idx <= end]


def turn_events(session: Session, turn: Turn) -> list[Event]:
    """Events belonging to ``turn`` (its inclusive event range)."""
    return events_in_range(session, turn.event_start, turn.event_end)


def _is_injected(text: str | None, injected_tags: Sequence[str], delegation_tag: str) -> RoleHint | None:
    """Classify a user-role payload. Delegation is checked before injected tags."""
    if not text:
        return None
    if delegation_tag and delegation_tag in text:
        return RoleHint.delegation
    stripped = text.lstrip()
    for tag in injected_tags:
        if stripped.startswith(tag):
            return RoleHint.injected
    return RoleHint.user


def _source_thread_id(text: str) -> str | None:
    """Pull ``<source_thread_id>…</source_thread_id>`` from a delegation wrapper."""
    start = text.find("<source_thread_id>")
    if start < 0:
        return None
    start += len("<source_thread_id>")
    end = text.find("</source_thread_id>", start)
    if end < 0:
        return None
    value = text[start:end].strip()
    return value or None


def build_turns(
    session: Session,
    *,
    injected_tags: Sequence[str] = (),
    delegation_tag: str = "<codex_delegation>",
) -> list[Turn]:
    """Heuristic turn builder for sources that do not emit explicit turn ids.

    A non-injected ``user_msg`` opens a turn; a usage-bearing event closes the
    current model call. ``model_change``-style updates arrive as ``Event.tool_args``
    on a ``meta`` event with ``raw_type == "model_change"`` (Pi).
    """
    turns: list[Turn] = []
    current: Turn | None = None
    call_start: int | None = None
    call_usage = None
    call_ts_start = None
    active_config = ModelConfig(model=session.model)

    def _close_call(end_idx: int, ts_end: datetime | None) -> None:
        nonlocal call_start, call_usage, call_ts_start
        if current is None or call_start is None:
            call_start = None
            call_usage = None
            call_ts_start = None
            return
        current.model_calls.append(
            ModelCall(
                response_id=session.events[end_idx].response_id if 0 <= end_idx < len(session.events) else None,
                turn_id=current.turn_id,
                event_start=call_start,
                event_end=end_idx,
                usage=call_usage,
                ts_start=call_ts_start,
                ts_end=ts_end,
            )
        )
        call_start = None
        call_usage = None
        call_ts_start = None

    def _open_turn(ev: Event, idx: int, hint: RoleHint) -> None:
        nonlocal current, call_start, call_usage, call_ts_start
        if current is not None:
            _close_call(idx - 1, ev.ts)
            current.event_end = max(current.event_start, idx - 1)
            current.ts_end = ev.ts
            if current.status is TurnStatus.open:
                current.status = TurnStatus.completed
            turns.append(current)
        source_tid = _source_thread_id(ev.text or "") if hint is RoleHint.delegation else None
        current = Turn(
            turn_id=ev.turn_id or f"{session.session_id}:turn{len(turns)}",
            session_id=session.session_id,
            source_start=ev.source,
            event_start=idx,
            event_end=idx,
            status=TurnStatus.open,
            user_input=UserInput(text=ev.text or "", kind=hint.value, source_thread_id=source_tid),
            model_config_active=active_config.model_copy(),
            model_config_revisions=[active_config.model_copy()],
            ts_start=ev.ts,
        )
        call_start = idx
        call_usage = None
        call_ts_start = ev.ts

    for ev in session.events:
        idx = ev.idx
        if ev.raw_type == "model_change" and ev.tool_args.get("model"):
            active_config = ModelConfig(
                provider=ev.tool_args.get("provider"),
                model=ev.tool_args.get("model"),
                reasoning_effort=ev.tool_args.get("reasoning_effort"),
            )
            if current is not None:
                current.model_config_revisions.append(active_config.model_copy())
                current.model_config_active = active_config.model_copy()
        if ev.kind is EventKind.user_msg:
            hint = ev.role_hint or _is_injected(ev.text, injected_tags, delegation_tag)
            if hint is None:
                hint = RoleHint.user
            ev.role_hint = hint
            if hint is RoleHint.injected:
                continue
            _open_turn(ev, idx, hint)
            continue
        if current is None:
            continue
        current.event_end = idx
        current.ts_end = ev.ts or current.ts_end
        if ev.kind is EventKind.compaction:
            current.compactions += 1
        if ev.kind is EventKind.assistant_msg and ev.phase == "final_answer":
            current.final_message = ev.text
            current.status = TurnStatus.completed
        if ev.usage is not None:
            call_usage = ev.usage
            _close_call(idx, ev.ts)
            call_start = idx + 1
            call_ts_start = ev.ts

    if current is not None:
        last_idx = session.events[-1].idx if session.events else current.event_start
        if call_start is not None and call_start <= last_idx:
            _close_call(last_idx, current.ts_end)
        current.event_end = last_idx
        turns.append(current)
    return turns


def observe_capabilities(session: Session, declared: CapabilitySet) -> Capabilities:
    """Fill the observed row from what this session actually carried."""
    events = session.events
    observed = CapabilitySet()
    if any(ev.usage is not None for ev in events):
        observed.per_call_usage = CapLevel.true
    if session.model or any(t.model_config_active.model for t in session.turns):
        observed.per_turn_model = CapLevel.true
    if any(t.model_config_active.reasoning_effort for t in session.turns):
        observed.reasoning_effort = CapLevel.true
    if any(ev.kind is EventKind.thinking and ev.text for ev in events):
        observed.reasoning_text = CapLevel.true
    if any(ev.usage is not None and ev.usage.reasoning_output for ev in events):
        observed.reasoning_tokens = CapLevel.true
    if session.has_any_timestamps:
        observed.timestamps = CapLevel.true if session.has_timestamps else CapLevel.partial
    if any(ev.ts_end is not None for ev in events):
        observed.op_timing = CapLevel.true
    if any(ev.classifications for ev in events):
        observed.parsed_commands = CapLevel.true
    if any(art.diff_head for ev in events for art in ev.artifacts):
        observed.file_diffs = CapLevel.true
    results_with_output = [ev for ev in events if ev.kind is EventKind.tool_result and ev.output]
    if results_with_output:
        # ``output_tokens_original`` is a truncation marker: the harness kept the
        # *size* of the full output but not the text, so full output stays partial.
        observed.full_tool_output = CapLevel.partial
        if any(ev.output_tokens_original for ev in results_with_output):
            observed.output_size_original = CapLevel.true
    if session.parent_session_id or any(t.user_input.source_thread_id for t in session.turns):
        observed.thread_linkage = CapLevel.true
    if any(ev.kind is EventKind.compaction for ev in events):
        observed.compaction_summaries = CapLevel.partial
        if any(ev.text for ev in events if ev.kind is EventKind.compaction):
            observed.compaction_summaries = CapLevel.true
    if any(t.final_message for t in session.turns) or any(
        ev.phase == "final_answer" for ev in events if ev.kind is EventKind.assistant_msg
    ):
        observed.final_answer_marker = CapLevel.true
    if any(t.context_window_tokens for t in session.turns):
        observed.context_window = CapLevel.true
    return Capabilities(declared=declared, observed=observed)


def declared_row(**levels: CapLevel | bool) -> CapabilitySet:
    """Build a declared CapabilitySet from field=True/False/CapLevel kwargs."""
    row = CapabilitySet()
    for name, value in levels.items():
        if name not in CAPABILITY_FIELDS:
            raise KeyError(f"unknown capability field: {name}")
        if isinstance(value, bool):
            setattr(row, name, CapLevel.true if value else CapLevel.false)
        else:
            setattr(row, name, value)
    return row


__all__ = [
    "build_turns",
    "declared_row",
    "events_in_range",
    "logical_events",
    "observe_capabilities",
    "turn_events",
]
