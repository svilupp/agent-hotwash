"""Reconcile Cursor run logs with the matching turns in a saved chat."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import TYPE_CHECKING

from agent_hotwash.canonical import build_turns, declared_row, observe_capabilities
from agent_hotwash.events import CapLevel, EventKind, RoleHint, TurnStatus
from agent_hotwash.sources._common import build_session

if TYPE_CHECKING:
    from agent_hotwash.events import Event, Trace, Turn

_TOOLS = (EventKind.tool_call, EventKind.tool_result)
_TEXT = (EventKind.assistant_msg, EventKind.thinking, EventKind.user_msg)
_COVERAGE = declared_row(
    per_turn_model=CapLevel.partial,
    reasoning_text=CapLevel.partial,
    reasoning_tokens=CapLevel.partial,
    timestamps=CapLevel.partial,
    op_timing=CapLevel.partial,
    file_diffs=CapLevel.partial,
    full_tool_output=CapLevel.partial,
    thread_linkage=CapLevel.partial,
    compaction_summaries=CapLevel.partial,
    final_answer_marker=CapLevel.partial,
    context_window=CapLevel.partial,
)


def _key(event: Event) -> tuple[EventKind, str] | None:
    if event.kind in _TOOLS and event.call_id:
        return event.kind, event.call_id
    if event.kind in _TEXT and event.text and event.role_hint is not RoleHint.injected:
        return event.kind, event.text.strip()
    return None


def _combine_turn(native: list[Event], stream: Trace) -> list[Event]:
    raw = stream.root.events
    raw_keys = Counter(key for ev in raw if (key := _key(ev)) is not None)
    native_keys = Counter(key for ev in native if (key := _key(ev)) is not None)
    coordinates: dict[tuple[EventKind, str], list[int]] = defaultdict(list)
    for index, event in enumerate(native):
        if (key := _key(event)) is not None and raw_keys[key] == native_keys[key]:
            coordinates[key].append(index)
    additions: dict[int, list[Event]] = defaultdict(list)
    before = len(native)
    for event in reversed(raw):
        key = _key(event)
        choices = coordinates.get(key) if key is not None else None
        if choices:
            index = choices.pop()
            canonical = native[index]
            before = index
            canonical.ts = canonical.ts or event.ts
            canonical.ts_end = canonical.ts_end or event.ts_end
            canonical.response_id = canonical.response_id or event.response_id
            if event.source:
                canonical.alternate_sources.extend([event.source, *event.alternate_sources])
            if event.phase == "final_answer":
                canonical.phase = "final_answer"
            continue
        if event.kind in _TOOLS or event.kind is EventKind.session_end or event.phase == "final_answer":
            additions[before].append(event)
    combined = []
    for index, event in enumerate(native):
        combined.extend(reversed(additions.get(index, [])))
        combined.append(event)
    combined.extend(reversed(additions.get(len(native), [])))
    return combined


def _tool_ids(events: list[Event]) -> set[str]:
    return {ev.call_id for ev in events if ev.kind in _TOOLS and ev.call_id}


def _matches(native: Trace, stream: Trace, call_ids: dict[str, set[str]]) -> list[Turn]:
    request_id = stream.provenance.harness_meta.get("request_id")
    by_request = [
        turn for turn in native.root.turns if request_id and native.root.events[turn.event_start].span_id == request_id
    ]
    if by_request:
        return by_request
    calls = _tool_ids(stream.root.events)
    return [turn for turn in native.root.turns if call_ids[turn.turn_id] & calls]


def _distinct_run(native: Trace, stream: Trace, saved_calls: set[str]) -> bool:
    if not stream.provenance.harness_meta.get("request_id") or _tool_ids(stream.root.events) & saved_calls:
        return False
    unkeyed = {
        turn.user_input.text.strip() for turn in native.root.turns if not native.root.events[turn.event_start].span_id
    }
    prompt = stream.root.turns[0].user_input.text.strip() if stream.root.turns else None
    return not unkeyed or bool(prompt and prompt not in unkeyed)


def reconcile(native: Trace, streams: list[Trace]) -> list[Trace]:
    """Keep the saved history, taking result boundaries and timing from proven run matches."""
    root = native.root
    known_turns = {turn.turn_id: turn for turn in root.turns}
    call_ids = {turn.turn_id: _tool_ids(root.events[turn.event_start : turn.event_end + 1]) for turn in root.turns}
    saved_calls = _tool_ids(root.events)
    matches: dict[str, Trace] = {}
    standalone: list[Trace] = []
    unresolved = 0
    for stream in sorted(
        streams,
        key=lambda t: (not any(e.kind is EventKind.session_end for e in t.root.events), t.trace_id),
    ):
        candidates = _matches(native, stream, call_ids) if stream.root.session_id == root.session_id else []
        if len(candidates) == 1 and candidates[0].turn_id not in matches:
            matches[candidates[0].turn_id] = stream
        elif not candidates and _distinct_run(native, stream, saved_calls):
            stream.provenance.notes.append("This invocation has its own request ID outside the saved checkpoint.")
            standalone.append(stream)
            continue
        else:
            unresolved += 1
            root.source_diagnostics.append(
                {
                    "source_path": str(stream.provenance.root_path),
                    "status": "cursor_stream_overlap_unresolved",
                }
            )
        native.provenance.files.extend(path for path in stream.provenance.files if path not in native.provenance.files)
    if unresolved:
        native.provenance.notes.append("Some same-chat run logs could not be matched to one saved turn.")
    if not matches:
        return [native, *standalone]

    combined: list[Event] = []
    next_index = 0
    for turn in root.turns:
        combined.extend(root.events[next_index : turn.event_start])
        next_index = turn.event_end + 1
        segment = root.events[turn.event_start : next_index]
        if (stream := matches.get(turn.turn_id)) is not None:
            segment = _combine_turn(segment, stream)
            root.source_artifacts.extend(stream.root.source_artifacts)
            root.source_diagnostics.extend(stream.root.source_diagnostics)
        combined.extend(segment)
    combined.extend(root.events[next_index:])
    coverage = sum(any(e.invocation_usage is not None for e in stream.root.events) for stream in matches.values())
    rebuilt = build_session(
        combined,
        root.agent,
        session_id=root.session_id,
        parent_session_id=root.parent_session_id,
        usage_reliable=coverage == len(root.turns),
    )
    rebuilt.turns = build_turns(rebuilt)
    for turn in rebuilt.turns:
        prior = known_turns[turn.turn_id]
        turn.context_window_tokens = prior.context_window_tokens
        rows = rebuilt.events[turn.event_start : turn.event_end + 1]
        terminal = next((event for event in reversed(rows) if event.kind is EventKind.session_end), None)
        turn.status = prior.status
        turn.source_end = prior.source_end
        if terminal is not None:
            turn.status = TurnStatus.completed if terminal.ok else TurnStatus.aborted
            turn.source_end = terminal.source
        for event in rows:
            event.turn_id = turn.turn_id
    root.events, root.turns, root.file_state = rebuilt.events, rebuilt.turns, rebuilt.file_state
    root.has_timestamps, root.has_any_timestamps = rebuilt.has_timestamps, rebuilt.has_any_timestamps
    root.ts_coverage, root.usage_reliable = rebuilt.ts_coverage, rebuilt.usage_reliable
    if not root.usage_reliable and coverage:
        root.degraded.append("usage_partial")
        native.provenance.notes.append("Invocation token totals cover only the matched metered runs.")
    prior_linkage = root.capabilities.observed.thread_linkage
    root.capabilities = observe_capabilities(root, _COVERAGE)
    root.capabilities.observed.thread_linkage = prior_linkage
    native.capabilities = root.capabilities
    native.provenance.notes.append("Run logs were matched to saved turns by request ID or tool call ID.")
    return [native, *standalone]
