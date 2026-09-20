"""Canonical turn builder, heuristic Pi-like turns, capabilities lattice."""

from __future__ import annotations

from datetime import UTC, datetime

from agent_hotwash.canonical import build_turns, declared_row, observe_capabilities
from agent_hotwash.events import (
    AgentKind,
    Capabilities,
    CapLevel,
    Event,
    EventKind,
    Session,
    Usage,
)


def test_pi_like_heuristic_turns_and_model_change() -> None:
    ts = datetime(2026, 1, 1, tzinfo=UTC)
    events = [
        Event(
            kind=EventKind.meta,
            idx=0,
            ts=ts,
            raw_type="model_change",
            tool_args={"model": "acme-mini", "provider": "acme"},
        ),
        Event(kind=EventKind.user_msg, idx=1, ts=ts, text="list the files"),
        Event(
            kind=EventKind.assistant_msg,
            idx=2,
            ts=ts,
            text="running ls",
            usage=Usage(input=10, output=4),
        ),
        Event(kind=EventKind.user_msg, idx=3, ts=ts, text="now explain the listing"),
        Event(
            kind=EventKind.assistant_msg,
            idx=4,
            ts=ts,
            text="app.py and README.md",
            usage=Usage(input=12, output=6),
        ),
    ]
    session = Session(
        session_id="pi1",
        agent=AgentKind.pi,
        model="acme-mini",
        events=events,
        has_any_timestamps=True,
        has_timestamps=True,
    )
    turns = build_turns(session)
    assert len(turns) == 2
    assert turns[0].user_input.text == "list the files"
    assert turns[0].model_config_active.model == "acme-mini"
    assert turns[0].model_config_active.provider == "acme"
    billed0 = [c for c in turns[0].model_calls if c.usage is not None]
    assert billed0
    assert billed0[0].usage is not None
    assert billed0[0].usage.input == 10
    assert turns[1].user_input.text.startswith("now explain")
    billed1 = [c for c in turns[1].model_calls if c.usage is not None]
    assert billed1
    assert turns[0].event_end < turns[1].event_start or turns[0].event_end <= turns[1].event_start


def test_turn_boundary_second_user_closes_first() -> None:
    events = [
        Event(kind=EventKind.user_msg, idx=0, text="first"),
        Event(kind=EventKind.assistant_msg, idx=1, text="ok", usage=Usage(input=1, output=1)),
        Event(kind=EventKind.user_msg, idx=2, text="second"),
        Event(kind=EventKind.assistant_msg, idx=3, text="done", usage=Usage(input=1, output=1)),
    ]
    session = Session(session_id="s", agent=AgentKind.unknown, events=events)
    turns = build_turns(session)
    assert [t.user_input.text for t in turns] == ["first", "second"]
    assert turns[0].event_start == 0
    assert turns[1].event_start == 2


def test_capabilities_declared_vs_observed_and_lattice_min() -> None:
    events = [
        Event(kind=EventKind.user_msg, idx=0, text="hi"),
        Event(kind=EventKind.thinking, idx=1, text="I should list files."),
        Event(kind=EventKind.assistant_msg, idx=2, text="ok", usage=Usage(input=8, output=2)),
    ]
    session = Session(
        session_id="s",
        agent=AgentKind.pi,
        model="acme-mini",
        events=events,
        has_any_timestamps=True,
        has_timestamps=True,
    )
    session.turns = build_turns(session)
    declared = declared_row(
        per_call_usage=True,
        per_turn_model=True,
        reasoning_effort=False,
        reasoning_text=True,
        timestamps=True,
        full_tool_output=CapLevel.partial,
    )
    caps = observe_capabilities(session, declared)
    assert caps.declared.reasoning_effort is CapLevel.false
    assert caps.declared.reasoning_text is CapLevel.true
    assert caps.observed.per_call_usage is CapLevel.true
    assert caps.observed.reasoning_text is CapLevel.true
    assert caps.observed.reasoning_effort is CapLevel.false
    assert caps.meets("reasoning_text") is True
    assert caps.meets("reasoning_effort") is False

    other = Capabilities(
        declared=declared_row(per_call_usage=True, reasoning_text=False, timestamps=True),
        observed=declared_row(per_call_usage=False, reasoning_text=False, timestamps=True),
    )
    merged = Capabilities.merge_min([caps, other])
    assert merged.declared.per_call_usage is CapLevel.true
    assert merged.declared.reasoning_text is CapLevel.false
    assert merged.observed.per_call_usage is CapLevel.false
    assert merged.declared.timestamps is CapLevel.true
