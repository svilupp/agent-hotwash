"""Model construction / defaults tests."""

from __future__ import annotations

from pathlib import Path

from agent_hotwash.events import (
    AgentKind,
    Event,
    EventKind,
    Provenance,
    Session,
    ToolCategory,
    Trace,
    Usage,
)


def test_event_minimal_and_defaults() -> None:
    ev = Event(kind=EventKind.user_msg)
    assert ev.idx == -1  # unassigned until build_session runs
    assert ev.ts is None
    assert ev.agent is AgentKind.unknown
    assert ev.tool_args == {}


def test_event_extra_ignored() -> None:
    ev = Event.model_validate({"kind": "tool_call", "totally_unknown": 1})  # type: ignore[arg-type]
    assert not hasattr(ev, "totally_unknown")


def test_usage_defaults() -> None:
    u = Usage()
    assert u.input is None
    assert u.cumulative is False


def test_session_and_trace_construction() -> None:
    sess = Session(session_id="s", agent=AgentKind.claude, events=[Event(kind=EventKind.meta)])
    assert sess.has_timestamps is False
    assert sess.usage_reliable is True
    prov = Provenance(
        source_format="codebench",
        detector_confidence="high",
        root_path=Path("/x"),
    )
    trace = Trace(trace_id="t", agent=AgentKind.claude, root=sess, provenance=prov)
    assert trace.subagents == []
    assert trace.resolved is None
    # ts optional everywhere: a fully-empty-usage event is valid
    assert ToolCategory.subagent == "subagent"
