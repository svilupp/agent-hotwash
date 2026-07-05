"""Tests for the native Claude Code session parser."""

from __future__ import annotations

from pathlib import Path

from agent_hotwash.events import AgentKind, EventKind
from agent_hotwash.sources.claude_native import (
    load_session_file,
    looks_like_native_claude,
)

FIXTURES = Path(__file__).parent / "fixtures" / "claude_native"
SESSION = FIXTURES / "proj" / "sess-fixture.jsonl"


def test_detects_native_claude() -> None:
    assert looks_like_native_claude(SESSION)


def test_basic_shape() -> None:
    trace = load_session_file(SESSION)
    assert trace.agent is AgentKind.claude
    assert trace.model == "claude-opus-4-8"
    assert trace.root.session_id == "sess-fixture"
    assert trace.root.has_timestamps


def test_user_and_tool_events() -> None:
    trace = load_session_file(SESSION)
    events = trace.root.events
    user_msgs = [e for e in events if e.kind is EventKind.user_msg]
    assert any(e.text == "Please add a greeting." for e in user_msgs)
    calls = [e for e in events if e.kind is EventKind.tool_call]
    results = [e for e in events if e.kind is EventKind.tool_result]
    # Read, Edit, Task in the root (subagent tools are separate).
    assert {"Read", "Edit", "Task"} <= {e.tool_name for e in calls}
    assert len(results) == len(calls)


def test_aux_line_types_tolerated() -> None:
    # file-history-snapshot line must not crash or produce a stray event.
    trace = load_session_file(SESSION)
    assert all(e.raw_type != "file-history-snapshot" for e in trace.root.events)


def test_file_ops_parsed() -> None:
    trace = load_session_file(SESSION)
    edit = next(e for e in trace.root.events if e.tool_name == "Edit")
    assert edit.path == "src/app.py"
    assert edit.lines_added is not None


def test_subagent_linked_from_file() -> None:
    trace = load_session_file(SESSION)
    assert len(trace.subagents) == 1
    sub = trace.subagents[0]
    assert sub.parent_session_id == "sess-fixture"
    assert "sub-abc" in sub.session_id
    sub_calls = [e for e in sub.events if e.kind is EventKind.tool_call]
    assert any(e.tool_name == "Bash" for e in sub_calls)


def test_meta_user_records_are_not_turns() -> None:
    # A ``isMeta`` user record (local-command caveat, injected agent-message)
    # is system context, not a real prompt -- it must not become a user_msg.
    trace = load_session_file(SESSION)
    user_texts = [e.text for e in trace.root.events if e.kind is EventKind.user_msg]
    assert not any(t and t.startswith("Caveat:") for t in user_texts)
    assert user_texts == ["Please add a greeting."]


def test_usage_summable() -> None:
    trace = load_session_file(SESSION)
    total_in = sum(e.usage.input for e in trace.root.events if e.usage and e.usage.input)
    assert total_in > 0


def test_usage_counted_once_per_request_id() -> None:
    # req1 spans two jsonl lines (a1 text, a2 tool_use) that repeat the same
    # requestId and the same message.usage. Usage must be attached only once, so
    # the request's 200 input tokens are counted a single time, not doubled.
    trace = load_session_file(SESSION)
    inputs = [e.usage.input for e in trace.root.events if e.usage and e.usage.input]
    assert len(inputs) == 4  # req1..req4, once each (not 5 assistant lines)
    assert sum(inputs) == 223  # 200 (req1, once) + 8 + 5 + 10
