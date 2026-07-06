"""Pure source-helper tests: flatten_text, tool_category_of, de_cumulate,
build_session."""

from __future__ import annotations

from agent_hotwash.events import AgentKind, Event, EventKind, ToolCategory, Usage
from agent_hotwash.sources._common import (
    build_session,
    de_cumulate,
    flatten_text,
    tool_category_of,
)


def test_flatten_text() -> None:
    assert flatten_text("hello") == "hello"
    assert flatten_text([{"text": "a"}, {"text": "b"}]) == "a b"
    assert flatten_text([{"type": "image"}]) == ""
    assert flatten_text(None) == ""
    assert flatten_text(123) == ""


def test_tool_category_of() -> None:
    assert tool_category_of("Read") is ToolCategory.read
    assert tool_category_of("Edit") is ToolCategory.write
    assert tool_category_of("Bash") is ToolCategory.execute
    assert tool_category_of("command_execution") is ToolCategory.execute  # codex synthetic
    assert tool_category_of("file_change") is ToolCategory.write
    assert tool_category_of("Agent") is ToolCategory.subagent
    assert tool_category_of("SomethingElse") is ToolCategory.other
    assert tool_category_of(None) is ToolCategory.other


def test_de_cumulate() -> None:
    usages = [
        Usage(input=10, output=5, cumulative=True),
        Usage(input=25, output=12, cumulative=True),
        Usage(input=25, output=20, cumulative=True),
    ]
    out = de_cumulate(usages)
    assert [u.input for u in out if u] == [10, 15, 0]
    assert [u.output for u in out if u] == [5, 7, 8]
    assert all(not u.cumulative for u in out if u)


def test_de_cumulate_passthrough_non_cumulative() -> None:
    usages = [Usage(input=3), None, Usage(input=4)]
    out = de_cumulate(usages)
    first, third = out[0], out[2]
    assert first is not None and first.input == 3
    assert out[1] is None
    assert third is not None and third.input == 4


def test_build_session_assigns_idx_and_links_errors() -> None:
    raw = [
        Event(kind=EventKind.user_msg, text="do it"),
        Event(
            kind=EventKind.tool_call,
            tool_name="Bash",
            tool_category=ToolCategory.execute,
            tool_args={"command": "frobnicate"},
            call_id="c1",
        ),
        Event(
            kind=EventKind.tool_result,
            call_id="c1",
            exit_code=127,
            error_text="bash: frobnicate: command not found",
        ),
    ]
    sess = build_session(raw, AgentKind.claude, session_id="s1")
    assert [e.idx for e in sess.events] == [0, 1, 2]
    assert sess.events[2].ok is False
    assert sess.events[2].error_category == "command_not_found"
    assert sess.events[0].agent is AgentKind.claude
    assert sess.has_timestamps is False


def test_build_session_parses_file_ops_and_categories() -> None:
    raw = [
        Event(
            kind=EventKind.tool_call,
            tool_name="Write",
            tool_args={"file_path": "a.py", "content": "l1\nl2\nl3"},
            call_id="w1",
        ),
        Event(
            kind=EventKind.tool_call,
            tool_name="Read",
            tool_args={"file_path": "a.py"},
            call_id="r1",
        ),
    ]
    sess = build_session(raw, AgentKind.claude, session_id="s2")
    write_ev = sess.events[0]
    assert write_ev.tool_category is ToolCategory.write  # inferred
    assert write_ev.path == "a.py"
    assert write_ev.lines_added == 3
    assert write_ev.tool_norm_args  # normalized args populated
    assert "a.py" in sess.file_state
    assert sess.file_state["a.py"].ever_read is True


def test_build_session_decumulates_usage() -> None:
    raw = [
        Event(kind=EventKind.assistant_msg, usage=Usage(input=100, cumulative=True)),
        Event(kind=EventKind.assistant_msg, usage=Usage(input=250, cumulative=True)),
    ]
    sess = build_session(raw, AgentKind.codex, session_id="s3")
    assert [e.usage.input for e in sess.events if e.usage] == [100, 150]
    assert all(not e.usage.cumulative for e in sess.events if e.usage)
