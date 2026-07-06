"""Per-file state builder tests."""

from __future__ import annotations

from agent_hotwash.events import Event, EventKind, ToolCategory
from agent_hotwash.primitives.filestate import build_file_state


def _call(idx: int, name: str, cat: ToolCategory, path: str) -> Event:
    return Event(kind=EventKind.tool_call, idx=idx, tool_name=name, tool_category=cat, path=path)


def test_read_then_edit_transitions() -> None:
    events = [
        _call(0, "Read", ToolCategory.read, "a.py"),
        _call(1, "Edit", ToolCategory.write, "a.py"),
        _call(2, "Edit", ToolCategory.write, "a.py"),
        _call(3, "Write", ToolCategory.write, "b.py"),
    ]
    fs = build_file_state(events)
    a = fs["a.py"]
    assert a.ever_read is True
    assert a.read_at == [0]
    assert a.edited_at == [1, 2]
    assert a.edit_count == 2
    assert a.write_count == 0
    assert a.last_op == "edit"

    b = fs["b.py"]
    assert b.ever_read is False
    assert b.write_count == 1
    assert b.edit_count == 0
    assert b.last_op == "write"


def test_edit_without_read() -> None:
    fs = build_file_state([_call(0, "Edit", ToolCategory.write, "x.py")])
    assert fs["x.py"].ever_read is False
    assert fs["x.py"].edited_at == [0]


def test_non_file_events_ignored() -> None:
    events = [
        Event(kind=EventKind.user_msg, idx=0, text="hi"),
        _call(1, "Bash", ToolCategory.execute, ""),  # no path -> skipped
    ]
    assert build_file_state(events) == {}
