"""Detector parity: canonical op_kind vs Claude tool names; windows skip meta."""

from __future__ import annotations

import agent_hotwash.detectors  # noqa: F401 — register detectors
from agent_hotwash.config import load_config
from agent_hotwash.detectors.registry import logical_events, run_detectors
from agent_hotwash.events import AgentKind, Event, EventKind, ToolCategory
from agent_hotwash.sources._common import build_session

CFG = load_config()
_N_META = 40


def _meta() -> Event:
    return Event(kind=EventKind.meta, raw_type="custom_tool_call", text="wrapper exec")


def _fail(call_id: str) -> Event:
    # RETRY_STORM needs at least one real failure among the repeats.
    return Event(kind=EventKind.tool_result, call_id=call_id, ok=False, exit_code=2, error_text="ls: cannot access")


def _canonical_events() -> list[Event]:
    metas = [_meta() for _ in range(_N_META)]
    execs = [
        Event(
            kind=EventKind.tool_call,
            tool_name="cmd.exec",
            op_kind="cmd.exec",
            tool_category=ToolCategory.execute,
            call_id=f"e{i}",
            tool_args={"command": "ls -la"},
        )
        for i in range(4)
    ]
    search = Event(
        kind=EventKind.tool_call,
        tool_name="cmd.search",
        op_kind="cmd.search",
        tool_category=ToolCategory.read,
        call_id="g1",
        tool_args={"pattern": "foo", "path": "src"},
    )
    edit = Event(
        kind=EventKind.tool_call,
        tool_name="file.edit",
        op_kind="file.edit",
        tool_category=ToolCategory.write,
        path="a.py",
        call_id="ed1",
        tool_args={"file_path": "a.py", "old_string": "a", "new_string": "b"},
    )
    return [
        Event(kind=EventKind.user_msg, text="no"),
        *metas,
        Event(kind=EventKind.user_msg, text="wrong"),
        Event(kind=EventKind.user_msg, text="still broken"),
        search,
        execs[0],
        _fail("e0"),
        *execs[1:],
        edit,
    ]


def _claude_events() -> list[Event]:
    metas = [_meta() for _ in range(_N_META)]
    execs = [
        Event(
            kind=EventKind.tool_call,
            tool_name="Bash",
            tool_category=ToolCategory.execute,
            call_id=f"e{i}",
            tool_args={"command": "ls -la"},
        )
        for i in range(4)
    ]
    search = Event(
        kind=EventKind.tool_call,
        tool_name="Grep",
        tool_category=ToolCategory.read,
        call_id="g1",
        tool_args={"pattern": "foo", "path": "src"},
    )
    edit = Event(
        kind=EventKind.tool_call,
        tool_name="Edit",
        tool_category=ToolCategory.write,
        call_id="ed1",
        tool_args={"file_path": "a.py", "old_string": "a", "new_string": "b"},
    )
    return [
        Event(kind=EventKind.user_msg, text="no"),
        *metas,
        Event(kind=EventKind.user_msg, text="wrong"),
        Event(kind=EventKind.user_msg, text="still broken"),
        search,
        execs[0],
        _fail("e0"),
        *execs[1:],
        edit,
    ]


def test_canonical_vs_claude_same_detectors_fire() -> None:
    canonical = build_session(_canonical_events(), AgentKind.codex, session_id="canon")
    claude = build_session(_claude_events(), AgentKind.claude, session_id="claude")

    assert sum(1 for e in canonical.events if e.kind is EventKind.meta) == _N_META
    assert all(e.kind is not EventKind.meta for e in logical_events(canonical))
    assert len(logical_events(canonical)) == len(canonical.events) - _N_META

    ids_c = {f.id for f in run_detectors(canonical, CFG)}
    ids_l = {f.id for f in run_detectors(claude, CFG)}
    shared = {"RETRY_STORM", "EDIT_WITHOUT_READ", "CORRECTION_LOOP"}
    assert shared <= ids_c
    assert shared <= ids_l
    # If meta were counted, 3 corrections would not fit in the 30-event window.
    n_logical = len(logical_events(canonical))
    assert n_logical < _N_META + 3
