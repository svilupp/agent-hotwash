"""Shared synthetic Trace/Session factory for the analytics + aggregate tests.

Kept intentionally generic so other work packages can reuse it: ``tf`` is a
namespace of tiny builders that go through the real ``build_session`` so derived
state (idx, file_state, error classification, usage de-cumulation) matches what
parsers produce.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal

import pytest

from agent_hotwash.events import (
    AgentKind,
    Event,
    EventKind,
    Provenance,
    ToolCategory,
    Trace,
    Usage,
)
from agent_hotwash.sources._common import build_session

_BASE = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)


def _ts(offset_s: float | None) -> datetime | None:
    return _BASE + timedelta(seconds=offset_s) if offset_s is not None else None


def user(text: str, *, at: float | None = None) -> Event:
    return Event(kind=EventKind.user_msg, text=text, ts=_ts(at))


def assistant(text: str = "", *, at: float | None = None) -> Event:
    return Event(kind=EventKind.assistant_msg, text=text, ts=_ts(at))


def thinking(text: str = "", *, at: float | None = None) -> Event:
    return Event(kind=EventKind.thinking, text=text, ts=_ts(at))


def compaction(*, at: float | None = None) -> Event:
    return Event(kind=EventKind.compaction, ts=_ts(at))


def tool(
    name: str,
    *,
    call_id: str,
    args: dict[str, Any] | None = None,
    category: ToolCategory | None = None,
    at: float | None = None,
) -> Event:
    return Event(
        kind=EventKind.tool_call,
        tool_name=name,
        tool_category=category,
        tool_args=args or {},
        call_id=call_id,
        ts=_ts(at),
    )


def result(
    *,
    call_id: str,
    ok: bool = True,
    exit_code: int | None = None,
    output: str | None = None,
    error_text: str | None = None,
    at: float | None = None,
) -> Event:
    return Event(
        kind=EventKind.tool_result,
        call_id=call_id,
        ok=ok,
        exit_code=exit_code,
        output=output,
        error_text=error_text,
        ts=_ts(at),
    )


def usage(**kw: Any) -> Usage:
    return Usage(**kw)


def with_usage(ev: Event, u: Usage) -> Event:
    ev.usage = u
    return ev


def session(
    events: list[Event],
    *,
    agent: AgentKind = AgentKind.claude,
    session_id: str = "s0",
    model: str | None = "claude-opus-4-8",
    parent_session_id: str | None = None,
    usage_reliable: bool = True,
):
    return build_session(
        events,
        agent,
        session_id=session_id,
        model=model,
        parent_session_id=parent_session_id,
        usage_reliable=usage_reliable,
    )


def trace(
    root,
    *,
    subagents=None,
    agent: AgentKind = AgentKind.claude,
    model: str | None = "claude-opus-4-8",
    experiment: str | None = None,
    instance_id: str | None = None,
    trace_id: str = "t0",
    resolved: bool | None = None,
    source_format: Literal["codebench", "claude_native", "codex_native"] = "claude_native",
    harness_meta: dict[str, Any] | None = None,
) -> Trace:
    prov = Provenance(
        source_format=source_format,
        detector_confidence="high",
        root_path=Path("/tmp/x"),
        harness_meta=harness_meta or {},
    )
    return Trace(
        trace_id=trace_id,
        agent=agent,
        model=model,
        experiment=experiment,
        instance_id=instance_id,
        root=root,
        subagents=subagents or [],
        provenance=prov,
        resolved=resolved,
    )


@pytest.fixture
def tf() -> SimpleNamespace:
    return SimpleNamespace(
        user=user,
        assistant=assistant,
        thinking=thinking,
        compaction=compaction,
        tool=tool,
        result=result,
        usage=usage,
        with_usage=with_usage,
        session=session,
        trace=trace,
        AgentKind=AgentKind,
        ToolCategory=ToolCategory,
    )


# ---------------------------------------------------------------------------
# WP4 (detectors) fixture factories. Uniquely named (``make_detector_*`` / a
# ``dt`` fixture namespace of ``mk_*`` builders) to stay collision-free.
# ---------------------------------------------------------------------------


def mk_user(text: str) -> Event:
    return Event(kind=EventKind.user_msg, text=text)


def mk_assistant(text: str) -> Event:
    return Event(kind=EventKind.assistant_msg, text=text)


def mk_thinking(text: str = "thinking...") -> Event:
    return Event(kind=EventKind.thinking, text=text)


def mk_compaction() -> Event:
    return Event(kind=EventKind.compaction)


def mk_call(
    tool_name: str,
    *,
    call_id: str | None = None,
    args: dict[str, Any] | None = None,
    category: ToolCategory | None = None,
) -> Event:
    return Event(
        kind=EventKind.tool_call,
        tool_name=tool_name,
        tool_category=category,
        tool_args=args or {},
        call_id=call_id,
    )


def mk_result(
    *,
    call_id: str | None = None,
    ok: bool = True,
    exit_code: int | None = None,
    error_text: str | None = None,
    output: str | None = None,
) -> Event:
    return Event(
        kind=EventKind.tool_result,
        call_id=call_id,
        ok=ok,
        exit_code=exit_code,
        error_text=error_text,
        output=output,
    )


def mk_read(path: str, *, call_id: str | None = None) -> Event:
    return mk_call("Read", call_id=call_id, args={"file_path": path})


def mk_edit(
    path: str, *, old: str = "old", new: str = "new", call_id: str | None = None, tool_name: str = "Edit"
) -> Event:
    return mk_call(tool_name, call_id=call_id, args={"file_path": path, "old_string": old, "new_string": new})


def mk_write(path: str, *, content: str = "x\n", call_id: str | None = None) -> Event:
    return mk_call("Write", call_id=call_id, args={"file_path": path, "content": content})


def mk_bash(cmd: str, *, call_id: str | None = None) -> Event:
    return mk_call("Bash", call_id=call_id, args={"command": cmd})


def make_detector_session(
    events: list[Event],
    *,
    session_id: str = "s1",
    agent: AgentKind = AgentKind.claude,
    with_timestamps: bool = False,
    gap_minutes: float = 1.0,
):
    """Build a fully-derived Session from raw events (WP4 fixture factory).

    When ``with_timestamps`` is set, monotonic timestamps spaced ``gap_minutes``
    apart are stamped so timestamp-gated detectors can be exercised.
    """
    if with_timestamps:
        base = datetime(2026, 3, 1, tzinfo=UTC)
        for i, ev in enumerate(events):
            ev.ts = base + timedelta(minutes=gap_minutes * i)
    return build_session(events, agent, session_id=session_id)


@pytest.fixture
def dt() -> SimpleNamespace:
    return SimpleNamespace(
        user=mk_user,
        assistant=mk_assistant,
        thinking=mk_thinking,
        compaction=mk_compaction,
        call=mk_call,
        result=mk_result,
        read=mk_read,
        edit=mk_edit,
        write=mk_write,
        bash=mk_bash,
        make=make_detector_session,
        AgentKind=AgentKind,
        ToolCategory=ToolCategory,
    )
