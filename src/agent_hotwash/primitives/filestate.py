"""Per-file state machine.

``FileState`` records, per path, the idx positions at which the file was read
and edited plus small derived counters. It powers the file-oriented detectors
(EDIT_THRASH #4, EDIT_WITHOUT_READ #5, FULL_FILE_REWRITE #6,
LOOKS_RIGHT_RUNS_WRONG #21, COMPACTION_AMNESIA #25).

The state is built once per :class:`~agent_hotwash.events.Session` by
``build_session`` in ``sources/_common.py`` and stashed on ``Session.file_state``.

This module never imports ``events`` at runtime (only under ``TYPE_CHECKING``) so
that ``events.py`` can reference ``FileState`` without an import cycle.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from collections.abc import Iterable

    from agent_hotwash.events import Event

FileOp = Literal["read", "write", "edit"]


class FileState(BaseModel):
    """Read/edit history for a single path, indexed by event ``idx``."""

    read_at: list[int] = Field(default_factory=list)
    edited_at: list[int] = Field(default_factory=list)
    last_op: FileOp | None = None
    edit_count: int = 0
    write_count: int = 0
    ever_read: bool = False


def build_file_state(events: Iterable[Event]) -> dict[str, FileState]:
    """Fold an ordered event stream into a ``{path: FileState}`` map.

    Only ``tool_call`` events whose ``path`` and ``tool_category`` are set
    contribute; the categorizer (``read``/``write``) plus the raw ``tool_name``
    (to separate ``Write`` from ``Edit``) drive the transitions.
    """
    from agent_hotwash.events import EventKind, ToolCategory

    states: dict[str, FileState] = {}
    for ev in events:
        if ev.kind is not EventKind.tool_call or not ev.path:
            continue
        cat = ev.tool_category
        st = states.setdefault(ev.path, FileState())
        if cat is ToolCategory.read:
            st.read_at.append(ev.idx)
            st.ever_read = True
            st.last_op = "read"
        elif cat is ToolCategory.write:
            name = (ev.tool_name or "").lower()
            if name == "write":
                st.write_count += 1
                st.last_op = "write"
            else:  # Edit / MultiEdit / edit / notebook edit
                st.edit_count += 1
                st.last_op = "edit"
            st.edited_at.append(ev.idx)
    return states
