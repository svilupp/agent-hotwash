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

    Prefers ``Event.artifacts`` when present (multi-path FileChange); falls
    back to the legacy single ``path`` + ``tool_name`` / ``op_kind``.
    """
    from agent_hotwash.events import ArtifactOp, EventKind, ToolCategory

    states: dict[str, FileState] = {}
    for ev in events:
        if ev.kind is not EventKind.tool_call:
            continue
        paths: list[tuple[str, FileOp]] = []
        if ev.artifacts:
            for art in ev.artifacts:
                if art.op in (ArtifactOp.read, ArtifactOp.search):
                    paths.append((art.path, "read"))
                elif art.op is ArtifactOp.add:
                    paths.append((art.path, "write"))
                else:
                    paths.append((art.path, "edit"))
        elif ev.path:
            cat = ev.tool_category
            if cat is ToolCategory.read:
                paths.append((ev.path, "read"))
            elif cat is ToolCategory.write:
                name = (ev.op_kind or ev.tool_name or "").lower()
                paths.append((ev.path, "write" if name in {"write", "file.write"} else "edit"))
        for path, op in paths:
            if not path:
                continue
            st = states.setdefault(path, FileState())
            if op == "read":
                st.read_at.append(ev.idx)
                st.ever_read = True
                st.last_op = "read"
            elif op == "write":
                st.write_count += 1
                st.last_op = "write"
                st.edited_at.append(ev.idx)
            else:
                st.edit_count += 1
                st.last_op = "edit"
                st.edited_at.append(ev.idx)
    return states
