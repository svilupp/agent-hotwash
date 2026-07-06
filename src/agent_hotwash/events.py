"""Normalized event model — the single stream every layer keys off.

Parsers are the only code that knows about raw agent formats; they emit these
models and everything downstream (analytics, smells, taxonomies, reporting)
reads them. All timestamps are optional: pi and codex code-bench runs have no
per-event wall clock, so every consumer must be correct when ``ts is None``.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from agent_hotwash.primitives.filestate import FileState

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class EventKind(StrEnum):
    session_start = "session_start"
    session_end = "session_end"
    user_msg = "user_msg"
    assistant_msg = "assistant_msg"
    thinking = "thinking"
    tool_call = "tool_call"
    tool_result = "tool_result"
    compaction = "compaction"
    meta = "meta"  # turn boundaries, token_count events, anything else


class AgentKind(StrEnum):
    claude = "claude"
    codex = "codex"
    pi = "pi"
    unknown = "unknown"


class ToolCategory(StrEnum):
    read = "read"
    write = "write"
    execute = "execute"
    planning = "planning"
    subagent = "subagent"
    other = "other"


# ---------------------------------------------------------------------------
# Token accounting
# ---------------------------------------------------------------------------


class Usage(BaseModel):
    """Token counts for one event (or a cumulative running total)."""

    model_config = ConfigDict(extra="ignore")

    input: int | None = None
    output: int | None = None
    cache_read: int | None = None
    cache_write: int | None = None
    # True when the source reports running totals (codex turn.completed) rather
    # than per-event deltas. ``de_cumulate`` rewrites these into deltas.
    cumulative: bool = False


# ---------------------------------------------------------------------------
# Event
# ---------------------------------------------------------------------------


class Event(BaseModel):
    """One row of the normalized stream. Fields are a superset; each parser
    fills only what its format carries."""

    model_config = ConfigDict(extra="ignore")

    kind: EventKind
    idx: int = -1  # 0-based position within its Session (assigned by builder)
    ts: datetime | None = None  # OPTIONAL — never assume present
    agent: AgentKind = AgentKind.unknown

    # message / thinking payloads
    text: str | None = None  # flattened text for user_msg/assistant_msg/thinking

    # tool_call
    tool_name: str | None = None
    tool_category: ToolCategory | None = None
    tool_args: dict[str, Any] = Field(default_factory=dict)  # normalized-ish raw args
    tool_norm_args: str | None = None  # arg-normalizer output (RETRY_STORM/NO_ADAPT_RETRY)
    call_id: str | None = None  # links tool_call <-> tool_result

    # tool_result
    ok: bool | None = None  # False == errored (is_error / nonzero exit)
    exit_code: int | None = None
    error_text: str | None = None
    output: str | None = None  # truncated to config cap
    error_category: str | None = None  # filled by error classifier at build time

    # file op convenience (write/edit/read) — parsed from tool_args at build time
    path: str | None = None
    lines_added: int | None = None
    lines_removed: int | None = None

    # span / graph
    span_id: str | None = None  # this event's own id (uuid in native formats)
    parent_span_id: str | None = None  # parentUuid / parent_tool_use_id
    usage: Usage | None = None

    raw_type: str | None = None  # original event/type string, for debugging


# ---------------------------------------------------------------------------
# Session and Trace containers
# ---------------------------------------------------------------------------


class Provenance(BaseModel):
    """Where a Trace came from and how confidently the format was detected."""

    model_config = ConfigDict(extra="ignore")

    source_format: Literal["codebench", "claude_native", "codex_native", "pi_native"]
    detector_confidence: Literal["high", "low"]
    root_path: Path
    files: list[Path] = Field(default_factory=list)  # every file that fed this trace
    harness_meta: dict[str, Any] = Field(default_factory=dict)  # run/metrics/verification json
    notes: list[str] = Field(default_factory=list)  # degradation notes


class Session(BaseModel):
    """One continuous agent conversation (one stdout stream / native file)."""

    model_config = ConfigDict(extra="ignore")

    session_id: str
    agent: AgentKind
    model: str | None = None
    events: list[Event] = Field(default_factory=list)
    parent_session_id: str | None = None  # set for linked subagent sessions
    # derived, computed once by the builder:
    file_state: dict[str, FileState] = Field(default_factory=dict)
    # ``has_timestamps`` is the STRICT flag: True only when a large-enough
    # fraction of events carry a wall clock (see ``ts_coverage``). Time-based
    # metrics gate on it so partial coverage (e.g. code-bench claude, which
    # timestamps only user records) does not yield misleading durations or
    # idle-gap false positives. ``has_any_timestamps`` is the loose flag.
    has_timestamps: bool = False
    has_any_timestamps: bool = False
    ts_coverage: float = 0.0  # fraction of events carrying a timestamp
    usage_reliable: bool = True  # False when stream usage was zero -> fell back


class Trace(BaseModel):
    """The analyzable unit: a root Session plus linked subagents + provenance."""

    model_config = ConfigDict(extra="ignore")

    trace_id: str  # stable: hash(root_path) or run_id for format A
    agent: AgentKind
    model: str | None = None
    experiment: str | None = None  # format A groups; None otherwise
    instance_id: str | None = None
    root: Session
    subagents: list[Session] = Field(default_factory=list)  # LINKED, not inlined
    provenance: Provenance
    resolved: bool | None = None  # ground-truth pass/fail from harness_meta (format A)


__all__ = [
    "AgentKind",
    "Event",
    "EventKind",
    "FileState",
    "Provenance",
    "Session",
    "ToolCategory",
    "Trace",
    "Usage",
]
