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


class CapLevel(StrEnum):
    """Capability lattice: ``false < partial < true``."""

    false = "false"
    partial = "partial"
    true = "true"


class TurnStatus(StrEnum):
    open = "open"
    completed = "completed"
    aborted = "aborted"


class RoleHint(StrEnum):
    user = "user"
    delegation = "delegation"
    injected = "injected"


class ArtifactOp(StrEnum):
    read = "read"
    search = "search"
    add = "add"
    update = "update"
    delete = "delete"
    move = "move"


class ThreadLinkKind(StrEnum):
    spawn = "spawn"
    fork = "fork"
    created = "created"


class PricingStatus(StrEnum):
    exact = "exact"
    estimated = "estimated"
    unknown = "unknown"


# Canonical op_kind vocabulary (mcp.<server>.<tool> is open-ended).
OP_KINDS = frozenset(
    {
        "cmd.read",
        "cmd.search",
        "cmd.list",
        "cmd.exec",
        "file.edit",
        "file.write",
        "file.delete",
        "web.search",
        "web.open",
        "agent.spawn",
        "agent.wait",
        "agent.message",
        "plan.update",
        "image.view",
        "other",
    }
)

_CAP_ORDER = {CapLevel.false: 0, CapLevel.partial: 1, CapLevel.true: 2}

# Capability field names — the validator and feature bank `requires` keys.
CAPABILITY_FIELDS = (
    "per_call_usage",
    "per_turn_model",
    "reasoning_effort",
    "reasoning_text",
    "reasoning_tokens",
    "timestamps",
    "op_timing",
    "parsed_commands",
    "file_diffs",
    "full_tool_output",
    "output_size_original",
    "thread_linkage",
    "compaction_summaries",
    "final_answer_marker",
    "context_window",
)


# ---------------------------------------------------------------------------
# Token accounting
# ---------------------------------------------------------------------------


class Usage(BaseModel):
    """Token counts for one event (or a cumulative running total).

    Normative billing fields: ``input`` is *uncached* input
    (``input_tokens - cached_input_tokens``), ``cache_read`` is cached input,
    ``cache_write`` is cache-write input, ``output`` is output tokens.
    ``reasoning_output`` is an informational subset of ``output`` and is never
    added into a token total or a cost.
    """

    model_config = ConfigDict(extra="ignore")

    input: int | None = None
    output: int | None = None
    cache_read: int | None = None
    cache_write: int | None = None
    reasoning_output: int | None = None
    # True when the source reports running totals (codex turn.completed) rather
    # than per-event deltas. ``de_cumulate`` rewrites these into deltas.
    cumulative: bool = False


# ---------------------------------------------------------------------------
# Canonical coordinates and ops
# ---------------------------------------------------------------------------


class SourceRef(BaseModel):
    """Immutable source coordinates, distinct from the canonical event idx."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    record_index: int
    ordinal: int | None = None
    byte_offset: int | None = None
    item_id: str | None = None


class ParsedCommand(BaseModel):
    """One ``parsed_cmd`` entry (Codex) or an empty classification."""

    model_config = ConfigDict(extra="ignore")

    type: str
    cmd: str | None = None
    path: str | None = None


class ArtifactInteraction(BaseModel):
    """One path touched by an op (read/search/add/update/delete/move)."""

    model_config = ConfigDict(extra="ignore")

    path: str
    op: ArtifactOp
    lines_added: int | None = None
    lines_removed: int | None = None
    diff_head: str | None = None


class ModelConfig(BaseModel):
    """Model / effort / collaboration settings active for a turn (last wins)."""

    model_config = ConfigDict(extra="ignore")

    provider: str | None = None
    model: str | None = None
    reasoning_effort: str | None = None
    collaboration_mode: str | None = None
    service_tier: str | None = None


class UserInput(BaseModel):
    """The user-facing input that opened a turn."""

    model_config = ConfigDict(extra="ignore")

    text: str = ""
    kind: Literal["user", "delegation", "injected", "none"] = "none"
    source_thread_id: str | None = None


class ModelCall(BaseModel):
    """One model response, keyed by ``response_id`` when the usage record arrived."""

    model_config = ConfigDict(extra="ignore")

    response_id: str | None = None  # None = unterminated trailing call
    turn_id: str | None = None
    event_start: int = 0
    event_end: int = 0  # inclusive
    usage: Usage | None = None
    ts_start: datetime | None = None
    ts_end: datetime | None = None


class Turn(BaseModel):
    """One user/delegation turn: lifecycle, model config, and its model calls."""

    model_config = ConfigDict(extra="ignore")

    turn_id: str
    session_id: str
    source_start: SourceRef | None = None
    source_end: SourceRef | None = None
    event_start: int = 0
    event_end: int = 0  # inclusive
    status: TurnStatus = TurnStatus.open
    user_input: UserInput = Field(default_factory=UserInput)
    model_config_active: ModelConfig = Field(default_factory=ModelConfig)
    model_config_revisions: list[ModelConfig] = Field(default_factory=list)
    context_window_tokens: int | None = None
    model_calls: list[ModelCall] = Field(default_factory=list)
    final_message: str | None = None
    compactions: int = 0
    ts_start: datetime | None = None
    ts_end: datetime | None = None


class CapabilitySet(BaseModel):
    """One side of a Capabilities pair (declared or observed)."""

    model_config = ConfigDict(extra="ignore")

    per_call_usage: CapLevel = CapLevel.false
    per_turn_model: CapLevel = CapLevel.false
    reasoning_effort: CapLevel = CapLevel.false
    reasoning_text: CapLevel = CapLevel.false
    reasoning_tokens: CapLevel = CapLevel.false
    timestamps: CapLevel = CapLevel.false
    op_timing: CapLevel = CapLevel.false
    parsed_commands: CapLevel = CapLevel.false
    file_diffs: CapLevel = CapLevel.false
    full_tool_output: CapLevel = CapLevel.false
    output_size_original: CapLevel = CapLevel.false
    thread_linkage: CapLevel = CapLevel.false
    compaction_summaries: CapLevel = CapLevel.false
    final_answer_marker: CapLevel = CapLevel.false
    context_window: CapLevel = CapLevel.false


def cap_min(*levels: CapLevel) -> CapLevel:
    """Lattice minimum: ``false < partial < true``."""
    if not levels:
        return CapLevel.false
    return min(levels, key=lambda lv: _CAP_ORDER[lv])


def cap_max(*levels: CapLevel) -> CapLevel:
    """Lattice maximum."""
    if not levels:
        return CapLevel.false
    return max(levels, key=lambda lv: _CAP_ORDER[lv])


class Capabilities(BaseModel):
    """Declared (format/version) vs observed (this session) capability rows."""

    model_config = ConfigDict(extra="ignore")

    declared: CapabilitySet = Field(default_factory=CapabilitySet)
    observed: CapabilitySet = Field(default_factory=CapabilitySet)

    def meets(self, name: str) -> bool:
        """True when the *declared* capability is not ``false``."""
        return getattr(self.declared, name, CapLevel.false) is not CapLevel.false

    @classmethod
    def merge_min(cls, rows: list[Capabilities]) -> Capabilities:
        """Lattice-min merge across a thread tree (C12)."""
        if not rows:
            return cls()
        declared = CapabilitySet()
        observed = CapabilitySet()
        for field in CAPABILITY_FIELDS:
            setattr(declared, field, cap_min(*(getattr(r.declared, field) for r in rows)))
            setattr(observed, field, cap_min(*(getattr(r.observed, field) for r in rows)))
        return cls(declared=declared, observed=observed)


class ThreadLink(BaseModel):
    """One parent→child edge in a thread tree, with ranked evidence."""

    model_config = ConfigDict(extra="ignore")

    child_id: str
    parent_id: str
    kind: ThreadLinkKind
    depth: int | None = None
    history_base: dict[str, Any] | None = None
    evidence: list[str] = Field(default_factory=list)
    confidence: Literal["high", "medium", "low"] = "high"
    notes: list[str] = Field(default_factory=list)


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
    ts_end: datetime | None = None
    agent: AgentKind = AgentKind.unknown

    # message / thinking payloads
    text: str | None = None  # flattened text for user_msg/assistant_msg/thinking
    phase: str | None = None  # AgentMessage.phase: commentary | final_answer
    role_hint: RoleHint | None = None

    # tool_call
    tool_name: str | None = None
    tool_category: ToolCategory | None = None
    tool_args: dict[str, Any] = Field(default_factory=dict)  # normalized-ish raw args
    tool_norm_args: str | None = None  # arg-normalizer output (RETRY_STORM/NO_ADAPT_RETRY)
    call_id: str | None = None  # links tool_call <-> tool_result
    op_kind: str | None = None
    classifications: list[ParsedCommand] = Field(default_factory=list)
    artifacts: list[ArtifactInteraction] = Field(default_factory=list)

    # tool_result
    ok: bool | None = None  # False == errored (is_error / nonzero exit)
    exit_code: int | None = None
    error_text: str | None = None
    output: str | None = None  # truncated to config cap (head+tail)
    error_category: str | None = None  # filled by error classifier at build time
    output_tokens_original: int | None = None

    # file op convenience (write/edit/read) — first artifact, for legacy readers
    path: str | None = None
    lines_added: int | None = None
    lines_removed: int | None = None

    # span / graph
    span_id: str | None = None  # this event's own id (uuid in native formats)
    parent_span_id: str | None = None  # parentUuid / parent_tool_use_id
    usage: Usage | None = None

    # source coordinates + grouping
    source: SourceRef | None = None
    turn_id: str | None = None
    response_id: str | None = None
    group_id: str | None = None  # wrapper interval id (custom_tool_call exec)

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
    harness_version: str | None = None
    thread_linkage: Literal["full", "partial", "none"] | None = None


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

    turns: list[Turn] = Field(default_factory=list)
    capabilities: Capabilities = Field(default_factory=Capabilities)
    harness_version: str | None = None
    thread_source: str | None = None
    replay_prefix: SourceRef | None = None  # end of inherited prefix, if any
    degraded: list[str] = Field(default_factory=list)


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
    links: list[ThreadLink] = Field(default_factory=list)
    capabilities: Capabilities = Field(default_factory=Capabilities)


__all__ = [
    "CAPABILITY_FIELDS",
    "OP_KINDS",
    "AgentKind",
    "ArtifactInteraction",
    "ArtifactOp",
    "CapLevel",
    "Capabilities",
    "CapabilitySet",
    "Event",
    "EventKind",
    "FileState",
    "ModelCall",
    "ModelConfig",
    "ParsedCommand",
    "PricingStatus",
    "Provenance",
    "RoleHint",
    "Session",
    "SourceRef",
    "ThreadLink",
    "ThreadLinkKind",
    "ToolCategory",
    "Trace",
    "Turn",
    "TurnStatus",
    "Usage",
    "UserInput",
    "cap_max",
    "cap_min",
]
