"""Detector registry, the ``Finding`` model, and the run entry point.

Detectors are pure functions ``(Session, Config) -> list[Finding]`` registered by
the :func:`detector` decorator. They never import each other and never mutate the
input Session. :func:`run_detectors` iterates a Trace's root + linked subagent
sessions (or a bare Session), runs every *enabled* detector, applies any config
severity override, and returns findings in a deterministic order.

The LLM-judge seam is the ``tier`` field on a registered detector: v1 registers
only ``tier="rule"``; a future ``detectors/llm/`` can register ``tier="llm"``
detectors emitting the same :class:`Finding` interface.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from agent_hotwash.events import Event, EventKind, Session, ToolCategory, Trace

if TYPE_CHECKING:
    from agent_hotwash.config import Config


class Severity(StrEnum):
    info = "info"
    low = "low"
    medium = "medium"
    high = "high"


# Ordering used by the ``--fail-on`` CI gate and any severity comparison.
_SEVERITY_ORDER = {Severity.info: 0, Severity.low: 1, Severity.medium: 2, Severity.high: 3}


def severity_rank(sev: Severity | str) -> int:
    """Numeric rank of a severity (info < low < medium < high)."""
    return _SEVERITY_ORDER[Severity(sev)]


class SpanRef(BaseModel):
    """A pointer into a session's event stream (a point or an inclusive range)."""

    model_config = ConfigDict(extra="ignore")

    session_id: str
    event_idx: int
    end_idx: int | None = None  # for range findings


class Finding(BaseModel):
    """One detector hit: what fired, where, why, and how sure we are."""

    model_config = ConfigDict(extra="ignore")

    id: str  # taxonomy/smell id, e.g. "EDIT_THRASH"
    kind: Literal["smell", "taxonomy"]
    severity: Severity
    confidence: Literal["high", "low"]  # "low" for the fuzzy detectors
    session_id: str
    spans: list[SpanRef] = Field(default_factory=list)
    evidence: dict[str, Any] = Field(default_factory=dict)
    message: str = ""


DetectorFn = Callable[[Session, "Config"], list[Finding]]


@dataclass(frozen=True)
class DetectorSpec:
    """Registry entry for one detector."""

    id: str
    kind: Literal["smell", "taxonomy"]
    tier: Literal["rule", "llm"]
    default_severity: Severity
    default_confidence: Literal["high", "low"]
    fn: DetectorFn
    doc: str = ""
    llm_candidate: bool = False


# Insertion-ordered registry (dict preserves order -> deterministic runs).
_REGISTRY: dict[str, DetectorSpec] = {}


def detector(
    detector_id: str,
    *,
    kind: Literal["smell", "taxonomy"],
    severity: Severity,
    confidence: Literal["high", "low"] = "high",
    tier: Literal["rule", "llm"] = "rule",
    llm_candidate: bool = False,
) -> Callable[[DetectorFn], DetectorFn]:
    """Register a detector function under ``detector_id``.

    ``confidence`` is the detector's default emitted confidence ("low" for the
    fuzzy taxonomies). ``llm_candidate`` flags a rule-tier detector that is a
    candidate for an LLM upgrade.
    """

    def deco(fn: DetectorFn) -> DetectorFn:
        if detector_id in _REGISTRY:
            raise ValueError(f"detector id already registered: {detector_id}")
        _REGISTRY[detector_id] = DetectorSpec(
            id=detector_id,
            kind=kind,
            tier=tier,
            default_severity=severity,
            default_confidence=confidence,
            fn=fn,
            doc=(fn.__doc__ or "").strip(),
            llm_candidate=llm_candidate,
        )
        return fn

    return deco


def get_registry() -> dict[str, DetectorSpec]:
    """The full detector registry (id -> spec), in registration order."""
    return dict(_REGISTRY)


def _apply_overrides(findings: list[Finding], config: Config) -> list[Finding]:
    overrides = config.detectors.severity
    if not overrides:
        return findings
    for f in findings:
        if f.id in overrides:
            f.severity = Severity(overrides[f.id])
    return findings


def _run_one_session(session: Session, config: Config) -> list[Finding]:
    out: list[Finding] = []
    for spec in _REGISTRY.values():
        if not config.detectors.is_enabled(spec.id):
            continue
        out.extend(spec.fn(session, config))
    return _apply_overrides(out, config)


def run_detectors(session_or_trace: Session | Trace, config: Config) -> list[Finding]:
    """Run every enabled detector over a Session, or a Trace's root + subagents.

    Findings are returned in a stable order: by originating session (root first,
    then subagents in order), then registration order, then span position.
    """
    findings: list[Finding] = []
    if isinstance(session_or_trace, Trace):
        sessions = [session_or_trace.root, *session_or_trace.subagents]
    else:
        sessions = [session_or_trace]
    for sess in sessions:
        findings.extend(_run_one_session(sess, config))
    return findings


# ---------------------------------------------------------------------------
# Shared, read-only helpers for detector functions. Kept here so smells.py and
# taxonomy.py never import each other.
# ---------------------------------------------------------------------------

_GREP_LIKE = {"grep", "glob", "find", "rg", "egrep", "ugrep"}
# Targeted-edit tool names (as opposed to a full-file Write). Includes codex's
# synthetic `file_change`, which applies a patch to an existing file.
_EDIT_TOOLS = {"edit", "multiedit", "notebookedit", "file_change"}


def span(session: Session, idx: int, end: int | None = None) -> SpanRef:
    """Build a :class:`SpanRef` tagged with the session id."""
    return SpanRef(session_id=session.session_id, event_idx=idx, end_idx=end)


def make_finding(
    spec_id: str,
    session: Session,
    *,
    kind: Literal["smell", "taxonomy"],
    severity: Severity,
    confidence: Literal["high", "low"],
    spans: list[SpanRef],
    evidence: dict[str, Any],
    message: str,
) -> Finding:
    return Finding(
        id=spec_id,
        kind=kind,
        severity=severity,
        confidence=confidence,
        session_id=session.session_id,
        spans=spans,
        evidence=evidence,
        message=message,
    )


def user_msgs(session: Session) -> list[Event]:
    return [e for e in session.events if e.kind is EventKind.user_msg]


def assistant_msgs(session: Session) -> list[Event]:
    return [e for e in session.events if e.kind is EventKind.assistant_msg]


def tool_calls(session: Session) -> list[Event]:
    return [e for e in session.events if e.kind is EventKind.tool_call]


def is_read_call(ev: Event) -> bool:
    return ev.kind is EventKind.tool_call and ev.tool_category is ToolCategory.read


def is_write_call(ev: Event) -> bool:
    return ev.kind is EventKind.tool_call and ev.tool_category is ToolCategory.write


def is_edit_tool(ev: Event) -> bool:
    """A targeted-edit tool (Edit/MultiEdit/NotebookEdit), not a full Write."""
    return is_write_call(ev) and (ev.tool_name or "").lower() in _EDIT_TOOLS


def is_grep_like(ev: Event) -> bool:
    return ev.kind is EventKind.tool_call and (ev.tool_name or "").lower() in _GREP_LIKE


def is_exec_call(ev: Event) -> bool:
    return ev.kind is EventKind.tool_call and ev.tool_category is ToolCategory.execute


def bash_command(ev: Event) -> str:
    """Shell command string of an execute tool_call ('' when not one)."""
    if not is_exec_call(ev):
        return ""
    args = ev.tool_args or {}
    for key in ("command", "cmd", "script"):
        val = args.get(key)
        if isinstance(val, str):
            return val
    return ""


def result_ok_by_call(session: Session) -> dict[str, bool | None]:
    """Map call_id -> the linked tool_result's ok flag."""
    out: dict[str, bool | None] = {}
    for ev in session.events:
        if ev.kind is EventKind.tool_result and ev.call_id is not None:
            out[ev.call_id] = ev.ok
    return out


def call_by_id(session: Session) -> dict[str, Event]:
    return {e.call_id: e for e in session.events if e.kind is EventKind.tool_call and e.call_id}


def failing_results(session: Session) -> list[Event]:
    """tool_result events that errored (ok is False)."""
    return [e for e in session.events if e.kind is EventKind.tool_result and e.ok is False]


def approx_tokens(text: str | None) -> int:
    """Cheap token estimate (~4 chars/token) for text without usage data."""
    if not text:
        return 0
    return max(1, len(text) // 4)


def total_tokens(session: Session) -> int:
    """Sum of input+output token deltas across events (0 when no usage)."""
    total = 0
    for ev in session.events:
        u = ev.usage
        if u is None:
            continue
        total += (u.input or 0) + (u.output or 0)
    return total


def peak_input_tokens(session: Session) -> int:
    """Largest single-request input (+cache_read) size, a proxy for peak context."""
    best = 0
    for ev in session.events:
        u = ev.usage
        if u is None:
            continue
        best = max(best, (u.input or 0) + (u.cache_read or 0))
    return best


def word_count(text: str | None) -> int:
    return len((text or "").split())


__all__ = [
    "DetectorFn",
    "DetectorSpec",
    "Finding",
    "Severity",
    "SpanRef",
    "detector",
    "get_registry",
    "make_finding",
    "run_detectors",
    "severity_rank",
    "span",
]
