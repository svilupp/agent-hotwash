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

import re
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from agent_hotwash.events import ArtifactOp, Event, EventKind, Session, ToolCategory, Trace
from agent_hotwash.primitives.commands import exit1_is_signal_free

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

_GREP_LIKE = {"grep", "glob", "find", "rg", "egrep", "ugrep", "cmd.search"}
# Targeted-edit tool names (as opposed to a full-file Write). Includes codex's
# synthetic `file_change` / canonical `file.edit`.
_EDIT_TOOLS = {"edit", "multiedit", "notebookedit", "file_change", "file.edit"}
_WRITE_TOOLS = {"write", "file.write"}


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
    return [e for e in logical_events(session) if e.kind is EventKind.user_msg]


def assistant_msgs(session: Session) -> list[Event]:
    return [e for e in logical_events(session) if e.kind is EventKind.assistant_msg]


def tool_calls(session: Session) -> list[Event]:
    return [e for e in logical_events(session) if e.kind is EventKind.tool_call]


def logical_events(session: Session) -> list[Event]:
    """Detector window: every event except ``meta`` (wrapper/settings noise)."""
    from agent_hotwash.canonical import logical_events as _logical

    return _logical(session)


def is_read_call(ev: Event) -> bool:
    return ev.kind is EventKind.tool_call and ev.tool_category is ToolCategory.read


def is_write_call(ev: Event) -> bool:
    return ev.kind is EventKind.tool_call and ev.tool_category is ToolCategory.write


def is_edit_tool(ev: Event) -> bool:
    """A targeted-edit tool (Edit/MultiEdit/file.edit), not a full Write.

    A FileChange whose artifacts are all ``add``/``delete`` creates or removes
    files — nothing pre-existing was edited — so it is not an edit tool call.
    """
    if not is_write_call(ev):
        return False
    if ev.artifacts and all(a.op in (ArtifactOp.add, ArtifactOp.delete) for a in ev.artifacts):
        return False
    if ev.op_kind == "file.edit":
        return True
    if ev.op_kind in ("file.write", "file.delete"):
        return False
    return (ev.tool_name or "").lower() in _EDIT_TOOLS


def edited_paths(ev: Event) -> list[str]:
    """Paths a write call modifies in place (``update`` artifacts; not
    add/delete/move). Falls back to ``ev.path`` when no artifacts are present."""
    if ev.artifacts:
        return [a.path for a in ev.artifacts if a.path and a.op is ArtifactOp.update]
    return [ev.path] if ev.path else []


def read_paths(ev: Event) -> list[str]:
    """Paths a read/search call touched (all ``read``/``search`` artifacts, or
    the legacy single ``ev.path`` for a read-category call)."""
    if ev.kind is not EventKind.tool_call:
        return []
    if ev.artifacts:
        return [a.path for a in ev.artifacts if a.path and a.op in (ArtifactOp.read, ArtifactOp.search)]
    if ev.tool_category is ToolCategory.read and ev.path:
        return [ev.path]
    return []


def is_under_dir(path: str, directory: str) -> bool:
    """``path`` lies inside ``directory`` (string prefix on a ``/`` boundary)."""
    d = directory.rstrip("/")
    return bool(d) and path.startswith(d + "/")


def is_grep_like(ev: Event) -> bool:
    if ev.kind is not EventKind.tool_call:
        return False
    if ev.op_kind == "cmd.search":
        return True
    return (ev.tool_name or "").lower() in _GREP_LIKE


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


def result_by_call(session: Session) -> dict[str, Event]:
    """Map call_id -> the linked tool_result event."""
    return {e.call_id: e for e in session.events if e.kind is EventKind.tool_result and e.call_id}


def failing_results(session: Session) -> list[Event]:
    """tool_result events that errored (ok is False)."""
    return [e for e in session.events if e.kind is EventKind.tool_result and e.ok is False]


# Exit statuses of a process killed by the harness/user (SIGINT / SIGTERM).
_KILLED_EXITS = {130, 143}
# A read/search command that exits 1 with at most this much output is a probe
# that found nothing, not a failure.
BENIGN_PROBE_OUTPUT_CHARS = 300


def is_killed_result(ev: Event) -> bool:
    """A tool_result of a process that was interrupted/killed (exit 130/143 or a
    trailing ``^C``) — not an error the agent should have handled."""
    if ev.kind is not EventKind.tool_result:
        return False
    if ev.exit_code in _KILLED_EXITS or ev.error_category == "cancelled":
        return True
    tail = (ev.error_text or ev.output or "").rstrip()
    return tail.endswith("^C")


def is_benign_failure(result: Event, call: Event | None = None) -> bool:
    """A failing ``tool_result`` (``ok is False``) that carries no real signal.

    Benign: ``no_match_probe`` results (rg/grep exit 1), killed/cancelled
    processes (exit 130/143, ``^C``), read/search commands exiting 1 with a
    tiny output, and commands whose exit 1 means "differs"/"false" (``diff``,
    ``cmp``, ``test``, ``git diff --check``). Returns ``False`` for results that
    did not fail at all.
    """
    if result.kind is not EventKind.tool_result or result.ok is not False:
        return False
    if result.error_category == "no_match_probe" or is_killed_result(result):
        return True
    if result.error_category == "file_not_found":
        return False  # a missing file is real signal, however short the output
    if result.exit_code != 1:
        return False
    command = bash_command(call) if call is not None else ""
    if command and exit1_is_signal_free(command):
        return True
    # Read/search/list probes exit 1 with (near-)empty output when nothing
    # matched (missing files were excluded above via ``file_not_found``).
    read_like = call is not None and (
        call.tool_category is ToolCategory.read or (call.op_kind or "") in ("cmd.read", "cmd.search", "cmd.list")
    )
    output = result.error_text or result.output or ""
    return read_like and len(output.strip()) <= BENIGN_PROBE_OUTPUT_CHARS


def is_commentary(ev: Event) -> bool:
    """A Codex ``phase == "commentary"`` assistant message (progress narration
    mid-turn, not the turn's answer)."""
    return ev.kind is EventKind.assistant_msg and ev.phase == "commentary"


def terminal_assistant_msgs(session: Session) -> list[Event]:
    """Assistant messages that close a turn.

    Codex marks these with ``phase == "final_answer"``. For harnesses without a
    phase marker (``phase is None``) the last assistant message before the next
    ``user_msg`` / a ``task_complete`` meta / the end of the stream is terminal.
    ``commentary`` messages are never terminal.
    """
    out: list[Event] = []
    pending: Event | None = None
    for ev in session.events:
        if ev.kind is EventKind.assistant_msg:
            if ev.phase == "final_answer":
                out.append(ev)
                pending = None
            elif ev.phase is None:
                pending = ev
            continue
        if ev.kind is EventKind.user_msg or (ev.kind is EventKind.meta and ev.raw_type == "task_complete"):
            if pending is not None:
                out.append(pending)
            pending = None
    if pending is not None:
        out.append(pending)
    return out


def is_turn_end(ev: Event) -> bool:
    """A ``task_complete`` meta marker (Codex) closing the current turn."""
    return ev.kind is EventKind.meta and ev.raw_type == "task_complete"


def logical_positions(session: Session) -> dict[int, int]:
    """Map ``Event.idx`` -> position in :func:`logical_events` (meta excluded),
    so distances between events can be measured in logical events."""
    return {ev.idx: i for i, ev in enumerate(logical_events(session))}


_SOURCE_EXT_RE = re.compile(
    r"\.(py|pyi|js|jsx|ts|tsx|mjs|cjs|json|ya?ml|toml|ini|cfg|conf|md|rst|sh|bash|zsh|go|rs|"
    r"java|kt|rb|php|c|h|cc|cpp|hpp|cs|css|scss|html?|sql|env|xml|vue|svelte|txt|lock|proto|"
    r"tf|dockerfile|mk|cmake|gradle|swift|m|mm|ex|exs|erl|hs|lua|pl|r|jl|dart|scala|clj|ipynb)$",
    re.IGNORECASE,
)


def is_source_like_path(path: str) -> bool:
    """A file path with a source-ish extension (not a directory or a bare name)."""
    base = path.rstrip("/").rsplit("/", 1)[-1]
    if base in ("", ".", ".."):
        return False
    return bool(_SOURCE_EXT_RE.search(base))


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
