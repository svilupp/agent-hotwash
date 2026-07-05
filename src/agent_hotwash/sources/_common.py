"""Shared source helpers.

=============================================================================
WP1 — pure helpers (this section).  WP2 adds the raw per-format decoders below.
=============================================================================

Everything here is pure (except ``iter_jsonl``, which reads a file). The Session
builder ``build_session`` turns a parser's raw event list into a fully derived
:class:`~agent_hotwash.events.Session`: it assigns ``idx``, links
tool_call<->tool_result, runs the error classifier, parses file ops, builds the
per-file state, de-cumulates usage, and sets the reliability flags.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from agent_hotwash.events import (
    AgentKind,
    Event,
    EventKind,
    Session,
    ToolCategory,
    Usage,
)
from agent_hotwash.primitives.argnorm import norm_args
from agent_hotwash.primitives.errors import classify_error
from agent_hotwash.primitives.filestate import build_file_state

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence
    from pathlib import Path

# Minimum fraction of events that must carry a timestamp for the strict
# ``Session.has_timestamps`` flag (and thus every time-based metric) to trust
# the wall clock. Below this, timing degrades to None rather than mislead.
TS_COVERAGE_MIN = 0.5

# ---------------------------------------------------------------------------
# jsonl + text
# ---------------------------------------------------------------------------

# Tool/item name -> coarse category. Matched case-sensitively first, then
# lower-cased. codex has no named tools; its item types are mapped as synthetic
# names (command_execution/file_change).
_TOOL_CATEGORY: dict[str, ToolCategory] = {
    # read
    "Read": ToolCategory.read,
    "Glob": ToolCategory.read,
    "Grep": ToolCategory.read,
    "LS": ToolCategory.read,
    "WebFetch": ToolCategory.read,
    "WebSearch": ToolCategory.read,
    "NotebookRead": ToolCategory.read,
    "read": ToolCategory.read,
    "grep": ToolCategory.read,
    "find": ToolCategory.read,
    "ls": ToolCategory.read,
    "glob": ToolCategory.read,
    "cat": ToolCategory.read,
    # write
    "Write": ToolCategory.write,
    "Edit": ToolCategory.write,
    "MultiEdit": ToolCategory.write,
    "NotebookEdit": ToolCategory.write,
    "write": ToolCategory.write,
    "edit": ToolCategory.write,
    "file_change": ToolCategory.write,  # codex synthetic
    # execute
    "Bash": ToolCategory.execute,
    "bash": ToolCategory.execute,
    "command_execution": ToolCategory.execute,  # codex synthetic
    # planning
    "TodoWrite": ToolCategory.planning,
    "Task": ToolCategory.planning,
    "EnterPlanMode": ToolCategory.planning,
    "ExitPlanMode": ToolCategory.planning,
    # subagent — claude and pi spawn subagents via a tool named `Agent`.
    "Agent": ToolCategory.subagent,
}


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """Yield decoded JSON objects from a `.jsonl` file, skipping blank/non-`{`
    lines and tolerating decode errors (mirrors the reference ``_iter_events``)."""
    with path.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                yield obj


def flatten_text(content: Any) -> str:
    """Flatten a message/thinking/tool_result ``content`` to plain text.

    Accepts a plain string or a list of content blocks (dicts carrying ``text``);
    anything else flattens to ``""``.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                t = block.get("text")
                if isinstance(t, str):
                    parts.append(t)
            elif isinstance(block, str):
                parts.append(block)
        return " ".join(parts)
    return ""


def tool_category_of(name: str | None) -> ToolCategory:
    """Coarse category for a tool/item name; ``other`` when unknown."""
    if not name:
        return ToolCategory.other
    if name in _TOOL_CATEGORY:
        return _TOOL_CATEGORY[name]
    return _TOOL_CATEGORY.get(name.lower(), ToolCategory.other)


# ---------------------------------------------------------------------------
# usage de-cumulation
# ---------------------------------------------------------------------------


def _delta(cur: int | None, prev: int | None) -> int | None:
    if cur is None:
        return None
    d = cur - (prev or 0)
    return d if d >= 0 else cur  # a reset (cur < prev) is treated as a fresh delta


def de_cumulate(usages: Sequence[Usage | None]) -> list[Usage | None]:
    """Rewrite a stream of usage objects into per-event deltas.

    Sources that report running totals (codex ``turn.completed``) set
    ``Usage.cumulative``; each such entry is converted to the delta from the
    previous cumulative entry, so the resulting per-event deltas remain summable.
    Non-cumulative usages pass through unchanged. Positions with no usage stay
    ``None``.
    """
    out: list[Usage | None] = []
    prev = Usage()
    for u in usages:
        if u is None or not u.cumulative:
            out.append(u)
            continue
        out.append(
            Usage(
                input=_delta(u.input, prev.input),
                output=_delta(u.output, prev.output),
                cache_read=_delta(u.cache_read, prev.cache_read),
                cache_write=_delta(u.cache_write, prev.cache_write),
                cumulative=False,
            )
        )
        prev = u
    return out


# ---------------------------------------------------------------------------
# file-op parsing
# ---------------------------------------------------------------------------

_PATH_KEYS = ("file_path", "path", "notebook_path", "filename", "filepath")


def _extract_path(args: dict[str, Any]) -> str | None:
    for key in _PATH_KEYS:
        val = args.get(key)
        if isinstance(val, str) and val:
            return val
    return None


def _count_lines(text: Any) -> int | None:
    if not isinstance(text, str) or not text:
        return None
    return text.count("\n") + 1


def _parse_file_op(ev: Event) -> None:
    """Populate ``path`` / ``lines_added`` / ``lines_removed`` on a file-op
    tool_call from its args (best-effort)."""
    args = ev.tool_args or {}
    ev.path = ev.path or _extract_path(args)
    name = (ev.tool_name or "").lower()
    if name == "write":
        ev.lines_added = _count_lines(args.get("content") or args.get("file_text"))
    elif name in ("edit", "multiedit"):
        ev.lines_added = _count_lines(args.get("new_string"))
        ev.lines_removed = _count_lines(args.get("old_string"))


# ---------------------------------------------------------------------------
# Session builder
# ---------------------------------------------------------------------------


def _command_of(ev: Event) -> str:
    args = ev.tool_args or {}
    for key in ("command", "cmd", "script"):
        val = args.get(key)
        if isinstance(val, str):
            return val
    return ""


def build_session(
    raw_events: list[Event],
    agent: AgentKind,
    *,
    session_id: str,
    model: str | None = None,
    parent_session_id: str | None = None,
    usage_reliable: bool = True,
) -> Session:
    """Turn a parser's raw event list into a fully derived Session.

    Order of operations (DESIGN §2.3): (1) keep source order, assign ``idx``;
    (2) link tool_call<->tool_result by ``call_id``; (3) classify errored
    results; (4) parse file ops; (5) build per-file state; (6) de-cumulate usage;
    (7) set flags.
    """
    events = list(raw_events)
    for i, ev in enumerate(events):
        ev.idx = i
        if ev.agent is AgentKind.unknown:
            ev.agent = agent

    # (2) link calls <-> results
    call_by_id: dict[str, Event] = {ev.call_id: ev for ev in events if ev.kind is EventKind.tool_call and ev.call_id}

    # (3) classify errored results, using the linked call for tool name/command
    for ev in events:
        if ev.kind is not EventKind.tool_result:
            continue
        call = call_by_id.get(ev.call_id) if ev.call_id else None
        if ev.ok is False or (ev.exit_code is not None and ev.exit_code != 0):
            ev.ok = False
            tool = (call.tool_name if call else ev.tool_name) or ""
            command = _command_of(call) if call else _command_of(ev)
            msg = ev.error_text or ev.output or ""
            category, _severity, _conf = classify_error(tool, ev.exit_code, msg, command)
            ev.error_category = category

    # (4) parse file ops
    for ev in events:
        if ev.kind is EventKind.tool_call and ev.tool_category is None:
            ev.tool_category = tool_category_of(ev.tool_name)
        if ev.kind is EventKind.tool_call and ev.tool_category in (
            ToolCategory.read,
            ToolCategory.write,
        ):
            _parse_file_op(ev)
        if ev.kind is EventKind.tool_call and ev.tool_norm_args is None and ev.tool_args:
            ev.tool_norm_args = norm_args(ev.tool_args)

    # (5) per-file state
    file_state = build_file_state(events)

    # (6) de-cumulate usage
    if any(ev.usage is not None and ev.usage.cumulative for ev in events):
        deltas = de_cumulate([ev.usage for ev in events])
        for ev, u in zip(events, deltas, strict=True):
            ev.usage = u

    # (7) flags — distinguish "some events stamped" from "enough to time by".
    # A source that stamps only a minority of events (code-bench claude stamps
    # only user records) gives sparse, misleading timing, so the strict
    # ``has_timestamps`` flag that gates every time-based metric requires the
    # stamped fraction to reach ``TS_COVERAGE_MIN``.
    stamped = sum(1 for ev in events if ev.ts is not None)
    ts_coverage = stamped / len(events) if events else 0.0
    has_any_timestamps = stamped > 0
    has_timestamps = ts_coverage >= TS_COVERAGE_MIN

    return Session(
        session_id=session_id,
        agent=agent,
        model=model,
        events=events,
        parent_session_id=parent_session_id,
        file_state=file_state,
        has_timestamps=has_timestamps,
        has_any_timestamps=has_any_timestamps,
        ts_coverage=ts_coverage,
        usage_reliable=usage_reliable,
    )


# =============================================================================
# WP2 — raw decoder helpers (shared by the per-format parsers below this banner).
# =============================================================================

# Tool_result output is stored up to this many chars; analytics can re-truncate
# to the config cap. Parsers have no Config, so this is a conservative default.
OUTPUT_TRUNCATE = 2000


def truncate(text: str | None, cap: int = OUTPUT_TRUNCATE) -> str | None:
    """Cap a tool output/error string to ``cap`` chars (best-effort, no config)."""
    if text is None:
        return None
    if len(text) <= cap:
        return text
    return text[:cap]


def parse_ts(value: Any) -> datetime | None:
    """Parse a timestamp into an aware ``datetime`` (UTC), or ``None``.

    Accepts ISO-8601 strings (with a trailing ``Z``) and epoch milliseconds
    (pi ``message.timestamp``). Anything unparseable yields ``None`` so the
    normalized stream degrades gracefully rather than raising.
    """
    if value is None:
        return None
    if isinstance(value, bool):  # guard: bool is an int subclass
        return None
    if isinstance(value, (int, float)):
        # epoch — treat large values as milliseconds, else seconds.
        seconds = value / 1000.0 if value > 1e11 else float(value)
        try:
            return datetime.fromtimestamp(seconds, tz=UTC)
        except (OverflowError, OSError, ValueError):
            return None
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(s)
        except ValueError:
            return None
        return dt if dt.tzinfo else dt.replace(tzinfo=UTC)
    return None


def total_stream_tokens(events: list[Event]) -> int:
    """Sum every non-null token field across a built session's events.

    Used by the codebench parser to decide whether stream usage is genuinely
    absent (all zero) and a ``metrics.json`` backfill is warranted.
    """
    total = 0
    for ev in events:
        u = ev.usage
        if u is None:
            continue
        for field in (u.input, u.output, u.cache_read, u.cache_write):
            if field:
                total += field
    return total
