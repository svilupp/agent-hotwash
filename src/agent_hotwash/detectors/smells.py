"""Layer-2 smells — cheap, config-thresholded heuristics.

Each smell is a pure ``(Session, Config) -> list[Finding]`` function keyed off a
single threshold from ``[smells]`` in the config. Noisy-but-useful signals; the
high-precision judgements live in :mod:`agent_hotwash.detectors.taxonomy`.

All twelve smells from the research doc are here. ``idle_gap`` is timestamp-gated
and no-ops (never false-positives) when the session has no wall clock.
"""

from __future__ import annotations

import itertools
import re
from typing import TYPE_CHECKING

from agent_hotwash.detectors.registry import (
    Severity,
    approx_tokens,
    bash_command,
    detector,
    is_grep_like,
    is_read_call,
    is_source_like_path,
    is_write_call,
    logical_events,
    logical_positions,
    make_finding,
    read_paths,
    span,
    tool_calls,
    total_tokens,
    user_msgs,
    word_count,
)
from agent_hotwash.events import EventKind
from agent_hotwash.primitives.commands import segment_head, split_segments, strip_shell_wrapper

if TYPE_CHECKING:
    from agent_hotwash.config import Config
    from agent_hotwash.detectors.registry import Finding
    from agent_hotwash.events import Session

_KIND = "smell"

# A `>`/`>>` output redirection and its target. The lookbehind/lookahead keep
# out `=>` (JS arrow), `>=` (SQL/compare), `->>` (JSON operator), `<>`; a
# bare `2>&1` yields no target token.
_REDIRECT_RE = re.compile(r"(?<![=<>\-!])>>?(?![>=])\s*(['\"]?)([^\s&|>'\"=]+)\1")
# echo/printf/cat are the shell's write-a-file verbs when redirected.
_WRITE_HEADS = {"echo", "printf", "cat"}
# Scratch targets: writing there is not editing the project.
_SCRATCH_PREFIXES = ("/dev/", "/tmp/", "/private/tmp/", "/var/folders/", "$tmpdir", "${tmpdir}")
_SCRATCH_SUFFIXES = (".log", ".out", ".err")
_SCRATCH_DIRS = {"tmp", "logs", "log", ".cache", "evidence", "artifacts"}


def _is_project_target(target: str) -> bool:
    low = target.strip("'\"").lower()
    if not low or low.startswith(_SCRATCH_PREFIXES) or low.endswith(_SCRATCH_SUFFIXES):
        return False
    return not any(part in _SCRATCH_DIRS for part in low.split("/")[:-1])


def _segment_mutates(seg: str) -> bool:
    """One shell segment (no `&&`/`|`/`;`/newline): stream editor or redirect?"""
    hr = segment_head(seg)
    if hr is None:
        return False
    head, rest = hr
    if head == "tee":
        return any(_is_project_target(t) for t in rest if not t.startswith("-"))
    if head == "sed":
        return any(t == "-i" or (t.startswith("-i") and not t.startswith("-in")) for t in rest)
    if head == "patch":
        return True
    if head == "dd":
        return any(t.startswith("of=") and _is_project_target(t[3:]) for t in rest)
    # Redirects: only the segment's first line — heredoc bodies below it are data.
    first_line = seg.split("\n", 1)[0]
    for _q, target in _REDIRECT_RE.findall(first_line):
        if not _is_project_target(target):
            continue
        if is_source_like_path(target) or head in _WRITE_HEADS:
            return True
    return False


def _mutates_file(cmd: str) -> bool:
    """True when a shell command edits a file outside the Edit tool.

    Anchored on command segments: ``tee``/``sed -i``/``patch``/``dd of=`` as a
    segment head, or a ``>``/``>>`` redirect to a non-scratch path that is
    either a source-like file or the output of ``echo``/``printf``/``cat``.
    Heredoc scripts piped into an interpreter (``python - <<'PY'``) are not
    file writes.
    """
    return any(_segment_mutates(seg) for seg in split_segments(strip_shell_wrapper(cmd)))


# A guard so a short session is not "100% monoculture".
_MONOCULTURE_MIN_CALLS = 10
# Legacy Codex decode path wraps every call as `exec`; it says nothing about the mix.
_MONOCULTURE_IGNORED_TOOLS = {"exec"}


@detector("bloated_opener", kind=_KIND, severity=Severity.info)
def bloated_opener(session: Session, config: Config) -> list[Finding]:
    """First user message is very large (an over-stuffed opening prompt)."""
    users = user_msgs(session)
    if not users:
        return []
    first = users[0]
    tokens = approx_tokens(first.text)
    thresh = config.smells.bloated_opener_tokens
    if tokens <= thresh:
        return []
    return [
        make_finding(
            "bloated_opener",
            session,
            kind=_KIND,
            severity=Severity.info,
            confidence="high",
            spans=[span(session, first.idx)],
            evidence={"opener_tokens": tokens, "threshold": thresh},
            message=f"Opening prompt ~{tokens} tokens (> {thresh}).",
        )
    ]


@detector("thin_prompt", kind=_KIND, severity=Severity.info)
def thin_prompt(session: Session, config: Config) -> list[Finding]:
    """Opening prompt is very short (too little context to act well)."""
    users = user_msgs(session)
    if not users:
        return []
    first = users[0]
    words = word_count(first.text)
    thresh = config.smells.thin_prompt_words
    if words >= thresh:
        return []
    return [
        make_finding(
            "thin_prompt",
            session,
            kind=_KIND,
            severity=Severity.info,
            confidence="high",
            spans=[span(session, first.idx)],
            evidence={"words": words, "threshold": thresh},
            message=f"Opening prompt only {words} words (< {thresh}).",
        )
    ]


def _first_edit_idx(session: Session) -> int | None:
    for ev in session.events:
        if is_write_call(ev):
            return ev.idx
    return None


@detector("cold_start_reads", kind=_KIND, severity=Severity.low)
def cold_start_reads(session: Session, config: Config) -> list[Finding]:
    """Many distinct files read before the first edit (read-heavy cold start).

    No-ops when the session never writes — a research/QA thread has no "cold
    start", its reads are the whole job. Counts distinct artifact paths (a
    compound ``sed a && sed b`` is two reads, three ``sed`` slices of one file
    are one); reads with no path count once each.
    """
    first_edit = _first_edit_idx(session)
    if first_edit is None:
        return []
    reads = [ev for ev in session.events if is_read_call(ev) and ev.idx < first_edit]
    if not reads:
        return []
    distinct: set[str] = set()
    for ev in reads:
        paths = read_paths(ev)
        distinct.update(paths if paths else {f"#{ev.idx}"})
    thresh = config.smells.cold_start_reads
    if len(distinct) <= thresh:
        return []
    return [
        make_finding(
            "cold_start_reads",
            session,
            kind=_KIND,
            severity=Severity.low,
            confidence="high",
            spans=[span(session, reads[0].idx, reads[-1].idx)],
            evidence={"reads_before_first_edit": len(distinct), "read_calls": len(reads), "threshold": thresh},
            message=f"{len(distinct)} distinct paths read before first edit (> {thresh}).",
        )
    ]


@detector("slow_to_action", kind=_KIND, severity=Severity.info)
def slow_to_action(session: Session, config: Config) -> list[Finding]:
    """Many LOGICAL events elapse before the first tool call (slow to act).

    Counted as the position in ``logical_events`` (meta wrappers such as
    ``task_started``/``turn_context``/``token_usage_record`` excluded), so a
    standard harness prelude never trips it.
    """
    logical = logical_events(session)
    calls = tool_calls(session)
    thresh = config.smells.slow_to_action_events
    count = logical_positions(session).get(calls[0].idx, calls[0].idx) if calls else len(logical)
    if count <= thresh:
        return []
    anchor = calls[0].idx if calls else max(0, len(session.events) - 1)
    return [
        make_finding(
            "slow_to_action",
            session,
            kind=_KIND,
            severity=Severity.info,
            confidence="high",
            spans=[span(session, anchor)],
            evidence={"events_to_first_tool_call": count, "threshold": thresh},
            message=f"{count} events before first tool call (> {thresh}).",
        )
    ]


@detector("overlong_trace", kind=_KIND, severity=Severity.low)
def overlong_trace(session: Session, config: Config) -> list[Finding]:
    """Trace exceeds the event / token / minute cap (an overlong session)."""
    s = config.smells
    n_events = len(session.events)
    tokens = total_tokens(session)
    reasons: dict[str, int] = {}
    if n_events > s.overlong_events:
        reasons["events"] = n_events
    if tokens > s.overlong_tokens:
        reasons["tokens"] = tokens
    minutes = _duration_minutes(session)
    if minutes is not None and minutes > s.overlong_minutes:
        reasons["minutes"] = int(minutes)
    if not reasons:
        return []
    return [
        make_finding(
            "overlong_trace",
            session,
            kind=_KIND,
            severity=Severity.low,
            confidence="high",
            spans=[span(session, 0, max(0, n_events - 1))],
            evidence={"exceeded": reasons, "events": n_events, "tokens": tokens},
            message="Overlong trace: " + ", ".join(f"{k}={v}" for k, v in reasons.items()) + ".",
        )
    ]


@detector("context_bloat_no_clear", kind=_KIND, severity=Severity.medium)
def context_bloat_no_clear(session: Session, config: Config) -> list[Finding]:
    """Context filled past a fraction of the window with no compaction/clear.

    Needs token usage; no-ops when usage is absent (peak stays 0).
    """
    window = config.analytics.context_window_tokens
    pct = config.smells.context_bloat_pct
    peak = 0
    peak_idx = 0
    for ev in session.events:
        u = ev.usage
        if u is None:
            continue
        cur = (u.input or 0) + (u.cache_read or 0)
        if cur > peak:
            peak, peak_idx = cur, ev.idx
    if peak == 0 or peak <= pct * window:
        return []
    if any(ev.kind is EventKind.compaction for ev in session.events):
        return []
    return [
        make_finding(
            "context_bloat_no_clear",
            session,
            kind=_KIND,
            severity=Severity.medium,
            confidence="high",
            spans=[span(session, peak_idx)],
            evidence={"peak_context_tokens": peak, "window": window, "pct": pct},
            message=f"Context reached ~{peak} tokens ({peak / window:.0%} of window) with no compaction.",
        )
    ]


@detector("tool_monoculture", kind=_KIND, severity=Severity.low)
def tool_monoculture(session: Session, config: Config) -> list[Finding]:
    """A single tool dominates the call mix (>= 10 calls; the legacy ``exec``
    wrapper name is ignored because it hides the real tool)."""
    calls = [ev for ev in tool_calls(session) if (ev.tool_name or "") not in _MONOCULTURE_IGNORED_TOOLS]
    if len(calls) < _MONOCULTURE_MIN_CALLS:
        return []
    counts: dict[str, int] = {}
    for ev in calls:
        counts[ev.tool_name or "?"] = counts.get(ev.tool_name or "?", 0) + 1
    top_name = max(counts, key=lambda k: counts[k])
    frac = counts[top_name] / len(calls)
    pct = config.smells.tool_monoculture_pct
    if frac <= pct:
        return []
    return [
        make_finding(
            "tool_monoculture",
            session,
            kind=_KIND,
            severity=Severity.low,
            confidence="high",
            spans=[span(session, calls[0].idx, calls[-1].idx)],
            evidence={"tool": top_name, "fraction": round(frac, 3), "calls": len(calls)},
            message=f"'{top_name}' is {frac:.0%} of {len(calls)} tool calls (> {pct:.0%}).",
        )
    ]


@detector("linear_scan_search", kind=_KIND, severity=Severity.low)
def linear_scan_search(session: Session, config: Config) -> list[Finding]:
    """Many file reads with zero grep/glob searches (scanning by hand)."""
    reads = [ev for ev in session.events if is_read_call(ev) and not is_grep_like(ev)]
    greps = [ev for ev in session.events if is_grep_like(ev)]
    thresh = config.smells.linear_scan_reads
    if greps or len(reads) <= thresh:
        return []
    return [
        make_finding(
            "linear_scan_search",
            session,
            kind=_KIND,
            severity=Severity.low,
            confidence="high",
            spans=[span(session, reads[0].idx, reads[-1].idx)],
            evidence={"reads": len(reads), "grep_glob": 0, "threshold": thresh},
            message=f"{len(reads)} reads with no grep/glob (> {thresh}).",
        )
    ]


@detector("bash_as_editor", kind=_KIND, severity=Severity.low)
def bash_as_editor(session: Session, config: Config) -> list[Finding]:
    """A file is mutated through shell redirection/sed/tee instead of Edit."""
    hits: list[int] = []
    for ev in session.events:
        cmd = bash_command(ev)
        if cmd and _mutates_file(cmd):
            hits.append(ev.idx)
    if not hits:
        return []
    return [
        make_finding(
            "bash_as_editor",
            session,
            kind=_KIND,
            severity=Severity.low,
            confidence="high",
            spans=[span(session, i) for i in hits],
            evidence={"occurrences": len(hits)},
            message=f"{len(hits)} shell command(s) mutate files instead of using Edit.",
        )
    ]


@detector("no_plan_dive", kind=_KIND, severity=Severity.low)
def no_plan_dive(session: Session, config: Config) -> list[Finding]:
    """A mutating op within the first K events with no prior thinking/plan."""
    k = config.smells.no_plan_dive_events
    saw_thinking = False
    for ev in session.events:
        if ev.idx >= k:
            break
        if ev.kind is EventKind.thinking or (ev.kind is EventKind.tool_call and _is_planning(ev)):
            saw_thinking = True
        if is_write_call(ev) and not saw_thinking:
            return [
                make_finding(
                    "no_plan_dive",
                    session,
                    kind=_KIND,
                    severity=Severity.low,
                    confidence="high",
                    spans=[span(session, ev.idx)],
                    evidence={"first_mutation_idx": ev.idx, "window": k},
                    message=f"Mutating op at event {ev.idx} within first {k} events, no planning first.",
                )
            ]
    return []


def _is_planning(ev) -> bool:
    from agent_hotwash.events import ToolCategory

    return ev.tool_category is ToolCategory.planning


@detector("runaway_todo", kind=_KIND, severity=Severity.low)
def runaway_todo(session: Session, config: Config) -> list[Finding]:
    """A todo list grows across TodoWrite calls but nothing is ever completed."""
    todo_calls = [ev for ev in tool_calls(session) if (ev.tool_name or "").lower() == "todowrite"]
    if len(todo_calls) < 2:
        return []
    max_len = 0
    completed = 0
    for ev in todo_calls:
        items = (ev.tool_args or {}).get("todos")
        if not isinstance(items, list):
            continue
        max_len = max(max_len, len(items))
        for it in items:
            if isinstance(it, dict) and str(it.get("status", "")).lower() == "completed":
                completed += 1
    grew = _todo_len(todo_calls[-1]) > _todo_len(todo_calls[0])
    if completed > 0 or max_len < 3 or not grew:
        return []
    return [
        make_finding(
            "runaway_todo",
            session,
            kind=_KIND,
            severity=Severity.low,
            confidence="high",
            spans=[span(session, todo_calls[0].idx, todo_calls[-1].idx)],
            evidence={"max_todos": max_len, "completed": 0, "todo_writes": len(todo_calls)},
            message=f"Todo list grew to {max_len} items with zero completions.",
        )
    ]


def _todo_len(ev) -> int:
    items = (ev.tool_args or {}).get("todos")
    return len(items) if isinstance(items, list) else 0


def _duration_minutes(session: Session) -> float | None:
    if not session.has_timestamps:
        return None
    ts = [ev.ts for ev in session.events if ev.ts is not None]
    if len(ts) < 2:
        return None
    return (max(ts) - min(ts)).total_seconds() / 60.0


@detector("idle_gap", kind=_KIND, severity=Severity.info)
def idle_gap(session: Session, config: Config) -> list[Finding]:
    """A long wall-clock gap between consecutive events.

    Timestamp-gated: no-ops entirely when the session has no timestamps.
    """
    if not session.has_timestamps:
        return []
    thresh = config.smells.idle_gap_minutes
    stamped = [(ev.idx, ev.ts) for ev in session.events if ev.ts is not None]
    biggest = 0.0
    gap_at = None
    for (_, prev_ts), (cur_idx, cur_ts) in itertools.pairwise(stamped):
        gap = (cur_ts - prev_ts).total_seconds() / 60.0
        if gap > biggest:
            biggest, gap_at = gap, cur_idx
    if gap_at is None or biggest <= thresh:
        return []
    return [
        make_finding(
            "idle_gap",
            session,
            kind=_KIND,
            severity=Severity.info,
            confidence="high",
            spans=[span(session, gap_at)],
            evidence={"gap_minutes": round(biggest, 2), "threshold": thresh},
            message=f"Idle gap of {biggest:.1f} min (> {thresh}).",
        )
    ]
