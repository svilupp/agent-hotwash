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
    is_write_call,
    make_finding,
    span,
    tool_calls,
    total_tokens,
    user_msgs,
    word_count,
)
from agent_hotwash.events import EventKind

if TYPE_CHECKING:
    from agent_hotwash.config import Config
    from agent_hotwash.detectors.registry import Finding
    from agent_hotwash.events import Session

_KIND = "smell"

# Stream-editing shell tools that mutate a file outside Edit (always count).
_STREAM_EDIT_RE = re.compile(r"(\btee\b)|(\bsed\b[^|]*\s-i\b)|(\bpatch\b)|(\bdd\b\s)")
# A `>`/`>>` redirection and its target (first whitespace/operator-delimited token).
_REDIRECT_RE = re.compile(r">>?\s*(['\"]?)([^\s&|>'\"]+)\1")
# Redirection only counts as "bash as editor" when it writes an actual project
# file — a source-ish extension — not /dev/null or a scratch log (`> out.log`).
_SOURCE_EXT_RE = re.compile(
    r"\.(py|pyi|js|jsx|ts|tsx|mjs|cjs|json|ya?ml|toml|ini|cfg|conf|md|rst|sh|bash|go|rs|"
    r"java|kt|rb|php|c|h|cc|cpp|hpp|cs|css|scss|html?|sql|env|xml|vue|svelte)$",
    re.IGNORECASE,
)
# echo/printf/cat (heredoc) are the shell's write-a-file verbs.
_WRITE_CMD_RE = re.compile(r"\b(echo|printf|cat)\b")


def _mutates_file(cmd: str) -> bool:
    """True when a shell command edits a file outside the Edit tool.

    Stream editors (tee/sed -i/patch/dd) always count. A redirection counts only
    when it targets a source-ish project file (not /dev/null or a scratch log) or
    is an echo/printf/cat heredoc writing that file.
    """
    if _STREAM_EDIT_RE.search(cmd):
        return True
    has_writer = bool(_WRITE_CMD_RE.search(cmd))
    for _q, target in _REDIRECT_RE.findall(cmd):
        low = target.lower()
        if low.startswith("/dev/"):
            continue
        if _SOURCE_EXT_RE.search(low) or has_writer:
            return True
    return False


# A guard so a session with a single tool call is not "100% monoculture".
_MONOCULTURE_MIN_CALLS = 5


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
    """Many file reads before the first edit (read-heavy cold start)."""
    first_edit = _first_edit_idx(session)
    reads = [ev.idx for ev in session.events if is_read_call(ev) and (first_edit is None or ev.idx < first_edit)]
    thresh = config.smells.cold_start_reads
    if len(reads) <= thresh:
        return []
    return [
        make_finding(
            "cold_start_reads",
            session,
            kind=_KIND,
            severity=Severity.low,
            confidence="high",
            spans=[span(session, reads[0], reads[-1])],
            evidence={"reads_before_first_edit": len(reads), "threshold": thresh},
            message=f"{len(reads)} reads before first edit (> {thresh}).",
        )
    ]


@detector("slow_to_action", kind=_KIND, severity=Severity.info)
def slow_to_action(session: Session, config: Config) -> list[Finding]:
    """Many events elapse before the first tool call (slow to act)."""
    calls = tool_calls(session)
    thresh = config.smells.slow_to_action_events
    count = calls[0].idx if calls else len(session.events)
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
    """A single tool dominates the call mix."""
    calls = tool_calls(session)
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
