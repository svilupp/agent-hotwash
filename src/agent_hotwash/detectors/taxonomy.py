"""Layer-3 failure taxonomies — 28 deterministic detectors.

Each is a pure ``(Session, Config) -> list[Finding]`` function built from the
primitives plus per-detector knobs from ``[taxonomy.<id>]`` in config. Ids are
uppercase; ``Config.taxonomy_knobs`` resolves them case-insensitively.

Confidence tiers (per the research doc):

* 23 rule-clean detectors emit ``confidence="high"``.
* 5 fuzzy detectors — ``CONTEXT_ROT``, ``ASSUMING_NOT_OBSERVING``,
  ``OVER_ENGINEERING``, ``GOAL_DRIFT``, ``STYLE_IMPOSITION`` — ship as rule
  versions emitting ``confidence="low"`` and are flagged ``llm_candidate=True``
  for a future LLM-tier upgrade behind the same :class:`Finding` interface.

Time-dependent detectors gate their time-based sub-conditions on
``session.has_timestamps`` and never false-positive when the clock is absent.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from agent_hotwash.detectors.registry import (
    Severity,
    assistant_msgs,
    bash_command,
    call_by_id,
    detector,
    failing_results,
    is_edit_tool,
    is_exec_call,
    is_grep_like,
    is_read_call,
    is_write_call,
    make_finding,
    result_ok_by_call,
    span,
    tool_calls,
    total_tokens,
    user_msgs,
)
from agent_hotwash.events import EventKind
from agent_hotwash.primitives.argnorm import edit_distance
from agent_hotwash.primitives.commands import classify_command
from agent_hotwash.primitives.lexicons import Lexicons, is_interrogative
from agent_hotwash.primitives.window import SlidingWindow

if TYPE_CHECKING:
    from agent_hotwash.config import Config
    from agent_hotwash.detectors.registry import Finding
    from agent_hotwash.events import Event, Session

_KIND = "taxonomy"

_REVERT_RE = re.compile(r"git\s+(checkout|revert)|reset\s+--hard|git\s+restore|git\s+clean\s+-", re.IGNORECASE)
_SKIP_RE = re.compile(r"@pytest\.mark\.(skip|xfail)|\bxfail\b|\.skip\s*\(|\bskip\b|@unittest\.skip", re.IGNORECASE)
_ASSERT_RE = re.compile(r"\bassert\b|\bexpect\s*\(")
_ASSUME_RE = re.compile(r"\bthe\s+\w+\s+(returns|is|does|will|should)\b", re.IGNORECASE)
_ACK_RE = re.compile(
    r"\b(error|fail|failed|failing|issue|problem|retry|retrying|fix|fixing|wrong|broke|broken|didn'?t|couldn'?t|can'?t|revert)\b",
    re.IGNORECASE,
)
_IMPORT_RE = re.compile(
    r"^\s*(?:import\s+([\w.]+)|from\s+([\w.]+)\s+import|(?:const|let|var).*require\(['\"]([^'\"]+)|import\s+.*from\s+['\"]([^'\"]+))",
    re.MULTILINE,
)
_FIX_INTENT_RE = re.compile(r"\b(fix|bug|repair|correct|patch|resolve)\b", re.IGNORECASE)
_OVERENG_RE = re.compile(
    r"\b(we could also|it would be better if|might as well|while we'?re at it|for good measure)\b", re.IGNORECASE
)
_PATH_RE = re.compile(r"[\w./-]+\.[A-Za-z0-9]{1,6}")
_TEST_PATH_RE = re.compile(r"(^|/)(test_|tests?/)|_test\.|\.test\.|\.spec\.", re.IGNORECASE)


def _knob(config: Config, det_id: str, key: str, default: Any) -> Any:
    return config.taxonomy_knobs(det_id).get(key, default)


def _edit_text(ev: Event) -> tuple[str, str]:
    """(old, new) text for an edit/write call — best effort across arg shapes."""
    args = ev.tool_args or {}
    old = str(args.get("old_string", "") or "")
    new = str(args.get("new_string", "") or args.get("content", "") or "")
    return old, new


def _outbound_text(ev: Event) -> str:
    """Text an event would emit outward (assistant text or tool-call args)."""
    if ev.kind is EventKind.assistant_msg:
        return ev.text or ""
    if ev.kind is EventKind.tool_call:
        parts = [ev.text or ""]
        for v in (ev.tool_args or {}).values():
            if isinstance(v, str):
                parts.append(v)
        return "\n".join(parts)
    return ev.text or ""


# ---------------------------------------------------------------------------
# 1. CONTEXT_ROT  (fuzzy -> low)
# ---------------------------------------------------------------------------
@detector("CONTEXT_ROT", kind=_KIND, severity=Severity.low, confidence="low", llm_candidate=True)
def context_rot(session: Session, config: Config) -> list[Finding]:
    """Error/correction/thrash rate spikes in the last third of the session.

    Fuzzy: rule proxy for quality decay as context fills; LLM-upgrade candidate.
    """
    events = session.events
    n = len(events)
    if n < 9:
        return []
    lex = Lexicons.from_config(config)
    third = n // 3

    def rate(lo: int, hi: int) -> int:
        c = 0
        for ev in events[lo:hi]:
            if (
                (ev.kind is EventKind.tool_result and ev.ok is False)
                or (ev.kind is EventKind.user_msg and ev.text and lex.correction.search(ev.text))
                or is_write_call(ev)
            ):
                c += 1
        return c

    first_r = rate(0, third)
    last_r = rate(n - third, n)
    ratio = float(_knob(config, "CONTEXT_ROT", "ratio", 2.0))
    ctx_pct = float(_knob(config, "CONTEXT_ROT", "ctx_pct", 0.60))
    window = config.analytics.context_window_tokens
    peak = (
        max(((e.usage.input or 0) + (e.usage.cache_read or 0)) for e in events if e.usage)
        if any(e.usage for e in events)
        else None
    )
    ctx_ok = True if peak is None else peak > ctx_pct * window
    if first_r < 1 or last_r < 2 or last_r < ratio * first_r or not ctx_ok:
        return []
    return [
        make_finding(
            "CONTEXT_ROT",
            session,
            kind=_KIND,
            severity=Severity.low,
            confidence="low",
            spans=[span(session, n - third, n - 1)],
            evidence={"first_third_rate": first_r, "last_third_rate": last_r, "ratio": ratio},
            message=f"Error/correction/thrash rate rose {first_r}->{last_r} across the session.",
        )
    ]


# ---------------------------------------------------------------------------
# 2. KITCHEN_SINK
# ---------------------------------------------------------------------------
@detector("KITCHEN_SINK", kind=_KIND, severity=Severity.low)
def kitchen_sink(session: Session, config: Config) -> list[Finding]:
    """Multiple unrelated task-intro user turns with no /clear between."""
    lex = Lexicons.from_config(config)
    intros = [ev for ev in user_msgs(session) if ev.text and lex.task_intro.search(ev.text)]
    min_intros = int(_knob(config, "KITCHEN_SINK", "min_task_intros", 2))
    if len(intros) < min_intros:
        return []
    lo, hi = intros[0].idx, intros[-1].idx
    if any(ev.kind is EventKind.compaction and lo < ev.idx < hi for ev in session.events):
        return []
    return [
        make_finding(
            "KITCHEN_SINK",
            session,
            kind=_KIND,
            severity=Severity.low,
            confidence="high",
            spans=[span(session, e.idx) for e in intros],
            evidence={"task_intros": len(intros)},
            message=f"{len(intros)} task-intro turns with no /clear between them.",
        )
    ]


# ---------------------------------------------------------------------------
# 3. CORRECTION_LOOP
# ---------------------------------------------------------------------------
@detector("CORRECTION_LOOP", kind=_KIND, severity=Severity.medium)
def correction_loop(session: Session, config: Config) -> list[Finding]:
    """User re-corrects repeatedly within a sliding event window."""
    lex = Lexicons.from_config(config)
    min_repeats = int(_knob(config, "CORRECTION_LOOP", "min_repeats", 3))
    window_events = int(_knob(config, "CORRECTION_LOOP", "window_events", 30))
    win = SlidingWindow[Any](window_events)
    hit = win.first_window_reaching(
        session.events,
        lambda e: e.kind is EventKind.user_msg and bool(e.text) and bool(lex.correction.search(e.text)),
        min_repeats,
    )
    if hit is None:
        return []
    start, end = hit
    return [
        make_finding(
            "CORRECTION_LOOP",
            session,
            kind=_KIND,
            severity=Severity.medium,
            confidence="high",
            spans=[span(session, session.events[start].idx, session.events[end].idx)],
            evidence={"min_repeats": min_repeats, "window_events": window_events},
            message=f">= {min_repeats} corrections within {window_events} events.",
        )
    ]


# ---------------------------------------------------------------------------
# 4. EDIT_THRASH
# ---------------------------------------------------------------------------
@detector("EDIT_THRASH", kind=_KIND, severity=Severity.medium)
def edit_thrash(session: Session, config: Config) -> list[Finding]:
    """Same file edited >= N times within M events with no read between."""
    min_edits = int(_knob(config, "EDIT_THRASH", "min_edits", 3))
    window = int(_knob(config, "EDIT_THRASH", "window_events", 20))
    out: list[Finding] = []
    for path, st in session.file_state.items():
        edits = st.edited_at
        if len(edits) < min_edits:
            continue
        for i in range(len(edits) - min_edits + 1):
            lo, hi = edits[i], edits[i + min_edits - 1]
            if hi - lo > window:
                continue
            if any(lo < r < hi for r in st.read_at):
                continue
            out.append(
                make_finding(
                    "EDIT_THRASH",
                    session,
                    kind=_KIND,
                    severity=Severity.medium,
                    confidence="high",
                    spans=[span(session, lo, hi)],
                    evidence={"path": path, "edits": min_edits, "span_events": hi - lo},
                    message=f"{path} edited {min_edits}x within {hi - lo} events, no read between.",
                )
            )
            break
    return out


# ---------------------------------------------------------------------------
# 5. EDIT_WITHOUT_READ
# ---------------------------------------------------------------------------
@detector("EDIT_WITHOUT_READ", kind=_KIND, severity=Severity.medium)
def edit_without_read(session: Session, config: Config) -> list[Finding]:
    """A targeted edit to a file that was never read first."""
    out: list[Finding] = []
    seen: set[str] = set()
    for ev in session.events:
        if not is_edit_tool(ev) or not ev.path or ev.path in seen:
            continue
        st = session.file_state.get(ev.path)
        # A prior read OR a prior write/edit (create-then-edit) grounds the edit,
        # mirroring FULL_FILE_REWRITE / OVER_ENGINEERING "existed" checks.
        grounded = st is not None and (any(r < ev.idx for r in st.read_at) or any(e < ev.idx for e in st.edited_at))
        if grounded:
            continue
        seen.add(ev.path)
        out.append(
            make_finding(
                "EDIT_WITHOUT_READ",
                session,
                kind=_KIND,
                severity=Severity.medium,
                confidence="high",
                spans=[span(session, ev.idx)],
                evidence={"path": ev.path},
                message=f"Edited {ev.path} without reading it first.",
            )
        )
    return out


# ---------------------------------------------------------------------------
# 6. FULL_FILE_REWRITE
# ---------------------------------------------------------------------------
@detector("FULL_FILE_REWRITE", kind=_KIND, severity=Severity.low)
def full_file_rewrite(session: Session, config: Config) -> list[Finding]:
    """Write over an existing file that is larger than N lines (vs a targeted Edit)."""
    min_lines = int(_knob(config, "FULL_FILE_REWRITE", "min_lines", 50))
    out: list[Finding] = []
    for ev in session.events:
        if not is_write_call(ev) or (ev.tool_name or "").lower() != "write" or not ev.path:
            continue
        st = session.file_state.get(ev.path)
        existed = st is not None and (any(r < ev.idx for r in st.read_at) or any(e < ev.idx for e in st.edited_at))
        if not existed:
            continue
        lines = ev.lines_added if ev.lines_added is not None else _content_lines(ev)
        if lines < min_lines:
            continue
        out.append(
            make_finding(
                "FULL_FILE_REWRITE",
                session,
                kind=_KIND,
                severity=Severity.low,
                confidence="high",
                spans=[span(session, ev.idx)],
                evidence={"path": ev.path, "lines": lines, "threshold": min_lines},
                message=f"Full rewrite of existing {ev.path} ({lines} lines).",
            )
        )
    return out


def _content_lines(ev: Event) -> int:
    content = (ev.tool_args or {}).get("content")
    return content.count("\n") + 1 if isinstance(content, str) and content else 0


# ---------------------------------------------------------------------------
# 7. RETRY_STORM
# ---------------------------------------------------------------------------
@detector("RETRY_STORM", kind=_KIND, severity=Severity.medium)
def retry_storm(session: Session, config: Config) -> list[Finding]:
    """An identical (tool, normalized-args) call is repeated >= N times."""
    min_repeats = int(_knob(config, "RETRY_STORM", "min_repeats", 4))
    groups: dict[tuple[str, str], list[int]] = {}
    for ev in tool_calls(session):
        key = (ev.tool_name or "?", ev.tool_norm_args or "")
        groups.setdefault(key, []).append(ev.idx)
    out: list[Finding] = []
    for (name, _args), idxs in groups.items():
        if len(idxs) < min_repeats:
            continue
        out.append(
            make_finding(
                "RETRY_STORM",
                session,
                kind=_KIND,
                severity=Severity.medium,
                confidence="high",
                spans=[span(session, idxs[0], idxs[-1])],
                evidence={"tool": name, "repeats": len(idxs)},
                message=f"'{name}' called with identical args {len(idxs)}x.",
            )
        )
    return out


# ---------------------------------------------------------------------------
# 8. NO_ADAPT_RETRY
# ---------------------------------------------------------------------------
@detector("NO_ADAPT_RETRY", kind=_KIND, severity=Severity.medium)
def no_adapt_retry(session: Session, config: Config) -> list[Finding]:
    """A failing call is retried with barely-changed args (no adaptation)."""
    min_repeats = int(_knob(config, "NO_ADAPT_RETRY", "min_repeats", 2))
    eps = int(_knob(config, "NO_ADAPT_RETRY", "arg_edit_distance_eps", 5))
    ok_by_call = result_ok_by_call(session)
    by_tool: dict[str, list[Event]] = {}
    for ev in tool_calls(session):
        failed = ev.call_id is not None and ok_by_call.get(ev.call_id) is False
        if failed:
            by_tool.setdefault(ev.tool_name or "?", []).append(ev)
    out: list[Finding] = []
    for name, evs in by_tool.items():
        i = 0
        while i < len(evs):
            cluster = [evs[i]]
            j = i + 1
            while (
                j < len(evs)
                and edit_distance(evs[j - 1].tool_norm_args or "", evs[j].tool_norm_args or "", cap=eps + 1) < eps
            ):
                cluster.append(evs[j])
                j += 1
            if len(cluster) >= min_repeats:
                out.append(
                    make_finding(
                        "NO_ADAPT_RETRY",
                        session,
                        kind=_KIND,
                        severity=Severity.medium,
                        confidence="high",
                        spans=[span(session, cluster[0].idx, cluster[-1].idx)],
                        evidence={"tool": name, "retries": len(cluster), "eps": eps},
                        message=f"'{name}' retried {len(cluster)}x with near-identical args after failure.",
                    )
                )
            i = j if j > i + 1 else i + 1
    return out


# ---------------------------------------------------------------------------
# 9. ASSUMING_NOT_OBSERVING  (fuzzy -> low)
# ---------------------------------------------------------------------------
@detector("ASSUMING_NOT_OBSERVING", kind=_KIND, severity=Severity.low, confidence="low", llm_candidate=True)
def assuming_not_observing(session: Session, config: Config) -> list[Finding]:
    """Assistant asserts program state with no prior read/grep/bash grounding.

    Fuzzy: keys on a claim regex; LLM-upgrade candidate.
    """
    grounded = False
    out: list[Finding] = []
    for ev in session.events:
        if is_read_call(ev) or is_exec_call(ev) or is_grep_like(ev):
            grounded = True
        claim = _ASSUME_RE.search(ev.text) if ev.kind is EventKind.assistant_msg and ev.text else None
        if claim is not None and not grounded:
            out.append(
                make_finding(
                    "ASSUMING_NOT_OBSERVING",
                    session,
                    kind=_KIND,
                    severity=Severity.low,
                    confidence="low",
                    spans=[span(session, ev.idx)],
                    evidence={"claim": claim.group(0)},
                    message="Assistant asserts state before any read/grep/bash observation.",
                )
            )
            break
    return out


# ---------------------------------------------------------------------------
# 10. TEST_GAMING
# ---------------------------------------------------------------------------
@detector("TEST_GAMING", kind=_KIND, severity=Severity.high)
def test_gaming(session: Session, config: Config) -> list[Finding]:
    """A test file is loosened (skip/xfail added, asserts removed) after a failing run."""
    out: list[Finding] = []
    failed_test_before = False
    for ev in session.events:
        if ev.kind is EventKind.tool_result and ev.ok is False and ev.error_category == "build_test_fail":
            failed_test_before = True
        if not is_write_call(ev) or not ev.path or not _TEST_PATH_RE.search(ev.path):
            continue
        if not failed_test_before:
            continue
        old, new = _edit_text(ev)
        added_skip = bool(_SKIP_RE.search(new)) and not _SKIP_RE.search(old)
        removed_assert = len(_ASSERT_RE.findall(old)) > len(_ASSERT_RE.findall(new))
        if not (added_skip or removed_assert):
            continue
        out.append(
            make_finding(
                "TEST_GAMING",
                session,
                kind=_KIND,
                severity=Severity.high,
                confidence="high",
                spans=[span(session, ev.idx)],
                evidence={"path": ev.path, "added_skip": added_skip, "removed_assert": removed_assert},
                message=f"Test file {ev.path} loosened after a failing test run.",
            )
        )
    return out


# ---------------------------------------------------------------------------
# 11. PANIC_REVERT
# ---------------------------------------------------------------------------
@detector("PANIC_REVERT", kind=_KIND, severity=Severity.high)
def panic_revert(session: Session, config: Config) -> list[Finding]:
    """A git revert/reset shortly after a failing test/build run."""
    window = int(_knob(config, "PANIC_REVERT", "window_events", 5))
    fail_idx: int | None = None
    for ev in session.events:
        if ev.kind is EventKind.tool_result and ev.ok is False and ev.error_category == "build_test_fail":
            fail_idx = ev.idx
        cmd = bash_command(ev)
        if cmd and _REVERT_RE.search(cmd) and fail_idx is not None and 0 <= ev.idx - fail_idx <= window:
            return [
                make_finding(
                    "PANIC_REVERT",
                    session,
                    kind=_KIND,
                    severity=Severity.high,
                    confidence="high",
                    spans=[span(session, fail_idx, ev.idx)],
                    evidence={"command": cmd.strip()[:120], "events_after_fail": ev.idx - fail_idx},
                    message="Revert/reset within a few events of a failing test.",
                )
            ]
    return []


# ---------------------------------------------------------------------------
# 12. ACTING_ON_QUESTION
# ---------------------------------------------------------------------------
@detector("ACTING_ON_QUESTION", kind=_KIND, severity=Severity.medium)
def acting_on_question(session: Session, config: Config) -> list[Finding]:
    """A user question is answered by editing files instead of replying."""
    lex = Lexicons.from_config(config)
    events = session.events
    out: list[Finding] = []
    for i, ev in enumerate(events):
        if ev.kind is not EventKind.user_msg or not ev.text or not is_interrogative(ev.text, lex):
            continue
        for nxt in events[i + 1 :]:
            if nxt.kind is EventKind.user_msg:
                break
            if is_write_call(nxt):
                out.append(
                    make_finding(
                        "ACTING_ON_QUESTION",
                        session,
                        kind=_KIND,
                        severity=Severity.medium,
                        confidence="high",
                        spans=[span(session, ev.idx, nxt.idx)],
                        evidence={"question": ev.text[:120]},
                        message="Treated a user question as an instruction to edit.",
                    )
                )
                break
    return out


# ---------------------------------------------------------------------------
# Error-category taxonomies (13-17) share a small helper.
# ---------------------------------------------------------------------------
def _category_hits(session: Session, category: str) -> list[Event]:
    return [e for e in failing_results(session) if e.error_category == category]


def _command_for_result(session: Session, result: Event) -> str:
    calls = call_by_id(session)
    call = calls.get(result.call_id) if result.call_id else None
    return bash_command(call) if call else ""


# 13. PERMISSION_FRICTION
@detector("PERMISSION_FRICTION", kind=_KIND, severity=Severity.low)
def permission_friction(session: Session, config: Config) -> list[Finding]:
    """Repeated permission-denied results for the same command."""
    min_repeats = int(_knob(config, "PERMISSION_FRICTION", "min_repeats", 2))
    hits = _category_hits(session, "permission")
    by_cmd: dict[str, list[Event]] = {}
    for r in hits:
        by_cmd.setdefault(_command_for_result(session, r) or "?", []).append(r)
    out: list[Finding] = []
    for cmd, rs in by_cmd.items():
        if len(rs) < min_repeats:
            continue
        out.append(
            make_finding(
                "PERMISSION_FRICTION",
                session,
                kind=_KIND,
                severity=Severity.low,
                confidence="high",
                spans=[span(session, rs[0].idx, rs[-1].idx)],
                evidence={"command": cmd[:120], "denials": len(rs)},
                message=f"{len(rs)} permission denials for the same command.",
            )
        )
    return out


# 14. SANDBOX_EGRESS_FAIL
@detector("SANDBOX_EGRESS_FAIL", kind=_KIND, severity=Severity.low)
def sandbox_egress_fail(session: Session, config: Config) -> list[Finding]:
    """A blocked network/egress or out-of-sandbox result."""
    hits = _category_hits(session, "sandbox_egress")
    if not hits:
        return []
    return [
        make_finding(
            "SANDBOX_EGRESS_FAIL",
            session,
            kind=_KIND,
            severity=Severity.low,
            confidence="high",
            spans=[span(session, h.idx) for h in hits],
            evidence={"occurrences": len(hits)},
            message=f"{len(hits)} sandbox egress/network-blocked failure(s).",
        )
    ]


# 15. TOOL_ARG_MALFORMED
@detector("TOOL_ARG_MALFORMED", kind=_KIND, severity=Severity.medium)
def tool_arg_malformed(session: Session, config: Config) -> list[Finding]:
    """A tool call rejected for a bad/invalid argument schema."""
    hits = _category_hits(session, "agent_syntax_error")
    if not hits:
        return []
    return [
        make_finding(
            "TOOL_ARG_MALFORMED",
            session,
            kind=_KIND,
            severity=Severity.medium,
            confidence="high",
            spans=[span(session, h.idx) for h in hits],
            evidence={"occurrences": len(hits)},
            message=f"{len(hits)} malformed tool-argument error(s).",
        )
    ]


# 16. MCP_TRANSPORT_ERR
@detector("MCP_TRANSPORT_ERR", kind=_KIND, severity=Severity.low)
def mcp_transport_err(session: Session, config: Config) -> list[Finding]:
    """An MCP transport/connection failure."""
    hits = _category_hits(session, "mcp_transport")
    if not hits:
        return []
    return [
        make_finding(
            "MCP_TRANSPORT_ERR",
            session,
            kind=_KIND,
            severity=Severity.low,
            confidence="high",
            spans=[span(session, h.idx) for h in hits],
            evidence={"occurrences": len(hits)},
            message=f"{len(hits)} MCP transport error(s).",
        )
    ]


# 17. RATE_LIMIT_LOOP
@detector("RATE_LIMIT_LOOP", kind=_KIND, severity=Severity.low)
def rate_limit_loop(session: Session, config: Config) -> list[Finding]:
    """Repeated rate-limit failures (retrying into a wall)."""
    min_repeats = int(_knob(config, "RATE_LIMIT_LOOP", "min_repeats", 2))
    hits = _category_hits(session, "rate_limit")
    if len(hits) < min_repeats:
        return []
    return [
        make_finding(
            "RATE_LIMIT_LOOP",
            session,
            kind=_KIND,
            severity=Severity.low,
            confidence="high",
            spans=[span(session, hits[0].idx, hits[-1].idx)],
            evidence={"occurrences": len(hits), "min_repeats": min_repeats},
            message=f"{len(hits)} rate-limit failures in a row.",
        )
    ]


# ---------------------------------------------------------------------------
# 18. PERFECTIONISM_LOOP
# ---------------------------------------------------------------------------
@detector("PERFECTIONISM_LOOP", kind=_KIND, severity=Severity.low)
def perfectionism_loop(session: Session, config: Config) -> list[Finding]:
    """Continued edits after the last passing test with no new user turn."""
    min_edits = int(_knob(config, "PERFECTIONISM_LOOP", "min_edits", 2))
    last_pass = _last_passing_test_idx(session)
    if last_pass is None:
        return []
    edits_after = [ev.idx for ev in session.events if is_write_call(ev) and ev.idx > last_pass]
    user_after = any(ev.kind is EventKind.user_msg and ev.idx > last_pass for ev in session.events)
    if user_after or len(edits_after) < min_edits:
        return []
    return [
        make_finding(
            "PERFECTIONISM_LOOP",
            session,
            kind=_KIND,
            severity=Severity.low,
            confidence="high",
            spans=[span(session, edits_after[0], edits_after[-1])],
            evidence={"edits_after_green": len(edits_after)},
            message=f"{len(edits_after)} edits after the last passing test, no new user turn.",
        )
    ]


def _last_passing_test_idx(session: Session) -> int | None:
    ok_by_call = result_ok_by_call(session)
    last: int | None = None
    for ev in session.events:
        cmd = bash_command(ev)
        if not cmd or classify_command(cmd) != "build_test":
            continue
        ok = ok_by_call.get(ev.call_id) if ev.call_id else ev.ok
        if ok is True:
            last = ev.idx
    return last


# ---------------------------------------------------------------------------
# 19. OVER_ENGINEERING  (fuzzy -> low)
# ---------------------------------------------------------------------------
@detector("OVER_ENGINEERING", kind=_KIND, severity=Severity.low, confidence="low", llm_candidate=True)
def over_engineering(session: Session, config: Config) -> list[Finding]:
    """A 'fix' task spawns many new files, or assistant proposes scope creep.

    Fuzzy: rule proxy for scope creep; LLM-upgrade candidate.
    """
    new_files_thresh = int(_knob(config, "OVER_ENGINEERING", "new_files", 3))
    users = user_msgs(session)
    fix_intent = bool(users and users[0].text and _FIX_INTENT_RE.search(users[0].text))
    new_files = _new_file_paths(session)
    creep_hits = ((ev, _OVERENG_RE.search(ev.text)) for ev in assistant_msgs(session) if ev.text)
    creep_msg, creep_match = next(((ev, m) for ev, m in creep_hits if m is not None), (None, None))
    if fix_intent and len(new_files) >= new_files_thresh:
        return [
            make_finding(
                "OVER_ENGINEERING",
                session,
                kind=_KIND,
                severity=Severity.low,
                confidence="low",
                spans=[span(session, users[0].idx)],
                evidence={"new_files": sorted(new_files), "intent": "fix"},
                message=f"'fix' task created {len(new_files)} new files.",
            )
        ]
    if creep_msg is not None and creep_match is not None:
        return [
            make_finding(
                "OVER_ENGINEERING",
                session,
                kind=_KIND,
                severity=Severity.low,
                confidence="low",
                spans=[span(session, creep_msg.idx)],
                evidence={"phrase": creep_match.group(0)},
                message="Assistant proposes out-of-scope extra work.",
            )
        ]
    return []


def _new_file_paths(session: Session) -> set[str]:
    """Paths first written (Write) with no prior read/edit — created from scratch."""
    created: set[str] = set()
    for ev in session.events:
        if is_write_call(ev) and (ev.tool_name or "").lower() == "write" and ev.path:
            st = session.file_state.get(ev.path)
            prior = st is not None and (any(r < ev.idx for r in st.read_at) or any(e < ev.idx for e in st.edited_at))
            if not prior:
                created.add(ev.path)
    return created


# ---------------------------------------------------------------------------
# 20. GOAL_DRIFT  (fuzzy -> low)
# ---------------------------------------------------------------------------
@detector("GOAL_DRIFT", kind=_KIND, severity=Severity.low, confidence="low", llm_candidate=True)
def goal_drift(session: Session, config: Config) -> list[Finding]:
    """Files touched barely overlap the paths named in the opening ask.

    Fuzzy: Jaccard path proxy; LLM-upgrade candidate.
    """
    eps = float(_knob(config, "GOAL_DRIFT", "jaccard_eps", 0.2))
    users = user_msgs(session)
    if not users or not users[0].text:
        return []
    asked = set(_PATH_RE.findall(users[0].text))
    touched = {_basename(p) for p in session.file_state}
    asked = {_basename(p) for p in asked}
    if not asked or not touched:
        return []
    jac = len(asked & touched) / len(asked | touched)
    if jac >= eps:
        return []
    return [
        make_finding(
            "GOAL_DRIFT",
            session,
            kind=_KIND,
            severity=Severity.low,
            confidence="low",
            spans=[span(session, users[0].idx)],
            evidence={"jaccard": round(jac, 3), "asked": sorted(asked), "touched": sorted(touched)},
            message=f"Files touched overlap the ask by only {jac:.0%}.",
        )
    ]


def _basename(path: str) -> str:
    return path.rsplit("/", 1)[-1]


# ---------------------------------------------------------------------------
# 21. LOOKS_RIGHT_RUNS_WRONG
# ---------------------------------------------------------------------------
@detector("LOOKS_RIGHT_RUNS_WRONG", kind=_KIND, severity=Severity.medium)
def looks_right_runs_wrong(session: Session, config: Config) -> list[Finding]:
    """A file is edited and declared done but never executed/tested afterward."""
    lex = Lexicons.from_config(config)
    last_edit, edited_paths = _last_edit(session)
    if last_edit is None:
        return []
    if not _completion_after(session, last_edit, lex):
        return []
    build_tests = _build_tests_after(session, last_edit)
    touches_file = any(any(_basename(p) in cmd for p in edited_paths) for cmd in build_tests)
    if touches_file:
        return []
    return [
        make_finding(
            "LOOKS_RIGHT_RUNS_WRONG",
            session,
            kind=_KIND,
            severity=Severity.medium,
            confidence="high",
            spans=[span(session, last_edit)],
            evidence={"edited_paths": sorted(edited_paths)},
            message="Edited files declared done but never executed/tested.",
        )
    ]


# ---------------------------------------------------------------------------
# 22. UNVERIFIED_COMPLETION
# ---------------------------------------------------------------------------
@detector("UNVERIFIED_COMPLETION", kind=_KIND, severity=Severity.medium)
def unverified_completion(session: Session, config: Config) -> list[Finding]:
    """A completion claim with no test/build run after the last edit."""
    lex = Lexicons.from_config(config)
    last_edit, _paths = _last_edit(session)
    if last_edit is None:
        return []
    if not _completion_after(session, last_edit, lex):
        return []
    if _build_tests_after(session, last_edit):
        return []
    return [
        make_finding(
            "UNVERIFIED_COMPLETION",
            session,
            kind=_KIND,
            severity=Severity.medium,
            confidence="high",
            spans=[span(session, last_edit)],
            evidence={"last_edit_idx": last_edit},
            message="Declared done with no test/build after the last edit.",
        )
    ]


def _last_edit(session: Session) -> tuple[int | None, set[str]]:
    last: int | None = None
    paths: set[str] = set()
    for ev in session.events:
        if is_write_call(ev):
            last = ev.idx
            if ev.path:
                paths.add(ev.path)
    return last, paths


def _completion_after(session: Session, after_idx: int, lex: Lexicons) -> bool:
    for ev in session.events:
        if ev.idx <= after_idx:
            continue
        if ev.kind in (EventKind.assistant_msg, EventKind.user_msg) and ev.text and lex.completion.search(ev.text):
            return True
    return False


def _build_tests_after(session: Session, after_idx: int) -> list[str]:
    cmds: list[str] = []
    for ev in session.events:
        if ev.idx <= after_idx:
            continue
        cmd = bash_command(ev)
        if cmd and classify_command(cmd) == "build_test":
            cmds.append(cmd)
    return cmds


# ---------------------------------------------------------------------------
# 23. SILENT_ERROR_SWALLOW
# ---------------------------------------------------------------------------
@detector("SILENT_ERROR_SWALLOW", kind=_KIND, severity=Severity.medium)
def silent_error_swallow(session: Session, config: Config) -> list[Finding]:
    """A tool failure is followed by an assistant turn that neither acknowledges
    it nor retries."""
    events = session.events
    out: list[Finding] = []
    for i, ev in enumerate(events):
        if ev.kind is not EventKind.tool_result or ev.ok is not False:
            continue
        acknowledged = False
        retried = False
        next_assistant: Event | None = None
        for nxt in events[i + 1 :]:
            if nxt.kind is EventKind.user_msg:
                break
            if nxt.kind is EventKind.tool_call:
                retried = True
                break
            if nxt.kind is EventKind.assistant_msg and next_assistant is None:
                next_assistant = nxt
                if nxt.text and _ACK_RE.search(nxt.text):
                    acknowledged = True
                break
        if next_assistant is not None and not acknowledged and not retried:
            out.append(
                make_finding(
                    "SILENT_ERROR_SWALLOW",
                    session,
                    kind=_KIND,
                    severity=Severity.medium,
                    confidence="high",
                    spans=[span(session, ev.idx, next_assistant.idx)],
                    evidence={"error_category": ev.error_category},
                    message="Tool failure not acknowledged and not retried.",
                )
            )
    return out


# ---------------------------------------------------------------------------
# 24. LINEAR_SCAN
# ---------------------------------------------------------------------------
@detector("LINEAR_SCAN", kind=_KIND, severity=Severity.low)
def linear_scan(session: Session, config: Config) -> list[Finding]:
    """Many manual file reads with zero grep/glob (should have searched)."""
    min_reads = int(_knob(config, "LINEAR_SCAN", "min_reads", 10))
    reads = [ev for ev in session.events if is_read_call(ev) and not is_grep_like(ev)]
    greps = [ev for ev in session.events if is_grep_like(ev)]
    if greps or len(reads) <= min_reads:
        return []
    return [
        make_finding(
            "LINEAR_SCAN",
            session,
            kind=_KIND,
            severity=Severity.low,
            confidence="high",
            spans=[span(session, reads[0].idx, reads[-1].idx)],
            evidence={"reads": len(reads), "grep_glob": 0, "threshold": min_reads},
            message=f"{len(reads)} manual reads with no grep/glob.",
        )
    ]


# ---------------------------------------------------------------------------
# 25. COMPACTION_AMNESIA
# ---------------------------------------------------------------------------
@detector("COMPACTION_AMNESIA", kind=_KIND, severity=Severity.medium)
def compaction_amnesia(session: Session, config: Config) -> list[Finding]:
    """After a compaction, a file read before it is re-read (redone work)."""
    comp_idx = next((ev.idx for ev in session.events if ev.kind is EventKind.compaction), None)
    if comp_idx is None:
        return []
    out: list[Finding] = []
    for path, st in session.file_state.items():
        pre = [r for r in st.read_at if r < comp_idx]
        post = [r for r in st.read_at if r > comp_idx]
        if pre and post:
            out.append(
                make_finding(
                    "COMPACTION_AMNESIA",
                    session,
                    kind=_KIND,
                    severity=Severity.medium,
                    confidence="high",
                    spans=[span(session, comp_idx, post[0])],
                    evidence={"path": path, "pre_read_idx": pre[-1], "post_read_idx": post[0]},
                    message=f"{path} re-read after compaction (already read before).",
                )
            )
    return out


# ---------------------------------------------------------------------------
# 26. STYLE_IMPOSITION  (fuzzy -> low)
# ---------------------------------------------------------------------------
@detector("STYLE_IMPOSITION", kind=_KIND, severity=Severity.low, confidence="low", llm_candidate=True)
def style_imposition(session: Session, config: Config) -> list[Finding]:
    """An edit imports a module that appears in no file the agent read.

    Fuzzy: rule proxy for fighting repo conventions; LLM-upgrade candidate.
    """
    read_blob = "\n".join(ev.output or "" for ev in session.events if ev.kind is EventKind.tool_result)
    if not read_blob.strip():
        return []
    out: list[Finding] = []
    for ev in session.events:
        if not is_write_call(ev):
            continue
        _old, new = _edit_text(ev)
        for match in _IMPORT_RE.finditer(new):
            mod = next((g for g in match.groups() if g), None)
            if not mod:
                continue
            top = mod.split(".")[0].split("/")[0]
            if top and top not in read_blob:
                out.append(
                    make_finding(
                        "STYLE_IMPOSITION",
                        session,
                        kind=_KIND,
                        severity=Severity.low,
                        confidence="low",
                        spans=[span(session, ev.idx)],
                        evidence={"module": mod, "path": ev.path},
                        message=f"Introduced import '{mod}' absent from all read files.",
                    )
                )
                return out
    return out


# ---------------------------------------------------------------------------
# 27. CREDENTIAL_LEAK
# ---------------------------------------------------------------------------
@detector("CREDENTIAL_LEAK", kind=_KIND, severity=Severity.high)
def credential_leak(session: Session, config: Config) -> list[Finding]:
    """A secret pattern appears in an outbound message or tool-call argument."""
    lex = Lexicons.from_config(config)
    out: list[Finding] = []
    for ev in session.events:
        if ev.kind not in (EventKind.assistant_msg, EventKind.tool_call):
            continue
        text = _outbound_text(ev)
        m = lex.secret.search(text)
        if m:
            out.append(
                make_finding(
                    "CREDENTIAL_LEAK",
                    session,
                    kind=_KIND,
                    severity=Severity.high,
                    confidence="high",
                    spans=[span(session, ev.idx)],
                    evidence={"match_prefix": m.group(0)[:8] + "..."},
                    message="A credential-shaped secret appears in an outbound payload.",
                )
            )
    return out


# ---------------------------------------------------------------------------
# 28. RUNAWAY_SESSION
# ---------------------------------------------------------------------------
@detector("RUNAWAY_SESSION", kind=_KIND, severity=Severity.medium)
def runaway_session(session: Session, config: Config) -> list[Finding]:
    """Trace blows past a hard event/token/minute cap with no positive outcome."""
    max_events = int(_knob(config, "RUNAWAY_SESSION", "max_events", 400))
    max_tokens = int(_knob(config, "RUNAWAY_SESSION", "max_tokens", 150_000))
    max_minutes = int(_knob(config, "RUNAWAY_SESSION", "max_minutes", 90))
    n = len(session.events)
    tokens = total_tokens(session)
    exceeded: dict[str, int] = {}
    if n > max_events:
        exceeded["events"] = n
    if tokens > max_tokens:
        exceeded["tokens"] = tokens
    minutes = _minutes(session)
    if minutes is not None and minutes > max_minutes:
        exceeded["minutes"] = int(minutes)
    if not exceeded or _ends_positive(session, config):
        return []
    return [
        make_finding(
            "RUNAWAY_SESSION",
            session,
            kind=_KIND,
            severity=Severity.medium,
            confidence="high",
            spans=[span(session, 0, max(0, n - 1))],
            evidence={"exceeded": exceeded},
            message="Session exceeds a hard cap with no positive outcome.",
        )
    ]


def _minutes(session: Session) -> float | None:
    if not session.has_timestamps:
        return None
    ts = [ev.ts for ev in session.events if ev.ts is not None]
    if len(ts) < 2:
        return None
    return (max(ts) - min(ts)).total_seconds() / 60.0


def _ends_positive(session: Session, config: Config) -> bool:
    lex = Lexicons.from_config(config)
    for ev in reversed(session.events):
        if ev.kind is EventKind.user_msg and ev.text:
            return bool(lex.positive.search(ev.text))
    if _last_passing_test_idx(session) is not None:
        for ev in session.events:
            if bash_command(ev) and "git commit" in bash_command(ev).lower():
                return True
    return False
