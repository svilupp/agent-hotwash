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

import itertools
import re
from typing import TYPE_CHECKING, Any

from agent_hotwash.detectors.registry import (
    Severity,
    assistant_msgs,
    bash_command,
    call_by_id,
    detector,
    edited_paths,
    failing_results,
    is_benign_failure,
    is_commentary,
    is_edit_tool,
    is_exec_call,
    is_grep_like,
    is_killed_result,
    is_read_call,
    is_source_like_path,
    is_turn_end,
    is_under_dir,
    is_write_call,
    logical_events,
    logical_positions,
    make_finding,
    read_paths,
    result_by_call,
    result_ok_by_call,
    span,
    terminal_assistant_msgs,
    tool_calls,
    total_tokens,
    user_msgs,
)
from agent_hotwash.events import EventKind, ToolCategory
from agent_hotwash.primitives.argnorm import edit_distance, norm_args
from agent_hotwash.primitives.commands import classify_command, split_segments, strip_shell_wrapper
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
# Words an assistant uses when it has noticed a failure. ``\w*error\w*`` covers
# `PermissionError` / `OSError`-style names quoted from a traceback.
_ACK_RE = re.compile(
    r"\b(\w*error\w*|\w*exception\w*|traceback|fail|fails|failed|failing|failure|issue|problem|retry|retrying|"
    r"fix|fixing|wrong|broke|broken|didn'?t|couldn'?t|can'?t|cannot|could\s+not|unable|revert|timed\s+out|timeout|"
    r"mismatch|not\s+found|failed\s+to|missing|no\s+match(?:es)?|blocked|blocks|denied|unavailable|crash(?:ed)?)\b",
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
_DOC_EXT_RE = re.compile(r"\.(md|rst|txt|adoc)$", re.IGNORECASE)
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
    events = logical_events(session)
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
            spans=[span(session, events[n - third].idx, events[n - 1].idx)],
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
    events = logical_events(session)
    win = SlidingWindow[Any](window_events)
    hit = win.first_window_reaching(
        events,
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
            spans=[span(session, events[start].idx, events[end].idx)],
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
    """A targeted in-place edit to a file the agent never looked at.

    Keys on every ``update`` artifact of an edit call (adds/deletes/moves are
    not edits of existing content). The edit is grounded when the exact path
    was read or written earlier (``file_state``), or when a read/search over a
    parent directory of it happened within the last ``read_window_events``
    logical events (``rg src/`` before editing ``src/x.py``).
    """
    window = int(_knob(config, "EDIT_WITHOUT_READ", "read_window_events", 50))
    # Guard: when the source cannot tell reads from other commands there is
    # nothing to ground against — stay silent rather than flag every edit.
    # A source that declares ``parsed_commands`` is judged even when it never
    # read anything (the strongest positive case); sources without a declared
    # row (codebench/claude) fall back to "did we observe any read at all".
    if (
        not session.capabilities.meets("parsed_commands")
        and not any(st.read_at for st in session.file_state.values())
        and not any(is_read_call(ev) or is_grep_like(ev) for ev in session.events)
    ):
        return []
    logical = logical_events(session)
    pos = logical_positions(session)
    out: list[Finding] = []
    seen: set[str] = set()
    for ev in session.events:
        if not is_edit_tool(ev):
            continue
        for path in edited_paths(ev):
            if path in seen:
                continue
            if _edit_grounded(session, logical, pos, path, ev, window):
                continue
            seen.add(path)
            out.append(
                make_finding(
                    "EDIT_WITHOUT_READ",
                    session,
                    kind=_KIND,
                    severity=Severity.medium,
                    confidence="high",
                    spans=[span(session, ev.idx)],
                    evidence={"path": path},
                    message=f"Edited {path} without reading it first.",
                )
            )
    return out


def _edit_grounded(
    session: Session, logical: list[Event], pos: dict[int, int], path: str, edit: Event, window: int
) -> bool:
    """Exact-path read/write before the edit, or a recent read of a parent dir.

    Fallback for compound reads the decoder could not attribute paths to
    (``sed … && rg … file``): a recent read/search command whose text names the
    file's basename also grounds the edit.
    """
    st = session.file_state.get(path)
    if st is not None and (any(r < edit.idx for r in st.read_at) or any(e < edit.idx for e in st.edited_at)):
        return True
    p = pos.get(edit.idx)
    if p is None:
        return False
    base = _basename(path)
    for prev in reversed(logical[max(0, p - window) : p]):
        for rp in read_paths(prev):
            if rp == path or is_under_dir(path, rp):
                return True
        if is_read_call(prev) or is_grep_like(prev):
            text = (prev.tool_args or {}).get("command") or (prev.tool_args or {}).get("cmd") or ""
            if len(base) >= 4 and isinstance(text, str) and base in text:
                return True
    return False


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
_POLLING_OP_RE = re.compile(r"wait|list|poll", re.IGNORECASE)


def _is_polling_op(ev: Event) -> bool:
    """Subagent orchestration and MCP wait/list/poll calls repeat by design."""
    if ev.tool_category is ToolCategory.subagent:
        return True
    name = ev.op_kind or ev.tool_name or ""
    return name.startswith("mcp.") and bool(_POLLING_OP_RE.search(name))


@detector("RETRY_STORM", kind=_KIND, severity=Severity.medium)
def retry_storm(session: Session, config: Config) -> list[Finding]:
    """The same (tool, normalized-args) call hammered >= N times in a short span.

    Keys on calls with non-empty normalized args (a FileChange with no args is
    not "the same call"), excluding subagent/MCP polling ops. The repeats must
    fall within ``window_events`` logical events with no file write between
    them (an edit->rerun loop is adaptation, not a storm) and at least one of
    them must have failed for real (benign probes and killed processes do not
    count).
    """
    min_repeats = int(_knob(config, "RETRY_STORM", "min_repeats", 4))
    window = int(_knob(config, "RETRY_STORM", "window_events", 40))
    pos = logical_positions(session)
    results = result_by_call(session)
    write_idxs = [ev.idx for ev in session.events if is_write_call(ev)]
    groups: dict[tuple[str, str], list[Event]] = {}
    for ev in tool_calls(session):
        if not ev.tool_norm_args or _is_polling_op(ev):
            continue
        groups.setdefault((ev.tool_name or "?", ev.tool_norm_args), []).append(ev)
    out: list[Finding] = []
    for (name, _args), evs in groups.items():
        if len(evs) < min_repeats:
            continue
        hit = _dense_failing_run(evs, pos, results, write_idxs, min_repeats, window)
        if hit is None:
            continue
        out.append(
            make_finding(
                "RETRY_STORM",
                session,
                kind=_KIND,
                severity=Severity.medium,
                confidence="high",
                spans=[span(session, hit[0].idx, hit[-1].idx)],
                evidence={"tool": name, "repeats": len(hit), "window_events": window},
                message=f"'{name}' called with identical args {len(hit)}x within {window} events.",
            )
        )
    return out


def _dense_failing_run(
    evs: list[Event],
    pos: dict[int, int],
    results: dict[str, Event],
    write_idxs: list[int],
    min_repeats: int,
    window: int,
) -> list[Event] | None:
    """Longest run of ``evs`` fitting in ``window`` logical events with no file
    write inside it (starting at the first run that reaches ``min_repeats``)
    that contains a real failure."""
    for i in range(len(evs) - min_repeats + 1):
        start = pos.get(evs[i].idx, evs[i].idx)
        run = [e for e in evs[i:] if pos.get(e.idx, e.idx) - start <= window]
        # Cut the run at the first file write between two members.
        trimmed = [run[0]]
        for e in run[1:]:
            if any(trimmed[-1].idx < w < e.idx for w in write_idxs):
                break
            trimmed.append(e)
        run = trimmed
        if len(run) < min_repeats:
            continue
        failed = any(
            (r := results.get(e.call_id or "")) is not None and r.ok is False and not is_benign_failure(r, e)
            for e in run
        )
        if failed:
            return run
    return None


# ---------------------------------------------------------------------------
# 8. NO_ADAPT_RETRY
# ---------------------------------------------------------------------------
_WS_RE = re.compile(r"\s+")


def _retry_key(ev: Event) -> str:
    """Args compared for "did the agent adapt?": the shell command alone for
    exec calls (``cwd`` and the duplicated ``cmd`` key are noise), otherwise the
    normalized args minus ``cwd``."""
    args = ev.tool_args or {}
    cmd = args.get("command") or args.get("cmd")
    if isinstance(cmd, str):
        return _WS_RE.sub(" ", cmd.strip())
    if "cwd" in args:
        return norm_args({k: v for k, v in args.items() if k != "cwd"})
    return ev.tool_norm_args or ""


@detector("NO_ADAPT_RETRY", kind=_KIND, severity=Severity.medium)
def no_adapt_retry(session: Session, config: Config) -> list[Finding]:
    """A failing call is immediately re-run with barely-changed args.

    Clusters only over CONSECUTIVE tool calls of the same tool — any other tool
    call, edit or user message in between means the agent did something before
    retrying, which is not "no adaptation". Killed/interrupted results (exit
    130/143, ``^C``) are not failures to adapt to. "Near-identical" is an edit
    distance below ``arg_edit_distance_eps`` that is also small relative to the
    command length (``bun run test`` -> ``bun run check`` is a different command).
    """
    min_repeats = int(_knob(config, "NO_ADAPT_RETRY", "min_repeats", 2))
    eps = int(_knob(config, "NO_ADAPT_RETRY", "arg_edit_distance_eps", 5))
    results = result_by_call(session)
    out: list[Finding] = []
    cluster: list[Event] = []

    def flush() -> None:
        if len(cluster) >= min_repeats:
            name = cluster[0].tool_name or "?"
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
        cluster.clear()

    for ev in logical_events(session):
        if ev.kind is EventKind.user_msg:
            flush()
            continue
        if ev.kind is not EventKind.tool_call:
            continue
        res = results.get(ev.call_id or "")
        failed = res is not None and res.ok is False and not is_killed_result(res)
        if not failed:
            flush()
            continue
        if cluster and (ev.tool_name != cluster[-1].tool_name or not _near_identical(cluster[-1], ev, eps)):
            flush()
        cluster.append(ev)
    flush()
    return out


def _near_identical(a: Event, b: Event, eps: int) -> bool:
    ka, kb = _retry_key(a), _retry_key(b)
    dist = edit_distance(ka, kb, cap=eps + 1)
    return dist < eps and dist <= max(1, min(len(ka), len(kb)) // 4)


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
    """The files the agent edited barely cover the source files named in the ask.

    Keys on source-like paths (must carry a code/config extension — not
    hostnames, e-mails or screenshots) in the opening user message, and on the
    basenames of paths that were EDITED (not merely read/listed). Recall =
    |asked ∩ edited| / |asked|; below ``recall_eps`` is drift.
    Fuzzy: path-recall proxy; LLM-upgrade candidate.
    """
    eps = float(_knob(config, "GOAL_DRIFT", "recall_eps", _knob(config, "GOAL_DRIFT", "jaccard_eps", 0.2)))
    users = user_msgs(session)
    if not users or not users[0].text:
        return []
    # Docs named in the ask (plans, READMEs) are there to be read, not edited.
    asked = {
        _basename(p) for p in _PATH_RE.findall(users[0].text) if is_source_like_path(p) and not _DOC_EXT_RE.search(p)
    }
    edited = {_basename(p) for p, st in session.file_state.items() if st.edited_at and is_source_like_path(p)}
    if not asked or not edited:
        return []
    recall = len(asked & edited) / len(asked)
    if recall >= eps:
        return []
    return [
        make_finding(
            "GOAL_DRIFT",
            session,
            kind=_KIND,
            severity=Severity.low,
            confidence="low",
            spans=[span(session, users[0].idx)],
            evidence={"recall": round(recall, 3), "asked": sorted(asked), "edited": sorted(edited)},
            message=f"Edited files cover only {recall:.0%} of the files named in the ask.",
        )
    ]


def _basename(path: str) -> str:
    return path.rsplit("/", 1)[-1]


# ---------------------------------------------------------------------------
# 21. LOOKS_RIGHT_RUNS_WRONG
# ---------------------------------------------------------------------------
@detector("LOOKS_RIGHT_RUNS_WRONG", kind=_KIND, severity=Severity.medium)
def looks_right_runs_wrong(session: Session, config: Config) -> list[Finding]:
    """A file is edited and declared done, but no later test/build run could
    have exercised it.

    A project-wide runner (``make check``, ``bun run test``, ``uv run pytest``
    with no file arguments) after the last edit counts as exercising every
    edited file; a file-targeted run counts when it names an edited file (or
    its ``test_<stem>`` counterpart). The completion claim must come from a
    terminal assistant message (``phase`` None or ``final_answer``).
    """
    lex = Lexicons.from_config(config)
    last_edit, paths = _last_edit(session)
    if last_edit is None:
        return []
    if not _completion_after(session, last_edit, lex):
        return []
    build_tests = _build_tests_after(session, last_edit)
    if any(_run_exercises(cmd, paths) for cmd in build_tests):
        return []
    return [
        make_finding(
            "LOOKS_RIGHT_RUNS_WRONG",
            session,
            kind=_KIND,
            severity=Severity.medium,
            confidence="high",
            spans=[span(session, last_edit)],
            evidence={"edited_paths": sorted(paths)},
            message="Edited files declared done but never executed/tested.",
        )
    ]


_PATH_TOKEN_RE = re.compile(r"(?:^|\s)(?:[\w.-]+/)*[\w.-]+\.[A-Za-z0-9]{1,6}(?=\s|$)")
# Code files a test runner can be pointed at (a `.toml` flow or `.json` config
# argument does not make the run file-targeted).
_CODE_EXT_RE = re.compile(r"\.(py|pyi|ts|tsx|js|jsx|mjs|cjs|go|rs|java|kt|rb|php|c|cc|cpp|cs|swift|exs?|jl)$", re.I)


def _run_exercises(cmd: str, edited: set[str]) -> bool:
    """Does a build/test command plausibly exercise one of ``edited``?

    True for project-wide runners (no code-file arguments in the build_test
    segment) and for file-targeted runs naming an edited file's stem.
    """
    inner = strip_shell_wrapper(cmd)
    segs = [s for s in split_segments(inner) if classify_command(s) == "build_test"] or [inner]
    for seg in segs:
        file_args = [t.strip() for t in _PATH_TOKEN_RE.findall(seg) if _CODE_EXT_RE.search(t.strip())]
        if not file_args:
            return True
        for p in edited:
            base, stem = _basename(p), _stem(p)
            if any(_basename(f) == base for f in file_args):
                return True
            if len(stem) >= 3 and any(stem in _stem(f) for f in file_args):
                return True
    return False


def _stem(path: str) -> str:
    base = _basename(path)
    stem = base.rsplit(".", 1)[0] if "." in base else base
    stem = stem.removeprefix("test_").removesuffix("_test").removesuffix(".test").removesuffix(".spec")
    return stem.lower()


# ---------------------------------------------------------------------------
# 22. UNVERIFIED_COMPLETION
# ---------------------------------------------------------------------------
@detector("UNVERIFIED_COMPLETION", kind=_KIND, severity=Severity.medium)
def unverified_completion(session: Session, config: Config) -> list[Finding]:
    """A completion claim with no test/build run after the last edit.

    The claim must come from a terminal assistant message (``phase`` None or
    ``final_answer``); any ``build_test`` command after the last edit (``make``,
    ``bun run check``, ``python -m pytest`` …) counts as verification.
    """
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
    """Index of the last write to a CODE file and every code path written.

    Docs/config-only edits (CHANGELOG.md, .gitignore, dist/index.html) need no
    test run, so they neither anchor nor count.
    """
    last: int | None = None
    paths: set[str] = set()
    for ev in session.events:
        if not is_write_call(ev):
            continue
        code_paths = [a.path for a in ev.artifacts if a.path and _CODE_EXT_RE.search(a.path)] or (
            [ev.path] if ev.path and not ev.artifacts and _CODE_EXT_RE.search(ev.path) else []
        )
        if code_paths:
            last = ev.idx
            paths.update(code_paths)
    return last, paths


def _completion_after(session: Session, after_idx: int, lex: Lexicons) -> bool:
    """An assistant completion claim after ``after_idx`` — only in messages that
    can be the turn's answer (``phase`` None or ``final_answer``), never in
    Codex ``commentary`` narration ("done reading, now editing…")."""
    for ev in session.events:
        if ev.idx <= after_idx or ev.kind is not EventKind.assistant_msg or is_commentary(ev):
            continue
        if ev.text and lex.completion.search(ev.text):
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
    """A real tool failure is followed by the turn's answer without the agent
    ever acknowledging it or acting on it.

    Scans forward from a failing ``tool_result``: any tool call means the agent
    kept working (not swallowed); any assistant text matching the ack lexicon
    means it noticed. Codex ``commentary`` messages do not end the scan — only
    the terminal message (``final_answer`` / last message before the turn ends)
    does. Benign failures (no-match probes, read commands exiting 1 with tiny
    output, killed processes) are not failures to acknowledge.
    """
    events = session.events
    calls = call_by_id(session)
    terminal = {e.idx for e in terminal_assistant_msgs(session)}
    out: list[Finding] = []
    for i, ev in enumerate(events):
        if ev.kind is not EventKind.tool_result or ev.ok is not False:
            continue
        if is_benign_failure(ev, calls.get(ev.call_id) if ev.call_id else None):
            continue
        acknowledged = False
        retried = False
        next_assistant: Event | None = None
        for nxt in events[i + 1 :]:
            if nxt.kind is EventKind.user_msg or is_turn_end(nxt):
                break
            if nxt.kind is EventKind.tool_call:
                retried = True
                break
            if nxt.kind is EventKind.assistant_msg:
                if nxt.text and _ACK_RE.search(nxt.text):
                    acknowledged = True
                    break
                if nxt.idx in terminal:
                    next_assistant = nxt
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
    """Right after a compaction the agent re-reads files it had already read.

    One finding per compaction event (anchored on the NEAREST preceding
    compaction). Only re-reads in the same turn and within ``window_events``
    logical events of the compaction count — a re-read hours later in a new
    task is not amnesia. Evidence lists the re-read paths (capped at 10).
    """
    window = int(_knob(config, "COMPACTION_AMNESIA", "window_events", 20))
    comps = [ev for ev in session.events if ev.kind is EventKind.compaction]
    if not comps:
        return []
    logical = logical_events(session)
    pos = logical_positions(session)
    out: list[Finding] = []
    for ci, comp in enumerate(comps):
        nxt_comp = comps[ci + 1].idx if ci + 1 < len(comps) else None
        cpos = pos.get(comp.idx)
        if cpos is None:
            continue
        reread: list[str] = []
        last_idx = comp.idx
        for ev in logical[cpos + 1 : cpos + 1 + window]:
            if nxt_comp is not None and ev.idx >= nxt_comp:
                break
            if ev.kind is EventKind.user_msg:
                break  # new turn — re-reads there are the new task's reads
            if ev.turn_id and comp.turn_id and ev.turn_id != comp.turn_id:
                break
            for path in read_paths(ev):
                st = session.file_state.get(path)
                if st is None or path in reread or not any(r < comp.idx for r in st.read_at):
                    continue
                reread.append(path)
                last_idx = ev.idx
        if not reread:
            continue
        out.append(
            make_finding(
                "COMPACTION_AMNESIA",
                session,
                kind=_KIND,
                severity=Severity.medium,
                confidence="high",
                spans=[span(session, comp.idx, last_idx)],
                evidence={"reread_paths": reread[:10], "reread_count": len(reread), "window_events": window},
                message=f"{len(reread)} already-read file(s) re-read within {window} events of a compaction.",
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
    """Trace blows past a hard event/token/active-minute cap and never reaches a
    positive outcome.

    Positive outcome = the last turn concluded (a ``final_answer`` message or a
    ``task_complete`` marker after the last user message), the user replied
    positively after the agent answered, or a passing test run followed by a
    commit. Minutes are ACTIVE time: gaps longer than ``idle_gap_minutes`` are
    excluded (an overnight pause is not runaway work). Tokens are uncached
    input + output summed over model calls.
    """
    max_events = int(_knob(config, "RUNAWAY_SESSION", "max_events", 400))
    max_tokens = int(_knob(config, "RUNAWAY_SESSION", "max_tokens", 150_000))
    max_minutes = int(_knob(config, "RUNAWAY_SESSION", "max_minutes", 90))
    idle_gap = float(_knob(config, "RUNAWAY_SESSION", "idle_gap_minutes", 10.0))
    n = len(logical_events(session))
    tokens = total_tokens(session)
    exceeded: dict[str, int] = {}
    if n > max_events:
        exceeded["events"] = n
    if tokens > max_tokens:
        exceeded["tokens"] = tokens
    minutes = _active_minutes(session, idle_gap)
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


def _active_minutes(session: Session, idle_gap_minutes: float) -> float | None:
    """Wall-clock minutes excluding gaps longer than ``idle_gap_minutes``."""
    if not session.has_timestamps:
        return None
    ts = sorted(ev.ts for ev in session.events if ev.ts is not None)
    if len(ts) < 2:
        return None
    active = 0.0
    for prev, cur in itertools.pairwise(ts):
        gap = (cur - prev).total_seconds() / 60.0
        if gap <= idle_gap_minutes:
            active += gap
    return active


def _ends_positive(session: Session, config: Config) -> bool:
    """Did the session reach a positive outcome?

    1. The last turn concluded: a ``final_answer`` assistant message or a
       ``task_complete`` marker after the last user message (Codex).
    2. The user replied positively AFTER the agent had answered (the last user
       message is otherwise just the task statement).
    3. A passing test run and a ``git commit`` somewhere in the session.
    """
    lex = Lexicons.from_config(config)
    events = session.events
    last_user = next((ev for ev in reversed(events) if ev.kind is EventKind.user_msg), None)
    last_user_idx = last_user.idx if last_user is not None else -1
    for ev in events:
        if ev.idx <= last_user_idx:
            continue
        if (ev.kind is EventKind.assistant_msg and ev.phase == "final_answer") or is_turn_end(ev):
            return True
    replied_positively = (
        last_user is not None
        and bool(last_user.text)
        and bool(lex.positive.search(last_user.text or ""))
        and any(ev.kind is EventKind.assistant_msg and ev.idx < last_user_idx for ev in events)
    )
    if replied_positively:
        return True
    if _last_passing_test_idx(session) is not None:
        for ev in events:
            if bash_command(ev) and "git commit" in bash_command(ev).lower():
                return True
    return False
