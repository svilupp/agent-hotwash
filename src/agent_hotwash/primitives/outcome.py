"""Outcome labeler — the deterministic success proxy.

There is no ground truth for most traces, so ``label_outcome`` infers a stable
``positive | negative | unknown`` label from end-of-session signals. For
code-bench (format A) runs the harness's ground-truth ``resolved`` is recorded
alongside (never overwritten) so we can measure the proxy's agreement with truth.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field

from agent_hotwash.events import EventKind, ToolCategory
from agent_hotwash.primitives.commands import classify_command
from agent_hotwash.primitives.lexicons import Lexicons

if TYPE_CHECKING:
    from agent_hotwash.config import Config
    from agent_hotwash.events import Event, Session, Trace

OutcomeLabel = Literal["positive", "negative", "unknown"]

_TEST_MARKERS = ("pytest", "vitest", "jest", "tsc", "unittest", " test", "test ")


class Outcome(BaseModel):
    """Inferred session outcome plus the reasons that decided it."""

    label: OutcomeLabel = "unknown"
    reasons: list[str] = Field(default_factory=list)
    ground_truth_resolved: bool | None = None  # from harness_meta (format A), never overwritten


def _bash_command(ev: Event) -> str | None:
    if ev.kind is not EventKind.tool_call or ev.tool_category is not ToolCategory.execute:
        return None
    args = ev.tool_args or {}
    for key in ("command", "cmd", "script"):
        val = args.get(key)
        if isinstance(val, str):
            return val
    return None


def _is_test_command(cmd: str) -> bool:
    low = cmd.lower()
    if classify_command(cmd) == "build_test":
        return True
    return any(m in low for m in _TEST_MARKERS)


def _last_test_result(session: Session) -> bool | None:
    """``True``/``False`` for the last test run's pass/fail, or ``None`` if there
    was none. The result ``ok`` is read from the linked tool_result when present,
    else from the tool_call's own ``ok``."""
    result_by_call: dict[str, bool | None] = {}
    for ev in session.events:
        if ev.kind is EventKind.tool_result and ev.call_id is not None:
            result_by_call[ev.call_id] = ev.ok
    last: bool | None = None
    for ev in session.events:
        cmd = _bash_command(ev)
        if cmd is None or not _is_test_command(cmd):
            continue
        ok = result_by_call.get(ev.call_id) if ev.call_id else None
        if ok is None:
            ok = ev.ok
        if ok is not None:
            last = ok
    return last


def _last_edit_idx(session: Session) -> int | None:
    last = None
    for ev in session.events:
        if ev.kind is EventKind.tool_call and ev.tool_category is ToolCategory.write:
            last = ev.idx
    return last


def _revert_after(session: Session, after_idx: int) -> bool:
    for ev in session.events:
        if ev.idx <= after_idx:
            continue
        cmd = _bash_command(ev)
        if not cmd:
            continue
        low = cmd.lower()
        if "git checkout" in low or "git revert" in low or "reset --hard" in low or "git restore" in low:
            return True
    return False


def _last_committed(session: Session) -> bool:
    for ev in session.events:
        cmd = _bash_command(ev)
        if cmd and "git commit" in cmd.lower() and ev.ok is not False:
            return True
    return False


def _last_user_text(session: Session) -> str:
    for ev in reversed(session.events):
        if ev.kind is EventKind.user_msg and ev.text:
            return ev.text
    return ""


def label_outcome(trace: Trace, config: Config) -> Outcome:
    """Label a trace's root session. Ordered, deterministic rules; the first
    matching signal wins."""
    lex = Lexicons.from_config(config)
    session = trace.root
    reasons: list[str] = []

    gt: bool | None = None
    hm = trace.provenance.harness_meta if trace.provenance else {}
    if isinstance(hm, dict):
        # The code-bench parser nests ground truth at harness_meta["verification"]
        # ["resolved"]; fall back to a flat "resolved" for other shapes.
        verification = hm.get("verification")
        resolved = verification.get("resolved") if isinstance(verification, dict) else None
        if resolved is None:
            resolved = hm.get("resolved")
        if resolved is not None:
            gt = bool(resolved)

    last_user = _last_user_text(session)
    last_test = _last_test_result(session)
    last_edit = _last_edit_idx(session)

    label: OutcomeLabel = "unknown"
    if last_user and lex.correction.search(last_user):
        label, why = "negative", "final user turn matches correction lexicon"
    elif last_test is False:
        label, why = "negative", "session ends on a failing test run"
    elif last_edit is not None and _revert_after(session, last_edit):
        label, why = "negative", "last edit was reverted"
    elif last_user and lex.positive.search(last_user):
        label, why = "positive", "final user turn matches positive lexicon"
    elif last_test is True:
        label, why = "positive", "session ends on a passing test run"
    elif _last_committed(session):
        label, why = "positive", "session includes a successful git commit"
    else:
        why = "no positive or negative end-of-session signal"
    reasons.append(why)

    return Outcome(label=label, reasons=reasons, ground_truth_resolved=gt)
