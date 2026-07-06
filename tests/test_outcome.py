"""Outcome-labeler tests on crafted mini-sessions."""

from __future__ import annotations

from pathlib import Path

from agent_hotwash.config import load_config
from agent_hotwash.events import (
    AgentKind,
    Event,
    EventKind,
    Provenance,
    ToolCategory,
    Trace,
)
from agent_hotwash.primitives.outcome import label_outcome
from agent_hotwash.sources._common import build_session

CFG = load_config()


def _trace(events: list[Event], harness_meta: dict | None = None) -> Trace:
    sess = build_session(events, AgentKind.claude, session_id="s")
    prov = Provenance(
        source_format="claude_native",
        detector_confidence="high",
        root_path=Path("/tmp/x"),
        harness_meta=harness_meta or {},
    )
    return Trace(trace_id="t", agent=AgentKind.claude, root=sess, provenance=prov)


def _bash(cmd: str, ok: bool, call_id: str) -> list[Event]:
    return [
        Event(
            kind=EventKind.tool_call,
            tool_name="Bash",
            tool_category=ToolCategory.execute,
            tool_args={"command": cmd},
            call_id=call_id,
        ),
        Event(kind=EventKind.tool_result, call_id=call_id, ok=ok),
    ]


def test_positive_passing_test() -> None:
    out = label_outcome(_trace(_bash("pytest -q", True, "c1")), CFG)
    assert out.label == "positive"


def test_negative_failing_test() -> None:
    out = label_outcome(_trace(_bash("pytest -q", False, "c1")), CFG)
    assert out.label == "negative"


def test_positive_lexicon_beats_nothing() -> None:
    events = [Event(kind=EventKind.user_msg, text="perfect, thanks!")]
    assert label_outcome(_trace(events), CFG).label == "positive"


def test_correction_lexicon_negative() -> None:
    events = [
        *_bash("pytest -q", True, "c1"),
        Event(kind=EventKind.user_msg, text="no, that's still broken"),
    ]
    # correction on final user turn wins over the earlier passing test
    assert label_outcome(_trace(events), CFG).label == "negative"


def test_positive_git_commit() -> None:
    events = _bash("git commit -m 'fix'", True, "c1")
    assert label_outcome(_trace(events), CFG).label == "positive"


def test_negative_revert_after_edit() -> None:
    events = [
        Event(
            kind=EventKind.tool_call,
            tool_name="Edit",
            tool_category=ToolCategory.write,
            tool_args={"file_path": "a.py"},
            call_id="e1",
        ),
        *_bash("git checkout -- a.py", True, "c1"),
    ]
    assert label_outcome(_trace(events), CFG).label == "negative"


def test_unknown_when_no_signal() -> None:
    events = [Event(kind=EventKind.user_msg, text="hello there")]
    assert label_outcome(_trace(events), CFG).label == "unknown"


def test_ground_truth_recorded_not_overwritten() -> None:
    # proxy says unknown, but harness resolved=True is recorded alongside
    out = label_outcome(_trace([Event(kind=EventKind.user_msg, text="hi")], {"resolved": True}), CFG)
    assert out.label == "unknown"
    assert out.ground_truth_resolved is True


def test_ground_truth_recorded_from_verification_nesting() -> None:
    # Real code-bench harness_meta nests ground truth at verification.resolved.
    hm = {"run": {"harness": "claude"}, "verification": {"resolved": False}, "metrics": {}}
    out = label_outcome(_trace([Event(kind=EventKind.user_msg, text="hi")], hm), CFG)
    assert out.ground_truth_resolved is False
