"""Tests for the native pi session parser."""

from __future__ import annotations

from pathlib import Path

from agent_hotwash.events import AgentKind, EventKind, ToolCategory
from agent_hotwash.sources.codex_native import looks_like_native_codex
from agent_hotwash.sources.detect import iter_traces
from agent_hotwash.sources.pi_native import (
    load_session_file,
    looks_like_native_pi,
)

SESSION = Path(__file__).parent / "fixtures" / "pi_native" / "session-fixture.jsonl"
CODEX_ROLLOUT = Path(__file__).parent / "fixtures" / "codex_native" / "rollout-fixture.jsonl"
CLAUDE_DIR = Path(__file__).parent / "fixtures" / "claude_native"


def test_detects_native_pi() -> None:
    assert looks_like_native_pi(SESSION)


def test_detection_does_not_misfire_on_other_formats() -> None:
    # A codex rollout (session_meta with a payload envelope) is not native pi.
    assert not looks_like_native_pi(CODEX_ROLLOUT)
    # And native pi is not mistaken for codex.
    assert not looks_like_native_codex(SESSION)
    for claude_file in CLAUDE_DIR.glob("*.jsonl"):
        assert not looks_like_native_pi(claude_file)


def test_session_id_and_model() -> None:
    trace = load_session_file(SESSION)
    assert trace.agent is AgentKind.pi
    assert trace.root.session_id == "eeeeeeee-0000-0000-0000-000000000005"
    assert trace.model == "acme-mini"  # from model_change record
    assert trace.root.has_timestamps  # epoch-ms message timestamps


def test_roles_split_into_events() -> None:
    trace = load_session_file(SESSION)
    kinds = [e.kind for e in trace.root.events]
    assert EventKind.user_msg in kinds
    assert EventKind.assistant_msg in kinds
    assert EventKind.thinking in kinds
    assert EventKind.tool_call in kinds
    assert EventKind.tool_result in kinds


def test_tools_categorized_and_linked() -> None:
    trace = load_session_file(SESSION)
    calls = [e for e in trace.root.events if e.kind is EventKind.tool_call]
    results = [e for e in trace.root.events if e.kind is EventKind.tool_result]
    bash_call = next(e for e in calls if e.call_id == "call_1")
    edit_call = next(e for e in calls if e.call_id == "call_2")
    assert bash_call.tool_category is ToolCategory.execute
    assert edit_call.tool_category is ToolCategory.write
    # file-op path recovered from edit args; feeds file_state.
    assert edit_call.path == "app.py"
    assert "app.py" in trace.root.file_state
    assert len(results) == len(calls)


def test_error_flag_and_classification() -> None:
    trace = load_session_file(SESSION)
    failed = [e for e in trace.root.events if e.kind is EventKind.tool_result and e.ok is False]
    assert len(failed) == 1
    assert failed[0].call_id == "call_2"  # top-level isError flag honored
    assert failed[0].error_category == "file_not_found"


def test_usage_summable_per_response() -> None:
    trace = load_session_file(SESSION)
    usages = [e.usage for e in trace.root.events if e.usage]
    # one usage per assistant response (3 responses), attached to first block.
    assert len(usages) == 3
    assert sum(u.input for u in usages if u.input) == 350
    assert sum(u.output for u in usages if u.output) == 70
    assert sum(u.cache_read for u in usages if u.cache_read) == 220


def test_iter_traces_detects_single_file_and_project_dir() -> None:
    # single file
    traces = list(iter_traces(SESSION))
    assert len(traces) == 1
    assert traces[0].agent is AgentKind.pi
    # project dir containing the session file
    dir_traces = list(iter_traces(SESSION.parent))
    assert len(dir_traces) == 1
    assert dir_traces[0].provenance.source_format == "pi_native"
