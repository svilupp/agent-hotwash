"""Tests for the native pi session parser."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import agent_hotwash.detectors  # noqa: F401 -- register detectors for skip test
from agent_hotwash.analytics import analyze
from agent_hotwash.config import load_config
from agent_hotwash.detectors.registry import run_detectors
from agent_hotwash.diagnostics.cost_views import session_invoice, tree_rollup
from agent_hotwash.events import AgentKind, EventKind, PricingStatus, ToolCategory
from agent_hotwash.semantic.pipeline import annotate_trace
from agent_hotwash.sources.codex_native import looks_like_native_codex
from agent_hotwash.sources.detect import iter_traces
from agent_hotwash.sources.pi_native import (
    estimate_usage_from_total,
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


def _write_pi(path: Path, records: list[dict]) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    return path


def _header(session_id: str, *, parent: Path | None = None, ts: str = "2026-08-01T00:00:00.000Z") -> dict:
    row = {
        "type": "session",
        "version": 3,
        "id": session_id,
        "timestamp": ts,
        "cwd": "/tmp/proj",
    }
    if parent is not None:
        row["parentSession"] = str(parent.resolve())
    return row


def _user(text: str = "go") -> dict:
    return {
        "type": "message",
        "timestamp": 1754006400000,
        "message": {"role": "user", "content": [{"type": "text", "text": text}], "timestamp": 1754006400000},
    }


def _assistant(*, model: str = "claude-fable-5", usage: dict | None = None, text: str = "ok") -> dict:
    return {
        "type": "message",
        "timestamp": 1754006401000,
        "message": {
            "role": "assistant",
            "content": [{"type": "text", "text": text}],
            "model": model,
            "provider": "anthropic",
            "usage": usage or {"input": 100, "output": 20, "cacheRead": 0, "cacheWrite": 0, "totalTokens": 120},
            "timestamp": 1754006401000,
        },
    }


def _model_change(model: str = "claude-fable-5") -> dict:
    return {
        "type": "model_change",
        "timestamp": "2026-08-01T00:00:00.100Z",
        "provider": "anthropic",
        "modelId": model,
    }


def _notification(agent_id: str, total_tokens: int, *, status: str = "completed") -> dict:
    return {
        "type": "custom_message",
        "customType": "subagent-notification",
        "timestamp": "2026-08-01T00:05:00.000Z",
        "details": {"id": agent_id, "totalTokens": total_tokens, "status": status, "toolUses": 3},
    }


def test_estimate_usage_mix() -> None:
    usage = estimate_usage_from_total(1_000_000)
    assert usage.cache_read == 800_000
    assert usage.output == 80_000
    assert usage.input == 120_000
    assert usage.cache_write == 0


def test_project_dir_rolls_parent_and_child_into_one_trace(tmp_path: Path) -> None:
    parent_id = "aaaaaaaa-0000-0000-0000-000000000001"
    child_id = "bbbbbbbb-0000-0000-0000-000000000002"
    parent = _write_pi(
        tmp_path / f"2026-08-01T00-00-00-000Z_{parent_id}.jsonl",
        [_header(parent_id), _model_change(), _user(), _assistant()],
    )
    _write_pi(
        tmp_path / f"2026-08-01T00-01-00-000Z_{child_id}.jsonl",
        [
            _header(child_id, parent=parent),
            {"type": "session_info", "name": "advisor#aa966699", "timestamp": "2026-08-01T00:01:00.000Z"},
            _model_change(),
            _user("child"),
            _assistant(usage={"input": 10, "output": 5, "cacheRead": 0, "cacheWrite": 0, "totalTokens": 15}),
        ],
    )
    traces = list(iter_traces(tmp_path))
    assert len(traces) == 1
    trace = traces[0]
    assert trace.trace_id == parent_id
    assert {s.session_id for s in trace.subagents} == {child_id}
    assert trace.provenance.thread_linkage == "full"
    assert trace.links[0].evidence == ["parentSession"]


def test_missing_child_estimated_from_notification(tmp_path: Path) -> None:
    parent_id = "aaaaaaaa-0000-0000-0000-000000000003"
    path = _write_pi(
        tmp_path / f"2026-08-01T00-00-00-000Z_{parent_id}.jsonl",
        [
            _header(parent_id),
            _model_change(),
            _user(),
            _assistant(),
            _notification("deadbeef-1111-222", 1_000_000),
            _notification("deadbeef-1111-222", 500_000, status="running"),  # max wins
        ],
    )
    trace = load_session_file(path)
    assert len(trace.subagents) == 1
    child = trace.subagents[0]
    assert child.session_id == "deadbeef-1111-222"
    assert child.usage_reliable is False
    assert "usage_estimated" in child.degraded
    assert child.events[0].usage is not None
    assert child.events[0].usage.cache_read == 800_000
    assert trace.provenance.thread_linkage == "partial"
    assert any("estimated" in n for n in trace.provenance.notes)

    cfg = load_config()
    analysis = analyze(trace, cfg)
    assert analysis.subagent_count == 1
    assert "usage_estimated" in analysis.degraded
    invoice = session_invoice(child, cfg)
    assert invoice.pricing_status is PricingStatus.estimated
    assert invoice.amount == pytest.approx(6.0)
    rollup, _carry, status = tree_rollup(trace, cfg)
    assert status == "partial"
    assert rollup.pricing_status is PricingStatus.estimated
    assert rollup.amount is not None and rollup.amount > 6.0


def test_persisted_child_does_not_double_count_notification(tmp_path: Path) -> None:
    parent_id = "aaaaaaaa-0000-0000-0000-000000000004"
    child_id = "cccccccc-0000-0000-0000-000000000005"
    parent = _write_pi(
        tmp_path / f"2026-08-01T00-00-00-000Z_{parent_id}.jsonl",
        [
            _header(parent_id),
            _model_change(),
            _user(),
            _assistant(),
            _notification("aa966699-88f4-4ba", 9_999_999),
        ],
    )
    _write_pi(
        tmp_path / f"2026-08-01T00-01-00-000Z_{child_id}.jsonl",
        [
            _header(child_id, parent=parent),
            {"type": "session_info", "name": "advisor#aa966699", "timestamp": "2026-08-01T00:01:00.000Z"},
            _model_change(),
            _user("child"),
            _assistant(usage={"input": 10, "output": 5, "cacheRead": 0, "cacheWrite": 0, "totalTokens": 15}),
        ],
    )
    traces = list(iter_traces(tmp_path))
    assert len(traces) == 1
    ids = {s.session_id for s in traces[0].subagents}
    assert ids == {child_id}
    assert "aa966699-88f4-4ba" not in ids
    assert traces[0].subagents[0].usage_reliable is True
    assert "usage_estimated" not in traces[0].subagents[0].degraded


def test_orphan_child_is_partial_root(tmp_path: Path) -> None:
    missing = tmp_path / "no-such-parent.jsonl"
    child_id = "dddddddd-0000-0000-0000-000000000006"
    _write_pi(
        tmp_path / f"2026-08-01T00-01-00-000Z_{child_id}.jsonl",
        [_header(child_id, parent=missing), _model_change(), _user(), _assistant()],
    )
    traces = list(iter_traces(tmp_path))
    assert len(traces) == 1
    assert traces[0].trace_id == child_id
    assert traces[0].provenance.thread_linkage == "partial"
    assert "thread_linkage" in traces[0].root.degraded
    assert any("parent not in input" in n for n in traces[0].provenance.notes)


def test_estimated_stub_skips_jev_and_detectors(tmp_path: Path) -> None:
    parent_id = "aaaaaaaa-0000-0000-0000-000000000007"
    path = _write_pi(
        tmp_path / f"2026-08-01T00-00-00-000Z_{parent_id}.jsonl",
        [_header(parent_id), _model_change(), _user(), _assistant(), _notification("cafe0001-aaaa", 1000)],
    )
    trace = load_session_file(path)
    stub = trace.subagents[0]
    tasks, _episodes, features, _caps = annotate_trace(trace, load_config(), mode="off")
    assert stub.session_id not in {t.session_id for t in tasks}
    assert features == []
    findings = run_detectors(trace, load_config())
    stub_findings = [f for f in findings if f.session_id == stub.session_id]
    assert stub_findings == []
