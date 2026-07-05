"""Tests for the code-bench run-dir parser and its three stdout decoders."""

from __future__ import annotations

from pathlib import Path

from agent_hotwash.events import AgentKind, EventKind
from agent_hotwash.sources.codebench import load_run_dir

FIXTURES = Path(__file__).parent / "fixtures" / "codebench"


def _kinds(events):
    return [e.kind for e in events]


def _results(events):
    return [e for e in events if e.kind is EventKind.tool_result]


def _calls(events):
    return [e for e in events if e.kind is EventKind.tool_call]


# --------------------------------------------------------------------------- codex


def test_codex_run_basic() -> None:
    trace = load_run_dir(FIXTURES / "codex_run")
    assert trace is not None
    assert trace.agent is AgentKind.codex
    assert trace.model == "gpt-5.5"
    assert trace.experiment == "fixture-codex"
    assert trace.instance_id == "demo-1"
    assert trace.resolved is True
    assert trace.provenance.source_format == "codebench"
    assert not trace.root.has_timestamps  # codex code-bench has no timestamps


def test_codex_calls_linked_and_errors_classified() -> None:
    trace = load_run_dir(FIXTURES / "codex_run")
    assert trace is not None
    events = trace.root.events
    calls, results = _calls(events), _results(events)
    assert len(calls) == 3  # 2 command_execution + 1 file_change
    assert len(results) == 3
    # the failing `cat missing.txt` is classified file_not_found.
    errored = [r for r in results if r.ok is False]
    assert len(errored) == 1
    assert errored[0].error_category == "file_not_found"
    assert errored[0].exit_code == 1


def test_codex_usage_decumulated_and_summable() -> None:
    trace = load_run_dir(FIXTURES / "codex_run")
    assert trace is not None
    # single cumulative turn -> de-cumulated to a delta; no negative fields.
    usages = [e.usage for e in trace.root.events if e.usage]
    assert usages
    total_out = sum(u.output for u in usages if u.output)
    assert total_out == 300
    assert all(not u.cumulative for u in usages)


def test_codex_file_change_path_extracted() -> None:
    trace = load_run_dir(FIXTURES / "codex_run")
    assert trace is not None
    fc = next(e for e in trace.root.events if e.tool_name == "file_change" and e.kind is EventKind.tool_call)
    assert fc.path == "src/app.py"


# --------------------------------------------------------------------------- claude


def test_claude_run_basic_and_blocks() -> None:
    trace = load_run_dir(FIXTURES / "claude_run")
    assert trace is not None
    assert trace.agent is AgentKind.claude
    assert trace.model == "claude-opus-4-8"
    assert trace.resolved is False
    # Only user records carry timestamps (sparse coverage), so the STRICT flag is
    # False while some timestamps are present. Timing degrades rather than mislead.
    assert not trace.root.has_timestamps
    assert trace.root.has_any_timestamps
    assert 0.0 < trace.root.ts_coverage < 0.5
    kinds = _kinds(trace.root.events)
    assert EventKind.thinking in kinds
    assert EventKind.assistant_msg in kinds


def test_claude_error_shapes_classified() -> None:
    trace = load_run_dir(FIXTURES / "claude_run")
    assert trace is not None
    results = _results(trace.root.events)
    by_cat = {r.error_category for r in results if r.ok is False}
    # Exit code 127 -> command_not_found; <tool_use_error> read-first -> edit_mismatch.
    assert "command_not_found" in by_cat
    assert "edit_mismatch" in by_cat


def test_claude_result_line_cost_lifted_to_provenance() -> None:
    trace = load_run_dir(FIXTURES / "claude_run")
    assert trace is not None
    # the stream-final `result` total_cost_usd is surfaced for cost provenance.
    assert trace.provenance.harness_meta.get("total_cost_usd") == 0.02


def test_claude_result_usage_fallback_when_assistant_usage_absent() -> None:
    # assistant records carry no usage -> decode the result line's usage instead
    # of falling back to metrics.json (which would flag usage_reliable=False).
    trace = load_run_dir(FIXTURES / "claude_run_no_usage")
    assert trace is not None
    assert trace.root.usage_reliable is True
    assert not any("metrics.json" in n for n in trace.provenance.notes)
    total = 0
    for e in trace.root.events:
        if e.usage:
            total += sum(x for x in (e.usage.input, e.usage.output, e.usage.cache_read, e.usage.cache_write) if x)
    assert total == 360  # 200 + 40 + 100 + 20 from the result line, not metrics (999*4)
    assert trace.provenance.harness_meta.get("total_cost_usd") == 3.33


def test_claude_subagents_linked_not_inlined() -> None:
    trace = load_run_dir(FIXTURES / "claude_run")
    assert trace is not None
    assert len(trace.subagents) == 1
    sub = trace.subagents[0]
    assert sub.parent_session_id == trace.root.session_id
    # subagent events are NOT in the root stream.
    assert all(e.parent_span_id is None for e in trace.root.events if e.kind is EventKind.tool_call)
    sub_kinds = _kinds(sub.events)
    assert EventKind.tool_call in sub_kinds


def test_claude_multiblock_usage_counted_once_and_output_from_result() -> None:
    # A single assistant response (msg_A) is emitted as three jsonl lines that
    # repeat the same message.id and the same usage. Regression guards:
    #  (1) input / cache_read / cache_write are counted ONCE per message.id
    #      (not once per block -> no 3x overcount), and
    #  (2) output comes from the run-final `result` line (500), not the summed
    #      per-message streaming placeholders (3 x 5 = 15).
    trace = load_run_dir(FIXTURES / "claude_run_multiblock")
    assert trace is not None
    usages = [e.usage for e in trace.root.events if e.usage]
    total_in = sum(u.input for u in usages if u.input)
    total_cr = sum(u.cache_read for u in usages if u.cache_read)
    total_cw = sum(u.cache_write for u in usages if u.cache_write)
    total_out = sum(u.output for u in usages if u.output)
    assert total_in == 1000
    assert total_cr == 2000
    assert total_cw == 100
    assert total_out == 500  # authoritative result-line output, not 15
    assert trace.root.usage_reliable is True


def test_claude_multiblock_cost_matches_provenance() -> None:
    # With usage billed once, the token x price estimate reproduces the harness
    # cost (opus-4-8 rates) closely instead of the ~2.3x cache overcount.
    from agent_hotwash.analytics import analyze
    from agent_hotwash.config import load_config

    trace = load_run_dir(FIXTURES / "claude_run_multiblock")
    assert trace is not None
    analysis = analyze(trace, load_config())
    assert analysis.cost is not None and analysis.cost_estimated is not None
    drift = abs(analysis.cost - analysis.cost_estimated) / analysis.cost
    assert drift < 0.2


# --------------------------------------------------------------------------- pi


def test_pi_zero_usage_backfills_from_metrics() -> None:
    trace = load_run_dir(FIXTURES / "pi_run_zerousage")
    assert trace is not None
    assert trace.agent is AgentKind.pi
    # stream usage is all zero -> fell back to metrics.json, flagged unreliable.
    assert trace.root.usage_reliable is False
    assert any("metrics.json" in n for n in trace.provenance.notes)
    total = 0
    for e in trace.root.events:
        if e.usage:
            total += sum(x for x in (e.usage.input, e.usage.output, e.usage.cache_read) if x)
    assert total > 0  # backfilled


def test_pi_toplevel_iserror_and_ignores_updates() -> None:
    trace = load_run_dir(FIXTURES / "pi_run_zerousage")
    assert trace is not None
    results = _results(trace.root.events)
    assert len(results) == 2  # *_update deltas ignored
    failed = [r for r in results if r.ok is False]
    assert len(failed) == 1
    assert failed[0].error_category == "build_test_fail"
