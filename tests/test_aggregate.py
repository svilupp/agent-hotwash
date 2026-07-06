"""Tests for cross-run aggregation (``aggregate``)."""

from __future__ import annotations

import pytest

from agent_hotwash.aggregate import _percentile, aggregate
from agent_hotwash.analytics import analyze
from agent_hotwash.config import load_config
from agent_hotwash.events import AgentKind


@pytest.fixture
def config():
    return load_config()


def _passing(tf, **kw):
    evs = [
        tf.user("run tests"),
        tf.tool("Bash", call_id="t1", args={"command": "pytest"}),
        tf.result(call_id="t1", ok=True),
    ]
    return analyze(tf.trace(tf.session(evs), **kw), load_config())


def _failing(tf, **kw):
    evs = [
        tf.user("run tests"),
        tf.tool("Bash", call_id="t1", args={"command": "pytest"}),
        tf.result(call_id="t1", ok=False, exit_code=1, output="1 failed"),
    ]
    return analyze(tf.trace(tf.session(evs), **kw), load_config())


def test_percentile_helper():
    assert _percentile([], 50) is None
    assert _percentile([5.0], 95) == 5.0
    assert _percentile([1.0, 2.0, 3.0, 4.0], 50) == pytest.approx(2.5)


def test_empty_aggregate():
    agg = aggregate([])
    assert agg.total_traces == 0
    assert agg.overall.success_rate is None


def test_success_rate_and_outcomes(tf):
    analyses = [_passing(tf), _passing(tf), _failing(tf)]
    agg = aggregate(analyses)
    assert agg.total_traces == 3
    assert agg.overall.success_rate == pytest.approx(2 / 3)
    assert agg.overall.outcome_histogram["positive"] == 2
    assert agg.overall.outcome_histogram["negative"] == 1


def test_grouping_by_agent_and_model(tf):
    a1 = _passing(tf, agent=AgentKind.claude, model="claude-opus-4-8")
    a2 = _failing(tf, agent=AgentKind.codex, model="gpt-5")
    agg = aggregate([a1, a2])
    assert set(agg.by_agent) == {"claude", "codex"}
    assert set(agg.by_model) == {"claude-opus-4-8", "gpt-5"}
    assert agg.by_agent["claude"].success_rate == 1.0
    assert agg.by_agent["codex"].success_rate == 0.0


def test_ground_truth_agreement(tf):
    # proxy positive + truth True -> agree; proxy negative + truth True -> disagree
    a1 = _passing(tf, resolved=True, harness_meta={"resolved": True})
    a2 = _failing(tf, resolved=True, harness_meta={"resolved": True})
    agg = aggregate([a1, a2])
    g = agg.overall
    assert g.ground_truth_success_rate == 1.0  # both resolved True
    assert g.proxy_truth_agreement == pytest.approx(0.5)


def test_tool_error_leaderboard_and_failures(tf):
    agg = aggregate([_failing(tf), _failing(tf)])
    g = agg.overall
    assert g.tool_error_leaderboard[0][0] == "Bash"
    assert g.tool_error_leaderboard[0][1] == 2
    assert sum(g.failure_histogram.values()) == 2


def test_most_thrashed_files(tf):
    evs = [
        tf.tool("Edit", call_id="e1", args={"file_path": "hot.py", "new_string": "x"}),
        tf.result(call_id="e1", ok=True),
        tf.tool("Edit", call_id="e2", args={"file_path": "hot.py", "new_string": "y"}),
        tf.result(call_id="e2", ok=True),
        tf.tool("Edit", call_id="e3", args={"file_path": "cold.py", "new_string": "z"}),
        tf.result(call_id="e3", ok=True),
    ]
    a = analyze(tf.trace(tf.session(evs)), load_config())
    agg = aggregate([a])
    assert agg.overall.most_thrashed_files[0] == ("hot.py", 2)


def test_cost_per_task(tf):
    a1 = analyze(
        tf.trace(tf.session([tf.with_usage(tf.assistant("a"), tf.usage(input=1_000_000))])),
        load_config(),
    )
    a2 = analyze(
        tf.trace(tf.session([tf.with_usage(tf.assistant("a"), tf.usage(input=3_000_000))])),
        load_config(),
    )
    agg = aggregate([a1, a2])
    assert agg.overall.total_cost is not None
    assert agg.overall.cost_per_task == pytest.approx(agg.overall.total_cost / 2)


def test_skipped_counts_empty_traces(tf):
    empty = analyze(tf.trace(tf.session([])), load_config())
    agg = aggregate([empty, _passing(tf)])
    assert agg.skipped == 1
    assert agg.total_traces == 2


def test_percentile_trace_length(tf):
    analyses = [_passing(tf), _failing(tf)]
    agg = aggregate(analyses)
    assert agg.overall.p50_trace_length == pytest.approx(3.0)  # both have 3 events


def test_duration_percentiles_present_with_timestamps(tf):
    evs = [
        tf.user("go", at=0),
        tf.tool("Bash", call_id="c1", args={"command": "ls"}, at=5),
        tf.result(call_id="c1", ok=True, at=10),
    ]
    a = analyze(tf.trace(tf.session(evs)), load_config())
    agg = aggregate([a])
    assert agg.overall.p50_duration_seconds == pytest.approx(10.0)
