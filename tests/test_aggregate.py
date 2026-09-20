"""Tests for cross-run aggregation (``aggregate``)."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from agent_hotwash.aggregate import _percentile, aggregate, monthly_rollup
from agent_hotwash.analytics import analyze
from agent_hotwash.config import load_config
from agent_hotwash.diagnostics.cost_views import CostView, CostViews, Diagnosis, Money, ResponseCharge
from agent_hotwash.events import AgentKind, PricingStatus, Usage
from agent_hotwash.report.model import RunResult


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


def _money(amount: float, view: CostView = CostView.invoice) -> Money:
    return Money(amount=amount, view=view, pricing_status=PricingStatus.exact)


def _charge(
    thread: str, rid: str, amount: float, ts: datetime, *, root: str, model: str = "m", effort: str = "high"
) -> ResponseCharge:
    return ResponseCharge(
        thread_id=thread,
        response_id=rid,
        ts_start=ts,
        model=model,
        effort=effort,
        invoice=_money(amount),
        root_task_id=root,
        usage=Usage(input=0),
    )


def _run_with_views(tf, charges, diagnoses, trace_id="t"):
    analysis = analyze(tf.trace(tf.session([tf.user("x")]), trace_id=trace_id), load_config())
    return RunResult(
        analysis=analysis,
        cost_views=CostViews(
            invoice=_money(sum(c.invoice.amount or 0 for c in charges)), per_response=charges, diagnoses=diagnoses
        ),
    )


def test_monthly_rollup_dedup_duplicated_child(tf):
    ts = datetime(2026, 3, 15, tzinfo=UTC)
    child_charge = _charge("child", "resp-1", 10.0, ts, root="root-task")
    parent_charge = _charge("root", "resp-root", 1.0, ts, root="root-task")
    dup_child = _charge("child", "resp-1", 10.0, ts, root="root-task")
    run_a = _run_with_views(tf, [parent_charge, child_charge], [], trace_id="a")
    run_b = _run_with_views(tf, [dup_child], [], trace_id="b")
    roll = monthly_rollup([run_a, run_b], timezone="UTC")
    assert len(roll.cells) == 1
    assert roll.cells[0].invoice_total == pytest.approx(11.0)
    assert roll.cells[0].n_tasks == 1


def test_monthly_rollup_month_boundary(tf):
    a = _charge("th", "r1", 4.0, datetime(2026, 1, 31, 23, 0, tzinfo=UTC), root="task-a")
    b = _charge("th", "r2", 5.0, datetime(2026, 2, 1, 0, 30, tzinfo=UTC), root="task-b")
    run = _run_with_views(tf, [a, b], [])
    roll = monthly_rollup([run], timezone="UTC")
    months = {c.month: c.invoice_total for c in roll.cells}
    assert months["2026-01"] == pytest.approx(4.0)
    assert months["2026-02"] == pytest.approx(5.0)


def test_monthly_rollup_three_measure_ranking(tf):
    ts = datetime(2026, 4, 1, tzinfo=UTC)
    charges = [
        _charge("t1", "a1", 1.0, ts, root="task-a"),
        _charge("t2", "b1", 1.0, ts, root="task-b"),
        _charge("t3", "c1", 1.0, ts, root="task-c"),
        _charge("t4", "d1", 100.0, ts, root="task-d"),
        _charge("t5", "e1", 10.0, ts, root="task-e"),
    ]
    diagnoses = [
        Diagnosis(
            id="DUP", view=CostView.invoice, amount=_money(1.0), pricing_status=PricingStatus.exact, spans=["task-a"]
        ),
        Diagnosis(
            id="DUP", view=CostView.invoice, amount=_money(1.0), pricing_status=PricingStatus.exact, spans=["task-b"]
        ),
        Diagnosis(
            id="DUP", view=CostView.invoice, amount=_money(1.0), pricing_status=PricingStatus.exact, spans=["task-c"]
        ),
        Diagnosis(
            id="BIG", view=CostView.invoice, amount=_money(100.0), pricing_status=PricingStatus.exact, spans=["task-d"]
        ),
        Diagnosis(
            id="CF",
            view=CostView.counterfactual,
            amount=Money(
                amount=70.0,
                view=CostView.counterfactual,
                pricing_status=PricingStatus.exact,
                amount_low=50.0,
                amount_high=90.0,
            ),
            pricing_status=PricingStatus.exact,
            spans=["task-e"],
        ),
    ]
    run = _run_with_views(tf, charges, diagnoses)
    roll = monthly_rollup([run], timezone="UTC")
    assert roll.ranked_by_task_count[0] == "DUP"
    assert roll.ranked_by_invoice[0] == "BIG"
    assert roll.ranked_by_counterfactual[0] == "CF"
    assert "ranked by" in (roll.overspend_statement or "")
