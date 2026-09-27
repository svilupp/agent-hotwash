"""Invoice overview uses response charges rather than trace-level rollups."""

from agent_hotwash.analytics import analyze
from agent_hotwash.config import Config, PriceEntry, load_config
from agent_hotwash.diagnostics.cost_views import CostView, CostViews, Money, ResponseCharge
from agent_hotwash.events import PricingStatus, Usage
from agent_hotwash.report.model import RunResult
from agent_hotwash.report.overview import SpendOverviewBuilder, build_spend_overview


def _run(tf, trace_id, charges):
    analysis = analyze(tf.trace(tf.session([tf.user("work")]), trace_id=trace_id), load_config())
    return RunResult(
        analysis=analysis,
        cost_views=CostViews(
            invoice=Money(amount=999, view=CostView.invoice, pricing_status=PricingStatus.exact),
            per_response=charges,
        ),
    )


def _charge(thread, response, amount, model, task, *, status=PricingStatus.exact, cache_write=0):
    return ResponseCharge(
        thread_id=thread,
        response_id=response,
        model=model,
        root_task_id=task,
        invoice=Money(amount=amount, view=CostView.invoice, pricing_status=status),
        usage=Usage(cache_write=cache_write),
    )


def test_dedup_actual_model_and_task_mix(tf):
    one = _charge("thread", "r1", 3, "model-a", "task-1", cache_write=1_000_000)
    two = _charge("thread", "r2", 7, "model-b", "task-1")
    three = _charge("other", "r3", 2, "model-a", "task-2")
    runs = [_run(tf, "trace-1", [one, two]), _run(tf, "trace-2", [one, three])]
    pricing = Config(
        pricing={"model-a": PriceEntry(input=1, output=1, cache_read=1, cache_write=2, as_of="2026-01-01")}
    )
    result = build_spend_overview(runs, pricing=pricing)
    assert result.invoice.amount == 12
    assert result.calls == 3
    assert [(row.name, row.invoice.amount) for row in result.by_model] == [("model-b", 7), ("model-a", 5)]
    assert result.top_tasks[0].task_id == "task-1"
    assert result.top_tasks[0].invoice.amount == 10
    assert {row.name for row in result.top_tasks[0].by_model} == {"model-a", "model-b"}
    assert result.cache_write_tokens == 1_000_000
    assert result.cache_write_charge is not None
    assert result.cache_write_charge.amount == 2


def test_unknown_prices_and_anonymous_limit(tf):
    anonymous = _charge("thread", None, 1, "m", "task")
    unknown = _charge("thread", "r2", None, "m", "task", status=PricingStatus.unknown)
    result = build_spend_overview([_run(tf, "a", [anonymous, unknown]), _run(tf, "b", [anonymous])])
    assert result.calls == 3
    assert result.anonymous_calls == 2
    assert result.invoice.amount == 2
    assert result.invoice.pricing_status is PricingStatus.unknown
    assert result.unknown_calls == 1
    assert result.cache_write_charge is None


def test_streamed_dict_projection_redacts_and_bounds_label(tf):
    run = _run(tf, "trace", [_charge("thread", "r", 4, "m", "task")])
    projection = run.model_dump(mode="json")
    projection["structure"] = {
        "tasks": [
            {"task_id": "task", "turns": [{"user_input": {"text": "Email me at jan@example.com. " + "long " * 40}}]}
        ]
    }
    builder = SpendOverviewBuilder(top_tasks=1)
    missing_text = _run(tf, "earlier", [_charge("other", "previous", 1, "m", "task")])
    builder.add_run(missing_text)
    builder.add_run(projection)
    result = builder.build()
    assert result.invoice.amount == 5
    assert result.top_tasks[0].trace_ids == ["earlier", "trace"]
    assert "jan@example.com" not in result.top_tasks[0].label
    assert "[email]" in result.top_tasks[0].label
    assert len(result.top_tasks[0].label) <= 100
