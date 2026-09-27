"""Costs reconcile; unknown usage, context changes and mixed operations stay explicit."""

import pytest

from agent_hotwash.config import DiagnosticsConfig, load_config
from agent_hotwash.diagnostics.handover_cache import CacheWait, build_cache_waits, cache_wait_cohorts
from agent_hotwash.events import Event, EventKind, SourceRef
from agent_hotwash.primitives.coordination import coordination_kind, is_coordination
from agent_hotwash.primitives.handovers import build_handovers
from agent_hotwash.report.html import render_html
from agent_hotwash.report.model import Report, ReportMeta
from agent_hotwash.runner import run_trace


def round_events(tf, record, calls, *, read=100, write=10, model="claude-opus-4-8", incomplete=False):
    usage = tf.with_usage(
        tf.assistant(at=record * 100),
        tf.usage(input=2, output=None if incomplete else 3, cache_read=read, cache_write=write),
    )
    usage.source = SourceRef(record_index=record)
    usage.usage_model = model
    for n, call in enumerate(calls):
        call.source = SourceRef(record_index=record, ordinal=n + 1)
    return [usage, *calls]


def test_coordination_subtypes_reconcile_and_preserve_mixed_costs(tf):
    events = [tf.user("Implement")]
    specs = [
        [("Agent", {})],
        [("steer_subagent", {})],
        [("get_subagent_result", {"wait": False})],
        [("get_subagent_result", {"wait": True})],
        [("Agent", {}), ("steer_subagent", {})],
        [("Agent", {"resume": "child"})],
    ]
    for i, spec in enumerate(specs):
        calls = [tf.tool(n, call_id=f"{i}-{j}", args=a) for j, (n, a) in enumerate(spec)]
        events += round_events(tf, i + 1, calls)
    # Unknown usage must not prevent pricing of complete responses in the same bucket.
    events += round_events(tf, 9, [tf.tool("Agent", call_id="unknown")], incomplete=True)
    trace = tf.trace(tf.session(events))
    result = run_trace(trace, load_config())
    rows = [r for r in result.tails.model_activity if r.category == "coordination"]
    assert {r.coordination_kind for r in rows} == {"spawn", "steer", "poll", "wait", "mixed", "resume"}
    assert sum(r.rounds for r in rows) == 7
    assert sum(r.tool_calls for r in rows) == 8
    assert sum(r.rounds for r in rows if r.estimated_cost is not None) == 6
    price, _ = load_config().price_lookup("claude-opus-4-8")
    assert price
    per = (2 * price.input + 3 * price.output + 100 * price.cache_read + 10 * price.cache_write) / 1e6
    assert sum(r.estimated_cost or 0 for r in rows) == pytest.approx(6 * per)
    html = render_html(Report.build([result], ReportMeta(tool_version="test")))
    assert "Coordination charges by operation" in html and "mixed" in html
    assert not is_coordination(tf.tool("mcp__slack__send_message", call_id="slack"))
    assert coordination_kind(tf.tool("mcp__codex_app__create_thread", call_id="app")) == "spawn"


def make_wait_trace(tf, seconds=300):
    events = [
        tf.user("Work"),
        tf.tool("Agent", call_id="spawn", args={"prompt": "Work"}),
        tf.result(call_id="spawn", output="Agent started in background"),
    ]
    events[-1].tool_args = {"agentId": "child"}
    wait = tf.tool("get_subagent_result", call_id="wait", args={"agent_id": "child", "wait": True}, at=100)
    events += round_events(tf, 1, [wait], read=40000, write=0)
    events += [tf.result(call_id="wait", output="Status: completed\nDone", at=100 + seconds)]
    events += round_events(tf, 5, [], read=0, write=20000)
    return tf.trace(tf.session(events))


def test_wait_prices_bands_and_incomplete_inputs(tf):
    cfg = load_config()
    for seconds, expected in [(299, "0 <= wait < 300s"), (300, "300 <= wait < 3600s"), (3600, "wait >= 3600s")]:
        trace = make_wait_trace(tf, seconds)
        row = build_cache_waits(trace, build_handovers(trace), min_write_tokens=10000, config=cfg)[0]
        price, _ = cfg.price_lookup(row.next_model)
        assert price
        assert row.duration_band == expected and row.wait_mode == "blocking"
        assert row.next_input_cost == pytest.approx((2 * price.input + 20000 * price.cache_write) / 1e6)
        assert row.rewrite_write_cost == pytest.approx(20000 * price.cache_write / 1e6)
        assert sum(row.next_input_cost_components.values()) == pytest.approx(row.next_input_cost)
        trace.root.events[-1].usage.input = None
        unknown = build_cache_waits(trace, build_handovers(trace), min_write_tokens=10000, config=cfg)[0]
        assert unknown.next_input_cost is None


def test_wait_context_edits_block_comparison_but_preserve_observed_cost(tf):
    cfg = load_config()
    cfg = cfg.model_copy(update={"diagnostics": DiagnosticsConfig(cache_wait_bands_seconds=(60, 600))})
    trace = make_wait_trace(tf)
    events = trace.root.events
    events.insert(-1, Event(kind=EventKind.meta, raw_type="context_edit"))
    trace = tf.trace(tf.session(events))
    result = run_trace(trace, cfg)
    row = result.analysis.cache_waits[0]
    assert row.context_edit_between and not row.comparable
    assert row.cache_rewrite_after_wait is None and row.rewrite_write_cost is None
    assert row.next_input_cost is not None and row.duration_band == "60 <= wait < 600s"
    report = Report.build([result], ReportMeta(tool_version="test"))
    assert report.cache_wait_cohorts[0].comparable == 0
    html = render_html(report)
    assert "Cache waits by duration and model" in html and "60 &lt;= wait &lt; 600s" in html


def test_cohort_deduplicates_multi_child_wait_and_shared_next_response():
    row = CacheWait(
        handover_id="a",
        parent_id="parent",
        wait_event_idx=1,
        next_event_start=10,
        next_model="m",
        comparable=True,
        cache_rewrite_after_wait=True,
        next_input_cost=3,
        rewrite_write_cost=2,
        wait_mode="blocking",
        duration_band="0 <= wait < 300s",
    )
    other_child = row.model_copy(update={"handover_id": "b"})
    second_wait = row.model_copy(update={"wait_event_idx": 3})
    c = cache_wait_cohorts([("trace", row), ("trace", other_child), ("trace", second_wait)])[0]
    assert (c.waits, c.comparable, c.rewrites, c.priced_responses) == (2, 2, 2, 1)
    assert c.next_input_cost == 3 and c.rewrite_write_cost == 2
    # A different trace with coincident coordinates is a different charge.
    assert cache_wait_cohorts([("a", row), ("b", row)])[0].next_input_cost == 6


def test_unknown_price_preserves_observation_and_cumulative_usage_blocks_it(tf):
    trace = make_wait_trace(tf)
    cfg = load_config().model_copy(update={"pricing": {}})
    for event in trace.root.events:
        if event.usage:
            event.usage_model = "unpriced-test-model"
    result = run_trace(trace, cfg)
    wait = result.analysis.cache_waits[0]
    assert wait.comparable and wait.cache_rewrite_after_wait
    assert wait.next_input_cost is None and wait.next_pricing_status == "unknown"
    rebuild = next(row for row in result.tails.incidents if row.kind == "cache_rebuilds")
    assert rebuild.evidence["observed_cache_write_cost"] is None
    assert rebuild.evidence["priced_transitions"] == 0
    trace.root.events[-1].usage.cumulative = True
    result = run_trace(trace, cfg)
    assert not result.analysis.cache_waits[0].comparable
    assert not any(row.kind == "cache_rebuilds" for row in result.tails.incidents)


@pytest.mark.parametrize("write, matches", [(9999, 0), (10000, 1)])
def test_cache_rebuild_minimum_write_boundary(tf, write, matches):
    events = [tf.user("Work")]
    events += round_events(tf, 1, [], read=40000, write=0)
    events += round_events(tf, 2, [], read=0, write=write)
    rows = run_trace(tf.trace(tf.session(events)), load_config()).tails.incidents
    rebuilds = [row for row in rows if row.kind == "cache_rebuilds"]
    assert len(rebuilds) == matches
    assert all(not row.exceeds_threshold for row in rebuilds)


def test_zero_minimum_cannot_turn_zero_cache_write_into_rewrite(tf):
    trace = make_wait_trace(tf)
    trace.root.events[-1].usage.cache_write = 0
    wait = build_cache_waits(trace, build_handovers(trace), min_write_tokens=0)[0]
    assert wait.comparable and wait.cache_rewrite_after_wait is False


@pytest.mark.parametrize("bands", [(0,), (-1,), (300, 300), (3600, 300), (float("nan"),), (float("inf"),)])
def test_bad_wait_bands_rejected(bands):
    with pytest.raises(ValueError):
        DiagnosticsConfig(cache_wait_bands_seconds=bands)


def test_cache_rebuild_requires_warm_predecessor_and_preserves_boundaries(tf):
    def make(*, boundary=False, compact=False, context_edit=False, missing=False, switched=False):
        events = [tf.user("Implement")]
        for i, read in enumerate((40000, 0, 45000, 0)):
            if i == 1 and boundary:
                events.append(tf.user("new request"))
            if i == 1 and compact:
                events.append(tf.compaction())
            if i == 1 and context_edit:
                events.append(Event(kind=EventKind.meta, raw_type="context_edit"))
            events += round_events(
                tf,
                i + 1,
                [tf.tool("Agent", call_id=str(i))],
                read=read,
                write=20000 if read == 0 else 0,
                model="different-model" if i == 1 and switched else "claude-opus-4-8",
            )
            if i == 0 and missing:
                events[-2].usage = None
        return tf.trace(tf.session(events))

    result = run_trace(make(), load_config())
    row = next(r for r in result.tails.incidents if r.kind == "cache_rebuilds")
    assert row.value == 2 and row.evidence["cache_write_tokens"] == 40000 and row.exceeds_threshold
    assert row.evidence["priced_transitions"] == 2 and row.evidence["observed_cache_write_cost"] > 0
    assert any(f.id == "TAIL_CACHE_REBUILDS" for f in result.findings)
    for args in ({"boundary": True}, {"compact": True}, {"context_edit": True}, {"missing": True}, {"switched": True}):
        rows = [r for r in run_trace(make(**args), load_config()).tails.incidents if r.kind == "cache_rebuilds"]
        assert sum(r.value for r in rows) == 1
    trace = make()
    trace.root.usage_reliable = False
    assert not any(r.kind == "cache_rebuilds" for r in run_trace(trace, load_config()).tails.incidents)


def test_pi_response_model_is_recorded_for_pricing():
    from agent_hotwash.sources.pi_native import _message

    events = _message(
        {
            "role": "assistant",
            "model": "claude-sonnet-4-6",
            "usage": {"input": 1, "output": 2, "cacheRead": 3, "cacheWrite": 4},
            "content": [{"type": "text", "text": "Done"}],
        }
    )
    assert events[0].usage_model == "claude-sonnet-4-6"


def test_pi_preserves_context_boundaries_for_cache_checks():
    from agent_hotwash.sources.pi_native import decode_pi_native

    events, _, _ = decode_pi_native(
        [
            {"type": "compaction", "summary": "Earlier context", "timestamp": "2026-09-01T12:00:00Z"},
            {"type": "context_edit", "targetId": "old", "replacement": None},
        ]
    )
    assert events[0].kind == EventKind.compaction and events[0].text == "Earlier context"
    assert events[0].source and events[1].source
    assert events[0].source.record_index == 0
    assert events[1].raw_type == "context_edit" and events[1].source.record_index == 1


def test_pi_empty_failed_response_cannot_be_skipped_for_cache_waits(tf):
    from agent_hotwash.sources.pi_native import _message

    empty = _message(
        {
            "role": "assistant",
            "model": "claude-opus-4-8",
            "content": [],
            "stopReason": "error",
            "usage": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
        }
    )
    assert len(empty) == 1 and empty[0].usage is None
    empty[0].source = SourceRef(record_index=4)
    trace = make_wait_trace(tf)
    events = trace.root.events
    events[-1:-1] = empty
    result = run_trace(tf.trace(tf.session(events)), load_config())
    wait = result.analysis.cache_waits[0]
    assert not wait.comparable and wait.next_input_cost is None
    assert not any(row.kind == "cache_rebuilds" for row in result.tails.incidents)
