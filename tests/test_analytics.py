"""Tests for L1 analytics (``analyze``)."""

from __future__ import annotations

import pytest

from agent_hotwash.analytics import analyze
from agent_hotwash.config import load_config
from agent_hotwash.events import ToolCategory


@pytest.fixture
def config():
    return load_config()


def test_empty_trace(tf, config):
    tr = tf.trace(tf.session([]))
    a = analyze(tr, config)
    assert a.root.event_count == 0
    assert a.root.tool_calls_total == 0
    assert a.root.tool_error_rate is None
    assert a.root.events_to_first_tool_call is None
    assert a.outcome.label == "unknown"
    assert "tokens" in a.degraded


def test_turn_and_tool_counts(tf, config):
    evs = [
        tf.user("please fix the bug", at=0),
        tf.assistant("on it", at=1),
        tf.thinking("hmm let me look", at=2),
        tf.tool("Read", call_id="c1", args={"file_path": "a.py"}, at=3),
        tf.result(call_id="c1", ok=True, at=4),
        tf.tool("Edit", call_id="c2", args={"file_path": "a.py", "new_string": "x\ny"}, at=5),
        tf.result(call_id="c2", ok=True, at=6),
        tf.tool("Bash", call_id="c3", args={"command": "ls"}, at=7),
        tf.result(call_id="c3", ok=True, at=8),
    ]
    a = analyze(tf.trace(tf.session(evs)), config)
    m = a.root
    assert m.user_turns == 1
    assert m.assistant_turns == 1
    assert m.thinking_events == 1
    assert m.thinking_chars > 0
    assert m.tool_calls_total == 3
    assert m.read_count == 1
    assert m.edit_count == 1
    assert m.bash_count == 1
    assert m.tools_by_category[ToolCategory.read.value] == 1
    assert m.events_to_first_tool_call == 3
    assert m.read_before_first_edit == 1
    assert m.unique_files_touched == 1
    assert m.lines_added == 2  # "x\ny" -> 2 lines


def test_edit_write_ratio(tf, config):
    evs = [
        tf.tool("Write", call_id="w1", args={"file_path": "a.py", "content": "x"}),
        tf.result(call_id="w1", ok=True),
        tf.tool("Edit", call_id="e1", args={"file_path": "a.py", "new_string": "y"}),
        tf.result(call_id="e1", ok=True),
        tf.tool("Edit", call_id="e2", args={"file_path": "a.py", "new_string": "z"}),
        tf.result(call_id="e2", ok=True),
    ]
    m = analyze(tf.trace(tf.session(evs)), config).root
    assert m.write_count == 1
    assert m.edit_count == 2
    assert m.edit_write_ratio == 2.0


def test_timestamps_absent_yields_none_and_degraded(tf, config):
    evs = [
        tf.user("do it"),
        tf.tool("Bash", call_id="c1", args={"command": "ls"}),
        tf.result(call_id="c1", ok=True),
    ]
    s = tf.session(evs)
    assert s.has_timestamps is False
    a = analyze(tf.trace(s), config)
    assert a.root.duration_seconds is None
    assert a.root.active_seconds is None
    assert "duration_seconds" in a.degraded


def test_timestamps_present_active_vs_idle(tf, config):
    # gap of 10 minutes between event 1 and 2 counts as idle (default gap 5 min).
    evs = [
        tf.user("go", at=0),
        tf.tool("Bash", call_id="c1", args={"command": "ls"}, at=10),
        tf.result(call_id="c1", ok=True, at=600 + 10),  # 600s gap -> idle
    ]
    m = analyze(tf.trace(tf.session(evs)), config).root
    assert m.duration_seconds == 610.0
    assert m.idle_seconds == 600.0
    assert m.active_seconds == 10.0


def test_error_metrics_and_streak(tf, config):
    evs = [
        tf.tool("Bash", call_id="c1", args={"command": "pytest"}),
        tf.result(call_id="c1", ok=False, exit_code=1, output="1 failed"),
        tf.tool("Bash", call_id="c2", args={"command": "pytest"}),
        tf.result(call_id="c2", ok=False, exit_code=1, output="1 failed"),
        tf.tool("Bash", call_id="c3", args={"command": "pytest"}),
        tf.result(call_id="c3", ok=True),
    ]
    m = analyze(tf.trace(tf.session(evs)), config).root
    assert m.tool_error_count == 2
    assert m.tool_results_total == 3
    assert m.tool_error_rate == pytest.approx(2 / 3)
    assert m.max_error_streak == 2
    assert m.errors_by_tool.get("Bash") == 2
    assert m.retry_after_error == 2  # both errors have a later tool_call
    assert m.test_run_count == 3
    assert m.test_pass_count == 1
    assert m.test_fail_count == 2
    assert m.test_pass_fail_transitions == 1  # fail,fail,pass -> one transition


def test_edit_test_cycles(tf, config):
    evs = [
        tf.tool("Edit", call_id="e1", args={"file_path": "a.py", "new_string": "x"}),
        tf.result(call_id="e1", ok=True),
        tf.tool("Bash", call_id="t1", args={"command": "pytest"}),
        tf.result(call_id="t1", ok=True),
        tf.tool("Edit", call_id="e2", args={"file_path": "a.py", "new_string": "y"}),
        tf.result(call_id="e2", ok=True),
        tf.tool("Bash", call_id="t2", args={"command": "pytest"}),
        tf.result(call_id="t2", ok=True),
    ]
    m = analyze(tf.trace(tf.session(evs)), config).root
    assert m.edit_test_cycles == 2
    assert m.distinct_bash_commands == 1  # both "pytest"


def test_tokens_and_cache_hit_ratio(tf, config):
    evs = [
        tf.with_usage(
            tf.assistant("a"),
            tf.usage(input=100, output=50, cache_read=300, cache_write=10),
        ),
        tf.with_usage(
            tf.assistant("b"),
            tf.usage(input=100, output=50, cache_read=100, cache_write=0),
        ),
    ]
    m = analyze(tf.trace(tf.session(evs)), config).root
    assert m.tokens.input == 200
    assert m.tokens.output == 100
    assert m.tokens.cache_read == 400
    # cache_read / (cache_read + input) = 400 / 600
    assert m.cache_hit_ratio == pytest.approx(400 / 600)


def test_cumulative_usage_de_cumulated(tf, config):
    evs = [
        tf.with_usage(tf.assistant("a"), tf.usage(input=100, output=20, cumulative=True)),
        tf.with_usage(tf.assistant("b"), tf.usage(input=250, output=60, cumulative=True)),
    ]
    # build_session de-cumulates: deltas are 100/20 then 150/40 -> totals 250/60
    m = analyze(tf.trace(tf.session(evs)), config).root
    assert m.tokens.input == 250
    assert m.tokens.output == 60


def test_cost_estimated_from_price_table(tf, config):
    evs = [
        tf.with_usage(tf.assistant("a"), tf.usage(input=1_000_000, output=1_000_000)),
    ]
    a = analyze(tf.trace(tf.session(evs), model="claude-opus-4-8"), config)
    # opus-4-8: input 5 + output 25 per MTok = 30
    assert a.cost == pytest.approx(30.0)
    assert a.cost_source == "estimated"
    # With no provenance, cost_estimated mirrors the headline cost.
    assert a.cost_estimated == pytest.approx(30.0)


def test_cost_priced_per_turn_model_when_thread_switches_models(config):
    """Two turns, two models: cost = Σ per-model prices, tokens conserved."""
    from agent_hotwash.config import Config
    from agent_hotwash.events import (
        AgentKind,
        Event,
        EventKind,
        ModelConfig,
        Session,
        Turn,
        TurnStatus,
        Usage,
        UserInput,
    )

    data = config.model_dump()
    data["pricing"] = {
        "model-a": {"input": 1.0, "output": 0.0, "cache_read": 0.0, "cache_write": 0.0},
        "model-b": {"input": 10.0, "output": 0.0, "cache_read": 0.0, "cache_write": 0.0},
    }
    cfg = Config.model_validate(data)
    events = [
        Event(kind=EventKind.user_msg, idx=0, text="first", turn_id="t1"),
        Event(kind=EventKind.meta, idx=1, turn_id="t1", usage=Usage(input=1_000_000, output=0)),
        Event(kind=EventKind.user_msg, idx=2, text="second", turn_id="t2"),
        Event(kind=EventKind.meta, idx=3, turn_id="t2", usage=Usage(input=1_000_000, output=0)),
    ]

    def _turn(tid: str, model: str, start: int, end: int) -> Turn:
        return Turn(
            turn_id=tid,
            session_id="s",
            event_start=start,
            event_end=end,
            status=TurnStatus.completed,
            user_input=UserInput(text="x", kind="user"),
            model_config_active=ModelConfig(model=model),
        )

    session = Session(
        session_id="s",
        agent=AgentKind.unknown,
        events=events,
        turns=[_turn("t1", "model-a", 0, 1), _turn("t2", "model-b", 2, 3)],
        model="model-a",
    )
    trace = tf_trace(session, model="model-a")
    a = analyze(trace, cfg)
    assert a.root.tokens.input == 2_000_000  # conserved
    assert {k: v.input for k, v in a.root.tokens_by_model.items()} == {"model-a": 1_000_000, "model-b": 1_000_000}
    assert a.cost_estimated == pytest.approx(1.0 + 10.0)  # not 2.0 (all at model-a) nor 20.0
    assert a.cost == pytest.approx(11.0)
    assert a.cost_source == "estimated"


def test_cost_uses_session_model_when_no_turns(tf, config):
    evs = [tf.with_usage(tf.assistant("a"), tf.usage(input=1_000_000, output=1_000_000))]
    a = analyze(tf.trace(tf.session(evs), model="claude-opus-4-8"), config)
    assert list(a.root.tokens_by_model) == ["claude-opus-4-8"]
    assert a.cost == pytest.approx(30.0)


def tf_trace(session, *, model: str):
    from pathlib import Path

    from agent_hotwash.events import AgentKind, Provenance, Trace

    return Trace(
        trace_id="t0",
        agent=AgentKind.unknown,
        model=model,
        root=session,
        provenance=Provenance(source_format="codex_native", detector_confidence="high", root_path=Path("/tmp/x")),
    )


def test_cost_estimated_alongside_provenance_for_drift(tf, config):
    # 1M input + 1M output on fable-5 => 10 + 50 = 60 estimated. Provenance
    # reports a different figure; both are exposed so a reader can see drift.
    evs = [tf.with_usage(tf.assistant("a"), tf.usage(input=1_000_000, output=1_000_000))]
    a = analyze(
        tf.trace(tf.session(evs, model="claude-fable-5"), harness_meta={"total_cost_usd": 66.0}),
        config,
    )
    assert a.cost == pytest.approx(66.0)
    assert a.cost_source == "provenance"
    assert a.cost_estimated == pytest.approx(60.0)


def test_cost_from_provenance_wins(tf, config):
    evs = [tf.with_usage(tf.assistant("a"), tf.usage(input=1_000_000, output=1_000_000))]
    a = analyze(
        tf.trace(tf.session(evs), harness_meta={"total_cost_usd": 1.23}),
        config,
    )
    assert a.cost == pytest.approx(1.23)
    assert a.cost_source == "provenance"


def test_cost_from_provenance_metrics_nesting(tf, config):
    # Realistic code-bench harness_meta: cost lives at metrics["cost"]["total"],
    # not a flat key. The provenance lookup must traverse the real nesting.
    evs = [tf.with_usage(tf.assistant("a"), tf.usage(input=1_000_000, output=1_000_000))]
    harness_meta = {
        "run": {"harness": "claude", "run_id": "r1"},
        "metrics": {
            "tokens": {"input": 14183, "output": 30785, "cache_read": 2950586, "cache_write": 77235},
            "cost": {"total": 2.79855175, "computed": 2.79855175, "cost_reported": 2.79126375},
        },
        "verification": {"resolved": True},
    }
    a = analyze(tf.trace(tf.session(evs), source_format="codebench", harness_meta=harness_meta), config)
    assert a.cost == pytest.approx(2.79855175)
    assert a.cost_source == "provenance"


def test_cost_provenance_prefers_parser_provided_over_metrics(tf, config):
    # The claude parser lifts the stream-final total_cost_usd to a flat key; it
    # wins over metrics["cost"] when both are present.
    evs = [tf.with_usage(tf.assistant("a"), tf.usage(input=1_000_000, output=1_000_000))]
    harness_meta = {
        "metrics": {"cost": {"total": 2.5}},
        "total_cost_usd": 2.79126375,
        "verification": {"resolved": False},
    }
    a = analyze(tf.trace(tf.session(evs), source_format="codebench", harness_meta=harness_meta), config)
    assert a.cost == pytest.approx(2.79126375)
    assert a.cost_source == "provenance"


def test_corrections_and_outcome_negative(tf, config):
    evs = [
        tf.user("fix it"),
        tf.tool("Edit", call_id="e1", args={"file_path": "a.py", "new_string": "x"}),
        tf.result(call_id="e1", ok=True),
        tf.user("no, that's wrong"),
    ]
    a = analyze(tf.trace(tf.session(evs)), config)
    assert a.root.corrections_count == 1
    assert a.outcome.label == "negative"


def test_outcome_positive_from_passing_test(tf, config):
    evs = [
        tf.user("run tests"),
        tf.tool("Bash", call_id="t1", args={"command": "pytest"}),
        tf.result(call_id="t1", ok=True),
    ]
    a = analyze(tf.trace(tf.session(evs)), config)
    assert a.outcome.label == "positive"


def test_ground_truth_recorded(tf, config):
    evs = [tf.user("go")]
    a = analyze(
        tf.trace(tf.session(evs), resolved=True, harness_meta={"resolved": True}),
        config,
    )
    assert a.resolved is True
    assert a.outcome.ground_truth_resolved is True


def test_subagent_breakdown(tf, config):
    root = tf.session([tf.tool("Task", call_id="a1", args={}), tf.result(call_id="a1", ok=True)])
    sub = tf.session(
        [tf.tool("Read", call_id="r1", args={"file_path": "a.py"}), tf.result(call_id="r1", ok=True)],
        session_id="sub1",
        parent_session_id="s0",
    )
    a = analyze(tf.trace(root, subagents=[sub]), config)
    assert a.subagent_count == 1
    assert a.subagent_fanout == 1
    assert a.subagents[0].is_subagent is True
    assert a.subagents[0].read_count == 1
    assert a.subagent_event_total == 2
