"""Extremes must survive small cohorts without inventing hangs or savings."""

from agent_hotwash.config import load_config
from agent_hotwash.diagnostics.tails import build_tails
from agent_hotwash.events import EventKind
from agent_hotwash.primitives.handovers import HandoverEvent, HandoverRecord
from agent_hotwash.report.csv_writer import render_csv
from agent_hotwash.report.html import render_html
from agent_hotwash.report.json_writer import report_to_dict
from agent_hotwash.report.model import Report, ReportMeta
from agent_hotwash.report.table import render_table
from agent_hotwash.runner import run_trace
from agent_hotwash.semantic.bank import load_bank
from agent_hotwash.semantic.tails import tail_state


def rows(trace, kind, handovers=None):
    return [r for r in build_tails(trace, handovers or []).incidents if r.kind == kind]


def test_latency_cohort_retains_top_and_ratio(tf):
    events = []
    for i in range(12):
        events.extend(
            [
                tf.tool("Bash", call_id=str(i), at=i * 1000),
                tf.result(call_id=str(i), at=i * 1000 + (300 if i == 11 else 1)),
            ]
        )
    trace = tf.trace(tf.session(events))
    tails = build_tails(trace, [])
    latency = [r for r in tails.incidents if r.kind == "tool_latency"]
    assert len(latency) == 10
    assert latency[0].value == 300
    assert latency[0].median_ratio == 300
    assert latency[0].cohort_n == 12
    assert tails.coverage["timed_calls"] == 12
    d = next(d for d in tails.distributions if d.cohort == "tool_latency:Bash")
    assert d.maximum == 300 and d.p99 == 300
    assert d.observed_sum == 311


def test_invalid_missing_and_ambiguous_timing(tf):
    trace = tf.trace(
        tf.session(
            [
                tf.tool("Bash", call_id="a", at=5),
                tf.result(call_id="a", at=1),
                tf.tool("Bash", call_id="b"),
                tf.result(call_id="b"),
                tf.tool("Bash", call_id="c", at=0),
                tf.result(call_id="c", at=10),
                tf.result(call_id="c", at=20),
            ]
        )
    )
    tails = build_tails(trace, [])
    assert not rows(trace, "tool_latency")
    assert tails.coverage["missing_or_invalid_call_timing"] == 2
    assert tails.coverage["unpaired_or_ambiguous_calls"] == 1


def test_waits_separate_and_overlaps_not_wall_time(tf):
    trace = tf.trace(
        tf.session(
            [
                tf.tool("Bash", call_id="a", at=0),
                tf.tool("Bash", call_id="b", at=1),
                tf.result(call_id="a", at=500),
                tf.result(call_id="b", at=501),
                tf.tool("get_subagent_result", call_id="c", at=501),
                tf.result(call_id="c", at=901),
            ]
        )
    )
    assert len(rows(trace, "tool_latency")) == 2
    assert rows(trace, "parent_wait")[0].value == 400


def retry_trace(tf, *, successes=False, user_boundary=False, changed=False):
    events = []
    for i in range(10):
        if user_boundary and i == 5:
            events.append(tf.user("New task"))
        events.extend(
            [
                tf.tool("Bash", call_id=str(i), args={"command": f"bad {i}" if changed else "bad"}),
                tf.result(call_id=str(i), ok=successes, error_text="unsupported --bad option"),
            ]
        )
    return tf.trace(tf.session(events))


def test_ten_tries_are_one_incident_with_all_evidence(tf):
    trace = retry_trace(tf)
    incident = rows(trace, "retry_attempts")[0]
    assert incident.value == 10 and incident.exceeds_threshold
    assert len(incident.event_indices) == 20
    assert incident.evidence["recovered"] is False
    assert "arguments" in tail_state(trace, incident)["incident"]["attempts"][0]
    assert len(rows(trace, "failure_chain")) == 1


def test_success_and_user_boundaries_do_not_create_retry_storm(tf):
    assert not rows(retry_trace(tf, successes=True), "retry_attempts")
    assert all(not r.exceeds_threshold for r in rows(retry_trace(tf, user_boundary=True), "retry_attempts"))
    assert not rows(retry_trace(tf, changed=True), "retry_attempts")
    assert rows(retry_trace(tf, changed=True), "failure_chain")[0].evidence["same_goal_unverified"]


def test_output_truncation_does_not_invent_original_size(tf):
    result = tf.result(call_id="a", output="tiny")
    result.output_truncated = True
    trace = tf.trace(tf.session([tf.tool("Bash", call_id="a"), result]))
    assert not rows(trace, "output_volume")
    trace.root.events[1].output_chars_original = 200_000
    assert rows(trace, "output_volume")[0].value == 200_000


def test_open_delegation_is_lower_bound_not_hang(tf):
    trace = tf.trace(tf.session([tf.tool("Agent", call_id="a", at=0), tf.assistant("still working", at=2000)]))
    row = HandoverRecord(
        id="h",
        trace_id=trace.trace_id,
        parent_id=trace.root.session_id,
        source_format="claude_native",
        status="running-at-capture",
        events=[HandoverEvent(kind="spawn", session_id=trace.root.session_id, event_idx=0, ts=trace.root.events[0].ts)],
    )
    incident = rows(trace, "delegation_open", [row])[0]
    assert incident.value == 2000
    assert incident.evidence["censored"] and not incident.evidence["hang_proven"]
    row.status = "terminated"
    assert not rows(trace, "delegation_open", [row])


def test_estimated_children_excluded(tf):
    trace = retry_trace(tf)
    trace.root.degraded.append("usage_estimated")
    assert not build_tails(trace, []).incidents


def test_polling_is_not_retry(tf):
    trace = retry_trace(tf)
    for ev in trace.root.events:
        if ev.kind is EventKind.tool_call:
            ev.tool_name = "get_subagent_result"
    assert not rows(trace, "retry_attempts")
    assert rows(trace, "poll_amplification")[0].value == 10


def test_pipeline_and_all_report_formats(tf):
    run = run_trace(retry_trace(tf), load_config(), semantic_mode="off")
    report = Report.build([run], ReportMeta(tool_version="test"))
    assert any(f.id == "TAIL_RETRY_ATTEMPTS" and f.kind == "observation" for f in run.findings)
    assert report_to_dict(report)["runs"][0]["tails"]["incidents"]
    assert "Execution extremes" in render_html(report)
    assert "Repeated identical invocation" in render_html(report)
    assert "Execution extremes" in render_table(report)
    assert "tail_threshold_crossings" in render_csv(report)
    assert {f.id for f in load_bank() if f.scope == "tail" and f.id.startswith("tail.execution.")} == {
        "tail.execution.same_blocker",
        "tail.execution.changed_approach",
    }


def test_tail_findings_respect_disable_severity_and_no_detectors(tf):
    trace = tf.trace(tf.session([tf.tool("Bash", call_id="a", at=0), tf.result(call_id="a", at=200)]))
    config = load_config()
    run = run_trace(trace, config, semantic_mode="off")
    found = [f for f in run.findings if f.id == "TAIL_TOOL_LATENCY"]
    assert len(found) == 1
    assert found[0].evidence["value"] == 200
    assert found[0].evidence["threshold"] == config.tails.tool_latency
    assert found[0].spans[0].event_idx == 0
    assert len(run.tails.incidents) > 0
    html = render_html(Report.build([run], ReportMeta(tool_version="test")))
    assert 'id="execution-extremes"' in html
    assert "tool latency crossed its threshold in 1 runs" in html

    disabled = config.model_copy(deep=True)
    disabled.detectors.disabled.append("TAIL_TOOL_LATENCY")
    assert not any(f.id == "TAIL_TOOL_LATENCY" for f in run_trace(trace, disabled).findings)

    overridden = config.model_copy(deep=True)
    overridden.detectors.severity["TAIL_TOOL_LATENCY"] = "high"
    assert any(f.id == "TAIL_TOOL_LATENCY" and f.severity == "high" for f in run_trace(trace, overridden).findings)

    skipped = run_trace(trace, config, detectors=False)
    assert not skipped.findings
    assert skipped.tails.incidents


def test_model_usage_is_uncached_and_reasoning_not_double_counted(tf):
    event = tf.with_usage(
        tf.assistant("response"), tf.usage(input=150_000, output=9_000, cache_read=200_000, reasoning_output=8_000)
    )
    trace = tf.trace(tf.session([event]))
    assert rows(trace, "model_input")[0].value == 150_000
    assert rows(trace, "model_output")[0].value == 9_000
    trace.root.usage_reliable = False
    assert not rows(trace, "model_output")


def test_edits_preserve_repair_cycle_label(tf):
    from agent_hotwash.events import ToolCategory

    events = []
    for i in range(10):
        events.extend(
            [
                tf.tool("test", call_id=f"t{i}"),
                tf.result(call_id=f"t{i}", ok=False),
                tf.tool("Edit", call_id=f"e{i}", category=ToolCategory.write),
                tf.result(call_id=f"e{i}"),
            ]
        )
    trace = tf.trace(tf.session(events))
    incident = rows(trace, "retry_attempts")[0]
    assert incident.evidence["intervening_edit_calls"] == 9
    assert incident.label == "Repeated verification with intervening edits"
    state = tail_state(trace, incident)
    assert any(a.get("intervening_edit") for a in state["incident"]["attempts"])
    assert state["incident"]["sampled"]


def test_configurable_thresholds(tf):
    from agent_hotwash.config import TailsConfig

    trace = tf.trace(tf.session([tf.tool("Bash", call_id="a", at=0), tf.result(call_id="a", at=30)]))
    assert not rows(trace, "tool_latency")[0].exceeds_threshold
    tails = build_tails(trace, [], TailsConfig(tool_latency=20))
    assert next(r for r in tails.incidents if r.kind == "tool_latency").exceeds_threshold


def test_context_replay_respects_request_boundaries(tf):
    events = [tf.user("first")]
    events += [tf.with_usage(tf.assistant(), tf.usage(cache_read=250_000)) for _ in range(5)]
    events += [tf.user("second")]
    events += [tf.with_usage(tf.assistant(), tf.usage(cache_read=250_000)) for _ in range(4)]
    trace = tf.trace(tf.session(events))
    replay = rows(trace, "context_replay")
    assert len(replay) == 1
    assert replay[0].value == 1_250_000
    assert replay[0].evidence["model_calls"] == 5


def test_delegation_fanout_does_not_count_continuations(tf):
    trace = tf.trace(
        tf.session(
            [
                tf.user("work"),
                tf.tool("Agent", call_id="a"),
                tf.tool("Agent", call_id="b"),
                tf.tool("resume", call_id="c"),
            ]
        )
    )
    handovers = [
        HandoverRecord(
            id=str(i),
            trace_id=trace.trace_id,
            parent_id=trace.root.session_id,
            source_format="claude_native",
            events=[
                HandoverEvent(kind="spawn", session_id=trace.root.session_id, event_idx=i),
                HandoverEvent(kind="resume", session_id=trace.root.session_id, event_idx=3),
            ],
        )
        for i in (1, 2)
    ]
    incident = rows(trace, "delegation_fanout", handovers)[0]
    assert incident.value == 2
    assert incident.evidence["request_event_idx"] == 0


def test_cache_creation_is_not_cached_replay_or_a_cache_invalidation_claim(tf):
    events = [tf.user("implement")]
    events += [tf.with_usage(tf.assistant(), tf.usage(cache_read=0, cache_write=25000)) for _ in range(5)]
    events += [tf.user("next request")]
    events += [tf.with_usage(tf.assistant(), tf.usage(cache_write=90000)) for _ in range(4)]
    cumulative = tf.usage(cache_write=900000, cumulative=True)
    events.append(tf.with_usage(tf.assistant(), cumulative))
    trace = tf.trace(tf.session(events))
    # The session builder normally normalizes cumulative counters to deltas.
    # Simulate a still-cumulative observation supplied directly to the detector.
    trace.root.events[-1].usage = cumulative
    creation = rows(trace, "cache_creation")
    assert len(creation) == 1 and creation[0].value == 125000
    assert creation[0].exceeds_threshold and creation[0].evidence["invalidation_not_proven"]
    assert not rows(trace, "context_replay")
    trace.root.usage_reliable = False
    assert not rows(trace, "cache_creation")
