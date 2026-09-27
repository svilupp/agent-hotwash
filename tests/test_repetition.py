"""Counterexamples for successful repetition and hybrid review decisions."""

from typing import cast

from agent_hotwash.config import load_config
from agent_hotwash.diagnostics.repetition import _check_command
from agent_hotwash.diagnostics.tails import TailAnalysis
from agent_hotwash.events import ToolCategory
from agent_hotwash.report.html import render_html
from agent_hotwash.report.model import Report, ReportMeta
from agent_hotwash.runner import run_trace
from agent_hotwash.semantic.client import SystemOneAsker
from agent_hotwash.semantic.repetition import assess_repetition
from agent_hotwash.semantic.results import FeatureValue
from agent_hotwash.semantic.tails import annotate_tails, tail_state


def review_trace(tf, *, boundary=False, overlap=True, background=False):
    events = [tf.user("Review src/auth.py for correctness.")]
    for i in range(3):
        if boundary and i:
            events.append(tf.user("Review it again."))
        path = "src/auth.py" if overlap else f"src/file{i}.py"
        events += [
            tf.tool(
                "Agent",
                call_id=f"r{i}",
                args={"prompt": f"Review {path} for correctness."},
                category=ToolCategory.subagent,
            ),
            tf.result(
                call_id=f"r{i}",
                output="Agent is running in background." if background else "Missing timeout in src/auth.py.",
            ),
        ]
    return tf.trace(tf.session(events))


def test_delegation_requires_overlap_completed_parent_return_and_same_request(tf):
    cfg = load_config()
    trace = review_trace(tf)
    result = run_trace(trace, cfg)
    rows = [r for r in result.tails.incidents if r.kind == "delegation_repetition"]
    assert [r.value for r in rows] == [2, 1]
    assert sum(r.exceeds_threshold for r in rows) == 1
    row = rows[0]
    assert row.evidence["prior_spawn_idx"] == 3
    assert row.evidence["prior_return_idx"] == 4
    assert row.evidence["new_spawn_idx"] == 5
    state = tail_state(trace, row)["incident"]
    assert state["context"]["complete_local_evidence"]
    assert state["review_pair"]["prior_return"] == "Missing timeout in src/auth.py."
    assert all(a["event_idx"] <= 5 for a in state["attempts"])
    for kwargs in ({"boundary": True}, {"overlap": False}, {"background": True}):
        assert not any(
            r.kind == "delegation_repetition" for r in run_trace(review_trace(tf, **kwargs), cfg).tails.incidents
        )


def test_check_repeats_require_final_zero_exit_exact_args_and_no_boundary(tf):
    events = [tf.user("Implement the feature.")]
    for i in range(3):
        events += [
            tf.tool("bash", call_id=str(i), args={"command": "make check"}, at=i * 100),
            tf.result(call_id=str(i), exit_code=0, at=i * 100 + 30),
        ]
    trace = tf.trace(tf.session(events))
    rows = [r for r in run_trace(trace, load_config()).tails.incidents if r.kind == "verification_repetition"]
    assert len(rows) == 1 and rows[0].value == 2
    assert rows[0].evidence["repeat_round_trip_seconds"] == 60
    assert rows[0].evidence["timed_repeats"] == 2
    # Tool-wrapper success is not proof that a check completed successfully.
    for ev in trace.root.events:
        ev.exit_code = None
    unknown = next(r for r in run_trace(trace, load_config()).tails.incidents if r.kind == "verification_repetition")
    assert unknown.evidence["unknown_exit_invocations"] == 3
    assert unknown.evidence["explicit_zero_exit_invocations"] == 0
    for ev in trace.root.events:
        if ev.call_id is not None:
            ev.exit_code = 1
    assert not any(r.kind == "verification_repetition" for r in run_trace(trace, load_config()).tails.incidents)
    for i in (2, 4, 6):
        events[i].exit_code = 0
    events.insert(3, tf.user("Recheck now."))
    events.insert(6, tf.user("Recheck once more."))
    assert not any(
        r.kind == "verification_repetition"
        for r in run_trace(tf.trace(tf.session(events)), load_config()).tails.incidents
    )


def review_values(same=0.95, reuse=0.05, fresh=0.05):
    return {
        f"tail.review.{key}": FeatureValue(id=f"tail.review.{key}", value=value)
        for key, value in [("same_question", same), ("reuses_findings", reuse), ("fresh_pass_reason", fresh)]
    }


def test_review_opportunity_needs_all_atoms_and_complete_context(tf):
    trace = review_trace(tf)
    row = next(r for r in run_trace(trace, load_config()).tails.incidents if r.kind == "delegation_repetition")
    state = tail_state(trace, row)
    for values, expected in [
        (review_values(), "supported_opportunity"),
        (review_values(same=0.05), "unclear"),
        (review_values(reuse=0.95), "reuse_visible"),
        (review_values(fresh=0.95), "justified_repeat"),
        (review_values(fresh=0.5), "unclear"),
        ({}, "unclear"),
    ]:
        assess_repetition(row, values, state)
        assert row.assessment == expected
    values = review_values()
    values["tail.review.fresh_pass_reason"].reason = "api_error"
    assess_repetition(row, values, state)
    assert row.assessment == "unclear"
    assert "could be carried" not in row.label
    state["incident"]["context"]["complete_local_evidence"] = False
    assess_repetition(row, review_values(), state)
    assert row.assessment == "unclear"


def test_clipping_and_compaction_prevent_negative_reason_claim(tf):
    trace = review_trace(tf)
    trace.root.events[4].output = "x" * 3000
    row = next(r for r in run_trace(trace, load_config()).tails.incidents if r.kind == "delegation_repetition")
    assert not tail_state(trace, row)["incident"]["context"]["complete_local_evidence"]
    trace.root.events[4].output = "one bug"
    trace.root.events[4].output_truncated = True
    assert not tail_state(trace, row)["incident"]["context"]["complete_local_evidence"]


def test_live_route_and_report_show_hybrid_assessment(tf, monkeypatch):
    cfg = load_config()
    trace = review_trace(tf)
    run = run_trace(trace, cfg)
    rows = [r for r in run.tails.incidents if r.kind == "delegation_repetition"]
    asked = []

    def ask_many(self, items):
        asked.extend(f.id for _, features in items for f in features)
        return [review_values() for _ in items]

    monkeypatch.setattr("agent_hotwash.semantic.tails.Annotator.ask_many", ask_many)
    run.features = annotate_tails(trace, TailAnalysis(incidents=rows), cfg, cast(SystemOneAsker, object()), mode="live")
    assert set(asked) == set(review_values())
    assert rows[0].assessment == "supported_opportunity"
    assert rows[1].assessment == "observation"
    html = render_html(Report.build([run], ReportMeta(tool_version="test")))
    assert "supported opportunity" in html and "Prior review findings could be carried forward" in html


def test_check_selector_excludes_format_and_generation_only():
    for command in (
        "make recipes",
        "make generate",
        "make install",
        "ruff format a.py",
        "uv run ruff check --fix a.py",
    ):
        assert not _check_command(command)
    for command in (
        "make check",
        "make recipes && make check",
        "uv run ruff format --check a.py",
        "cd edge && npx vitest run",
    ):
        assert _check_command(command)


def test_review_prefilter_does_not_turn_implementation_into_review(tf):
    trace = review_trace(tf)
    for ev in trace.root.events:
        if ev.tool_name == "Agent":
            ev.tool_args["subagent_type"] = "implementer"
    assert not any(r.kind == "delegation_repetition" for r in run_trace(trace, load_config()).tails.incidents)


def test_background_check_returns_are_not_completed_invocations(tf):
    events = [tf.user("Run checks")]
    for i in range(3):
        events += [
            tf.tool("exec", call_id=str(i), args={"cmd": "make check"}),
            tf.result(call_id=str(i), output="Process running with session ID 123"),
        ]
    assert not any(
        r.kind == "verification_repetition"
        for r in run_trace(tf.trace(tf.session(events)), load_config()).tails.incidents
    )


def test_unchanged_pending_gate_separates_targets_and_progress(tf, monkeypatch):
    from agent_hotwash.events import SourceRef

    def trace_for(*, targets=False, progress=False):
        events = [tf.user("Implement the feature")]
        for i in range(12):
            usage = tf.with_usage(tf.assistant(), tf.usage(input=1, output=1, cache_read=1, cache_write=0))
            usage.source = SourceRef(record_index=i + 1)
            call = tf.tool(
                "get_subagent_result", call_id=str(i), args={"agent_id": str(i) if targets else "child", "wait": False}
            )
            call.source = SourceRef(record_index=i + 1, ordinal=1)
            events += [
                usage,
                call,
                tf.result(
                    call_id=str(i),
                    output="Agent is still running. Use wait: true." + (f" Progress {i}" if progress else ""),
                ),
            ]
        return tf.trace(tf.session(events))

    monitoring = False

    def ask_many(self, items):
        return [
            {
                f.id: FeatureValue(
                    id=f.id, value=0.95 if f.id.endswith("blocking_wait_offered") or monitoring else 0.05
                )
                for f in features
            }
            for _, features in items
        ]

    monkeypatch.setattr("agent_hotwash.semantic.tails.Annotator.ask_many", ask_many)
    cfg = load_config()
    for targets, progress, expected in [(False, False, 11), (True, False, 0), (False, True, 0)]:
        trace = trace_for(targets=targets, progress=progress)
        row = next(r for r in run_trace(trace, cfg).tails.incidents if r.kind == "status_probes")
        assert row.evidence["unchanged_pending_results"] == expected
        annotate_tails(trace, TailAnalysis(incidents=[row]), cfg, cast(SystemOneAsker, object()), mode="live")
        assert (row.assessment == "supported_opportunity") is (expected == 11)
    monitoring = True
    trace = trace_for()
    row = next(r for r in run_trace(trace, cfg).tails.incidents if r.kind == "status_probes")
    annotate_tails(trace, TailAnalysis(incidents=[row]), cfg, cast(SystemOneAsker, object()), mode="live")
    assert row.assessment == "justified_repeat"


def test_pending_progress_ignores_only_recognized_elapsed_clock(tf):
    for change_tools in (False, True):
        events = [tf.user("Implement the feature.")]
        for i in range(5):
            output = (
                f"Agent: child\nType: implementer | Status: running | Tool uses: {i if change_tools else 2} "
                f"| 47.3k token | Context: 19% | Duration: {100 + i}.0s (running)\n"
                "Description: Work\n\nAgent is still running. Use wait: true."
            )
            events += [
                tf.tool("get_subagent_result", call_id=str(i), args={"agent_id": "child", "wait": False}),
                tf.result(call_id=str(i), output=output),
            ]
        row = next(
            r
            for r in run_trace(tf.trace(tf.session(events)), load_config()).tails.incidents
            if r.kind == "status_probes"
        )
        assert row.evidence["unchanged_pending_results"] == 0
        assert row.evidence["unchanged_pending_progress_results"] == (0 if change_tools else 4)
