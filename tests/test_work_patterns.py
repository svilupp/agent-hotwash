"""Preserve repetition evidence and avoid neighboring-event cost attribution."""

from agent_hotwash.config import load_config
from agent_hotwash.diagnostics.work import status_probe
from agent_hotwash.events import SourceRef, ToolCategory
from agent_hotwash.report.html import render_html
from agent_hotwash.report.model import Report, ReportMeta
from agent_hotwash.runner import run_trace
from agent_hotwash.sources._common import output_metadata, truncate


def at_source(ev, record, ordinal=0):
    ev.source = SourceRef(record_index=record, ordinal=ordinal)
    return ev


def test_fingerprint_preserves_differences_after_truncation():
    a, b = "x" * 5000 + "a", "x" * 5000 + "b"
    assert truncate(a) == truncate(b)
    assert output_metadata(a)["output_sha256"] != output_metadata(b)["output_sha256"]
    assert output_metadata(a)["output_chars_original"] == 5001
    assert output_metadata("")["output_lines_original"] == 0
    assert output_metadata(None) == {}


def test_repeated_large_output_needs_full_identity_and_request_boundary(tf):
    events = [tf.user("inspect")]
    for i in range(3):
        result = tf.result(call_id=str(i), output=truncate("x" * 12000))
        for key, value in output_metadata("x" * 12000).items():
            setattr(result, key, value)
        events += [tf.tool("read", call_id=str(i), args={"path": "a.py"}, category=ToolCategory.read), result]
    trace = tf.trace(tf.session(events))
    result = run_trace(trace, load_config())
    rows = [r for r in result.tails.incidents if r.kind == "output_repetition"]
    assert len(rows) == 1 and rows[0].value == 24000
    assert any(f.id == "TAIL_OUTPUT_REPETITION" for f in result.findings)
    for ev in trace.root.events:
        ev.output_sha256 = None
    assert not any(r.kind == "output_repetition" for r in run_trace(trace, load_config()).tails.incidents)

    # Identical values from different requests cannot establish one repeated-work incident.
    for ev in events:
        if ev.output_truncated:
            ev.output_sha256 = output_metadata("x" * 12000)["output_sha256"]
    events.insert(3, tf.user("read it again now"))
    events.insert(6, tf.user("check once more"))
    assert not any(
        r.kind == "output_repetition" for r in run_trace(tf.trace(tf.session(events)), load_config()).tails.incidents
    )


def test_interleaved_status_probes_keep_pending_model_usage(tf):
    events = [tf.user("implement feature")]
    for i in range(12):
        usage = tf.with_usage(tf.assistant(), tf.usage(input=10, output=5, cache_read=1000, cache_write=0))
        poll = tf.tool("get_subagent_result", call_id=f"p{i}", args={"agent_id": "child", "wait": False})
        events += [
            at_source(usage, i * 4 + 1),
            at_source(poll, i * 4 + 1, 1),
            tf.result(call_id=f"p{i}", output="Agent is still running. Use wait: true."),
            tf.tool("read", call_id=f"r{i}", args={"path": str(i)}, category=ToolCategory.read),
            tf.result(call_id=f"r{i}", output="different work"),
        ]
    result = run_trace(tf.trace(tf.session(events)), load_config())
    row = next(r for r in result.tails.incidents if r.kind == "status_probes")
    assert row.value == 12 and row.exceeds_threshold
    assert row.evidence["pending_results"] == 12
    assert row.evidence["unchanged_pending_results"] == 11
    assert row.evidence["pending_only_model_rounds"] == 12
    assert row.evidence["pending_only_model_usage"]["cache_read"] == 12000
    assert sum(a.rounds for a in result.tails.model_activity) == 12
    assert not status_probe(tf.tool("get_subagent_result", call_id="block", args={"wait": True}))
    assert not status_probe(tf.tool("write_stdin", call_id="input", args={"chars": "y\n"}))
    assert not status_probe(tf.tool("write_stdin", call_id="wait", args={"chars": "", "yield_time_ms": 30000}))
    assert not status_probe(tf.tool("write_stdin", call_id="default", args={"chars": ""}))
    assert status_probe(tf.tool("write_stdin", call_id="poll", args={"chars": "", "yield_time_ms": 0}))


def test_inspection_counts_model_rounds_and_excludes_ambiguous_usage(tf):
    events = [tf.user("inspect three files")]
    for i in range(3):
        events.append(
            at_source(tf.with_usage(tf.assistant(), tf.usage(input=100, output=10, cache_read=0, cache_write=0)), i + 1)
        )
        for j in range(2):
            events.append(
                at_source(
                    tf.tool("read", call_id=f"{i}-{j}", args={"path": f"{i}-{j}"}, category=ToolCategory.read),
                    i + 1,
                    j + 1,
                )
            )
    trace = tf.trace(tf.session(events))
    result = run_trace(trace, load_config())
    row = next(r for r in result.tails.incidents if r.kind == "inspection_rounds")
    assert row.value == 3 and row.evidence["tool_calls"] == 6
    assert result.tails.model_activity[0].estimated_cost is not None
    assert (
        abs(
            sum(result.tails.model_activity[0].estimated_cost_components.values())
            - result.tails.model_activity[0].estimated_cost
        )
        < 1e-10
    )
    html = render_html(Report.build([result], ReportMeta(tool_version="test")))
    assert "Model work mix" in html and "3/3 usage observations" in html

    # An adjacent usage record must not be assigned to a later tool request.
    trace.root.events[2].source = SourceRef(record_index=100)
    trace.root.events[3].source = SourceRef(record_index=101)
    result = run_trace(trace, load_config())
    assert next(a for a in result.tails.model_activity if a.category == "inspection").rounds == 2

    # Two usage events in the same source record are ambiguous, not two rounds.
    trace.root.events[2].usage = tf.usage(input=100)
    trace.root.events[2].source = trace.root.events[1].source
    result = run_trace(trace, load_config())
    assert result.tails.coverage["attributed_model_rounds"] == 2


def test_unreliable_usage_does_not_get_priced(tf):
    events = [at_source(tf.with_usage(tf.assistant(), tf.usage(input=999, output=99, cache_read=0, cache_write=0)), 1)]
    trace = tf.trace(tf.session(events, usage_reliable=False))
    assert not run_trace(trace, load_config()).tails.model_activity


def test_batching_requires_first_reads_in_distinct_rounds_before_edits(tf, monkeypatch):
    from typing import cast

    from agent_hotwash.diagnostics.tails import build_tails
    from agent_hotwash.semantic.client import SystemOneAsker
    from agent_hotwash.semantic.results import FeatureValue
    from agent_hotwash.semantic.tails import annotate_tails, tail_state

    def make_trace(*, batch=False, edit=False):
        events = [tf.user("Compare a.py, b.py, and c.py.")]
        for i, name in enumerate(("a.py", "b.py", "c.py")):
            record = 1 if batch else i + 1
            if not batch or i == 0:
                events.append(at_source(tf.with_usage(tf.assistant(), tf.usage(input=1)), record))
            if edit and i == 1:
                events.append(tf.tool("edit", call_id="edit", category=ToolCategory.write))
            events += [
                at_source(
                    tf.tool("read", call_id=name, args={"path": name}, category=ToolCategory.read), record, i + 1
                ),
                tf.result(call_id=name, output=f"contents of {name}"),
            ]
        # Later unrelated discovery must not manufacture additional named first reads.
        for i in range(3):
            events += [
                at_source(tf.with_usage(tf.assistant(), tf.usage(input=1)), 10 + i),
                at_source(tf.tool("read", call_id=str(i), args={"path": "a.py"}, category=ToolCategory.read), 10 + i),
                tf.result(call_id=str(i), output="more a.py"),
            ]
        return tf.trace(tf.session(events))

    asked = []

    def ask_many(self, items):
        answers = []
        for _, features in items:
            asked.extend(f.id for f in features)
            answers.append({f.id: FeatureValue(id=f.id, value=0.95) for f in features})
        return answers

    monkeypatch.setattr("agent_hotwash.semantic.tails.Annotator.ask_many", ask_many)
    cfg = load_config()
    cfg = cfg.model_copy(update={"tails": cfg.tails.model_copy(update={"inspection_rounds": 3})})
    for batch, edit, expected in [(False, False, 3), (True, False, 1), (False, True, 1)]:
        trace = make_trace(batch=batch, edit=edit)
        tails = build_tails(trace, [], cfg.tails)
        row = next(r for r in tails.incidents if r.kind == "inspection_rounds")
        state = tail_state(trace, row)["incident"]
        assert state["context"]["preknown_distinct_rounds"] == expected
        assert any(a.get("diagnostic") == "contents of a.py" for a in state["attempts"])
        asked.clear()
        annotate_tails(trace, tails, cfg, cast(SystemOneAsker, object()), mode="live")
        assert ("tail.work.known_targets_batchable" in asked) is (expected == 3)
        assert ("could be batched" in row.label) is (expected == 3)


def test_meta_usage_requires_explicit_response_join(tf):
    from agent_hotwash.events import Event, EventKind

    call = tf.tool("read", call_id="r", args={"path": "a.py"}, category=ToolCategory.read)
    call.response_id = "response-1"
    usage = Event(
        kind=EventKind.meta, response_id="response-1", usage=tf.usage(input=12, output=3, cache_read=5, cache_write=0)
    )
    trace = tf.trace(tf.session([call, usage]))
    result = run_trace(trace, load_config())
    assert result.tails.model_activity[0].category == "inspection"
    assert result.tails.model_activity[0].usage.input == 12
    usage.response_id = None
    assert not run_trace(trace, load_config()).tails.model_activity
