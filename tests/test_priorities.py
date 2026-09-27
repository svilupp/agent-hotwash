"""Actionability must not promote unknown evidence or strand its source links."""

import json
from html.parser import HTMLParser

from agent_hotwash.config import load_config
from agent_hotwash.diagnostics.tails import TailIncident
from agent_hotwash.report.html import render_html
from agent_hotwash.report.json_writer import render_json
from agent_hotwash.report.model import Report, ReportMeta
from agent_hotwash.report.table import render_table
from agent_hotwash.runner import run_trace
from agent_hotwash.semantic.results import FeatureSet, FeatureValue


def incident(key, kind, *, value=10, assessment="observation", cost=None):
    return TailIncident(
        id=key,
        kind=kind,
        session_id="s0",
        event_indices=[1, 2],
        value=value,
        unit="seconds" if kind == "tool_latency" else "transitions",
        threshold=2,
        exceeds_threshold=value >= 2,
        cohort=kind,
        label=kind,
        action="Use the offered blocking wait.",
        assessment=assessment,
        evidence={"observed_cache_write_cost": cost, "priced_transitions": int(value) if cost is not None else 0},
    )


def report_with(tf, rows):
    run = run_trace(tf.trace(tf.session([tf.user("work")])), load_config(), semantic_mode="off")
    run.tails.incidents = rows
    return Report.build([run], ReportMeta(tool_version="test"))


def test_supported_changes_precede_large_observations_and_exclude_justified_repeats(tf):
    report = report_with(
        tf,
        [
            incident("unclear", "status_probes", value=500),
            incident("justified", "status_probes", value=600, assessment="justified_repeat"),
            incident("supported", "status_probes", value=4, assessment="supported_opportunity"),
            incident("slow", "tool_latency", value=99999),
        ],
    )
    assert [p.status for p in report.priorities] == ["supported", "investigate"]
    first = report.priorities[0]
    assert first.incidents == 1 and first.examples[0].target_id == "tail-supported"
    assert first.verify and first.owner and first.limit
    assert report.priorities[1].observed_cost is None
    text = render_table(report)
    assert text.index("Actions to review") < text.index("Per-run analysis")


def test_costs_preserve_missingness_and_counts_deduplicate_incidents(tf):
    report = report_with(tf, [incident("unknown", "cache_rebuilds", cost=None)])
    assert report.priorities[0].observed_cost is None
    report.runs.append(report.runs[0])
    report.runs[0].tails.incidents.append(incident("zero", "cache_rebuilds", cost=0))
    action = report.priorities[0]
    assert action.incidents == 2 and action.affected_runs == 1
    assert action.observed_cost == 0
    assert "10 transitions priced" in action.evidence


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids = set()
        self.refs = set()

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if attrs.get("id"):
            self.ids.add(attrs["id"])
        if tag == "a" and attrs.get("href", "").startswith("#"):
            self.refs.add(attrs["href"][1:])


def test_action_examples_outside_top_twenty_still_render_and_all_links_resolve(tf):
    rows = [incident(f"many-{i}", "cache_rebuilds", value=100 - i, cost=1) for i in range(25)]
    rows.append(incident("costliest", "cache_rebuilds", value=2, cost=500))
    report = report_with(tf, rows)
    assert report.priorities[0].examples[0].target_id == "tail-costliest"
    doc = render_html(report)
    links = Links()
    links.feed(doc)
    assert links.refs <= links.ids
    assert "tail-costliest" in links.ids
    assert "Verify improvement:" in doc and "Full source record and classifier evidence" in doc


def test_serialized_priorities_follow_later_expense_review_without_rebuild(tf):
    report = report_with(tf, [])
    run = report.runs[0]
    run.analysis.cost = 20
    run.analysis.cost_estimated = True
    report = Report.build([run], report.meta)
    assert report.priorities[0].status == "investigate"
    report.expense_tail.runs[0].assessment = "requested_verified_work_observed"
    data = json.loads(render_json(report))
    assert data["priorities"] == []
    assert Report.model_validate_json(render_json(report)).priorities == []


def test_censored_delegation_is_a_visibility_action_without_savings(tf):
    report = report_with(tf, [incident("open", "delegation_open", value=1000)])
    action = report.priorities[0]
    assert action.status == "measure" and action.observed_cost is None
    assert "at least" in action.evidence and "not proof of a hang" in action.limit


def test_call_excerpts_redact_before_clipping_and_escape_html(tf):
    secret = "sk-" + "A" * 40
    trace = tf.trace(
        tf.session(
            [
                tf.user("run checks"),
                tf.tool("Bash", call_id="slow", args={"cmd": "x" * 590 + secret}, at=0),
                tf.result(call_id="slow", ok=False, output="<script>alert(1)</script> " + secret, at=500),
            ]
        )
    )
    run = run_trace(trace, load_config(), semantic_mode="off")
    row = next(r for r in run.tails.incidents if r.kind == "tool_latency")
    assert "[secret]" in row.evidence["request_excerpt"] and "sk-" not in row.evidence["request_excerpt"]
    assert secret not in row.evidence["result_excerpt"]
    doc = render_html(Report.build([run], ReportMeta(tool_version="test")))
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in doc
    assert "<script>alert(1)</script>" not in doc
    assert "Error reported" in doc and "Invocation" in doc


def test_classifier_failures_are_visible_without_becoming_supported_changes(tf):
    report = report_with(tf, [incident("unreviewed", "status_probes", assessment="unclear")])
    report.runs[0].features = [
        FeatureSet(
            scope="tail",
            object_id="unreviewed",
            values={
                "failed": FeatureValue(id="failed", reason="api_error"),
                "uncertain": FeatureValue(id="uncertain", value=0.5, reason="low_support"),
                "answered": FeatureValue(id="answered", value=0.9),
                "missing": FeatureValue(id="missing", reason="insufficient_observability"),
                "fact": FeatureValue(id="fact", source="fact", value=True),
            },
        )
    ]
    # Duplicate input records cannot inflate coverage.
    report.runs.append(report.runs[0])
    coverage = report.classifier_coverage["tail"]
    assert coverage.questions == 4 and coverage.answered == 2
    assert coverage.uncertain == 1 and coverage.api_errors == 1 and coverage.unavailable == 1
    assert coverage.objects == 1 and coverage.fully_answered_objects == 0
    assert not report.priorities
    html = render_html(report)
    assert "Classifier review is incomplete" in html
    assert "Review incomplete:" in html
    assert "Classifier checks" in html and "Request failed" in html and "Evidence against in excerpt" not in html
    assert "Supported in excerpt" in html and "Uncertain" in html
    assert "2/4 feature answers received (1 uncertain); 1 API errors; 1 unavailable" in html
    assert "Review incomplete" in render_table(report)
    data = json.loads(render_json(report))
    assert data["classifier_coverage"]["tail"] == coverage.model_dump()
    assert Report.model_validate_json(render_json(report)).classifier_coverage == report.classifier_coverage


def test_semantic_off_does_not_imply_successful_classifier_review(tf):
    report = report_with(tf, [incident("slow", "tool_latency")])
    assert not report.classifier_coverage
    assert "No classifier feature records are available" in render_html(report)
