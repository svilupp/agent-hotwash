"""Cost tails route inspection; expensive never implies inappropriate."""

from unittest.mock import Mock

from agent_hotwash.config import TailsConfig, load_config
from agent_hotwash.diagnostics.expensive import expensive_tail
from agent_hotwash.report.html import render_html
from agent_hotwash.report.json_writer import report_to_dict
from agent_hotwash.report.model import Report, ReportMeta
from agent_hotwash.runner import run_trace
from agent_hotwash.semantic.bank import load_bank
from agent_hotwash.semantic.expensive import review_expensive
from agent_hotwash.semantic.pipeline import Annotator
from agent_hotwash.semantic.results import FeatureValue


def run(tf, i, cost, basis="estimated"):
    result = run_trace(
        tf.trace(tf.session([tf.user("Fix the API"), tf.assistant("Done")]), trace_id=str(i)),
        load_config(),
        semantic_mode="off",
    )
    result.analysis.cost = cost
    result.analysis.cost_source = basis
    return result


def test_five_percent_and_ties(tf):
    runs = [run(tf, i, i + 1) for i in range(40)]
    runs[-3].analysis.cost = 39
    tail = expensive_tail(runs)
    assert len(tail.runs) == 3
    assert [r.cost for r in tail.runs] == [40, 39, 39]
    assert tail.runs[0].cohort_n == 40
    assert tail.runs[0].cutoff == 39
    assert tail.selected_cost_share["estimated"] == 118 / 821
    html = render_html(Report.build(runs, ReportMeta(tool_version="test")))
    assert "Expense concentration" in html
    assert 'id="most-expensive-5percent"' in html


def test_cost_bases_missing_and_duplicates_separate(tf):
    runs = [run(tf, 1, 10), run(tf, 2, 20, "provenance"), run(tf, 3, None), run(tf, 4, 0)]
    runs.append(runs[0])
    tail = expensive_tail(runs)
    assert len(tail.runs) == 2
    assert tail.cohort_sizes == {"estimated": 1, "provenance": 1}
    assert tail.missing_cost_runs == 1 and tail.zero_cost_runs == 1 and tail.duplicate_trace_ids == 1


def test_expense_context_excluded_from_unreviewed_json(tf):
    r = run(tf, 1, 10)
    report = Report.build([r], ReportMeta(tool_version="test"))
    assert r.expense_context["review"]["work_items"]
    assert "expense_context" not in report_to_dict(report)["runs"][0]
    assert report.expense_tail.runs[0].assessment == "not_reviewed"


def test_review_supported_not_optimality_and_recovery_precedence(tf, monkeypatch):
    config = load_config()
    candidate = run(tf, 1, 10)
    candidate.expense_context["review"]["root_token_coverage"] = 1.0
    report = Report.build([candidate], ReportMeta(tool_version="test"))
    feats = [f for f in load_bank() if f.scope == "expense"]

    def answers(_self, items):
        return [
            {
                f.id: FeatureValue(id=f.id, value=0.9 if "recovery" not in f.id else 0.1, positive_threshold=0.7)
                for f in feats
            }
            for _ in items
        ]

    monkeypatch.setattr(Annotator, "ask_many", answers)
    review_expensive(report, config, Mock(), mode="cached")
    assert report.expense_tail.runs[0].assessment == "requested_verified_work_observed"
    assert "child spend" in report.expense_tail.runs[0].action

    def recovery(_self, items):
        return [{f.id: FeatureValue(id=f.id, value=0.9, positive_threshold=0.7) for f in feats} for _ in items]

    monkeypatch.setattr(Annotator, "ask_many", recovery)
    review_expensive(report, config, Mock(), mode="cached")
    assert report.expense_tail.runs[0].assessment == "investigate_blocked_recovery"


def test_low_support_and_budget(tf, monkeypatch):
    report = Report.build([run(tf, 1, 10)], ReportMeta(tool_version="test"))
    feats = [f for f in load_bank() if f.scope == "expense"]

    def answers(_self, items):
        return [{f.id: FeatureValue(id=f.id, value=0.5, reason="low_support") for f in feats} for _ in items]

    monkeypatch.setattr(Annotator, "ask_many", answers)
    config = load_config()
    review_expensive(report, config, Mock(), mode="cached")
    assert report.expense_tail.runs[0].assessment == "insufficient_evidence"
    report.expense_tail.runs[0].assessment = "not_reviewed"
    config = config.model_copy(update={"tails": TailsConfig(max_expense_reviews=0)})
    review_expensive(report, config, Mock(), mode="cached")
    assert report.expense_tail.reviewed_runs == 0
    assert report.expense_tail.runs[0].assessment == "not_reviewed"


def test_skill_wrapper_does_not_replace_user_scope(tf):
    result = run_trace(
        tf.trace(
            tf.session(
                [
                    tf.user(
                        '<skill name="x">'
                        + "internal instructions " * 150
                        + "</skill>\nMigrate the API and verify clients."
                    ),
                    tf.assistant("Updated API and clients; integration tests passed."),
                ]
            )
        ),
        load_config(),
        semantic_mode="off",
    )
    text = result.expense_context["review"]["work_items"][0]["request"]["text"]
    assert text == "Migrate the API and verify clients."
    assert "internal instructions" not in text


def test_different_turns_cannot_support_one_expense_claim(tf, monkeypatch):
    from agent_hotwash.semantic.expensive import expense_context

    trace = tf.trace(
        tf.session(
            [
                tf.user("Migrate the API and update all clients"),
                tf.with_usage(tf.assistant("I could not finish the migration."), tf.usage(input=200, output=100)),
                tf.user("Rename one local variable"),
                tf.with_usage(tf.assistant("Renamed it; make test passed."), tf.usage(input=300, output=100)),
            ]
        )
    )
    run = run_trace(trace, load_config(), semantic_mode="off")
    state = expense_context(trace, run.tails, load_config())["review"]
    assert len(state["work_items"]) == 2
    assert state["work_items"][0]["request"]["text"] == "Rename one local variable"
    assert state["work_items"][0]["return"]["text"] == "Renamed it; make test passed."
    run.analysis.cost = 10
    run.analysis.cost_source = "estimated"
    report = Report.build([run], ReportMeta(tool_version="test"))
    feats = [f for f in load_bank() if f.scope == "expense"]

    def crossed(_self, items):
        return [
            {
                f.id: FeatureValue(
                    id=f.id,
                    value=0.1
                    if f.id
                    in {"expense.outcome.matched_scope_and_verification", "expense.execution.recovery_dominates"}
                    else 0.9,
                    positive_threshold=0.7,
                )
                for f in feats
            }
            for _ in items
        ]

    monkeypatch.setattr(Annotator, "ask_many", crossed)
    review_expensive(report, load_config(), Mock(), mode="cached")
    assert report.expense_tail.runs[0].assessment == "verified_return_scope_unclear"
    assert report.expense_tail.runs[0].workload["reviewed_root_token_share"] == 1.0


def test_without_token_coverage_cannot_support_spend(tf, monkeypatch):
    report = Report.build([run(tf, 1, 10)], ReportMeta(tool_version="test"))
    feats = [f for f in load_bank() if f.scope == "expense"]

    def high(_self, items):
        return [
            {
                f.id: FeatureValue(id=f.id, value=0.1 if "recovery" in f.id else 0.9, positive_threshold=0.7)
                for f in feats
            }
            for _ in items
        ]

    monkeypatch.setattr(Annotator, "ask_many", high)
    review_expensive(report, load_config(), Mock(), mode="cached")
    assert report.expense_tail.runs[0].assessment == "verified_return_scope_unclear"
    assert report.expense_tail.runs[0].workload["reviewed_root_token_share"] == 0.0


def test_conflicting_scope_answer_cannot_support_spend(tf, monkeypatch):
    candidate = run(tf, 1, 10)
    candidate.expense_context["review"]["root_token_coverage"] = 1.0
    report = Report.build([candidate], ReportMeta(tool_version="test"))
    feats = [f for f in load_bank() if f.scope == "expense"]

    def conflicting(_self, items):
        return [
            {
                f.id: FeatureValue(
                    id=f.id,
                    value=0.1 if f.id in {"expense.scope.broad_work", "expense.execution.recovery_dominates"} else 0.9,
                    positive_threshold=0.7,
                )
                for f in feats
            }
            for _ in items
        ]

    monkeypatch.setattr(Annotator, "ask_many", conflicting)
    review_expensive(report, load_config(), Mock(), mode="cached")
    assert report.expense_tail.runs[0].assessment == "verified_return_scope_unclear"
