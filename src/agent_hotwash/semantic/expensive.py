"""Review high-spend runs using matched request/return work items."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from agent_hotwash.events import Capabilities, Event, EventKind, RoleHint
from agent_hotwash.semantic.bank import load_bank
from agent_hotwash.semantic.pipeline import Annotator
from agent_hotwash.semantic.redact import redact_state
from agent_hotwash.semantic.results import FeatureSet, FeatureValue

if TYPE_CHECKING:
    from agent_hotwash.config import Config
    from agent_hotwash.diagnostics.tails import TailAnalysis
    from agent_hotwash.events import Trace
    from agent_hotwash.report.model import Report
    from agent_hotwash.semantic.client import SystemOneAsker


def _excerpt(ev: Event, session_id: str) -> dict[str, Any]:
    text = ev.text or ""
    if ev.kind is EventKind.user_msg:
        text = re.sub(r"<skill\b[^>]*>.*?</skill>", "", text, flags=re.S).strip()
    bounded = text if len(text) <= 1800 else text[:900] + "\n[excerpt gap]\n" + text[-900:]
    return {
        "session_id": session_id,
        "event_idx": ev.idx,
        "source": ev.source.model_dump() if ev.source else None,
        "text": bounded,
        "truncated": len(text) > 1800,
    }


def expense_context(trace: Trace, tails: TailAnalysis, config: Config) -> dict[str, Any]:
    """Bounded work items with exact user-turn boundaries and observed token cover."""
    session = trace.root
    events = session.events
    user_positions = [
        i for i, ev in enumerate(events) if ev.kind is EventKind.user_msg and ev.role_hint in {None, RoleHint.user}
    ]
    recoveries = [r for r in tails.incidents if r.kind in {"retry_attempts", "failure_chain"}]
    work_items: list[dict[str, Any]] = []
    for number, start in enumerate(user_positions):
        end = user_positions[number + 1] if number + 1 < len(user_positions) else len(events)
        block = events[start:end]
        if not block:
            continue
        returns = [e for e in block if e.kind is EventKind.assistant_msg and e.phase != "commentary" and e.text]
        reply = returns[-1] if returns else None
        tokens = (
            sum(
                (e.usage.input or 0) + (e.usage.output or 0) + (e.usage.cache_read or 0) + (e.usage.cache_write or 0)
                for e in block
                if e.usage is not None and not e.usage.cumulative
            )
            if session.usage_reliable
            else None
        )
        first_idx, last_idx = block[0].idx, block[-1].idx
        work_items.append(
            {
                "request": _excerpt(block[0], session.session_id),
                "return": _excerpt(reply, session.session_id) if reply else None,
                "root_tokens_observed": tokens,
                "tool_calls": sum(e.kind is EventKind.tool_call for e in block),
                "failed_results": sum(e.kind is EventKind.tool_result and e.ok is False for e in block),
                "recovery": [
                    {"kind": r.kind, "attempts": r.value, "incident_id": r.id}
                    for r in recoveries
                    if r.session_id == session.session_id
                    and r.event_indices
                    and first_idx <= min(r.event_indices) <= last_idx
                ][:5],
                "start_idx": first_idx,
                "end_idx": last_idx,
            }
        )
    ranked = sorted(work_items, key=lambda item: (-(item["root_tokens_observed"] or 0), item["start_idx"]))
    chosen = ranked[:8]
    total_tokens = sum(item["root_tokens_observed"] or 0 for item in work_items)
    selected_tokens = sum(item["root_tokens_observed"] or 0 for item in chosen)
    state = {
        "review": {
            "work_items": chosen,
            "work_item_count": len(work_items),
            "selected_work_item_count": len(chosen),
            "root_token_coverage": selected_tokens / total_tokens if total_tokens > 0 else 0.0,
            "root_token_coverage_known": total_tokens > 0,
            "selection_basis": "largest observed root token totals",
            "child_spend_attributed": False,
        }
    }
    return redact_state(state, list(config.lexicons.secret))


def _supported(values: dict[str, FeatureValue], key: str) -> bool:
    fv = values[key]
    return (
        fv.reason is None
        and not fv.abstains
        and isinstance(fv.value, (int, float))
        and float(fv.value) >= (fv.positive_threshold or 0.7)
    )


def review_expensive(
    report: Report, config: Config, asker: SystemOneAsker, *, mode: str, allow_unredacted: bool = False
) -> None:
    """Evidence of matched work, without a cost-optimality verdict."""
    features = [f for f in load_bank() if f.scope == "expense"]
    by_id = {r.analysis.trace_id: r for r in report.runs}
    annotator = Annotator(asker, config, Capabilities(), mode=mode, allow_unredacted=allow_unredacted)
    selected = report.expense_tail.runs[: config.tails.max_expense_reviews]
    report.expense_tail.review_budget = config.tails.max_expense_reviews
    items = []
    for row in selected:
        state = by_id[row.trace_id].expense_context
        review = state.get("review", {})
        askable = [f for f in features if review.get("work_items")]
        items.append((state, askable))
    answers = annotator.ask_many(items)
    for row, (state, _), values in zip(selected, items, answers, strict=True):
        run = by_id[row.trace_id]
        for feature in features:
            if feature.id not in values:
                values[feature.id] = FeatureValue(id=feature.id, reason="insufficient_observability")
        fs = FeatureSet(scope="expense", object_id=row.trace_id, values=values)
        run.features = [f for f in (run.features or []) if not (f.scope == "expense" and f.object_id == row.trace_id)]
        run.features.append(fs)
        row.feature_set_id = row.trace_id
        row.evidence = state
        row.assessment = "insufficient_evidence"
        row.action = "Inspect request, return, and child cost evidence before judging spend."
        review = state.get("review", {})
        coverage = review.get("root_token_coverage")
        row.workload["reviewed_root_token_share"] = coverage
        row.workload["reviewed_work_items"] = review.get("selected_work_item_count")
        row.workload["total_work_items"] = review.get("work_item_count")
        if _supported(values, "expense.execution.recovery_dominates"):
            row.assessment = "investigate_blocked_recovery"
            row.action = "Inspect the cited incomplete deliverable and stop repeated recovery at its prerequisite."
        elif (
            _supported(values, "expense.scope.broad_work")
            and _supported(values, "expense.outcome.verification_reported")
            and _supported(values, "expense.outcome.matched_scope_and_verification")
            and isinstance(coverage, (int, float))
            and coverage >= 0.8
        ):
            row.assessment = "requested_verified_work_observed"
            row.action = (
                "Requested work and a matching checked return are visible; inspect child spend and execution tails."
            )
        elif _supported(values, "expense.outcome.verification_reported"):
            row.assessment = "verified_return_scope_unclear"
            row.action = "The checked return is visible; compare it with the costly work items and child activity."
    report.expense_tail.reviewed_runs = len(selected)
