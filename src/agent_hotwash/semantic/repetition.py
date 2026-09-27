"""Small semantic inputs and explicit conjunctions for repeated successful work."""

from __future__ import annotations

from typing import Any

from agent_hotwash.diagnostics.tails import TailIncident
from agent_hotwash.events import EventKind, ToolCategory, Trace
from agent_hotwash.semantic.results import FeatureValue


def repetition_state(trace: Trace, row: TailIncident, base: dict[str, Any]) -> dict[str, Any]:
    session = next(s for s in [trace.root, *trace.subagents] if s.session_id == row.session_id)
    by_idx = {e.idx: e for e in session.events}
    context = base["incident"]["context"]
    clipped = False

    def bound(text: str, n: int = 2400) -> str:
        nonlocal clipped
        if len(text) <= n:
            return text
        clipped = True
        return text[: n // 2] + "\n[excerpt gap]\n" + text[-n // 2 :]

    lo, hi = min(row.event_indices), max(row.event_indices)
    if row.kind == "delegation_repetition":
        prior = by_idx[row.evidence["prior_spawn_idx"]]
        returned = by_idx[row.evidence["prior_return_idx"]]
        current = by_idx[row.evidence["new_spawn_idx"]]
        prior_prompt = prior.tool_args.get("prompt") or prior.tool_args.get("message") or ""
        next_prompt = current.tool_args.get("prompt") or current.tool_args.get("message") or ""
        reply = returned.tool_args.get("result") or returned.output or returned.text or ""
        base["incident"]["review_pair"] = {
            "prior_request": bound(str(prior_prompt)),
            "prior_return": bound(str(reply)),
            "next_request": bound(str(next_prompt)),
            "shared_files": row.evidence["shared_request_files"],
        }
        context["review_requests_complete"] = len(str(prior_prompt)) <= 2400 and len(str(next_prompt)) <= 2400
        clipped |= bool(returned.output_truncated)
        lo, hi = returned.idx, current.idx
    intervening = [e for e in session.events if lo < e.idx < hi and e.kind == EventKind.tool_call]
    context["intervening_calls"] = [
        {
            "event_idx": e.idx,
            "tool": e.tool_name,
            "category": e.tool_category,
            "arguments": bound(str(e.tool_args), 800),
        }
        for e in intervening[:12]
    ]
    context["intervening_call_count"] = len(intervening)
    if row.kind == "verification_repetition":
        context["repeat_invocations"] = row.value
        context["exit_status_coverage"] = {
            key: row.evidence[key] for key in ("explicit_zero_exit_invocations", "unknown_exit_invocations")
        }
    context["intervening_compactions"] = sum(e.kind == EventKind.compaction for e in session.events if lo < e.idx < hi)
    context["intervening_write_calls"] = sum(e.tool_category == ToolCategory.write for e in intervening)
    request_idx = row.evidence.get("request_event_idx", -1)
    # An excerpted request cannot establish that the user did not ask for an
    # independent review. Preserve the full request only within the bound.
    request = by_idx.get(request_idx)
    request_text = request.text or "" if request else ""
    base["incident"]["request"] = bound(request_text, 2400)
    context["complete_local_evidence"] = bool(request_text) and not (
        clipped
        or len(intervening) > 12
        or base["incident"]["sampled"]
        or context["source_output_clipped"]
        or context["intervening_compactions"]
    )
    context["excerpted"] = clipped or len(intervening) > 12
    return base


def supported(values: dict[str, FeatureValue], key: str, *, positive: bool = True) -> bool:
    value = values.get(key)
    if not value or value.abstains or value.reason is not None or not isinstance(value.value, (int, float)):
        return False
    return (
        value.value >= (value.positive_threshold if value.positive_threshold is not None else 0.7)
        if positive
        else value.value <= (value.negative_threshold if value.negative_threshold is not None else 0.3)
    )


def assess_repetition(row: TailIncident, values: dict[str, FeatureValue], state: dict[str, Any]) -> None:
    """Negative/absent evidence alone never establishes avoidable work."""
    row.assessment = "unclear"
    row.assessment_reasons = ["A measured repeat does not establish redundant work."]
    if row.kind == "delegation_repetition":
        row.label = "Repeated delegation over review targets"
        row.action = "Compare review scope and artifact changes; carry forward prior findings and their disposition."
        same, reuse, fresh = (f"tail.review.{s}" for s in ("same_question", "reuses_findings", "fresh_pass_reason"))
        if not state["incident"]["context"].get("review_requests_complete", False):
            row.assessment_reasons = ["Review request excerpts cannot establish the complete task scope."]
            return
        if supported(values, same) and supported(values, fresh):
            row.assessment = "justified_repeat"
            row.label = "Repeated delegation with a visible fresh-review reason"
            row.action = "Preserve the independent or changed-state review; retain its specific scope and findings."
            row.assessment_reasons = [fresh]
        elif supported(values, same) and supported(values, reuse):
            row.assessment = "reuse_visible"
            row.label = "Follow-up review carries prior findings"
            row.assessment_reasons = [same, reuse]
        elif (
            supported(values, same)
            and supported(values, reuse, positive=False)
            and supported(values, fresh, positive=False)
            and state["incident"]["context"]["complete_local_evidence"]
        ):
            row.assessment = "supported_opportunity"
            row.label = "Prior review findings could be carried forward"
            row.action = (
                "Include the earlier findings and their disposition in the new review request; "
                "focus on unresolved questions. This does not establish that the review can be skipped."
            )
            row.assessment_reasons = [same, f"negative:{reuse}", f"negative:{fresh}", "complete_local_evidence"]
    elif row.kind == "verification_repetition":
        row.label = "Repeated check commands after non-error tool returns"
        row.action = "Check whether inputs changed or repeated validation was requested before proposing check reuse."
        for key in ("tail.check.changed_inputs", "tail.check.explicit_repeat_reason"):
            if supported(values, key) and (
                key == "tail.check.explicit_repeat_reason" or state["incident"]["context"]["complete_local_evidence"]
            ):
                row.assessment = "justified_repeat"
                row.label = "Repeated check with a visible rerun reason"
                row.action = (
                    "Keep validation after relevant changes or requested repeated trials; compare remaining scope."
                )
                row.assessment_reasons = [key]
                break
