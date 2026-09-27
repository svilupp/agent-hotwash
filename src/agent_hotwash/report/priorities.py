"""A shared action queue for human and machine reports.

Evidence-supported changes precede investigations, then visibility repairs.
Ordering is editorial, not an estimate of savings. Burdens can overlap.
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from agent_hotwash.diagnostics.expensive import ExpenseTail
    from agent_hotwash.diagnostics.tails import TailIncident
    from agent_hotwash.report.model import RunResult


class ActionExample(BaseModel):
    trace_id: str
    target_id: str
    label: str
    session_id: str | None = None
    event_idx: int | None = None


class ActionReviewContext(BaseModel):
    """Factual candidate details for a later semantic review pass."""

    observation: dict[str, str] = Field(default_factory=dict)
    proposal: dict[str, str] = Field(default_factory=dict)
    verification: dict[str, str] = Field(default_factory=dict)


class Priority(BaseModel):
    id: str
    status: Literal["supported", "investigate", "measure"]
    title: str
    owner: str
    affected_runs: int
    incidents: int
    evidence: str
    next_step: str
    verify: str
    limit: str
    observed_cost: float | None = None
    cost_basis: str | None = None
    examples: list[ActionExample] = Field(default_factory=list)
    review_context: ActionReviewContext | None = None


def metric(value: float, unit: str) -> str:
    if unit == "seconds":
        if value >= 3600:
            return f"{value / 3600:,.1f} h"
        if value >= 60:
            return f"{value / 60:,.1f} min"
        return f"{value:,.1f} s"
    return f"{value:,.0f} {unit}"


# Fixed order keeps incomparable dollars, durations and repeat counts separate.
_GUIDANCE = {
    "cache_rebuilds": (
        "Investigate repeated full cache writes",
        "Harness / prompt assembly",
        "Capture prompt-prefix fingerprints and cache policy at the linked responses. "
        "If an unchanged prefix is repeatedly written, test stable prompt scaffolding "
        "or scheduling independent work before waits.",
        "Compare cache-write tokens and total input charges per completed task; "
        "preserve task checks and completion latency.",
        "Prefix identity and expiry are unobserved. Charges overlap the work mix and are not proven savings.",
    ),
    "retry_attempts": (
        "Bound repeated recovery attempts",
        "Agent recovery policy",
        "Inspect the diagnostic between attempts. Require a changed hypothesis "
        "or a bounded stop-and-report path before retrying.",
        "Compare attempts per resolved blocker and recovery completion rate; keep necessary retries.",
        "Similar attempts can be required by changing state; repetition alone does not establish waste.",
    ),
    "tool_latency": (
        "Inspect the slowest tool operations",
        "Tool execution",
        "Open the longest call and its result. Separate useful work from external waiting, "
        "then test narrower scope, progress reporting, or a timeout.",
        "Compare p95 tool round trips for the same operation and workload, alongside successful completion rate.",
        "Call-to-result time is not CPU time or recoverable wall time; concurrent intervals can overlap.",
    ),
    "parent_wait": (
        "Check what the parent is waiting for",
        "Agent orchestration",
        "Inspect the linked wait and child return. Move independent work before the wait where dependencies permit.",
        "Compare parent blocked time and end-to-end task latency for matched workloads; retain required child results.",
        "A long blocking wait can be appropriate. Its duration alone does not justify "
        "more polling or a shorter deadline.",
    ),
    "delegation_open": (
        "Make delegation completion observable",
        "Trace collection",
        "Record explicit completion, cancellation, and parent consumption for the linked children "
        "before tuning deadlines.",
        "Reduce the share of delegations with unknown completion while preserving their full lifecycle.",
        "These durations end at capture. They are lower bounds on the observation window, not proof of a hang.",
    ),
}

_SUPPORTED = {
    "status_probes": (
        "Replace unchanged status polling",
        "Agent orchestration",
        "Compare pending-only model rounds per completed delegation and p95 completion latency.",
    ),
    "inspection_rounds": (
        "Batch independent, already named reads",
        "Agent tool use",
        "Compare inspection rounds for the same named targets; preserve dependent discovery.",
    ),
    "delegation_repetition": (
        "Carry prior findings into follow-up reviews",
        "Agent orchestration",
        "Check that earlier findings and their disposition reach the next reviewer; preserve independent review scope.",
    ),
}


# Describe the candidate's target and test plan without asserting a verdict.
# Other action types use the narrower generic projection during review.
ACTION_REVIEW_CONTEXT: dict[str, ActionReviewContext] = {
    "status_probes": ActionReviewContext(
        observation={
            "operation": "Model rounds used only to probe pending delegation progress.",
            "legitimate_alternative": "Some status checks inform a different immediate action.",
        },
        proposal={
            "target": "Use a blocking wait or completion notification for unchanged pending progress.",
            "trigger": (
                "A pending-progress probe returns no change and the tool offers a blocking wait or notification."
            ),
            "scope": "Only unchanged pending delegation progress; retain checks that inform another action.",
            "trial": "Compare the alternative on comparable delegations.",
            "rollback": "Restore progress probes if completion latency or task quality worsens.",
        },
        verification={
            "effect": "Pending-only model rounds per completed delegation.",
            "guard": "p95 completion latency and accepted task completion.",
        },
    ),
    "expense-estimated": ActionReviewContext(
        observation={
            "operation": "Whole-run charge assessment for selected costly tasks.",
            "missing_datum": "Accepted scope and outcome relative to requested work.",
            "legitimate_alternative": "A costly run may contain requested, verified work.",
        },
        proposal={
            "target": "Compare scope, deliverables, verification, and child activity before changing budgets.",
            "trigger": "A selected run has an unclear scope or outcome assessment.",
            "scope": "The linked selected runs, not all expensive tasks.",
        },
        verification={
            "effect": "Cost per accepted task on matched scope.",
            "guard": "Verification and completion rates.",
        },
    ),
    "tool_latency": ActionReviewContext(
        observation={
            "operation": "A tool call with long call-to-result elapsed time.",
            "missing_datum": "Useful work versus external waiting inside the operation.",
            "legitimate_alternative": "The call may perform necessary work or overlap concurrent work.",
        },
        proposal={
            "target": "Inspect the slowest call, then trial narrower scope, progress, or timeout only where warranted.",
            "trigger": "A matched tool operation remains slow after its result is inspected.",
            "scope": "The same operation and workload, not every tool call.",
        },
        verification={
            "effect": "p95 round-trip time for matched operations.",
            "guard": "Successful task completion rate.",
        },
    ),
    "parent_wait": ActionReviewContext(
        observation={
            "operation": "Parent waiting for a delegated child result.",
            "missing_datum": "Whether independent work was available before the linked wait.",
            "legitimate_alternative": "The parent may require the child result before continuing.",
        },
        proposal={
            "target": "Inspect dependencies and move independent work before a wait where possible.",
            "trigger": "A linked parent wait has independent work that does not depend on the child result.",
            "scope": "Matched parent-child workflows with verified independent work.",
        },
        verification={
            "effect": "Parent blocked time and end-to-end task latency.",
            "guard": "Required child results and accepted task completion.",
        },
    ),
    "cache_rebuilds": ActionReviewContext(
        observation={
            "operation": "Repeated cache-write transitions in request groups.",
            "missing_datum": "Prompt-prefix identity, expiry, and cache policy at the linked responses.",
            "legitimate_alternative": "A changed prefix or expiry may require a fresh cache write.",
        },
        proposal={
            "target": "Record prefix fingerprints, then test stable scaffolding only if an unchanged prefix repeats.",
            "trigger": "An unchanged prompt prefix is shown to be repeatedly written.",
            "scope": "The matching prompt-assembly path, not all cache writes.",
        },
        verification={
            "effect": "Cache-write tokens and input charges per completed task.",
            "guard": "Task checks and completion latency.",
        },
    ),
    "retry_attempts": ActionReviewContext(
        observation={
            "operation": "Repeated recovery attempts for a tool failure or blocker.",
            "missing_datum": "Whether diagnostic or relevant state changed between attempts.",
            "legitimate_alternative": "A retry can be necessary after the environment changes.",
        },
        proposal={
            "target": "Require a changed hypothesis or bounded stop-and-report path before repeating a failed attempt.",
            "trigger": "A subsequent retry lacks an identified change in hypothesis or observed state.",
            "scope": "The repeated failed operation, while allowing necessary retries after changed state.",
        },
        verification={
            "effect": "Attempts per resolved blocker.",
            "guard": "Recovery completion rate and necessary retries.",
        },
    ),
    "delegation_open": ActionReviewContext(
        observation={
            "operation": "Delegated child task lifecycle in the trace collector.",
            "missing_datum": (
                "A terminal completion or cancellation event and whether the parent consumed the child result."
            ),
            "legitimate_alternative": "The child may finish after trace capture; an open window does not prove a hang.",
        },
        proposal={
            "target": "Record terminal status and parent consumption for each linked child task.",
            "trigger": "A child task was delegated and linked to its parent trace.",
            "scope": "Trace collection for linked child tasks; no timeout or scheduling change.",
            "trial": "Instrument a bounded set of comparable delegated tasks.",
            "rollback": "Remove the added trace fields if they impair collection.",
        },
        verification={
            "effect": "Share of linked delegations with known terminal state and parent consumption.",
            "guard": "Preserve complete task lifecycle and accepted task outcomes.",
        },
    ),
}


def build_priorities(runs: list[RunResult], expense: ExpenseTail) -> list[Priority]:
    groups: dict[str, list[tuple[str, TailIncident]]] = defaultdict(list)
    supported: dict[str, list[tuple[str, TailIncident]]] = defaultdict(list)
    seen: set[tuple[str, str]] = set()
    for run in runs:
        for row in run.tails.incidents:
            key = (run.analysis.trace_id, row.id)
            if key in seen:
                continue
            seen.add(key)
            if row.assessment == "supported_opportunity" and row.kind in _SUPPORTED:
                supported[row.kind].append((key[0], row))
            elif (
                row.exceeds_threshold
                and row.kind in _GUIDANCE
                and row.assessment not in {"justified_repeat", "reuse_visible"}
            ):
                groups[row.kind].append((key[0], row))

    out = []

    def add(
        kind: str, members: list[tuple[str, TailIncident]], status: Literal["supported", "investigate", "measure"]
    ) -> None:
        members.sort(key=lambda pair: (-pair[1].value, pair[0], pair[1].id))
        if status == "supported":
            title, owner, verify = _SUPPORTED[kind]
            step = members[0][1].action
            limit = (
                "Supported by bounded trace evidence and JeV features. "
                "Confirm behavior on comparable tasks before rollout."
            )
        else:
            title, owner, step, verify, limit = _GUIDANCE[kind]
        example_rows = members
        cost = None
        basis = None
        largest = members[0][1]
        evidence = f"Largest observed case: {metric(largest.value, largest.unit)}."
        if kind == "status_probes" and all("pending_only_model_rounds" in row.evidence for _, row in members):
            rounds = sum(row.evidence["pending_only_model_rounds"] for _, row in members)
            unchanged = sum(row.evidence.get("unchanged_pending_progress_results", 0) for _, row in members)
            evidence = (
                f"{unchanged:,} repeated unchanged pending-progress results; "
                f"{rounds:,} model rounds requested only probes that returned pending status."
            )
        if kind == "cache_rebuilds":
            amounts = [row.evidence.get("observed_cache_write_cost") for _, row in members]
            known = [v for v in amounts if isinstance(v, (int, float))]
            cost = sum(known) if known else None
            basis = "estimated cache-write charges"
            priced = sum(row.evidence.get("priced_transitions", 0) for _, row in members)
            total = sum(row.value for _, row in members)
            evidence = (
                f"{total:,.0f} transitions in request groups crossing the threshold; {priced:,.0f} transitions priced."
            )
            # Show the costliest observed request groups, not just the most resets.
            example_rows = sorted(
                members,
                key=lambda pair: (
                    -(pair[1].evidence.get("observed_cache_write_cost") or 0),
                    -pair[1].value,
                    pair[1].id,
                ),
            )
        if kind == "delegation_open":
            evidence = f"Longest window without observed completion: at least {metric(largest.value, largest.unit)}."
        examples = []
        for tid, row in example_rows[:3]:
            label = metric(row.value, row.unit)
            if row.evidence.get("tool"):
                label = f"{row.evidence['tool']}: {label}"
            if kind == "cache_rebuilds" and row.evidence.get("observed_cache_write_cost") is not None:
                label += f"; ${row.evidence['observed_cache_write_cost']:.2f} cache-write charge"
            if kind == "delegation_open":
                label = "At least " + label
            examples.append(
                ActionExample(
                    trace_id=tid,
                    target_id=f"tail-{row.id}",
                    label=label,
                    session_id=row.session_id,
                    event_idx=row.event_indices[0] if row.event_indices else None,
                )
            )
        out.append(
            Priority(
                id=f"{status}-{kind}",
                status=status,
                title=title,
                owner=owner,
                affected_runs=len({tid for tid, _ in members}),
                incidents=len(members),
                evidence=evidence,
                next_step=step,
                verify=verify,
                limit=limit,
                observed_cost=cost,
                cost_basis=basis,
                examples=examples,
                review_context=ACTION_REVIEW_CONTEXT.get(kind),
            )
        )

    for kind in _SUPPORTED:
        if supported[kind]:
            add(kind, supported[kind], "supported")
    for kind in _GUIDANCE:
        if groups[kind]:
            add(kind, groups[kind], "measure" if kind == "delegation_open" else "investigate")

    expense_groups: dict[str, list] = defaultdict(list)
    for row in expense.runs:
        if row.assessment != "requested_verified_work_observed":
            expense_groups[row.basis].append(row)
    for basis, rows in sorted(expense_groups.items()):
        rows.sort(key=lambda r: (-r.cost, r.trace_id))
        reviewed = sum(row.assessment != "not_reviewed" for row in rows)
        out.append(
            Priority(
                id=f"investigate-expense-{basis}",
                status="investigate",
                title="Check costly runs against accepted scope",
                owner="Task review",
                affected_runs=len(rows),
                incidents=len(rows),
                evidence=f"{len(rows)} selected runs still need scope or outcome review; "
                f"{reviewed} have a review assessment.",
                next_step="Compare requested scope, accepted deliverables, verification, and child activity "
                "in the linked runs before reducing budgets.",
                verify="Compare cost per accepted task on matched scope, "
                "keeping verification and completion rates visible.",
                limit="Unreviewed or unclear scope is not evidence of unnecessary spend. "
                "Whole-run costs overlap other action cards.",
                observed_cost=sum(row.cost for row in rows),
                cost_basis=f"{basis} whole-run charges",
                review_context=ACTION_REVIEW_CONTEXT.get("expense-estimated"),
                examples=[
                    ActionExample(
                        trace_id=row.trace_id,
                        target_id=f"expense-{row.trace_id}",
                        label=f"${row.cost:.2f}: {row.assessment.replace('_', ' ')}",
                    )
                    for row in rows[:3]
                ],
            )
        )
    # Visibility repairs follow the actionable changes and investigations.
    return sorted(out, key=lambda row: {"supported": 0, "investigate": 1, "measure": 2}[row.status])
