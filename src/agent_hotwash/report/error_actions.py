"""Bounded, renderer-independent grouping of failed results by next decision.

Each failed result has one primary action group. This is a presentation layer
on top of the supported failure leaf; it does not infer a new failure cause.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, Field

from agent_hotwash.primitives.failures import FailureRecord

if TYPE_CHECKING:
    from agent_hotwash.report.model import RunResult

ActionId = Literal[
    "expected_iteration",
    "expected_observation",
    "task_scope",
    "model_choice",
    "context_cli",
    "agent_tool_use",
    "orchestration",
    "harness_reliability",
    "product_defect",
    "unknown",
]


class ErrorActionExample(BaseModel):
    trace_id: str
    stable_id: str
    leaf: str
    owner: str
    action: str
    diagnostic: str
    command: str | None = None


class ErrorActionGroup(BaseModel):
    id: ActionId
    title: str
    next_step: str
    urgency: Literal["low", "review"]
    count: int = 0
    leaf_counts: dict[str, int] = Field(default_factory=dict)
    owner_counts: dict[str, int] = Field(default_factory=dict)
    disposition_counts: dict[str, int] = Field(default_factory=dict)
    examples: list[ErrorActionExample] = Field(default_factory=list)


class ErrorActionSummary(BaseModel):
    total: int
    unknown_count: int
    unknown_share: float
    groups: list[ErrorActionGroup] = Field(default_factory=list)


# Ordered for stable reports. Zero-count groups are omitted from results.
_GROUPS: dict[ActionId, tuple[str, str, Literal["low", "review"]]] = {
    "expected_iteration": (
        "Expected coding iteration",
        "Inspect the failing check and repair the code; compare only persistent failures after normal iteration.",
        "low",
    ),
    "expected_observation": (
        "Expected negative observations",
        "Record the negative finding or stop condition without treating it as an agent error.",
        "low",
    ),
    "task_scope": (
        "Clarify task scope or prerequisites",
        "Narrow the task or state the missing prerequisite and requested identifier explicitly.",
        "review",
    ),
    "model_choice": (
        "Consider a model change",
        "Compare models on matched tasks after repeated, attributable agent failures.",
        "review",
    ),
    "context_cli": (
        "Improve context, skills, or CLI guidance",
        "Provide the accepted command, available dependency, and current file or schema context.",
        "review",
    ),
    "agent_tool_use": (
        "Fix agent tool and edit behavior",
        "Inspect rejected calls and edits; require reading current state and changing the approach before retrying.",
        "review",
    ),
    "orchestration": (
        "Check orchestration and subagents",
        "Inspect the handoff, child result, and retry policy before changing delegation.",
        "review",
    ),
    "harness_reliability": (
        "Check harness or tool reliability",
        "Inspect the tool contract, provider status, transport, and verifier evidence.",
        "review",
    ),
    "product_defect": (
        "Fix product runtime or build failures",
        "Inspect the affected task outcome and repair the product failure before optimizing agent cost.",
        "review",
    ),
    "unknown": (
        "Inspect unresolved failures",
        "Open the sampled diagnostic and trace before choosing an intervention.",
        "review",
    ),
}

_LEAF_GROUP: dict[str, ActionId] = {
    "static_format_finding": "expected_iteration",
    "static_lint_finding": "expected_iteration",
    "static_type_finding": "expected_iteration",
    "test_assertion_failure": "expected_iteration",
    "product_build_failure": "product_defect",
    "compound_check_attributed": "expected_iteration",
    "prompt_definition_finding": "context_cli",
    "expected_no_match": "expected_observation",
    "expected_stop_precondition": "task_scope",
    "wrong_requested_identifier": "context_cli",
    "required_target_missing": "context_cli",
    "cli_invocation_rejected": "context_cli",
    "read_range_invalid": "context_cli",
    "command_unavailable": "context_cli",
    "agent_validator_sql_schema": "context_cli",
    "agent_validator_data_shape": "agent_tool_use",
    "tool_contract_rejected": "agent_tool_use",
    "edit_match_missing": "agent_tool_use",
    "post_action_verifier_mismatch": "harness_reliability",
    "external_dependency_or_transport": "harness_reliability",
    "credential_refresh_blocked": "harness_reliability",
    "iam_permission_missing": "harness_reliability",
    "product_runtime_failure": "product_defect",
}


class ErrorActionAccumulator:
    """Consume one run at a time; retain only counts and bounded examples."""

    def __init__(self, *, examples_per_group: int = 3) -> None:
        self.examples_per_group = max(0, examples_per_group)
        self._counts: Counter[ActionId] = Counter()
        self._leaves: dict[ActionId, Counter[str]] = {}
        self._owners: dict[ActionId, Counter[str]] = {}
        self._dispositions: dict[ActionId, Counter[str]] = {}
        self._examples: dict[ActionId, list[ErrorActionExample]] = {}

    def add(self, failure: FailureRecord) -> None:
        self._add_values(
            leaf=failure.leaf,
            owner=failure.owner,
            disposition=failure.disposition,
            observation=failure.observation,
            trace_id=failure.provenance.trace_id,
            stable_id=failure.provenance.stable_id,
            action=failure.action,
            diagnostic=failure.diagnostic,
            command=failure.command,
        )

    def add_compact(self, failure: Mapping[Any, Any], *, trace_id: str = "") -> None:
        """Add a failure from an incremental JSON parser, without validation."""
        provenance = failure.get("provenance") or {}
        if not isinstance(provenance, Mapping):
            provenance = {}
        self._add_values(
            leaf=str(failure.get("leaf") or "unresolved"),
            owner=str(failure.get("owner") or "unknown"),
            disposition=str(failure.get("disposition") or "unresolved"),
            observation=bool(failure.get("observation", False)),
            trace_id=str(provenance.get("trace_id") or trace_id),
            stable_id=str(provenance.get("stable_id") or ""),
            action=str(failure.get("action") or ""),
            diagnostic=str(failure.get("diagnostic") or ""),
            command=str(failure["command"]) if failure.get("command") else None,
        )

    def _add_values(
        self,
        *,
        leaf: str,
        owner: str,
        disposition: str,
        observation: bool,
        trace_id: str,
        stable_id: str,
        action: str,
        diagnostic: str,
        command: str | None,
    ) -> None:
        group = _group_for(leaf, disposition, observation)
        self._counts[group] += 1
        self._leaves.setdefault(group, Counter())[leaf] += 1
        self._owners.setdefault(group, Counter())[owner] += 1
        self._dispositions.setdefault(group, Counter())[disposition] += 1
        examples = self._examples.setdefault(group, [])
        if len(examples) < self.examples_per_group:
            examples.append(
                ErrorActionExample(
                    trace_id=trace_id,
                    stable_id=stable_id,
                    leaf=leaf,
                    owner=owner,
                    action=action[:240],
                    diagnostic=diagnostic[:240],
                    command=command[:160] if command else None,
                )
            )

    def add_run(self, run: RunResult | Mapping[str, Any]) -> None:
        if isinstance(run, Mapping):
            analysis = run.get("analysis") or {}
            if not isinstance(analysis, Mapping):
                return
            trace_id = str(analysis.get("trace_id") or "")
            failures = analysis.get("failures")
            if not isinstance(failures, list):
                return
            for failure in failures:
                if isinstance(failure, Mapping):
                    self.add_compact(failure, trace_id=trace_id)
            return
        for failure in run.analysis.failures:
            self.add(failure)

    def groups(self) -> list[ErrorActionGroup]:
        return [
            ErrorActionGroup(
                id=group,
                title=title,
                next_step=next_step,
                urgency=urgency,
                count=self._counts[group],
                leaf_counts=dict(self._leaves[group]),
                owner_counts=dict(self._owners[group]),
                disposition_counts=dict(self._dispositions[group]),
                examples=list(self._examples.get(group, [])),
            )
            for group, (title, next_step, urgency) in _GROUPS.items()
            if self._counts[group]
        ]

    def summary(self) -> ErrorActionSummary:
        total = self._counts.total()
        unknown_count = self._counts["unknown"]
        return ErrorActionSummary(
            total=total,
            unknown_count=unknown_count,
            unknown_share=unknown_count / total if total else 0.0,
            groups=self.groups(),
        )


def _group_for(leaf: str, disposition: str, observation: bool) -> ActionId:
    if leaf in {
        "static_format_finding",
        "static_lint_finding",
        "static_type_finding",
        "test_assertion_failure",
        "compound_check_attributed",
    }:
        return "expected_iteration"
    if leaf == "expected_stop_precondition":
        return "task_scope"
    if disposition == "expected" or observation:
        return "expected_observation"
    return _LEAF_GROUP.get(leaf, "unknown")


def group_failure_actions(
    rows: Iterable[tuple[str, Iterable[FailureRecord | Mapping[str, Any]]]],
) -> list[ErrorActionGroup]:
    """Group ``(trace_id, failures)`` rows without materializing all failures.

    The supplied trace id is a run key for callers; each failure's provenance is
    retained as the authoritative sample trace id.
    """
    acc = ErrorActionAccumulator()
    for trace_id, failures in rows:
        for failure in failures:
            if isinstance(failure, FailureRecord):
                acc.add(failure)
            else:
                acc.add_compact(failure, trace_id=trace_id)
    return acc.groups()


__all__ = [
    "ErrorActionAccumulator",
    "ErrorActionExample",
    "ErrorActionGroup",
    "ErrorActionSummary",
    "group_failure_actions",
]
