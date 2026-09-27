"""Classifier coverage distinguishes missing answers from uncertain answers."""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

from pydantic import BaseModel

if TYPE_CHECKING:
    from agent_hotwash.report.model import RunResult
    from agent_hotwash.semantic.results import FeatureValue


class ClassifierCoverage(BaseModel):
    objects: int = 0
    fully_answered_objects: int = 0
    questions: int = 0
    answered: int = 0
    uncertain: int = 0
    api_errors: int = 0
    unavailable: int = 0

    def summary(self) -> str:
        return (
            f"{self.answered}/{self.questions} feature answers received "
            f"({self.uncertain} uncertain); {self.api_errors} API errors; "
            f"{self.unavailable} unavailable. "
            f"{self.fully_answered_objects}/{self.objects} cases have every selected feature answered."
        )


def classifier_coverage(runs: list[RunResult]) -> dict[str, ClassifierCoverage]:
    # Deduplicate repeated run/feature records, but keep scopes and objects distinct.
    objects: dict[tuple[str, str, str], dict[str, FeatureValue]] = defaultdict(dict)
    for run in runs:
        for fs in run.features or []:
            for value in fs.values.values():
                if value.source == "jev":
                    objects[(fs.scope, run.analysis.trace_id, fs.object_id)][value.id] = value
    out: dict[str, ClassifierCoverage] = {}
    for (scope, _, _), values in objects.items():
        coverage = out.setdefault(scope, ClassifierCoverage())
        answered = sum(v.value is not None and v.reason in {None, "low_support"} for v in values.values())
        errors = sum(v.reason == "api_error" for v in values.values())
        coverage.objects += 1
        coverage.fully_answered_objects += int(answered == len(values))
        coverage.questions += len(values)
        coverage.answered += answered
        coverage.uncertain += sum(
            v.value is not None and v.reason in {None, "low_support"} and v.abstains for v in values.values()
        )
        coverage.api_errors += errors
        coverage.unavailable += len(values) - answered - errors
    return dict(sorted(out.items()))
