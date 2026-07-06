"""Report data model.

A :class:`Report` bundles everything a run produced — one :class:`RunResult`
per analyzed trace (its :class:`Analysis` plus the :class:`Finding` list), the
cross-run :class:`Aggregate`, a finding histogram keyed by finding *id* across
all runs, and reproducibility metadata (tool version, config path, timestamp,
input paths). Writers (json/csv/table/html) render this one object; they never
re-run the pipeline.
"""

from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from agent_hotwash.aggregate import Aggregate, aggregate
from agent_hotwash.analytics import Analysis
from agent_hotwash.detectors.registry import Finding, Severity, severity_rank

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence


class RunResult(BaseModel):
    """One analyzed trace: its analytics plus the findings detectors emitted."""

    analysis: Analysis
    findings: list[Finding] = Field(default_factory=list)

    @property
    def finding_histogram(self) -> dict[str, int]:
        """Count of findings by id within this run."""
        c: Counter[str] = Counter(f.id for f in self.findings)
        return dict(c)

    def max_severity(self) -> Severity | None:
        """Highest finding severity in this run (``None`` if no findings)."""
        if not self.findings:
            return None
        return max((f.severity for f in self.findings), key=severity_rank)


class ReportMeta(BaseModel):
    """Reproducibility metadata: how and when this report was produced."""

    tool_version: str
    generated_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    config_path: str | None = None
    inputs: list[str] = Field(default_factory=list)
    detectors_enabled: bool = True


class Report(BaseModel):
    """The full render surface: run results + aggregate + finding histograms."""

    meta: ReportMeta
    runs: list[RunResult] = Field(default_factory=list)
    aggregate: Aggregate = Field(default_factory=Aggregate)

    # Finding id -> total count across all runs.
    finding_histogram: dict[str, int] = Field(default_factory=dict)
    # Finding id -> {severity: count} across all runs.
    finding_severity: dict[str, dict[str, int]] = Field(default_factory=dict)

    @classmethod
    def build(
        cls,
        runs: Sequence[RunResult],
        meta: ReportMeta,
    ) -> Report:
        """Assemble a Report from run results, computing the aggregate and the
        cross-run finding histograms."""
        runs = list(runs)
        agg = aggregate([r.analysis for r in runs])

        hist: Counter[str] = Counter()
        sev: dict[str, Counter[str]] = {}
        for r in runs:
            for f in r.findings:
                hist[f.id] += 1
                sev.setdefault(f.id, Counter())[f.severity.value] += 1

        return cls(
            meta=meta,
            runs=runs,
            aggregate=agg,
            finding_histogram=dict(hist.most_common()),
            finding_severity={k: dict(v) for k, v in sev.items()},
        )

    def total_findings(self) -> int:
        return sum(self.finding_histogram.values())

    def max_severity(self) -> Severity | None:
        sevs = [f.severity for r in self.runs for f in r.findings]
        return max(sevs, key=severity_rank) if sevs else None


def findings_at_or_above(findings: Iterable[Finding], threshold: Severity) -> list[Finding]:
    """Findings whose severity is >= ``threshold`` (for the ``--fail-on`` gate)."""
    t = severity_rank(threshold)
    return [f for f in findings if severity_rank(f.severity) >= t]


__all__ = [
    "Report",
    "ReportMeta",
    "RunResult",
    "findings_at_or_above",
]
