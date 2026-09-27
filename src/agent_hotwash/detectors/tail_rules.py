"""Registry-backed findings for measured execution extremes."""

from __future__ import annotations

from agent_hotwash.config import Config
from agent_hotwash.detectors.registry import Finding, Severity, SpanRef, trace_tail_detector
from agent_hotwash.diagnostics.tails import TailAnalysis
from agent_hotwash.events import Trace

_SEVERITIES = {
    "tool_latency": Severity.low,
    "parent_wait": Severity.low,
    "delegation_lifetime": Severity.info,
    "delegation_open": Severity.info,
    "retry_attempts": Severity.medium,
    "failure_chain": Severity.medium,
    "output_volume": Severity.low,
    "poll_amplification": Severity.low,
    "model_input": Severity.low,
    "model_output": Severity.low,
    "delegation_fanout": Severity.low,
    "context_replay": Severity.low,
    "cache_creation": Severity.low,
    "status_probes": Severity.low,
    "inspection_rounds": Severity.info,
    "output_repetition": Severity.low,
    "delegation_repetition": Severity.low,
    "verification_repetition": Severity.low,
    "cache_rebuilds": Severity.low,
}


def _register(kind: str, severity: Severity) -> None:
    detector_id = f"TAIL_{kind.upper()}"

    @trace_tail_detector(detector_id, severity=severity, confidence="low" if kind == "delegation_open" else "high")
    def detect(_trace: Trace, config: Config, tails: TailAnalysis) -> list[Finding]:
        """Report a threshold-crossing execution observation with its measurement."""
        if not config.detectors.is_enabled(detector_id):
            return []
        return [
            Finding(
                id=detector_id,
                kind="observation",
                severity=severity,
                confidence="low" if kind == "delegation_open" else "high",
                session_id=row.session_id,
                spans=[
                    SpanRef(
                        session_id=row.session_id,
                        event_idx=row.event_indices[0],
                        end_idx=row.event_indices[-1] if len(row.event_indices) > 1 else None,
                    )
                ],
                evidence={
                    "incident_id": row.id,
                    "value": row.value,
                    "unit": row.unit,
                    "threshold": row.threshold,
                    "cohort": row.cohort,
                    "cohort_n": row.cohort_n,
                    "median_ratio": row.median_ratio,
                    "sources": [source.model_dump() for source in row.sources],
                    **row.evidence,
                },
                message=f"{row.label}: {row.value:,.1f} {row.unit}. {row.action}",
            )
            for row in tails.incidents
            if row.kind == kind and row.exceeds_threshold
        ]


for _kind, _severity in _SEVERITIES.items():
    _register(_kind, _severity)
