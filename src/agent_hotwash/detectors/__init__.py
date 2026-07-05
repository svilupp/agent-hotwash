"""Detector package: registry + rule-tier smell and taxonomy detectors.

Importing this package registers every detector as a side effect (the ``smells``
and ``taxonomy`` modules run their ``@detector`` decorators at import time), so
``run_detectors`` and ``get_registry`` see the full set without any explicit
registration call.
"""

from __future__ import annotations

from agent_hotwash.detectors import smells, taxonomy
from agent_hotwash.detectors.registry import (
    DetectorSpec,
    Finding,
    Severity,
    SpanRef,
    detector,
    get_registry,
    make_finding,
    run_detectors,
    severity_rank,
    span,
)

__all__ = [
    "DetectorSpec",
    "Finding",
    "Severity",
    "SpanRef",
    "detector",
    "get_registry",
    "make_finding",
    "run_detectors",
    "severity_rank",
    "smells",
    "span",
    "taxonomy",
]
