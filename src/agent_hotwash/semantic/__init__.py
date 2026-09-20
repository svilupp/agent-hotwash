"""Harness-blind semantic layer: System One asker, bank, redaction, derived features."""

from agent_hotwash.semantic.bank import FeatureDef, criteria_hash, load_bank, validate_bank
from agent_hotwash.semantic.client import CacheMissError, SystemOneAsker
from agent_hotwash.semantic.redact import REDACTION_VERSION, redact_state
from agent_hotwash.semantic.results import (
    FeatureSet,
    FeatureValue,
    declared_success_without_observed_verification,
    investigation_then_change,
    recovered,
    stuck_window,
    thrashing_window,
)

redact = redact_state

__all__ = [
    "REDACTION_VERSION",
    "CacheMissError",
    "FeatureDef",
    "FeatureSet",
    "FeatureValue",
    "SystemOneAsker",
    "criteria_hash",
    "declared_success_without_observed_verification",
    "investigation_then_change",
    "load_bank",
    "recovered",
    "redact",
    "redact_state",
    "stuck_window",
    "thrashing_window",
    "validate_bank",
]
