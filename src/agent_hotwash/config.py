"""Frozen configuration model + loader.

The core reads a single immutable :class:`Config`; it never reaches for globals.
Every threshold, lexicon and price lives in ``config/defaults.toml``;
``load_config`` deep-merges an optional user TOML over those defaults and
validates the result.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

# Repo-root default config. config.py lives at src/agent_hotwash/config.py, so
# three parents up is the repo root.
_DEFAULTS_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "defaults.toml"


class SmellsConfig(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    bloated_opener_tokens: int = 2000
    thin_prompt_words: int = 8
    cold_start_reads: int = 8
    slow_to_action_events: int = 6
    overlong_events: int = 400
    overlong_tokens: int = 150_000
    overlong_minutes: int = 90
    context_bloat_pct: float = 0.70
    tool_monoculture_pct: float = 0.80
    linear_scan_reads: int = 10
    no_plan_dive_events: int = 4
    idle_gap_minutes: float = 5.0


class AnalyticsConfig(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    idle_gap_minutes: float = 5.0
    output_truncate_chars: int = 2000
    context_window_tokens: int = 200_000


class PriceEntry(BaseModel):
    """Per-MTok USD price for a model (used to estimate cost)."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    input: float = 0.0
    output: float = 0.0
    cache_read: float = 0.0
    cache_write: float = 0.0


class LexiconConfig(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    correction: list[str] = Field(default_factory=list)
    completion: list[str] = Field(default_factory=list)
    positive: list[str] = Field(default_factory=list)
    task_intro: list[str] = Field(default_factory=list)
    interrogative: list[str] = Field(default_factory=list)
    secret: list[str] = Field(default_factory=list)  # raw regex patterns


class DetectorsConfig(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    # When non-empty, ONLY these ids run. Otherwise everything not in `disabled`.
    enabled: list[str] = Field(default_factory=list)
    disabled: list[str] = Field(default_factory=list)
    # id -> severity override (e.g. {"EDIT_THRASH": "high"})
    severity: dict[str, str] = Field(default_factory=dict)

    def is_enabled(self, detector_id: str) -> bool:
        if self.enabled:
            return detector_id in self.enabled and detector_id not in self.disabled
        return detector_id not in self.disabled


class Config(BaseModel):
    """The one frozen config object the whole pipeline reads."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    smells: SmellsConfig = Field(default_factory=SmellsConfig)
    # Per-detector knobs keyed by taxonomy id (kept as dicts: ~28 detectors, each
    # with its own free-form knob set). Detectors read via `taxonomy_knobs`.
    taxonomy: dict[str, dict[str, Any]] = Field(default_factory=dict)
    lexicons: LexiconConfig = Field(default_factory=LexiconConfig)
    pricing: dict[str, PriceEntry] = Field(default_factory=dict)
    analytics: AnalyticsConfig = Field(default_factory=AnalyticsConfig)
    detectors: DetectorsConfig = Field(default_factory=DetectorsConfig)

    def taxonomy_knobs(self, detector_id: str) -> dict[str, Any]:
        """Knob dict for a taxonomy detector (empty if none configured).

        Lookup is case-insensitive on the id so ``[taxonomy.edit_thrash]`` in
        TOML serves the ``EDIT_THRASH`` detector.
        """
        if detector_id in self.taxonomy:
            return self.taxonomy[detector_id]
        return self.taxonomy.get(detector_id.lower(), {})

    def price_for(self, model: str | None) -> PriceEntry | None:
        """Price entry for a model name, trying exact then prefix match, then a
        ``default`` entry if present."""
        if model:
            if model in self.pricing:
                return self.pricing[model]
            for key, entry in self.pricing.items():
                if key != "default" and model.startswith(key):
                    return entry
        return self.pricing.get("default")


def _deep_merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``over`` onto a copy of ``base`` (dicts merge, scalars
    and lists replace)."""
    out = dict(base)
    for k, v in over.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _read_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as fh:
        return tomllib.load(fh)


def load_config(path: Path | None = None) -> Config:
    """Load the frozen config: defaults merged with an optional user TOML."""
    data = _read_toml(_DEFAULTS_PATH)
    if path is not None:
        data = _deep_merge(data, _read_toml(Path(path)))
    return Config.model_validate(data)


__all__ = [
    "AnalyticsConfig",
    "Config",
    "DetectorsConfig",
    "LexiconConfig",
    "PriceEntry",
    "SmellsConfig",
    "load_config",
]
