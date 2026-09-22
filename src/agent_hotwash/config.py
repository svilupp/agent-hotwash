"""Frozen configuration model + loader.

The core reads a single immutable :class:`Config`; it never reaches for globals.
Every threshold, lexicon and price lives in ``config/defaults.toml``;
``load_config`` deep-merges an optional user TOML over those defaults and
validates the result.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from agent_hotwash.events import PricingStatus

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
    """Per-MTok USD price for a model (used to estimate cost).

    An ``as_of`` date marks an exact dated row; without it (or when the entry
    was reached via prefix/default fallback) monetary diagnoses stay disabled
    and totals are labelled ``estimated``.
    """

    model_config = ConfigDict(extra="ignore", frozen=True)

    input: float = 0.0
    output: float = 0.0
    cache_read: float = 0.0
    cache_write: float = 0.0
    as_of: str | None = None  # ISO date; required for pricing_status=exact


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


class EpisodeStructureConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    max_model_calls: int = 6
    reads_before_edit_boundary: int = 2
    idle_gap_minutes: float = 5.0
    state_token_budget: int = 20_000


class StructureConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    injected_tags_user: list[str] = Field(default_factory=lambda: ["<recommended_plugins>", "<environment_context>"])
    delegation_tag: str = "<codex_delegation>"
    episodes: EpisodeStructureConfig = Field(default_factory=EpisodeStructureConfig)


class SemanticConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    mode: Literal["off", "cached", "live"] = "live"
    model: str = "jev-1.13.0"
    cache_dir: str = "~/.cache/agent-hotwash/systemone"
    max_questions_per_request: int = 15
    redact: bool = True
    allow_unredacted: bool = False  # live refuses unless this override is set
    # Global live-mode budget: 1200 requests/min (20 req/s), 250k tokens/s.
    requests_per_second: float = 20.0  # <= 0 disables client-side limiting
    burst: int = Field(default=20, ge=1)  # token-bucket depth
    max_concurrency: int = Field(default=12, ge=1)  # concurrent in-flight requests per process
    max_retries: int = Field(default=3, ge=0)  # on 429 / 5xx, exponential backoff (Retry-After honoured)
    timeout_s: float = Field(default=20.0, gt=0)


class DiagnosticsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    context_pressure_pct: float = 0.6
    min_support: int = 20
    timezone: str = "UTC"


def _tier_identity(key: str) -> tuple[str, str]:
    """Specificity class + normalised identity for a ``[tiers]`` key."""
    k = key.lower()
    if k == "default":
        return ("default", "default")
    if k.endswith(":*"):
        return ("model_star", k[:-2])
    if "*" in k:
        return ("glob", k)
    if ":" in k:
        return ("exact", k)
    return ("model", k)


class TiersConfig(BaseModel):
    """Versioned POLICY map of model+effort → integer rank (C11).

    Lookup precedence: exact ``model:effort`` > ``model:*`` > anchored glob
    (``prefix*``) > ``default``. Equal specificity is rejected at load time.
    """

    model_config = ConfigDict(extra="allow", frozen=True)

    version: int = 1

    def _ranks(self) -> dict[str, int]:
        data = self.model_dump()
        data.pop("version", None)
        out: dict[str, int] = {}
        for key, val in data.items():
            if isinstance(val, int):
                out[str(key)] = val
        return out

    @model_validator(mode="after")
    def _reject_equal_specificity(self) -> TiersConfig:
        ranks = self._ranks()
        seen: dict[tuple[str, str], str] = {}
        for key in ranks:
            slot = _tier_identity(key)
            if slot in seen and seen[slot] != key:
                raise ValueError(f"tiers keys {seen[slot]!r} and {key!r} have equal specificity")
            seen[slot] = key
        return self

    def rank(self, model: str | None, effort: str | None) -> int | None:
        """Resolve a rank, or ``None`` when nothing matches."""
        ranks = self._ranks()
        if not model:
            return ranks.get("default")
        effort_key = f"{model}:{effort}" if effort else None
        if effort_key and effort_key in ranks:
            return ranks[effort_key]
        # Case-insensitive exact / model:* fallback (keys are stored as written).
        ranks_l = {k.lower(): v for k, v in ranks.items()}
        if effort_key and effort_key.lower() in ranks_l:
            return ranks_l[effort_key.lower()]
        star = f"{model}:*"
        if star in ranks:
            return ranks[star]
        if star.lower() in ranks_l:
            return ranks_l[star.lower()]
        # Anchored globs: "gpt-5.6-*" matches gpt-5.6-luna; not unanchored.
        matches: list[tuple[int, int]] = []  # (specificity, rank)
        for key, val in ranks.items():
            if key in ("default",) or ":" in key:
                continue
            if "*" not in key:
                continue
            pattern = re.escape(key).replace(r"\*", ".*")
            if re.fullmatch(pattern, model):
                spec = len(key.replace("*", ""))
                matches.append((spec, val))
        if matches:
            matches.sort(key=lambda x: -x[0])
            if len(matches) > 1 and matches[0][0] == matches[1][0] and matches[0][1] != matches[1][1]:
                raise ValueError(f"ambiguous tiers glob match for model {model!r}")
            return matches[0][1]
        return ranks.get("default")


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
    structure: StructureConfig = Field(default_factory=StructureConfig)
    semantic: SemanticConfig = Field(default_factory=SemanticConfig)
    diagnostics: DiagnosticsConfig = Field(default_factory=DiagnosticsConfig)
    tiers: TiersConfig = Field(default_factory=TiersConfig)

    def taxonomy_knobs(self, detector_id: str) -> dict[str, Any]:
        """Knob dict for a taxonomy detector (empty if none configured).

        Lookup is case-insensitive on the id so ``[taxonomy.edit_thrash]`` in
        TOML serves the ``EDIT_THRASH`` detector.
        """
        if detector_id in self.taxonomy:
            return self.taxonomy[detector_id]
        return self.taxonomy.get(detector_id.lower(), {})

    def price_for(self, model: str | None) -> PriceEntry | None:
        """Price entry for a model name: exact key, then longest prefix, then ``default``."""
        entry, _status = self.price_lookup(model)
        return entry

    def price_lookup(self, model: str | None) -> tuple[PriceEntry | None, PricingStatus]:
        """Return ``(entry, pricing_status)``.

        * exact named row with ``as_of`` → ``exact``
        * exact named row without ``as_of``, or prefix/default fallback → ``estimated``
        * prefix fallback uses the longest matching key
        * nothing matches (not even default) → ``unknown``
        """
        if model and model in self.pricing:
            entry = self.pricing[model]
            status = PricingStatus.exact if entry.as_of else PricingStatus.estimated
            return entry, status
        if model:
            # Longest prefix wins so ``claude-fable-5-1[1m]`` does not take the
            # ``claude-fable-5`` row (and ``claude-opus-4`` does not steal 4.x).
            best: PriceEntry | None = None
            best_len = -1
            for key, entry in self.pricing.items():
                if key != "default" and model.startswith(key) and len(key) > best_len:
                    best = entry
                    best_len = len(key)
            if best is not None:
                return best, PricingStatus.estimated
        default = self.pricing.get("default")
        if default is not None:
            return default, PricingStatus.estimated
        return None, PricingStatus.unknown


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
    "DiagnosticsConfig",
    "EpisodeStructureConfig",
    "LexiconConfig",
    "PriceEntry",
    "SemanticConfig",
    "SmellsConfig",
    "StructureConfig",
    "TiersConfig",
    "load_config",
]
