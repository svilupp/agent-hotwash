"""Effort-tier policy lookup. Model class and effort stay separate (C11)."""

from __future__ import annotations

from agent_hotwash.config import Config, TiersConfig

HIGH_EFFORT_NAMES = frozenset({"high", "xhigh", "max"})


def model_class(model: str | None) -> str | None:
    """Model identity without effort mixed in."""
    if not model:
        return None
    return model.split(":")[0] if ":" in model else model


def effort_rank(model: str | None, effort: str | None, config: Config | TiersConfig) -> int | None:
    """Look up ``config.tiers.rank(model, effort)``."""
    tiers = config if isinstance(config, TiersConfig) else config.tiers
    return tiers.rank(model, effort)


def is_high_effort(model: str | None, effort: str | None, config: Config) -> bool:
    """True when the selected effort is at or above the policy's ``high`` rank."""
    observed = effort_rank(model, effort, config)
    if observed is None:
        return (effort or "").lower() in HIGH_EFFORT_NAMES
    high = effort_rank(model, "high", config)
    if high is not None:
        return observed >= high
    return observed >= 3


__all__ = ["HIGH_EFFORT_NAMES", "effort_rank", "is_high_effort", "model_class"]
