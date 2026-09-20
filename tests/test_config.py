"""Config load + merge + accessor tests."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from agent_hotwash.config import Config, SemanticConfig, StructureConfig, TiersConfig, load_config
from agent_hotwash.events import PricingStatus


def test_defaults_load() -> None:
    cfg = load_config()
    assert cfg.smells.overlong_events == 400
    assert cfg.smells.bloated_opener_tokens == 2000
    assert cfg.analytics.idle_gap_minutes == 5.0
    assert cfg.semantic.mode == "live"
    assert cfg.lexicons.correction  # non-empty


def test_config_is_frozen() -> None:
    cfg = load_config()
    with pytest.raises(Exception):  # noqa: B017 - pydantic ValidationError on frozen set
        cfg.smells.overlong_events = 1  # type: ignore[misc]


def test_taxonomy_knobs_case_insensitive() -> None:
    cfg = load_config()
    knobs = cfg.taxonomy_knobs("EDIT_THRASH")
    assert knobs["min_edits"] == 3
    assert knobs["window_events"] == 20
    assert cfg.taxonomy_knobs("edit_thrash") == knobs
    assert cfg.taxonomy_knobs("NOPE_MISSING") == {}


def test_price_lookup_exact_prefix_default() -> None:
    cfg = load_config()
    default_input = cfg.pricing["default"].input
    exact = cfg.price_for("claude-opus-4-8")  # exact id
    prefix = cfg.price_for("claude-opus-4-8[1m]")  # only matches by prefix
    unknown = cfg.price_for("some-unknown-model")
    none_model = cfg.price_for(None)
    assert exact is not None and exact.input == 5.0 and exact.output == 25.0
    assert prefix is not None and prefix.output == 25.0  # prefix match
    assert unknown is not None and unknown.input == default_input
    assert none_model is not None and none_model.input == default_input


def test_dated_codex_price_rows_are_exact() -> None:
    cfg = load_config()
    expected = {
        "gpt-5.6-luna": (0.20, 1.20, 0.02, 0.25),
        "gpt-5.6-sol": (4.0, 20.0, 0.40, 5.0),
        "gpt-6-astra": (10.0, 50.0, 1.0, 12.5),
    }
    for model, rates in expected.items():
        entry, status = cfg.price_lookup(model)
        assert status is PricingStatus.exact, model
        assert entry is not None
        assert entry.as_of == "2026-09-19"
        assert (entry.input, entry.output, entry.cache_read, entry.cache_write) == rates


def test_dated_anthropic_price_rows_are_exact() -> None:
    cfg = load_config()
    fable51 = (10.0, 50.0, 0.25, 12.5)
    fable5 = (10.0, 50.0, 1.0, 12.5)
    opus5 = (5.0, 25.0, 0.5, 6.25)
    opus4 = (15.0, 75.0, 1.5, 18.75)
    sonnet5 = (2.0, 10.0, 0.20, 2.50)
    sonnet4 = (3.0, 15.0, 0.30, 3.75)
    haiku45 = (1.0, 5.0, 0.1, 1.25)
    haiku35 = (0.80, 4.0, 0.08, 1.0)
    expected = {
        "claude-fable-5-1": fable51,
        "claude-mythos-5-1": fable51,
        "claude-fable-5": fable5,
        "claude-mythos-5": fable5,
        "claude-opus-5": opus5,
        "claude-opus-4-8": opus5,
        "claude-opus-4-7": opus5,
        "claude-opus-4-6": opus5,
        "claude-opus-4-5": opus5,
        "claude-opus-4-5-20251101": opus5,
        "claude-opus-4-1": opus4,
        "claude-opus-4-1-20250805": opus4,
        "claude-opus-4": opus4,
        "claude-opus-4-20250514": opus4,
        "claude-sonnet-5": sonnet5,
        "claude-sonnet-4-6": sonnet4,
        "claude-sonnet-4-5": sonnet4,
        "claude-sonnet-4-5-20250929": sonnet4,
        "claude-sonnet-4": sonnet4,
        "claude-sonnet-4-20250514": sonnet4,
        "claude-haiku-4-5": haiku45,
        "claude-haiku-4-5-20251001": haiku45,
        "claude-3-5-haiku": haiku35,
        "claude-3-5-haiku-20241022": haiku35,
    }
    for model, rates in expected.items():
        entry, status = cfg.price_lookup(model)
        assert status is PricingStatus.exact, model
        assert entry is not None
        assert entry.as_of == "2026-09-20", model
        assert (entry.input, entry.output, entry.cache_read, entry.cache_write) == rates, model


def test_prefix_price_lookup_stays_estimated() -> None:
    cfg = load_config()
    # No dated row for terra; gpt-5 prefix match is estimated only.
    entry, status = cfg.price_lookup("gpt-5.6-terra")
    assert status is PricingStatus.estimated
    assert entry is not None
    assert entry.as_of is None


def test_longest_prefix_price_lookup() -> None:
    cfg = load_config()
    fable51, status51 = cfg.price_lookup("claude-fable-5-1[1m]")
    fable5, status5 = cfg.price_lookup("claude-fable-5[1m]")
    opus48, status48 = cfg.price_lookup("claude-opus-4-8[1m]")
    opus4, status4 = cfg.price_lookup("claude-opus-4[1m]")
    assert status51 is PricingStatus.estimated
    assert status5 is PricingStatus.estimated
    assert status48 is PricingStatus.estimated
    assert status4 is PricingStatus.estimated
    assert fable51 is not None and fable51.cache_read == 0.25
    assert fable5 is not None and fable5.cache_read == 1.0
    assert opus48 is not None and opus48.input == 5.0
    assert opus4 is not None and opus4.input == 15.0


def test_imported_codebench_price_entries() -> None:
    # Undated leftover from the code-bench import (per MTok USD).
    cfg = load_config()
    gpt = cfg.price_for("gpt-5.5")
    assert gpt is not None
    assert (gpt.input, gpt.output, gpt.cache_read, gpt.cache_write) == (5.0, 30.0, 0.5, 0.0)


def test_user_override_deep_merges(tmp_path: Path) -> None:
    user = tmp_path / "user.toml"
    user.write_text(
        "[smells]\noverlong_events = 42\n\n[taxonomy.edit_thrash]\nmin_edits = 9\n",
        encoding="utf-8",
    )
    cfg = load_config(user)
    # overridden
    assert cfg.smells.overlong_events == 42
    assert cfg.taxonomy_knobs("edit_thrash")["min_edits"] == 9
    # untouched siblings survive the merge
    assert cfg.smells.bloated_opener_tokens == 2000
    assert cfg.taxonomy_knobs("edit_thrash")["window_events"] == 20


def test_detectors_enable_disable_logic() -> None:
    cfg = Config.model_validate({"detectors": {"disabled": ["FOO"]}})
    assert cfg.detectors.is_enabled("BAR")
    assert not cfg.detectors.is_enabled("FOO")
    cfg2 = Config.model_validate({"detectors": {"enabled": ["ONLY_ME"]}})
    assert cfg2.detectors.is_enabled("ONLY_ME")
    assert not cfg2.detectors.is_enabled("ANYTHING_ELSE")


def test_structure_and_semantic_reject_unknown_keys() -> None:
    with pytest.raises(ValidationError):
        StructureConfig.model_validate({"not_a_real_key": 1})
    with pytest.raises(ValidationError):
        SemanticConfig.model_validate({"mode": "off", "unexpected": True})


def test_tiers_precedence_and_equal_specificity() -> None:
    tiers = TiersConfig.model_validate(
        {
            "version": 1,
            "gpt-5.6-luna:high": 3,
            "gpt-5.6-luna:*": 9,
            "gpt-5.6-*": 8,
            "default": 1,
        }
    )
    assert tiers.rank("gpt-5.6-luna", "high") == 3
    assert tiers.rank("gpt-5.6-luna", "low") == 9
    assert tiers.rank("gpt-5.6-sol", "xhigh") == 8
    assert tiers.rank("other-model", "high") == 1
    with pytest.raises(ValidationError):
        TiersConfig.model_validate({"gpt-5.6-*": 1, "GPT-5.6-*": 2})


def test_unknown_model_pricing_status_unknown() -> None:
    cfg = Config.model_validate({"pricing": {}})
    entry, status = cfg.price_lookup("totally-unknown-xyz")
    assert entry is None
    assert status is PricingStatus.unknown
