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


def test_prefix_price_lookup_stays_estimated() -> None:
    cfg = load_config()
    # No dated row for terra; gpt-5 prefix match is estimated only.
    entry, status = cfg.price_lookup("gpt-5.6-terra")
    assert status is PricingStatus.estimated
    assert entry is not None
    assert entry.as_of is None


def test_imported_codebench_price_entries() -> None:
    # Real rates imported from code-bench pricing.toml (per MTok USD).
    cfg = load_config()
    fable = cfg.price_for("claude-fable-5")
    assert fable is not None
    assert (fable.input, fable.output, fable.cache_read, fable.cache_write) == (10.0, 50.0, 1.0, 12.5)

    sonnet = cfg.price_for("claude-sonnet-5")
    assert sonnet is not None
    assert (sonnet.input, sonnet.output) == (3.0, 15.0)

    # gpt-5.5 has no cache-write class in the source -> defaults to 0.0.
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
