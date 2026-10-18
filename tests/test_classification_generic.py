"""Classification criteria stay generic — no local files, repos, or fingerprints."""

from __future__ import annotations

import hashlib
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src" / "agent_hotwash"
_FEATURES = _SRC / "semantic" / "features"

# Historical private identifiers are fingerprinted so the guard does not publish them.
# Each pair is (case-folded length, SHA-256); comparisons retain substring semantics.
_BANNED_FINGERPRINTS = (
    (13, "9e13c39caa954d1f154f22cbfb1b868a176189ab5772f576f5725339321ba61b"),
    (11, "66534f4f96f80d54bf5c4af2adc5550f88b87ad56b31e839b996810c02d131bc"),
    (5, "8aacd6d8996436e1843ced224d413369fa00517f2cda04e145bae3b44ec19ca2"),
    (21, "3688cf5cf3c499f057c77adab22374fecaddde967a295d77141734f1e908ab8a"),
    (7, "b05fe1cfde72b58b573db5aaf53365bac2fe38cadc35c41eab6649c482d38aad"),
    (11, "fe908ba570044dcc0f9b75d11c196b61185dc2c26d8967242e058c752cfbf578"),
    (5, "15a1d868d59c09a6d95489009db707aeae9a5aa413a9363d7b9ff825465fb58b"),
    (10, "758d12d345ec0abbf0b1818fa90cd3d407f882bca3f51908df2984f74ca70f23"),
    (10, "fbf8f523a9c45e220e953e3e1ff5bbbe7ba42d78e01c0980c43cd52c10cb669b"),
    (16, "0d66c1f7fb255ad9b3bc0e95ad7387871d4a96f606e455cfcc19b74e7172d5e4"),
    (9, "7b4b233253d61207a5b7dabaebb465b6daac4a74fde46090fa9772bde7ca34fa"),
)

_CLASSIFIER_GLOBS = (
    "primitives/commands.py",
    "primitives/errors.py",
    "structure/facts.py",
    "structure/ledger.py",
    "detectors/taxonomy.py",
    "detectors/smells.py",
    "semantic/features/*.toml",
)


def _iter_classifier_files() -> list[Path]:
    out: list[Path] = []
    for pattern in _CLASSIFIER_GLOBS:
        out.extend(sorted(_SRC.glob(pattern)))
    return out


def test_classifiers_have_no_local_specifics() -> None:
    missing = [str(p.relative_to(_ROOT)) for p in _iter_classifier_files() if not p.is_file()]
    assert not missing, missing
    for path in _iter_classifier_files():
        text = path.read_text(encoding="utf-8")
        folded = text.lower()
        for length, fingerprint in _BANNED_FINGERPRINTS:
            assert not any(
                hashlib.sha256(folded[start : start + length].encode()).hexdigest() == fingerprint
                for start in range(max(0, len(folded) - length + 1))
            ), f"{path.relative_to(_ROOT)} contains a historical private identifier"


def test_feature_examples_do_not_name_this_repo_config() -> None:
    for path in sorted(_FEATURES.glob("*.toml")):
        text = path.read_text(encoding="utf-8")
        assert "config/defaults.toml" not in text, path.name
