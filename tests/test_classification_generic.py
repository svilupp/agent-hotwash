"""Classification criteria stay generic — no local files, repos, or fingerprints."""

from __future__ import annotations

from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src" / "agent_hotwash"
_FEATURES = _SRC / "semantic" / "features"

# Tokens that identify this machine, a private repo, or a real session.
_BANNED = (
    "browser-pilot",
    "window-shop",
    "wikow",
    "architecture-practice",
    "martin@",
    "ai-fallback",
    "977.8",
    "/Users/jan",
    "Crucial X9",
    "flows/login.toml",
    "svilupp--",
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
        hits = [tok for tok in _BANNED if tok.lower() in text.lower()]
        assert not hits, f"{path.relative_to(_ROOT)} leaks {hits}"


def test_feature_examples_do_not_name_this_repo_config() -> None:
    for path in sorted(_FEATURES.glob("*.toml")):
        text = path.read_text(encoding="utf-8")
        assert "config/defaults.toml" not in text, path.name
