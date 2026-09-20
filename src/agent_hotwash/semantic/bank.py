"""Declarative JeV feature bank loader and validator (§6.1)."""

from __future__ import annotations

import hashlib
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agent_hotwash.events import CAPABILITY_FIELDS
from agent_hotwash.semantic.jev import canonical_json

_ID_RE = re.compile(r"^(task|turn|episode)\.[a-z_]+\.[a-z_]+$")
_FEATURES_DIR = Path(__file__).resolve().parent / "features"
_SENTENCE_RE = re.compile(r"[.?!]")
_CAPABILITY_SET = frozenset(CAPABILITY_FIELDS)


@dataclass
class FeatureDef:
    """One feature in the milestone bank."""

    id: str
    scope: str
    primitive: str
    version: int = 1
    requires: list[str] = field(default_factory=list)
    question: str = ""
    inspect: list[str] = field(default_factory=list)
    compare: list[str] = field(default_factory=list)
    focus: str = ""
    criteria: dict[str, Any] = field(default_factory=dict)
    options: list[str] = field(default_factory=list)
    levels: list[str] = field(default_factory=list)


_BACKTICK_RE = re.compile(r"`([^`]+)`")
_DEFAULT_INSPECT = {
    "task": ["task.request"],
    "episode": ["episode.ops", "episode.messages", "task.request"],
    "turn": ["messages[0].text", "ledger.deliverables", "ledger.last_answer", "ledger.artifacts"],
}
_DEFAULT_FOCUS = {
    "task": (
        "Judge only the inspect paths. The required output is `task.request` "
        "(plus `task.amendments` / `task.deliverables` when those paths are named). "
        "Do not infer the request from later episode ops or messages."
    ),
    "episode": ("Judge only this episode from the inspect paths. Do not relabel task intent from ops."),
    "turn": (
        "Compare `messages[0].text` only to the named `ledger` fields. Same-topic wording is not a named referent."
    ),
}


def criteria_hash(feature: FeatureDef) -> str:
    """Stable hash of criterion text; bump ``version`` when this changes."""
    payload = {
        "id": feature.id,
        "question": feature.question,
        "criteria": feature.criteria,
        "primitive": feature.primitive,
        "inspect": feature.inspect,
        "compare": feature.compare,
        "focus": feature.focus,
    }
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def _as_str_list(val: Any) -> list[str]:
    if val is None:
        return []
    if isinstance(val, str):
        return [val] if val.strip() else []
    if isinstance(val, list):
        return [str(x) for x in val if str(x).strip()]
    return []


def _unique_paths(paths: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for raw in paths:
        path = raw.strip().strip("`")
        if path and path not in seen:
            seen.add(path)
            out.append(path)
    return out


def inspect_paths(feat: FeatureDef) -> list[str]:
    """Backticked JSON paths JeV should read on the state object."""
    raw = list(feat.inspect) or _BACKTICK_RE.findall(feat.question) or list(_DEFAULT_INSPECT.get(feat.scope, []))
    return [f"`{p}`" for p in _unique_paths(raw)]


def question_instructions(feat: FeatureDef) -> dict[str, Any]:
    """JeV ``instructions`` object: ``{question, inspect, focus, compare?}``."""
    inspect = inspect_paths(feat)
    body: dict[str, Any] = {"question": feat.question}
    if inspect:
        body["inspect"] = inspect[0] if len(inspect) == 1 else inspect
    compare = [f"`{p}`" for p in _unique_paths(list(feat.compare))]
    if compare:
        body["compare"] = compare
    focus = feat.focus.strip() if feat.focus else _DEFAULT_FOCUS.get(feat.scope, "")
    if focus:
        body["focus"] = focus
    return body


def _criterion_ok(name: str, block: Any, *, feature_id: str) -> None:
    if not isinstance(block, dict):
        raise ValueError(f"{feature_id}: criterion {name!r} must be a table")
    what = block.get("what")
    if not isinstance(what, str) or not what.strip():
        raise ValueError(f"{feature_id}: criterion {name!r} needs a `what` sentence")
    if _SENTENCE_RE.search(what) is None:
        raise ValueError(f"{feature_id}: criterion {name!r} `what` must be a sentence")
    examples = block.get("examples")
    if not isinstance(examples, list) or len(examples) < 2:
        raise ValueError(f"{feature_id}: criterion {name!r} needs ≥ 2 examples")
    if any(not isinstance(ex, str) or not ex.strip() for ex in examples):
        raise ValueError(f"{feature_id}: criterion {name!r} examples must be strings")


def validate_bank(features: list[FeatureDef]) -> None:
    """Raise ``ValueError`` if the bank violates §6.1 rules."""
    seen: set[str] = set()
    for feat in features:
        if not _ID_RE.match(feat.id):
            raise ValueError(f"invalid feature id {feat.id!r}")
        if feat.id in seen:
            raise ValueError(f"duplicate feature id {feat.id!r}")
        seen.add(feat.id)
        if feat.scope not in {"task", "turn", "episode"}:
            raise ValueError(f"{feat.id}: invalid scope {feat.scope!r}")
        if feat.primitive not in {"noul", "choice", "score"}:
            raise ValueError(f"{feat.id}: invalid primitive {feat.primitive!r}")
        extra = [r for r in feat.requires if r not in _CAPABILITY_SET]
        if extra:
            raise ValueError(f"{feat.id}: unknown requires {extra}")
        if not feat.criteria:
            raise ValueError(f"{feat.id}: criteria missing")
        for name, block in feat.criteria.items():
            _criterion_ok(str(name), block, feature_id=feat.id)
        if feat.primitive == "choice":
            options = feat.options or list(feat.criteria.keys())
            if "other" not in options:
                raise ValueError(f"{feat.id}: choice must include `other`")
            if len(options) > 8:
                raise ValueError(f"{feat.id}: choice has {len(options)} options; max 8 including other")
        if feat.primitive == "score":
            levels = feat.levels or list(feat.criteria.keys())
            if not 3 <= len(levels) <= 6:
                raise ValueError(f"{feat.id}: score needs 3-6 levels, got {len(levels)}")


def _from_mapping(item: dict[str, Any]) -> FeatureDef:
    criteria = item.get("criteria") or {}
    if not isinstance(criteria, dict):
        criteria = {}
    # TOML may produce non-str keys; normalise.
    criteria = {str(k): v for k, v in criteria.items()}
    primitive = str(item.get("primitive") or "noul")
    options = list(criteria.keys()) if primitive == "choice" else []
    levels = list(criteria.keys()) if primitive == "score" else []
    requires = item.get("requires") or []
    if not isinstance(requires, list):
        requires = []
    return FeatureDef(
        id=str(item.get("id") or ""),
        scope=str(item.get("scope") or ""),
        primitive=primitive,
        version=int(item.get("version") or 1),
        requires=[str(r) for r in requires],
        question=str(item.get("question") or ""),
        inspect=_as_str_list(item.get("inspect")),
        compare=_as_str_list(item.get("compare")),
        focus=str(item.get("focus") or ""),
        criteria=criteria,
        options=options,
        levels=levels,
    )


def load_bank(path: Path | None = None) -> list[FeatureDef]:
    """Load ``semantic/features/*.toml`` and validate."""
    root = path or _FEATURES_DIR
    features: list[FeatureDef] = []
    for toml_path in sorted(root.glob("*.toml")):
        with toml_path.open("rb") as fh:
            data = tomllib.load(fh)
        rows = data.get("feature") or []
        if isinstance(rows, dict):
            rows = [rows]
        for item in rows:
            if isinstance(item, dict):
                features.append(_from_mapping(item))
    validate_bank(features)
    return features


__all__ = [
    "FeatureDef",
    "criteria_hash",
    "inspect_paths",
    "load_bank",
    "question_instructions",
    "validate_bank",
]
