"""Feature bank overlay on native System One definitions."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from systemoneprompts import check_definition, load_definition
from systemoneprompts import wire_questions as _wire_questions
from systemoneprompts.diagnostics import SystemOnePromptsError, errors_of
from systemoneprompts.json_values import canonical_json

from agent_hotwash.events import CAPABILITY_FIELDS

_ID_RE = re.compile(r"^(task|turn|episode)\.[a-z_]+\.[a-z_]+$")
_FEATURES_DIR = Path(__file__).resolve().parent / "features"
_SCOPE_FILES = ("task.toml", "episode.toml", "turn.toml")
_SENTENCE_RE = re.compile(r"[.?!]")
_CAPABILITY_SET = frozenset(CAPABILITY_FIELDS)


@dataclass
class FeatureDef:
    """One bank feature: native System One question plus hotwash overlay."""

    id: str
    scope: str
    primitive: str
    version: int = 1
    requires: list[str] = field(default_factory=list)
    question: Any = field(default_factory=dict)
    criteria: Any = field(default_factory=dict)
    options: list[str] = field(default_factory=list)
    levels: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if isinstance(self.question, str):
            self.question = {
                "type": self.primitive,
                "instructions": {"question": self.question},
                "criteria": self.criteria,
            }
        elif not self.question:
            self.question = {
                "type": self.primitive,
                "instructions": {"question": ""},
                "criteria": self.criteria,
            }

    @property
    def question_text(self) -> str:
        ins = self.question.get("instructions") if isinstance(self.question, dict) else None
        if isinstance(ins, dict):
            return str(ins.get("question") or "")
        return str(ins or "")

    @property
    def inspect(self) -> list[str]:
        return _path_list(_instructions(self).get("inspect"))

    @property
    def compare(self) -> list[str]:
        return _path_list(_instructions(self).get("compare"))

    @property
    def focus(self) -> str:
        focus = _instructions(self).get("focus")
        return str(focus).strip() if isinstance(focus, str) else ""


def _instructions(feat: FeatureDef) -> dict[str, Any]:
    ins = feat.question.get("instructions") if isinstance(feat.question, dict) else None
    return ins if isinstance(ins, dict) else {}


def _path_list(raw: Any) -> list[str]:
    if raw is None:
        return []
    items = [raw] if isinstance(raw, str) else raw if isinstance(raw, list) else []
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        path = str(item).strip().strip("`")
        if path and path not in seen:
            seen.add(path)
            out.append(path)
    return out


def criteria_hash(feature: FeatureDef) -> str:
    """Stable hash of criterion text; bump ``version`` when this changes."""
    payload = {
        "id": feature.id,
        "question": feature.question_text,
        "criteria": feature.criteria,
        "primitive": feature.primitive,
        "inspect": feature.inspect,
        "compare": feature.compare,
        "focus": feature.focus,
    }
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


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


def _criteria_blocks(feat: FeatureDef) -> list[tuple[str, Any]]:
    criteria = feat.criteria
    if isinstance(criteria, list):
        return [(str(block.get("name") if isinstance(block, dict) else i), block) for i, block in enumerate(criteria)]
    if isinstance(criteria, dict):
        return [(str(name), block) for name, block in criteria.items()]
    return []


def validate_bank(features: list[FeatureDef], *, routing: dict[str, list[str]] | None = None) -> None:
    """Raise ``ValueError`` if the bank violates hotwash overlay rules."""
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
        blocks = _criteria_blocks(feat)
        if not blocks:
            raise ValueError(f"{feat.id}: criteria missing")
        for name, block in blocks:
            _criterion_ok(name, block, feature_id=feat.id)
        if feat.primitive == "choice":
            options = feat.options or [name for name, _ in blocks]
            if "other" not in options:
                raise ValueError(f"{feat.id}: choice must include `other`")
            if len(options) > 8:
                raise ValueError(f"{feat.id}: choice has {len(options)} options; max 8 including other")
        if feat.primitive == "score":
            levels = feat.levels or [name for name, _ in blocks]
            if not 3 <= len(levels) <= 6:
                raise ValueError(f"{feat.id}: score needs 3-6 levels, got {len(levels)}")
    if routing:
        for super_id, children in routing.items():
            if super_id not in seen:
                raise ValueError(f"routing key {super_id!r} is not a question")
            for child in children:
                if child not in seen:
                    raise ValueError(f"routing child {child!r} is not a question")


def _overlay(qid: str, data: dict[str, Any]) -> tuple[int, list[str]]:
    row = data.get(qid) if isinstance(data.get(qid), dict) else {}
    version = int(row.get("version") or 1) if isinstance(row, dict) else 1
    requires = row.get("requires") if isinstance(row, dict) else []
    if not isinstance(requires, list):
        requires = []
    return version, [str(r) for r in requires]


def _from_native(qid: str, question: dict[str, Any], *, scope: str, version: int, requires: list[str]) -> FeatureDef:
    primitive = str(question.get("type") or "noul")
    criteria = question.get("criteria")
    if primitive == "choice" and isinstance(criteria, dict):
        options = [str(k) for k in criteria]
        levels: list[str] = []
    elif primitive == "score" and isinstance(criteria, list):
        options = []
        levels = [str(block.get("name") or i) for i, block in enumerate(criteria) if isinstance(block, dict)]
    else:
        options, levels = [], []
        if not isinstance(criteria, (dict, list)):
            criteria = {}
    return FeatureDef(
        id=qid,
        scope=scope,
        primitive=primitive,
        version=version,
        requires=requires,
        question=dict(question),
        criteria=criteria if criteria is not None else {},
        options=options,
        levels=levels,
    )


def _load_scope(path: Path) -> tuple[list[FeatureDef], dict[str, list[str]], dict[str, str]]:
    definition = load_definition(str(path))
    diagnostics = check_definition(definition)
    bad = errors_of(diagnostics) + [d for d in diagnostics if d.code == "unguaranteed-backtick"]
    if bad:
        raise SystemOnePromptsError(bad)
    data = definition.data if isinstance(definition.data, dict) else {}
    scope = str(data.get("scope") or path.stem)
    features_data = data.get("features")
    overlay: dict[str, Any] = features_data if isinstance(features_data, dict) else {}
    question_ids = set(definition.questions)
    overlay_ids = set(overlay)
    if overlay_ids != question_ids:
        missing = sorted(question_ids - overlay_ids)
        extra = sorted(overlay_ids - question_ids)
        bits = []
        if missing:
            bits.append(f"missing [data.features] for {missing}")
        if extra:
            bits.append(f"unknown [data.features] ids {extra}")
        raise ValueError(f"{path.name}: " + "; ".join(bits))
    features = [
        _from_native(qid, question, scope=scope, version=ver, requires=req)
        for qid, question in definition.questions.items()
        for ver, req in [_overlay(qid, overlay)]
    ]
    routing_data = data.get("routing")
    raw_routing: dict[str, Any] = routing_data if isinstance(routing_data, dict) else {}
    routing: dict[str, list[str]] = {}
    for key, children in raw_routing.items():
        if isinstance(children, list):
            routing[str(key)] = [str(c) for c in children]
    return features, routing, dict(definition.requires)


@dataclass(frozen=True)
class LoadedBank:
    features: list[FeatureDef]
    routing: dict[str, list[str]]
    requires: dict[str, dict[str, str]]


def load_feature_bank(path: Path | None = None) -> LoadedBank:
    """Load ``features/{task,episode,turn}.toml`` via systemoneprompts."""
    root = path or _FEATURES_DIR
    features: list[FeatureDef] = []
    routing: dict[str, list[str]] = {}
    requires: dict[str, dict[str, str]] = {}
    for name in _SCOPE_FILES:
        toml_path = root / name
        if not toml_path.is_file():
            continue
        scope_feats, scope_routing, scope_requires = _load_scope(toml_path)
        features.extend(scope_feats)
        if scope_routing:
            routing.update(scope_routing)
        requires[toml_path.stem] = scope_requires
    validate_bank(features, routing=routing)
    return LoadedBank(features=features, routing=routing, requires=requires)


def load_bank(path: Path | None = None) -> list[FeatureDef]:
    """Load the milestone bank and validate the hotwash overlay."""
    return load_feature_bank(path).features


def intent_routing(path: Path | None = None) -> dict[str, list[str]]:
    """Intent super-family → subtype ids from ``[data.routing]``."""
    return load_feature_bank(path).routing


def scope_requires(scope: str, path: Path | None = None) -> dict[str, str]:
    """``[requires]`` table for one scope file."""
    return load_feature_bank(path).requires.get(scope, {})


def wire_questions(feats: Sequence[FeatureDef]) -> dict[str, dict[str, Any]]:
    """Wire form: ``{id: {type, instructions, criteria}}``."""
    return _wire_questions({feat.id: feat.question for feat in feats})


__all__ = [
    "FeatureDef",
    "LoadedBank",
    "criteria_hash",
    "intent_routing",
    "load_bank",
    "load_feature_bank",
    "scope_requires",
    "validate_bank",
    "wire_questions",
]
