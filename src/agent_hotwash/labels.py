"""JSONL label store (PLAN §9.7) and eval report."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field
from systemoneprompts.json_values import canonical_json

from agent_hotwash.canonical import build_turns
from agent_hotwash.config import Config
from agent_hotwash.events import Trace
from agent_hotwash.semantic.bank import FeatureDef, criteria_hash, load_bank
from agent_hotwash.structure.digest import DIGEST_SCHEMA_VERSION, build_digest
from agent_hotwash.structure.episodes import segment_episodes
from agent_hotwash.structure.tasks import segment_tasks

ProvenanceKind = Literal["real", "synthetic"]
ScopeKind = Literal["task", "turn", "episode"]

BURNED_SUFFIX = ".burned"


class LabelRecord(BaseModel):
    """One label-store row."""

    model_config = ConfigDict(extra="ignore")

    item_id: str
    root_trace_id: str
    scope: ScopeKind
    object_id: str
    source_kind: str
    digest_hash: str
    digest_schema_version: int = 1
    feature_id: str
    feature_version: int = 1
    criteria_hash: str
    answer: Any = None
    confidence: float | None = None
    skip_reason: str | None = None
    annotator: str = "unknown"
    ts: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    split: str = "dev"
    provenance: ProvenanceKind = "real"


RecordKey = tuple[str, str, int, str, str]


def record_key(rec: LabelRecord) -> RecordKey:
    """Resume key: item_id, feature_id, feature_version, criteria_hash, annotator.

    The annotator is part of the identity so a second annotator labelling the
    same item is new work (double-labelling, §9.8), not a resume-skip.
    """
    return (rec.item_id, rec.feature_id, rec.feature_version, rec.criteria_hash, rec.annotator)


class SplitConflict(ValueError):
    """A ``root_trace_id`` already pinned to one split was offered under another."""

    def __init__(self, root_trace_id: str, pinned: str, requested: str) -> None:
        self.root_trace_id = root_trace_id
        self.pinned = pinned
        self.requested = requested
        super().__init__(f"split for root_trace_id {root_trace_id!r} is pinned to {pinned!r}; got {requested!r}")


def pinned_splits(records: list[LabelRecord]) -> dict[str, str]:
    """``root_trace_id → split`` as first recorded in the store (§9.7: split pinned by root)."""
    out: dict[str, str] = {}
    for rec in records:
        out.setdefault(rec.root_trace_id, rec.split)
    return out


def check_split_pinned(pinned: dict[str, str], root_trace_id: str, split: str) -> None:
    """Raise :class:`SplitConflict` when ``split`` disagrees with the pinned one."""
    current = pinned.get(root_trace_id)
    if current is not None and current != split:
        raise SplitConflict(root_trace_id, current, split)


def burned_path(store: Path) -> Path:
    return store.with_name(store.name + BURNED_SUFFIX)


def load_records(path: Path) -> list[LabelRecord]:
    if not path.is_file():
        return []
    rows: list[LabelRecord] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        data = json.loads(line)
        if isinstance(data, dict):
            rows.append(LabelRecord.model_validate(data))
    return rows


def labelled_keys(records: list[LabelRecord]) -> set[RecordKey]:
    return {record_key(r) for r in records}


def append_records(path: Path, records: list[LabelRecord]) -> None:
    """Append rows; refuses to write a row whose split contradicts the store's pin."""
    if records:
        pinned = pinned_splits(load_records(path))
        for rec in records:
            check_split_pinned(pinned, rec.root_trace_id, rec.split)
            pinned.setdefault(rec.root_trace_id, rec.split)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        for rec in records:
            fh.write(rec.model_dump_json() + "\n")


def make_item_id(root_trace_id: str, scope: str, object_id: str) -> str:
    return f"{root_trace_id}:{scope}:{object_id}"


def pending_record(
    *,
    root_trace_id: str,
    scope: ScopeKind,
    object_id: str,
    source_kind: str,
    digest: dict[str, Any] | None,
    digest_schema_version: int,
    feature: FeatureDef,
    annotator: str,
    split: str,
    provenance: ProvenanceKind,
) -> LabelRecord:
    digest_obj = digest or {}
    return LabelRecord(
        item_id=make_item_id(root_trace_id, scope, object_id),
        root_trace_id=root_trace_id,
        scope=scope,
        object_id=object_id,
        source_kind=source_kind,
        digest_hash=hashlib_sha(digest_obj),
        digest_schema_version=digest_schema_version,
        feature_id=feature.id,
        feature_version=feature.version,
        criteria_hash=criteria_hash(feature),
        annotator=annotator,
        split=split,
        provenance=provenance,
    )


def collect_drafts(
    trace: Trace,
    config: Config,
    *,
    annotator: str,
    split: str,
    provenance: ProvenanceKind,
    feature_ids: set[str] | None = None,
) -> list[LabelRecord]:
    """One pending record per (object, feature) for labelling."""
    session = trace.root
    if not session.turns:
        session.turns = build_turns(
            session,
            injected_tags=config.structure.injected_tags_user,
            delegation_tag=config.structure.delegation_tag,
        )
    tasks = segment_tasks(session, config, semantic_mode="off")
    episodes = segment_episodes(session, tasks, config)
    bank = load_bank()
    if feature_ids:
        bank = [f for f in bank if f.id in feature_ids]
    source_kind = trace.provenance.source_format
    root_id = trace.trace_id
    drafts: list[LabelRecord] = []
    task_by_id = {t.task_id: t for t in tasks}
    for task in tasks:
        state = {
            "task": {
                "request": task.ledger.request,
                "amendments": list(task.ledger.amendments),
                "deliverables": list(task.ledger.deliverables),
                "status": task.ledger.status,
            }
        }
        for feat in bank:
            if feat.scope != "task":
                continue
            drafts.append(
                pending_record(
                    root_trace_id=root_id,
                    scope="task",
                    object_id=task.task_id,
                    source_kind=source_kind,
                    digest=state,
                    digest_schema_version=DIGEST_SCHEMA_VERSION,
                    feature=feat,
                    annotator=annotator,
                    split=split,
                    provenance=provenance,
                )
            )
    for ep in episodes:
        task = task_by_id.get(ep.task_id)
        digest = build_digest(task, ep, session, config) if task is not None else {"episode": ep.episode_id}
        for feat in bank:
            if feat.scope != "episode":
                continue
            drafts.append(
                pending_record(
                    root_trace_id=root_id,
                    scope="episode",
                    object_id=ep.episode_id,
                    source_kind=source_kind,
                    digest=digest,
                    digest_schema_version=DIGEST_SCHEMA_VERSION,
                    feature=feat,
                    annotator=annotator,
                    split=split,
                    provenance=provenance,
                )
            )
    for turn in session.turns:
        state = {"turn": {"turn_id": turn.turn_id, "text": turn.user_input.text}}
        for feat in bank:
            if feat.scope != "turn":
                continue
            drafts.append(
                pending_record(
                    root_trace_id=root_id,
                    scope="turn",
                    object_id=turn.turn_id,
                    source_kind=source_kind,
                    digest=state,
                    digest_schema_version=DIGEST_SCHEMA_VERSION,
                    feature=feat,
                    annotator=annotator,
                    split=split,
                    provenance=provenance,
                )
            )
    return drafts


def hashlib_sha(obj: Any) -> str:
    import hashlib

    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()


def _answers_agree(left: Any, right: Any) -> bool:
    if left == right:
        return True
    try:
        return float(left) == float(right)
    except (TypeError, ValueError):
        return canonical_json(left) == canonical_json(right)


def _is_positive(answer: Any) -> bool:
    if answer is None:
        return False
    if isinstance(answer, bool):
        return answer
    if isinstance(answer, (int, float)):
        return float(answer) >= 0.5
    if isinstance(answer, str):
        lowered = answer.lower()
        if lowered in {"false", "no", "none", "other", "unstated", "0_direct"}:
            return False
        if lowered.startswith(("0_", "0.")):
            return False
        if lowered in {"true", "yes"}:
            return True
        try:
            return float(answer) >= 0.5
        except ValueError:
            return lowered not in {"", "skip"}
    if isinstance(answer, dict):
        if "noul" in answer:
            try:
                return float(answer["noul"]) >= 0.5
            except (TypeError, ValueError):
                return False
        if "score" in answer:
            return _is_positive(answer.get("score"))
        choice = answer.get("choice")
        return _is_positive(choice)
    return bool(answer)


def eval_store(records: list[LabelRecord]) -> dict[str, Any]:
    """Per-feature agreement (double-labelled) and positive rates."""
    by_feat: dict[str, list[LabelRecord]] = {}
    for rec in records:
        by_feat.setdefault(rec.feature_id, []).append(rec)

    features: dict[str, Any] = {}
    for fid, rows in sorted(by_feat.items()):
        labelled = [r for r in rows if r.skip_reason is None and r.answer is not None]
        skipped = len(rows) - len(labelled)
        positives = sum(1 for r in labelled if _is_positive(r.answer))
        # Double-label: same item_id + feature version/hash, different annotators.
        pairs: dict[tuple[str, int, str], list[LabelRecord]] = {}
        for rec in labelled:
            pairs.setdefault((rec.item_id, rec.feature_version, rec.criteria_hash), []).append(rec)
        compared = 0
        agreed = 0
        for group in pairs.values():
            annotators: dict[str, list[LabelRecord]] = {}
            for rec in group:
                annotators.setdefault(rec.annotator, []).append(rec)
            if len(annotators) < 2:
                continue
            names = sorted(annotators)
            a = annotators[names[0]][0].answer
            b = annotators[names[1]][0].answer
            compared += 1
            if _answers_agree(a, b):
                agreed += 1
        features[fid] = {
            "n": len(labelled),
            "skipped": skipped,
            "positive_rate": (positives / len(labelled)) if labelled else None,
            "double_labelled": compared,
            "agreement": (agreed / compared) if compared else None,
        }

    return {
        "n_records": len(records),
        "n_features": len(features),
        "features": features,
    }


__all__ = [
    "LabelRecord",
    "RecordKey",
    "SplitConflict",
    "append_records",
    "burned_path",
    "check_split_pinned",
    "collect_drafts",
    "eval_store",
    "labelled_keys",
    "load_records",
    "make_item_id",
    "pending_record",
    "pinned_splits",
    "record_key",
]
