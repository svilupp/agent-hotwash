"""Task candidates and turn-relationship edges (§5.2)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from agent_hotwash.canonical import turn_events
from agent_hotwash.config import Config
from agent_hotwash.events import Session, Turn
from agent_hotwash.structure.ledger import Ledger, update_ledger

_CANDIDATE_KINDS = frozenset({"user", "delegation"})

EdgeKind = Literal["continues", "corrects", "depends_on", "sibling_same_area", "unrelated"]
Initiator = Literal["human", "parent_agent", "harness", "unknown"]
Confidence = Literal["high", "medium", "low"]

_IDENTITY_KEYS = (
    "identity",
    "task_identity",
    "turn.relationship.task_identity",
)
_CORRECTS_KEYS = (
    "corrects",
    "corrects_prior",
    "turn.relationship.corrects_prior",
)
_REFERENCES_KEYS = (
    "references",
    "references_prior_result",
    "turn.relationship.references_prior_result",
)
_COMPONENT_KEYS = (
    "same_component",
    "turn.relationship.same_component",
)


class Task(BaseModel):
    """One unit of requested work, possibly spanning several candidate turns."""

    model_config = ConfigDict(extra="ignore")

    task_id: str
    session_id: str
    initiator: Initiator = "unknown"
    parent_task: str | None = None
    turns: list[Turn] = Field(default_factory=list)
    ledger: Ledger = Field(default_factory=Ledger)
    edge_to_prev: EdgeKind | None = None
    edge_confidence: Confidence | None = None
    spawn_episode_id: str | None = None


def _candidate_turns(session: Session) -> list[Turn]:
    return [t for t in session.turns if t.user_input.kind in _CANDIDATE_KINDS]


def _initiator_of(turn: Turn) -> Initiator:
    kind = turn.user_input.kind
    if kind == "delegation":
        return "parent_agent"
    if kind == "user":
        return "human"
    if kind == "injected":
        return "harness"
    return "unknown"


def _lookup(block: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in block:
            return block[key]
    return None


def _as_choice(val: Any) -> str | None:
    if val is None:
        return None
    if isinstance(val, str):
        return val
    if isinstance(val, dict):
        choice = val.get("choice") or val.get("value")
        return str(choice) if choice is not None else None
    return str(val)


def _as_noul(val: Any) -> float:
    if val is None:
        return 0.0
    if isinstance(val, bool):
        return 1.0 if val else 0.0
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, dict) and val.get("noul") is not None:
        try:
            return float(val["noul"])
        except (TypeError, ValueError):
            return 0.0
    return 0.0


def _noul_true(val: Any) -> bool:
    return _as_noul(val) >= 0.5


def map_relationship_edge(answers: dict[str, Any] | None) -> tuple[EdgeKind, Confidence]:
    """Map ``(identity, corrects, references, same_component)`` → graph edge."""
    if not answers:
        return "continues", "low"
    identity = _as_choice(_lookup(answers, _IDENTITY_KEYS)) or "unclear"
    corrects = _noul_true(_lookup(answers, _CORRECTS_KEYS))
    references = _noul_true(_lookup(answers, _REFERENCES_KEYS))
    same_component = _noul_true(_lookup(answers, _COMPONENT_KEYS))

    if identity in ("other", "unclear"):
        return "continues", "low"
    if identity == "same_deliverable":
        if corrects:
            return "corrects", "high"
        return "continues", "high"
    if identity == "distinct_deliverable":
        if references:
            return "depends_on", "high"
        if same_component:
            return "sibling_same_area", "high"
        return "unrelated", "high"
    return "continues", "low"


def _new_task(session: Session, turn: Turn, n: int, edge: EdgeKind | None, conf: Confidence | None) -> Task:
    task = Task(
        task_id=f"{session.session_id}:task{n}",
        session_id=session.session_id,
        initiator=_initiator_of(turn),
        turns=[turn],
        ledger=Ledger(),
        edge_to_prev=edge,
        edge_confidence=conf,
    )
    update_ledger(task.ledger, turn, turn_events(session, turn))
    return task


def _answers_for(relationship_answers: dict[str, Any] | None, turn_id: str) -> dict[str, Any] | None:
    if not relationship_answers:
        return None
    block = relationship_answers.get(turn_id)
    return block if isinstance(block, dict) else None


def segment_tasks(
    session: Session,
    config: Config,
    *,
    semantic_mode: str,
    relationship_answers: dict[str, Any] | None = None,
) -> list[Task]:
    """Split a session into tasks.

    ``semantic_mode == "off"`` yields one task per session (all candidate turns).
    Otherwise each candidate turn is a task whose ``edge_to_prev`` is mapped from
    recorded relationship answers (or ``continues`` / low confidence when missing).
    """
    _ = config.structure  # candidates already classified; knobs reserved for later WPs
    candidates = _candidate_turns(session)
    if semantic_mode == "off":
        turns = candidates or list(session.turns)
        task = Task(
            task_id=f"{session.session_id}:task0",
            session_id=session.session_id,
            initiator=_initiator_of(turns[0]) if turns else "unknown",
            turns=list(session.turns) if session.turns else turns,
            ledger=Ledger(),
        )
        for turn in task.turns:
            update_ledger(task.ledger, turn, turn_events(session, turn))
        return [task]

    if not candidates:
        return []

    tasks: list[Task] = []
    first = candidates[0]
    tasks.append(_new_task(session, first, 0, None, None))
    for turn in candidates[1:]:
        edge, conf = map_relationship_edge(_answers_for(relationship_answers, turn.turn_id))
        tasks.append(_new_task(session, turn, len(tasks), edge, conf))

    # Attach intervening non-candidate turns to the open task.
    if session.turns:
        by_id = {t.turn_id: i for i, task in enumerate(tasks) for t in task.turns}
        current: int | None = None
        assigned: dict[int, list[Turn]] = {i: [] for i in range(len(tasks))}
        for turn in session.turns:
            if turn.turn_id in by_id:
                current = by_id[turn.turn_id]
            if current is None:
                current = 0
            assigned[current].append(turn)
        for i, task in enumerate(tasks):
            if assigned[i]:
                task.turns = assigned[i]
    return tasks


__all__ = ["Task", "map_relationship_edge", "segment_tasks"]
