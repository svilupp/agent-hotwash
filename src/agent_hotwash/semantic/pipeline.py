"""Run structure segmentation and optional System One annotation for a trace.

Harness-blind: no ``sources.*`` imports and no ``AgentKind`` / ``source_format``.

Order of work for one trace (PLAN §5-6):

1. turn relationship — one round per candidate turn, *sequential* because
   each round reads the ledger updated by the previous turn;
2. tasks → episodes (deterministic);
3. task features (super-families first, routed subtypes second), episode
   features (one digest each) and turn features — all independent, so they are
   asked *concurrently* up to ``semantic.max_concurrency`` threads sharing the
   asker's rate limiter.

A transport failure that survives the client's retries marks the affected
features ``reason="api_error"`` and the run continues; authentication failures
and ``cached``-mode misses propagate (the caller decides how to fail).
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

from systemoneprompts.cache import CacheMissError
from systemoneprompts.client import TypeSafeClientError, TypeSafeHttpError
from systemoneprompts.diagnostics import SystemOnePromptsError

from agent_hotwash.canonical import build_turns, turn_events
from agent_hotwash.semantic.bank import FeatureDef, load_feature_bank, wire_questions
from agent_hotwash.semantic.client import SystemOneAsker, _unwrap_cache_miss
from agent_hotwash.semantic.project import is_heavy_inspect
from agent_hotwash.semantic.results import FeatureSet, FeatureValue, Reason
from agent_hotwash.structure.digest import DIGEST_SCHEMA_VERSION, build_digest
from agent_hotwash.structure.episodes import Episode, attach_delegation, segment_episodes
from agent_hotwash.structure.ledger import Ledger, update_ledger
from agent_hotwash.structure.tasks import Task, segment_tasks

if TYPE_CHECKING:
    from collections.abc import Sequence

    from agent_hotwash.config import Config
    from agent_hotwash.events import Capabilities, Session, Trace, Turn

_REL_PREFIX = "turn.relationship."
_REL_IDENTITY = "turn.relationship.task_identity"
_SCOPE_BREADTH = "task.scope.breadth"
_ESCAPE_IDENTITIES = ("other", "unclear")
_PHASE_ACTIVITY = "episode.phase.activity"
_PHASE_PURPOSE = "episode.phase.purpose"
_CANDIDATE_KINDS = frozenset({"user", "delegation"})
_INTENT_GATE = 0.5
_FATAL_HTTP = frozenset({401, 403})


def feature_question(feat: FeatureDef) -> dict[str, Any]:
    """Wire question body: ``{type, instructions, criteria}``."""
    return wire_questions((feat,))[feat.id]


def questions_from_features(feats: Sequence[FeatureDef]) -> dict[str, dict[str, Any]]:
    return wire_questions(feats)


def _unanswered(feat: FeatureDef, reason: Reason) -> FeatureValue:
    return FeatureValue(id=feat.id, version=feat.version, reason=reason, source="jev")


def _partition_inspect(feats: Sequence[FeatureDef]) -> list[list[FeatureDef]]:
    """Keep op-body / whole-facts questions off the same request as phase labels."""
    light: list[FeatureDef] = []
    heavy: list[FeatureDef] = []
    for feat in feats:
        paths = list(feat.inspect) + list(feat.compare)
        if any(is_heavy_inspect(path) for path in paths):
            heavy.append(feat)
        else:
            light.append(feat)
    return [group for group in (light, heavy) if group]


def _is_fatal(exc: BaseException) -> bool:
    if _unwrap_cache_miss(exc) is not None:
        return True
    if isinstance(exc, TypeSafeHttpError) and exc.status in _FATAL_HTTP:
        return True
    if isinstance(exc, TypeSafeClientError):
        return any(d.code == "missing-credentials" for d in exc.diagnostics)
    return False


class Annotator:
    """Asks bank features against states with capability gating and redaction."""

    def __init__(
        self,
        asker: SystemOneAsker,
        config: Config,
        caps: Capabilities,
        *,
        mode: str,
        allow_unredacted: bool,
    ) -> None:
        self.asker = asker
        self.config = config
        self.caps = caps
        self.mode = mode
        # ``--allow-unredacted`` is the C9 override that lets ``live`` run when
        # ``semantic.redact = false`` is configured; it never disables redaction.
        self.allow_unredacted = allow_unredacted or config.semantic.allow_unredacted
        self.redact = config.semantic.redact
        self.concurrency = max(1, config.semantic.max_concurrency)

    def ask(self, state: dict[str, Any], feats: Sequence[FeatureDef]) -> dict[str, FeatureValue]:
        out: dict[str, FeatureValue] = {}
        askable: list[FeatureDef] = []
        for feat in feats:
            if feat.requires and not all(self.caps.meets(name) for name in feat.requires):
                out[feat.id] = _unanswered(feat, "insufficient_observability")
            else:
                askable.append(feat)
        if not askable:
            return out
        for group in _partition_inspect(askable):
            out.update(self._ask_group(state, group))
        return out

    def _ask_group(self, state: dict[str, Any], feats: Sequence[FeatureDef]) -> dict[str, FeatureValue]:
        out: dict[str, FeatureValue] = {}
        try:
            raw = self.asker.ask(
                state,
                wire_questions(feats),
                redact=self.redact,
                allow_unredacted=self.allow_unredacted,
            )
        except CacheMissError:
            raise
        except Exception as exc:
            if _is_fatal(exc):
                raise
            if isinstance(exc, SystemOnePromptsError):
                return {feat.id: _unanswered(feat, "api_error") for feat in feats}
            raise
        for feat in feats:
            out[feat.id] = self.asker.to_feature_value(feat, raw.get(feat.id), state, redact=self.redact)
        return out

    def ask_many(self, items: Sequence[tuple[dict[str, Any], Sequence[FeatureDef]]]) -> list[dict[str, FeatureValue]]:
        """``ask`` over independent (state, feats) pairs, concurrently."""
        if len(items) <= 1 or self.concurrency == 1:
            return [self.ask(state, feats) for state, feats in items]
        with ThreadPoolExecutor(max_workers=min(self.concurrency, len(items))) as pool:
            return list(pool.map(lambda it: self.ask(it[0], it[1]), items))


# ---------------------------------------------------------------------------
# per-scope steps
# ---------------------------------------------------------------------------


def _ensure_turns(session: Session, config: Config) -> None:
    if session.turns:
        return
    session.turns = build_turns(
        session,
        injected_tags=config.structure.injected_tags_user,
        delegation_tag=config.structure.delegation_tag,
    )


def _identity_choice(fv: FeatureValue | None) -> str | None:
    if fv is None or fv.value is None:
        return None
    val = fv.value
    if isinstance(val, dict):
        val = val.get("choice") or val.get("value")
    return str(val) if val is not None else None


def _relationship_answers(
    annotator: Annotator, session: Session, candidates: Sequence[Turn], rel_feats: Sequence[FeatureDef]
) -> dict[str, dict[str, Any]]:
    """One sequential round per candidate turn after the first (ledger-dependent).

    §5.2 / C3: ``task_identity`` is asked first; the three verifier Nouls are
    asked only when identity is neither ``other`` nor ``unclear`` (never chain a
    verifier off the escape hatch). An unanswered identity counts as unclear.
    """
    answers: dict[str, dict[str, Any]] = {}
    if len(candidates) < 2 or not rel_feats:
        return answers
    identity_feats = [f for f in rel_feats if f.id == _REL_IDENTITY]
    verifier_feats = [f for f in rel_feats if f.id != _REL_IDENTITY]
    ledger = Ledger()
    for i, turn in enumerate(candidates):
        if i > 0:
            state = _relationship_state(ledger, turn)
            if identity_feats:
                values = annotator.ask(state, identity_feats)
                identity = _identity_choice(values.get(_REL_IDENTITY))
                if verifier_feats and identity not in (None, *_ESCAPE_IDENTITIES):
                    values.update(annotator.ask(state, verifier_feats))
            else:
                values = annotator.ask(state, rel_feats)
            answers[turn.turn_id] = {fid: fv.value for fid, fv in values.items()}
        update_ledger(ledger, turn, turn_events(session, turn))
    return answers


def _relationship_state(ledger: Ledger, turn: Turn) -> dict[str, Any]:
    """State for turn-relationship questions: ``messages[0].text`` vs ``ledger.*``."""
    dump = ledger.model_dump(mode="json")
    arts = dump.get("artifacts")
    if isinstance(arts, (set, list, tuple)):
        dump["artifacts"] = sorted(str(a) for a in arts)
    return {
        "ledger": dump,
        "messages": [{"kind": turn.user_input.kind, "text": turn.user_input.text}],
        "digest_schema_version": DIGEST_SCHEMA_VERSION,
    }


def _task_state(task: Task) -> dict[str, Any]:
    return {
        "task": {
            "request": task.ledger.request,
            "amendments": list(task.ledger.amendments),
            "deliverables": list(task.ledger.deliverables),
            "status": task.ledger.status,
        },
        "digest_schema_version": DIGEST_SCHEMA_VERSION,
    }


def _task_features(
    annotator: Annotator,
    tasks: Sequence[Task],
    task_feats: Sequence[FeatureDef],
    routing: dict[str, list[str]],
) -> list[FeatureSet]:
    """Super-family questions for every task, then routed subtypes (≥ 0.5)."""
    if not tasks or not task_feats:
        return []
    subtypes = frozenset(sid for ids in routing.values() for sid in ids)
    first_round = [f for f in task_feats if f.id not in subtypes]
    supers = [f for f in first_round if f.id != _SCOPE_BREADTH]
    breadth = [f for f in first_round if f.id == _SCOPE_BREADTH]
    states = [_task_state(t) for t in tasks]
    values_per_task = annotator.ask_many([(s, supers) for s in states]) if supers else [{} for _ in states]
    if breadth:
        for values, extra in zip(values_per_task, annotator.ask_many([(s, breadth) for s in states]), strict=True):
            values.update(extra)

    routed_items: list[tuple[int, dict[str, Any], list[FeatureDef]]] = []
    for i, values in enumerate(values_per_task):
        routed: list[FeatureDef] = []
        for super_id, children in routing.items():
            fv = values.get(super_id)
            score = fv.value if fv is not None and isinstance(fv.value, (int, float)) else 0.0
            if isinstance(score, bool):
                score = 0.0
            if score >= _INTENT_GATE:
                wanted = set(children)
                routed.extend(f for f in task_feats if f.id in wanted)
        if routed:
            routed_items.append((i, states[i], routed))
    for (i, _state, _feats), extra in zip(
        routed_items, annotator.ask_many([(s, f) for _i, s, f in routed_items]), strict=True
    ):
        values_per_task[i].update(extra)
    return [
        FeatureSet(scope="task", object_id=task.task_id, values=values)
        for task, values in zip(tasks, values_per_task, strict=True)
    ]


def _episode_features(
    annotator: Annotator,
    session: Session,
    tasks: Sequence[Task],
    episodes: Sequence[Episode],
    episode_feats: Sequence[FeatureDef],
) -> list[FeatureSet]:
    if not episodes or not episode_feats:
        return []
    task_by_id = {t.task_id: t for t in tasks}
    fallback = tasks[0] if tasks else None
    items: list[tuple[dict[str, Any], Sequence[FeatureDef]]] = []
    for ep in episodes:
        task = task_by_id.get(ep.task_id) or fallback
        state = (
            {"episode": {"id": ep.episode_id}} if task is None else build_digest(task, ep, session, annotator.config)
        )
        items.append((state, episode_feats))
    return [
        FeatureSet(scope="episode", object_id=ep.episode_id, values=values)
        for ep, values in zip(episodes, annotator.ask_many(items), strict=True)
    ]


def _turn_features(annotator: Annotator, turns: Sequence[Turn], turn_feats: Sequence[FeatureDef]) -> list[FeatureSet]:
    if not turns or not turn_feats:
        return []
    items = [
        (
            {
                "messages": [{"kind": t.user_input.kind, "text": t.user_input.text}],
                "turn_id": t.turn_id,
                "digest_schema_version": DIGEST_SCHEMA_VERSION,
            },
            turn_feats,
        )
        for t in turns
    ]
    return [
        FeatureSet(scope="turn", object_id=t.turn_id, values=values)
        for t, values in zip(turns, annotator.ask_many(items), strict=True)
    ]


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


def annotate_trace(
    trace: Trace,
    config: Config,
    *,
    mode: str,
    allow_unredacted: bool = False,
    asker: SystemOneAsker | None = None,
) -> tuple[list[Task], list[Episode], list[FeatureSet], Capabilities]:
    """Segment structure and, when ``mode`` is ``cached``/``live``, call System One.

    Returns ``(tasks, episodes, feature_sets, capabilities)`` for **every**
    session in the tree (root first, then children in link order). A delegated
    child's tasks are bound to the parent episode that spawned its thread
    (``Task.parent_task`` / ``spawn_episode_id``, PLAN WP3b) when
    ``trace.links`` names the parent. An ``asker`` may be injected (the runner
    shares one rate-limited asker per process).
    """
    sessions = _sessions_parents_first(trace)
    for session in sessions:
        _ensure_turns(session, config)
    caps = trace.capabilities if trace.subagents else trace.root.capabilities

    annotator: Annotator | None = None
    if mode in {"cached", "live"}:
        if asker is None:
            from agent_hotwash.semantic.ratelimit import RateLimiter

            asker = SystemOneAsker(
                config.semantic.model,
                config.semantic.cache_dir,
                mode=mode,
                max_questions=config.semantic.max_questions_per_request,
                max_retries=config.semantic.max_retries,
                timeout_s=config.semantic.timeout_s,
                limiter=RateLimiter(config.semantic.requests_per_second, config.semantic.burst),
                secret_patterns=list(config.lexicons.secret),
            )
        annotator = Annotator(asker, config, caps, mode=mode, allow_unredacted=allow_unredacted)

    parent_of = {link.child_id: link.parent_id for link in trace.links}
    tasks: list[Task] = []
    episodes: list[Episode] = []
    feature_sets: list[FeatureSet] = []
    episodes_by_session: dict[str, list[Episode]] = {}
    for session in sessions:
        s_tasks, s_episodes, s_sets = _annotate_session(annotator, session, config, mode=mode)
        parent_id = parent_of.get(session.session_id)
        parent_eps = episodes_by_session.get(parent_id) if parent_id else None
        if parent_eps:
            for task in s_tasks:
                attach_delegation(task, parent_eps, session.session_id)
        episodes_by_session[session.session_id] = s_episodes
        tasks += s_tasks
        episodes += s_episodes
        feature_sets += s_sets
    return tasks, episodes, feature_sets, caps


def _sessions_parents_first(trace: Trace) -> list[Session]:
    """Root, then children ordered so every parent precedes its children."""
    order = [trace.root, *trace.subagents]
    depth = {trace.root.session_id: 0}
    parent_of = {link.child_id: link.parent_id for link in trace.links}
    for _ in range(len(order)):  # depth propagation; trees are shallow
        for s in order:
            p = parent_of.get(s.session_id)
            if p in depth and s.session_id not in depth:
                depth[s.session_id] = depth[p] + 1
    return sorted(order, key=lambda s: depth.get(s.session_id, len(order)))


def _annotate_session(
    annotator: Annotator | None,
    session: Session,
    config: Config,
    *,
    mode: str,
) -> tuple[list[Task], list[Episode], list[FeatureSet]]:
    """Tasks, episodes and (when ``annotator`` is set) features for one session."""
    if annotator is None:
        tasks = segment_tasks(session, config, semantic_mode=mode)
        return tasks, segment_episodes(session, tasks, config), []

    loaded = load_feature_bank()
    bank = loaded.features
    rel_feats = [f for f in bank if f.id.startswith(_REL_PREFIX)]
    task_feats = [f for f in bank if f.scope == "task"]
    episode_feats = [f for f in bank if f.scope == "episode"]
    turn_feats = [f for f in bank if f.scope == "turn" and not f.id.startswith(_REL_PREFIX)]

    candidates = [t for t in session.turns if t.user_input.kind in _CANDIDATE_KINDS]
    relationship = _relationship_answers(annotator, session, candidates, rel_feats)
    tasks = segment_tasks(session, config, semantic_mode=mode, relationship_answers=relationship)
    episodes = segment_episodes(session, tasks, config)

    feature_sets = _task_features(annotator, tasks, task_feats, loaded.routing)
    episode_sets = _episode_features(annotator, session, tasks, episodes, episode_feats)
    apply_phase_labels(episodes, episode_sets)
    feature_sets += episode_sets
    feature_sets += _turn_features(annotator, session.turns, turn_feats)
    return tasks, episodes, feature_sets


def apply_phase_labels(episodes: Sequence[Episode], feature_sets: Sequence[FeatureSet]) -> None:
    """Copy the answered ``episode.phase.*`` Choices onto each atom's display label.

    Phase is a label, not a boundary (C4): atoms, spend and provenance are
    untouched; only ``phase_activity`` / ``phase_purpose`` are filled so
    display grouping and PHASE_SPEND evidence see the resolved values.
    """
    by_id = {fs.object_id: fs for fs in feature_sets if fs.scope == "episode"}
    for ep in episodes:
        fs = by_id.get(ep.episode_id)
        if fs is None:
            continue
        activity = _identity_choice(fs.values.get(_PHASE_ACTIVITY))
        purpose = _identity_choice(fs.values.get(_PHASE_PURPOSE))
        if activity is not None:
            ep.phase_activity = activity
        if purpose is not None:
            ep.phase_purpose = purpose


__all__ = ["Annotator", "annotate_trace", "apply_phase_labels", "feature_question", "questions_from_features"]
