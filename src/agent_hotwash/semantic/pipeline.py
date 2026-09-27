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
features ``reason="api_error"`` and the run continues; authentication/billing failures
and ``cached``-mode misses propagate (the caller decides how to fail).
"""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

from systemoneprompts.cache import CacheMissError
from systemoneprompts.client import TypeSafeClientError, TypeSafeHttpError
from systemoneprompts.diagnostics import SystemOnePromptsError

from agent_hotwash.canonical import build_turns, turn_events
from agent_hotwash.events import EventKind
from agent_hotwash.primitives.failures import FailureRecord, build_failure_records
from agent_hotwash.primitives.handovers import HandoverRecord, build_handovers
from agent_hotwash.semantic.bank import FeatureDef, load_feature_bank, wire_questions
from agent_hotwash.semantic.client import SystemOneAsker, _unwrap_cache_miss
from agent_hotwash.semantic.project import is_heavy_inspect, project_for_questions
from agent_hotwash.semantic.results import FeatureSet, FeatureValue, Reason
from agent_hotwash.structure.digest import DIGEST_SCHEMA_VERSION, build_digest
from agent_hotwash.structure.episodes import Episode, attach_delegation, segment_episodes
from agent_hotwash.structure.ledger import Ledger, update_ledger
from agent_hotwash.structure.tasks import Task, segment_tasks

if TYPE_CHECKING:
    from agent_hotwash.config import Config
    from agent_hotwash.events import Capabilities, Session, Trace, Turn

_REL_PREFIX = "turn.relationship."
_REL_IDENTITY = "turn.relationship.task_identity"
_SCOPE_BREADTH = "task.scope.breadth"
_ESCAPE_IDENTITIES = ("other", "unclear")
_PHASE_ACTIVITY = "episode.phase.activity"
_PHASE_PURPOSE = "episode.phase.purpose"
_PRODUCE_ARTIFACT = "episode.progress.produces_requested_artifact"
_PRODUCE_GATE = 0.7
_CANDIDATE_KINDS = frozenset({"user", "delegation"})
_INTENT_GATE = 0.5
_FATAL_HTTP = frozenset({401, 402, 403})
_FAILURE_DEPENDENT = frozenset(
    {
        "failure.intent.identifier_matches",
        "failure.context.recipe_followed",
        "failure.recovery.corrected_argument",
        "failure.recovery.repeated_without_evidence",
        "failure.recovery.stopped_on_access_block",
        "failure.recovery.passed_without_product_edit",
    }
)
_FAILURE_GUIDANCE_FEATURES = frozenset(
    {
        "failure.context.recipe_documented",
        "failure.context.recipe_followed",
        "failure.context.generated_only",
    }
)


def feature_question(feat: FeatureDef) -> dict[str, Any]:
    """Wire question body: ``{type, instructions, criteria}``."""
    return wire_questions((feat,))[feat.id]


def questions_from_features(feats: Sequence[FeatureDef]) -> dict[str, dict[str, Any]]:
    return wire_questions(feats)


def visible_state(state: dict[str, Any], feats: Sequence[FeatureDef]) -> dict[str, Any]:
    """Inspect-projected digest: the payload live JeV and protocol raters share."""
    return project_for_questions(state, questions_from_features(feats))


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


def _noul_value(fv: FeatureValue | None) -> float | None:
    if fv is None or fv.reason is not None or fv.value is None:
        return None
    val = fv.value
    if isinstance(val, dict) and "noul" in val:
        val = val["noul"]
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        return None
    return float(val)


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


def _task_for_event(tasks: Sequence[Task], event_idx: int) -> Task | None:
    for task in tasks:
        if any(turn.event_start <= event_idx <= turn.event_end for turn in task.turns):
            return task
    return tasks[0] if tasks else None


def _failure_context(session: Session, event_idx: int, *, after: bool) -> list[dict[str, Any]]:
    lo, hi = (event_idx + 1, event_idx + 13) if after else (max(0, event_idx - 12), event_idx)
    rows: list[dict[str, Any]] = []
    for ev in session.events:
        if not lo <= ev.idx < hi or ev.kind not in {EventKind.tool_call, EventKind.tool_result}:
            continue
        command = None
        if ev.kind is EventKind.tool_call:
            command = next(
                (ev.tool_args.get(k) for k in ("command", "cmd", "script") if isinstance(ev.tool_args.get(k), str)),
                None,
            )
        rows.append(
            {
                "event_idx": ev.idx,
                "kind": ev.kind.value,
                "tool": ev.tool_name or ev.op_kind or "?",
                "command": command or "",
                "ok": ev.ok,
                "result": (ev.error_text or ev.output or "")[:1200],
                "artifacts": [a.path for a in ev.artifacts],
            }
        )
    return rows


def _failure_state(record: FailureRecord, session: Session, tasks: Sequence[Task]) -> dict[str, Any]:
    task = _task_for_event(tasks, record.provenance.event_idx)
    request = task.ledger.request if task is not None else ""
    stop_rules = [line.strip() for line in request.splitlines() if "stop" in line.lower() and "if" in line.lower()]
    excerpt = record.result_excerpt
    midpoint = min(len(excerpt), 2000)
    guidance: list[str] = []
    for event in session.events:
        if event.idx >= record.provenance.event_idx or event.kind is not EventKind.user_msg or not event.text:
            continue
        lowered = event.text.lower()
        if not any(marker in lowered for marker in ("agents.md", "claude.md", "<instructions>", "project-doc")):
            continue
        lines = [line.strip() for line in event.text.splitlines() if line.strip()]
        relevant = [
            line
            for line in lines
            if any(
                marker in line.lower()
                for marker in ("make ", "pytest", "test", "lint", "format", "typecheck", "generated", "workspace")
            )
        ]
        guidance.extend(relevant[:30])
    guidance = list(dict.fromkeys(guidance))[:40]
    return {
        "task": {"request": request, "explicit_stop_rules": stop_rules},
        # Current checkout guidance is deliberately not substituted for trace
        # history. Only guidance actually present before the failure is used.
        "repo": {
            "guidance_status": "observed" if guidance else "unavailable",
            "guidance_excerpt": guidance,
        },
        "failure": {
            "tool": record.tool,
            "op_kind": record.op_kind or "",
            "command": record.command or "",
            "result": {
                "exit": record.exit_code,
                "error_text": record.diagnostic,
                "out_head": excerpt[:midpoint],
                "out_tail": excerpt[-2000:] if len(excerpt) > midpoint else "",
                "output_truncated": record.output_truncated,
            },
            "shell": {
                "segments": record.shell_segments,
                "failing_segment": record.failing_segment,
                "attribution_reliable": record.attribution_reliable,
            },
        },
        "context": {
            "prior_relevant_ops": _failure_context(session, record.provenance.event_idx, after=False),
            "after": _failure_context(session, record.provenance.event_idx, after=True),
        },
        "digest_schema_version": DIGEST_SCHEMA_VERSION,
    }


def _positive(values: dict[str, FeatureValue], feature_id: str) -> bool:
    value = values.get(feature_id)
    if value is None or value.reason is not None or value.abstains:
        return False
    raw = value.value
    return isinstance(raw, (int, float)) and not isinstance(raw, bool) and float(raw) >= 0.7


def _failure_feature_eligible(feat: FeatureDef, record: FailureRecord, session: Session, state: dict[str, Any]) -> bool:
    fid = feat.id
    if fid == "failure.check.tool_contract_rejected":
        return not record.command and record.signals.get("tool_contract_rejected") is True
    if fid == "failure.check.edit_match_missing":
        return not record.command and record.signals.get("edit_match_missing") is True
    if fid == "failure.check.diagnostic_attributed":
        return len(record.shell_segments) > 1 and not record.attribution_reliable
    if fid == "failure.context.child_owned_target":
        return bool(session.parent_session_id and state["task"]["request"])
    if fid == "failure.recovery.same_cause_persisted":
        return record.recovery.attempts > 0
    return True


def _failure_features(
    annotator: Annotator,
    trace_id: str,
    session: Session,
    tasks: Sequence[Task],
    failure_feats: Sequence[FeatureDef],
) -> list[FeatureSet]:
    """Ask independent failure questions wide, then real dependencies only."""
    records = build_failure_records(trace_id, session)
    if not records or not failure_feats:
        return []
    first = [feat for feat in failure_feats if feat.id not in _FAILURE_DEPENDENT]
    by_id = {feat.id: feat for feat in failure_feats}
    states = [_failure_state(record, session, tasks) for record in records]
    first_items: list[tuple[dict[str, Any], Sequence[FeatureDef]]] = []
    skipped_per: list[dict[str, FeatureValue]] = []
    for record, state in zip(records, states, strict=True):
        guidance_available = state["repo"]["guidance_status"] == "observed"

        askable = [
            feat
            for feat in first
            if _failure_feature_eligible(feat, record, session, state)
            and (guidance_available or feat.id not in _FAILURE_GUIDANCE_FEATURES)
        ]
        skipped = {
            feat.id: _unanswered(feat, "insufficient_observability")
            for feat in first
            if feat.id not in {candidate.id for candidate in askable}
        }
        first_items.append((state, askable))
        skipped_per.append(skipped)
    values_per = annotator.ask_many(first_items)
    for values, skipped in zip(values_per, skipped_per, strict=True):
        values.update(skipped)
    dependent_items: list[tuple[int, dict[str, Any], list[FeatureDef]]] = []
    for i, (state, values) in enumerate(zip(states, values_per, strict=True)):
        wanted: list[str] = []
        if _positive(values, "failure.intent.user_named_identifier"):
            wanted.append("failure.intent.identifier_matches")
        if _positive(values, "failure.context.recipe_documented"):
            wanted.append("failure.context.recipe_followed")
        if state["context"]["after"]:
            wanted.append("failure.recovery.repeated_without_evidence")
            wanted.append("failure.recovery.passed_without_product_edit")
            record = records[i]
            if record.signals.get("cli_argument_rejected") is True or _positive(
                values, "failure.check.invocation_rejected"
            ):
                wanted.append("failure.recovery.corrected_argument")
            if record.signals.get("auth_refresh_blocked") is True or record.signals.get("iam_permission_named") is True:
                wanted.append("failure.recovery.stopped_on_access_block")
        feats = [by_id[fid] for fid in wanted if fid in by_id]
        if feats:
            dependent_items.append((i, state, feats))
    if dependent_items:
        extras = annotator.ask_many([(state, feats) for _i, state, feats in dependent_items])
        for (i, _state, _feats), extra in zip(dependent_items, extras, strict=True):
            values_per[i].update(extra)
    return [
        FeatureSet(scope="failure", object_id=record.provenance.stable_id, values=values)
        for record, values in zip(records, values_per, strict=True)
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
        s_tasks, s_episodes, s_sets = _annotate_session(annotator, trace.trace_id, session, config, mode=mode)
        parent_id = parent_of.get(session.session_id)
        parent_eps = episodes_by_session.get(parent_id) if parent_id else None
        if parent_eps:
            for task in s_tasks:
                attach_delegation(task, parent_eps, session.session_id)
        episodes_by_session[session.session_id] = s_episodes
        tasks += s_tasks
        episodes += s_episodes
        feature_sets += s_sets
    if annotator is not None:
        handover_feats = [f for f in load_feature_bank().features if f.scope == "handover"]
        feature_sets += _handover_features(annotator, trace, build_handovers(trace), handover_feats)
    return tasks, episodes, feature_sets, caps


def _handover_features(
    annotator: Annotator, trace: Trace, handovers: Sequence[HandoverRecord], feats: Sequence[FeatureDef]
) -> list[FeatureSet]:
    """Ask only questions whose literal evidence is present in this trace."""
    if not feats or not handovers:
        return []
    sessions = {s.session_id: s for s in [trace.root, *trace.subagents]}
    items: list[tuple[dict[str, Any], Sequence[FeatureDef]]] = []
    ids: list[str] = []
    skipped: list[dict[str, FeatureValue]] = []
    for row in handovers:
        parent = sessions[row.parent_id]
        spawn = next((e for e in row.events if e.kind == "spawn"), None)
        call = parent.events[spawn.event_idx] if spawn else None
        request = (call.tool_args.get("prompt") or call.tool_args.get("message")) if call else None
        request = request if isinstance(request, str) and row.request_visibility == "plaintext" else ""
        final_event = next((e for e in reversed(row.events) if e.kind == "final"), None)
        reply = ""
        if final_event and final_event.session_id in sessions:
            source_event = sessions[final_event.session_id].events[final_event.event_idx]
            reply = source_event.text or source_event.output or ""
        if row.reply_visibility != "plaintext":
            reply = ""
        steer_texts = []
        for steer in (e for e in row.events if e.kind == "steer"):
            source_event = sessions[steer.session_id].events[steer.event_idx]
            text = source_event.tool_args.get("message") or source_event.tool_args.get("prompt")
            if isinstance(text, str) and text:
                steer_texts.append(text)
        state = {
            "request": {"text": request},
            "steer": {"text": ""},
            "contract": {"before": request, "final": "\n".join([request, *steer_texts])},
            "reply": {"text": reply},
            "child": {"status": row.status, "failures": row.failure_ids, "checks": []},
            "parent": {"after": []},
        }
        allowed = []
        hidden: dict[str, FeatureValue] = {}
        for feat in feats:
            fid = feat.id
            unavailable = (
                ".steer." in fid
                or ".parent." in fid
                or fid == "handover.child.unsupported_completion"
                or (fid.startswith("handover.request.") and not request)
                or (fid == "handover.request.dependency_named" and not row.request_files)
                or (fid.startswith("handover.reply.") and not reply)
                or (fid == "handover.reply.blocker_next_step" and not row.failure_ids)
                or (fid == "handover.reply.request_coverage" and not request)
            )
            if unavailable:
                hidden[fid] = _unanswered(feat, "insufficient_observability")
            else:
                allowed.append(feat)
        ids.append(row.id)
        skipped.append(hidden)
        items.append((state, allowed))
    answers = annotator.ask_many(items)
    result = [
        FeatureSet(scope="handover", object_id=oid, values={**vals, **gap})
        for oid, vals, gap in zip(ids, answers, skipped, strict=True)
    ]
    steer_feats = [feat for feat in feats if feat.id.startswith("handover.steer.")]
    steer_items: list[tuple[dict[str, Any], Sequence[FeatureDef]]] = []
    steer_ids: list[str] = []
    for row in handovers:
        if row.request_visibility != "plaintext":
            continue
        parent = sessions[row.parent_id]
        spawn = next((e for e in row.events if e.kind == "spawn"), None)
        if spawn is None:
            continue
        before = parent.events[spawn.event_idx].tool_args.get("prompt") or parent.events[spawn.event_idx].tool_args.get(
            "message"
        )
        if not isinstance(before, str):
            continue
        for i, steer in enumerate(e for e in row.events if e.kind == "steer"):
            source = sessions[steer.session_id].events[steer.event_idx]
            message = source.tool_args.get("message") or source.tool_args.get("prompt")
            if not isinstance(message, str) or not message or message.startswith("gAAAA"):
                continue
            steer_items.append(({"contract": {"before": before}, "steer": {"text": message}}, steer_feats))
            steer_ids.append(f"{row.id}:steer:{i}")
            before += "\n" + message
    if steer_items:
        result.extend(
            FeatureSet(scope="handover", object_id=oid, values=values)
            for oid, values in zip(steer_ids, annotator.ask_many(steer_items), strict=True)
        )
    return result


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
    trace_id: str,
    session: Session,
    config: Config,
    *,
    mode: str,
) -> tuple[list[Task], list[Episode], list[FeatureSet]]:
    """Tasks, episodes and (when ``annotator`` is set) features for one session."""
    if "usage_estimated" in session.degraded:
        # Notification-only stubs have no transcript; do not spend JeV on them.
        return [], [], []
    if annotator is None:
        tasks = segment_tasks(session, config, semantic_mode=mode)
        return tasks, segment_episodes(session, tasks, config), []

    loaded = load_feature_bank()
    bank = loaded.features
    rel_feats = [f for f in bank if f.id.startswith(_REL_PREFIX)]
    task_feats = [f for f in bank if f.scope == "task"]
    episode_feats = [f for f in bank if f.scope == "episode"]
    turn_feats = [f for f in bank if f.scope == "turn" and not f.id.startswith(_REL_PREFIX)]
    failure_feats = [f for f in bank if f.scope == "failure"]

    candidates = [t for t in session.turns if t.user_input.kind in _CANDIDATE_KINDS]
    relationship = _relationship_answers(annotator, session, candidates, rel_feats)
    tasks = segment_tasks(session, config, semantic_mode=mode, relationship_answers=relationship)
    episodes = segment_episodes(session, tasks, config)

    feature_sets = _task_features(annotator, tasks, task_feats, loaded.routing)
    episode_sets = _episode_features(annotator, session, tasks, episodes, episode_feats)
    apply_phase_labels(episodes, episode_sets)
    feature_sets += episode_sets
    feature_sets += _turn_features(annotator, session.turns, turn_feats)
    feature_sets += _failure_features(annotator, trace_id, session, tasks, failure_feats)
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
        produce = _noul_value(fs.values.get(_PRODUCE_ARTIFACT))
        if purpose in {"investigate", "orient"} and produce is not None and produce >= _PRODUCE_GATE:
            purpose = "produce"
            old = fs.values.get(_PHASE_PURPOSE)
            if old is not None:
                new_val: Any = "produce"
                if isinstance(old.value, dict):
                    new_val = {**old.value, "choice": "produce"}
                fs.values[_PHASE_PURPOSE] = old.model_copy(update={"value": new_val, "source": "derived"})
        if activity is not None:
            ep.phase_activity = activity
        if purpose is not None:
            ep.phase_purpose = purpose


__all__ = [
    "Annotator",
    "annotate_trace",
    "apply_phase_labels",
    "feature_question",
    "questions_from_features",
    "visible_state",
]
