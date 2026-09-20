"""MECE waste partition — one primary cause per span (§7.2)."""

from __future__ import annotations

import math
from typing import Any

from agent_hotwash.config import Config
from agent_hotwash.diagnostics.cost_views import (
    CostView,
    Diagnosis,
    Money,
    invoice_of,
    monetary_enabled,
    reasoning_spend_of,
)
from agent_hotwash.diagnostics.effort import is_high_effort
from agent_hotwash.events import EventKind, PricingStatus, Session
from agent_hotwash.primitives.errors import classify_impediment
from agent_hotwash.semantic.results import FeatureSet, FeatureValue, stuck_window, thrashing_window
from agent_hotwash.structure.episodes import Episode
from agent_hotwash.structure.tasks import Task

_REQUIRE_FEATURES = (
    "episode.reasoning.requires_diagnosis",
    "episode.reasoning.requires_design_tradeoff",
    "episode.reasoning.requires_long_context_synthesis",
    "episode.reasoning.requires_domain_knowledge",
)

_NOVEL = "episode.progress.novel_output"
_TARGETS = "episode.progress.targets_named_component"
_DECLARES = "episode.claim.declares_success"
_CITES = "episode.claim.cites_verification"
_DEMAND = "episode.reasoning.demand"
_ACTIVITY = "episode.phase.activity"
_PURPOSE = "episode.phase.purpose"
_OUTCOME = "episode.outcome.kind"


def _index(features: list[FeatureSet] | None) -> dict[str, FeatureSet]:
    return {fs.object_id: fs for fs in (features or []) if fs.object_id}


def _fv(fs: FeatureSet | None, feature_id: str) -> FeatureValue | None:
    if fs is None:
        return None
    return fs.values.get(feature_id)


def _jev_abstain(fv: FeatureValue | None) -> bool:
    return bool(fv is not None and fv.abstains)


def _noul(fv: FeatureValue | None, default: float | None = None) -> float | None:
    if fv is None or fv.value is None:
        return default
    val = fv.value
    if isinstance(val, bool):
        return 1.0 if val else 0.0
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, dict) and val.get("noul") is not None:
        try:
            return float(val["noul"])
        except (TypeError, ValueError):
            return default
    return default


def _choice(fv: FeatureValue | None) -> str | None:
    if fv is None or fv.value is None:
        return None
    val = fv.value
    if isinstance(val, str):
        return val
    if isinstance(val, dict):
        c = val.get("choice") or val.get("value")
        return str(c) if c is not None else None
    return str(val)


def _prob_argmax(probabilities: Any) -> int | None:
    if not isinstance(probabilities, dict) or not probabilities:
        return None
    best_p: float | None = None
    best_i: int | None = None
    for key, raw in probabilities.items():
        try:
            idx = int(key)
            prob = float(raw)
        except (TypeError, ValueError):
            continue
        if best_p is None or prob > best_p:
            best_p = prob
            best_i = idx
    return best_i


def _demand_level(fv: FeatureValue | None) -> int | None:
    """Integer score level: round expected value; argmax breaks a .5 tie."""
    if fv is None or fv.value is None:
        return None
    val = fv.value
    if isinstance(val, (int, float)) and not isinstance(val, bool):
        score = float(val)
        lo = math.floor(score)
        hi = math.ceil(score)
        if lo == hi:
            return int(lo)
        if abs((score - lo) - (hi - score)) < 1e-9:
            argmax = _prob_argmax(fv.answer.get("probabilities") if isinstance(fv.answer, dict) else None)
            if argmax is not None:
                return argmax
        return int(hi if (hi - score) <= (score - lo) else lo)
    text = str(val)
    if text.startswith("0"):
        return 0
    if text.startswith("1"):
        return 1
    if text.startswith("2"):
        return 2
    if text.startswith("3"):
        return 3
    return None


def _fact(ep: Episode, key: str, default: Any = None) -> Any:
    return ep.facts.get(key, default)


def _atom_payload(ep: Episode, fs: FeatureSet | None) -> dict[str, Any]:
    novel = _noul(_fv(fs, _NOVEL))
    payload = dict(ep.facts)
    payload.setdefault("ops", list(ep.ops))
    payload.setdefault("artifacts", list(ep.artifacts))
    payload.setdefault("paths", list(ep.artifacts))
    if novel is not None:
        payload.setdefault("novel_output", novel)
    return payload


def _ep_invoice(ep: Episode, session: Session, config: Config) -> Money:
    model = session.model
    turn = next((t for t in session.turns if t.turn_id == ep.turn_id), None)
    if turn and turn.model_config_active and turn.model_config_active.model:
        model = turn.model_config_active.model
    entry, status = config.price_lookup(model)
    return invoice_of(ep.usage, entry, status)


def _low(val: float | None) -> bool:
    return val is not None and val < 0.3


def _overlap_count(val: Any) -> int | None:
    """``artifact_overlap`` is a list of paths on real atoms and an int in hand-written facts."""
    if val is None:
        return None
    if isinstance(val, (list, tuple, set)):
        return len(val)
    if isinstance(val, bool):
        return int(val)
    if isinstance(val, (int, float)):
        return int(val)
    return None


def _money_diag(
    did: str,
    money: Money,
    *,
    spans: list[str],
    evidence: dict[str, Any],
    informational: bool = False,
    waste: bool = True,
) -> Diagnosis:
    return Diagnosis(
        id=did,
        view=money.view,
        amount=money,
        pricing_status=money.pricing_status,
        evidence=evidence,
        spans=spans,
        group_id=f"{did}:{spans[0] if spans else 'x'}",
        informational=informational,
        counts_as_agent_waste=waste,
    )


def _external_kind(ep: Episode, session: Session) -> str | None:
    kind = _fact(ep, "env_impediment") or _fact(ep, "impediment")
    if kind:
        return str(kind)
    for ev in session.events:
        if ep.event_start <= ev.idx <= ep.event_end:
            found = classify_impediment(ev.error_text or ev.output or ev.text)
            if found:
                return found
    return None


def _after_compaction(ep: Episode, session: Session) -> bool:
    if ep.termination == "compaction" or _fact(ep, "after_compaction"):
        return True
    return any(ev.kind is EventKind.compaction and ev.idx <= ep.event_start for ev in session.events)


def partition_waste(
    session: Session,
    tasks: list[Task],
    episodes: list[Episode],
    features: list[FeatureSet] | None,
    existing_findings: list[Any],
    config: Config,
) -> list[Diagnosis]:
    """One primary waste cause per episode span, plus cross-cutting indicators."""
    feat = _index(features)
    claimed: set[str] = set()
    out: list[Diagnosis] = []
    finding_ids = {getattr(f, "id", None) for f in existing_findings}

    def _try(ep: Episode, did: str, money: Money, evidence: dict[str, Any], *, waste: bool = True) -> bool:
        if ep.episode_id in claimed:
            return False
        if not monetary_enabled(money.pricing_status):
            return False
        claimed.add(ep.episode_id)
        if _after_compaction(ep, session) and did == "DUPLICATE_WORK":
            evidence = {**evidence, "downgraded": True, "compaction_exculpatory": True}
        out.append(_money_diag(did, money, spans=[ep.episode_id], evidence=evidence, waste=waste))
        if did == "DUPLICATE_WORK" and evidence.get("downgraded"):
            out[-1].corroborates = ["COMPACTION_AMNESIA"]
        return True

    # --- EXTERNAL_BLOCK (primary but not agent waste) ---------------------
    for ep in episodes:
        kind = _external_kind(ep, session)
        if not kind:
            continue
        money = _ep_invoice(ep, session, config)
        if not monetary_enabled(money.pricing_status):
            continue
        _try(ep, "EXTERNAL_BLOCK", money, {"impediment": kind}, waste=False)

    # --- POST_COMPLETION_WORK ---------------------------------------------
    completed_at: int | None = None
    for i, ep in enumerate(episodes):
        fs = feat.get(ep.episode_id)
        declares = _noul(_fv(fs, _DECLARES), _fact(ep, "declares_success", 0.0))
        cites = _noul(_fv(fs, _CITES), 1.0 if _fact(ep, "cites_verification") else 0.0)
        if _jev_abstain(_fv(fs, _DECLARES)) or _jev_abstain(_fv(fs, _CITES)):
            continue
        if (declares or 0) >= 0.7 and (cites or 0) >= 0.7:
            completed_at = i
            break
    if completed_at is not None:
        later = episodes[completed_at + 1 :]
        trailing_cites = False
        for ep in later:
            fs = feat.get(ep.episode_id)
            if _jev_abstain(_fv(fs, _CITES)):
                continue
            cites = _noul(_fv(fs, _CITES), 1.0 if _fact(ep, "cites_verification") else 0.0)
            if (cites or 0) >= 0.7:
                trailing_cites = True
        if later and not trailing_cites:
            no_user = all(ep.trigger != "user_request" for ep in later)
            outstanding = any(_fact(ep, "outstanding_failure") for ep in later)
            if no_user and not outstanding:
                for ep in later:
                    money = _ep_invoice(ep, session, config)
                    _try(
                        ep,
                        "POST_COMPLETION_WORK",
                        money,
                        {"after_completion_index": completed_at, "note": "observed, not savings"},
                    )

    # --- DUPLICATE_WORK ---------------------------------------------------
    for ep in episodes:
        fs = feat.get(ep.episode_id)
        if _jev_abstain(_fv(fs, _NOVEL)):
            continue
        novel = _noul(_fv(fs, _NOVEL), float(_fact(ep, "novel_output", 1.0)))
        repeats = float(_fact(ep, "repeat_count", 0) or _fact(ep, "failure_signature_repeats", 0) or 0)
        reread = float(_fact(ep, "paths_reread_unchanged", 0) or 0)
        if (repeats >= 2 or reread >= 1) and novel is not None and novel < 0.3:
            money = _ep_invoice(ep, session, config)
            # lower bound: this atom; upper bound same atom when we cannot split calls
            lower = money
            evidence = {
                "repeat_count": repeats,
                "paths_reread_unchanged": reread,
                "novel_output": novel,
                "lower_bound": lower.as_dict(),
                "upper_bound": money.as_dict(),
            }
            money = Money(
                amount=money.amount,
                view=CostView.invoice,
                pricing_status=money.pricing_status,
                amount_low=lower.amount,
                amount_high=money.amount,
            )
            _try(ep, "DUPLICATE_WORK", money, evidence)

    # --- INEFFECTIVE_ITERATION --------------------------------------------
    atoms = [_atom_payload(ep, feat.get(ep.episode_id)) for ep in episodes]
    k_stuck = 2
    k_thrash = 2
    has_stuck = stuck_window(k_stuck, atoms)
    has_thrash = thrashing_window(k_thrash, atoms)
    no_adapt = "NO_ADAPT_RETRY" in finding_ids or any(_fact(ep, "no_adapt_retry") for ep in episodes)
    if has_stuck or has_thrash or no_adapt:
        window_eps = [ep for ep in episodes if ep.episode_id not in claimed]
        if has_stuck:
            # consecutive tail matching stuck criteria
            window_eps = [
                ep
                for ep in episodes
                if float(_fact(ep, "failure_signature_repeats", 0) or 0) > 0
                and float(_noul(_fv(feat.get(ep.episode_id), _NOVEL), float(_fact(ep, "novel_output", 1))) or 1) < 0.3
            ] or window_eps
        for ep in window_eps:
            if any(_jev_abstain(_fv(feat.get(ep.episode_id), fid)) for fid in (_NOVEL,)):
                continue
            money = _ep_invoice(ep, session, config)
            _try(
                ep,
                "INEFFECTIVE_ITERATION",
                money,
                {"stuck_window": has_stuck, "thrashing_window": has_thrash, "no_adapt_retry": no_adapt},
            )

    # --- IRRELEVANT_WORK --------------------------------------------------
    for ep in episodes:
        fs = feat.get(ep.episode_id)
        if _jev_abstain(_fv(fs, _TARGETS)) or _jev_abstain(_fv(fs, _PURPOSE)):
            continue
        purpose = _choice(_fv(fs, _PURPOSE)) or ep.phase_purpose or _fact(ep, "phase_purpose")
        if purpose == "orient":
            continue
        targets = _noul(_fv(fs, _TARGETS))
        overlap = _overlap_count(_fact(ep, "artifact_overlap", None))
        if overlap is None:
            overlap = 0 if _low(targets) else 1
        if _low(targets) and overlap == 0:
            money = _ep_invoice(ep, session, config)
            _try(ep, "IRRELEVANT_WORK", money, {"targets_named_component": targets, "artifact_overlap": overlap})

    # --- EXCESS_REASONING_TIER --------------------------------------------
    for ep in episodes:
        fs = feat.get(ep.episode_id)
        if _jev_abstain(_fv(fs, _DEMAND)) or any(_jev_abstain(_fv(fs, r)) for r in _REQUIRE_FEATURES):
            continue
        demand = _demand_level(_fv(fs, _DEMAND))
        if demand is None:
            raw = _fact(ep, "reasoning_demand")
            demand = int(raw) if isinstance(raw, (int, float)) else None
        requires_low = True
        for rid in _REQUIRE_FEATURES:
            n = _noul(_fv(fs, rid), float(_fact(ep, rid.split(".")[-1], 0.0)))
            if n is None or n >= 0.3:
                requires_low = False
                break
        turn = next((t for t in session.turns if t.turn_id == ep.turn_id), None)
        model = (turn.model_config_active.model if turn and turn.model_config_active else None) or session.model
        effort = turn.model_config_active.reasoning_effort if turn and turn.model_config_active else None
        high = is_high_effort(model, effort, config)
        if demand is not None and demand <= 1 and requires_low and high:
            entry, status = config.price_lookup(model)
            money = reasoning_spend_of(ep.usage, entry, status)
            if monetary_enabled(status):
                evidence = {
                    "demand": demand,
                    "effort": effort,
                    "model": model,
                    "label": money.label,
                    "caveats": [
                        "latency and tool-call cost not included",
                        "avoidable amount unknown without matched runs",
                    ],
                }
                _try(ep, "EXCESS_REASONING_TIER", money, evidence)

    # --- COORDINATION_OVERHEAD --------------------------------------------
    for ep in episodes:
        coord_ops = [op for op in ep.ops if op in {"agent.spawn", "agent.wait", "agent.message"}]
        if not coord_ops:
            continue
        reuse = _fact(ep, "child_result_reuse")
        reuse_n = float(reuse) if isinstance(reuse, (int, float)) else (1.0 if reuse else 0.0)
        if reuse_n >= 0.5:
            continue
        money = _ep_invoice(ep, session, config)
        _try(ep, "COORDINATION_OVERHEAD", money, {"ops": coord_ops, "child_result_reuse": reuse_n})

    # --- Cross-cutting (no dollars) ---------------------------------------
    trailing = episodes[-2:] if len(episodes) >= 2 else episodes[-1:]
    if trailing:
        no_progress = True
        for ep in trailing:
            fs = feat.get(ep.episode_id)
            outcome = _choice(_fv(fs, _OUTCOME)) or _fact(ep, "outcome") or ep.phase_purpose
            if (
                outcome not in {"no_observable_progress", None}
                and _fact(ep, "no_observable_progress") is not True
                and outcome in {"goal_step_completed", "partial_progress"}
            ):
                no_progress = False
            if _fact(ep, "no_observable_progress") is False:
                no_progress = False
        if no_progress and any(
            _fact(ep, "no_observable_progress")
            or _choice(_fv(feat.get(ep.episode_id), _OUTCOME)) == "no_observable_progress"
            for ep in trailing
        ):
            out.append(
                Diagnosis(
                    id="LOW_YIELD_TAIL",
                    view=CostView.invoice,
                    amount=None,
                    pricing_status=PricingStatus.unknown,
                    evidence={"trailing": [ep.episode_id for ep in trailing]},
                    spans=[ep.episode_id for ep in trailing],
                    group_id="low_yield_tail",
                    informational=True,
                    counts_as_agent_waste=False,
                )
            )

    if "RUNAWAY_SESSION" in finding_ids:
        out.append(
            Diagnosis(
                id="RUNAWAY_SESSION",
                view=CostView.invoice,
                amount=None,
                pricing_status=PricingStatus.unknown,
                evidence={"source": "detector"},
                spans=[session.session_id],
                group_id="runaway",
                informational=True,
                counts_as_agent_waste=False,
            )
        )
    if "CONTEXT_ROT" in finding_ids:
        late_progress = False
        if episodes:
            last = episodes[len(episodes) * 2 // 3 :]
            for ep in last:
                fs = feat.get(ep.episode_id)
                novel = _noul(_fv(fs, _NOVEL), float(_fact(ep, "novel_output", 0.0)))
                produces = _noul(_fv(fs, "episode.progress.produces_requested_artifact"), 0.0)
                if (novel or 0) >= 0.5 or (produces or 0) >= 0.5:
                    late_progress = True
        if late_progress:
            out.append(
                Diagnosis(
                    id="CONTEXT_ROT",
                    view=CostView.invoice,
                    amount=None,
                    pricing_status=PricingStatus.unknown,
                    evidence={"suppressed": True, "late_progress": True},
                    spans=[session.session_id],
                    group_id="context_rot",
                    informational=True,
                    reason="suppressed",
                    counts_as_agent_waste=False,
                )
            )
        else:
            out.append(
                Diagnosis(
                    id="CONTEXT_ROT",
                    view=CostView.invoice,
                    amount=None,
                    pricing_status=PricingStatus.unknown,
                    evidence={"upgraded": True, "source": "detector"},
                    spans=[session.session_id],
                    group_id="context_rot",
                    informational=True,
                    counts_as_agent_waste=False,
                )
            )

    return out


__all__ = ["partition_waste"]
