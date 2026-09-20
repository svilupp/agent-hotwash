"""Finding-group precedence with existing detectors (§7.5)."""

from __future__ import annotations

from typing import Any

from agent_hotwash.config import Config
from agent_hotwash.diagnostics.cost_views import CostView, Diagnosis, Money
from agent_hotwash.events import PricingStatus
from agent_hotwash.semantic.results import FeatureSet, declared_success_without_observed_verification
from agent_hotwash.structure.episodes import Episode
from agent_hotwash.structure.tasks import Task

_SEMANTIC_EDGES = frozenset({"unrelated", "sibling_same_area", "depends_on", "corrects"})


def _index(features: list[FeatureSet] | None) -> dict[str, FeatureSet]:
    return {fs.object_id: fs for fs in (features or []) if fs.object_id}


def _noul(fs: FeatureSet | None, feature_id: str) -> float:
    if fs is None:
        return 0.0
    fv = fs.values.get(feature_id)
    if fv is None or fv.value is None:
        return 0.0
    val = fv.value
    if isinstance(val, bool):
        return 1.0 if val else 0.0
    if isinstance(val, (int, float)):
        return float(val)
    return 0.0


def _answered(fs: FeatureSet | None, feature_id: str) -> bool:
    """True when the feature was actually answered (no abstention reason, non-null value)."""
    if fs is None:
        return False
    fv = fs.values.get(feature_id)
    return fv is not None and fv.reason is None and fv.value is not None


def _from_finding(finding: Any) -> Diagnosis:
    spans = []
    for sp in getattr(finding, "spans", []) or []:
        sid = getattr(sp, "session_id", "")
        idx = getattr(sp, "event_idx", "")
        spans.append(f"{sid}:{idx}")
    fid = getattr(finding, "id", "UNKNOWN")
    return Diagnosis(
        id=fid,
        view=CostView.invoice,
        amount=None,
        pricing_status=PricingStatus.unknown,
        evidence=dict(getattr(finding, "evidence", None) or {}),
        spans=spans or [getattr(finding, "session_id", "")],
        group_id=fid,
        informational=fid
        in {"RUNAWAY_SESSION", "CONTEXT_ROT", "KITCHEN_SINK", "UNVERIFIED_COMPLETION", "COMPACTION_AMNESIA"},
        counts_as_agent_waste=False,
        tier=getattr(finding, "kind", None),
    )


def apply_precedence(
    diagnoses: list[Diagnosis],
    *,
    tasks: list[Task],
    episodes: list[Episode],
    features: list[FeatureSet] | None,
    existing_findings: list[Any],
    config: Config,
    semantic_on: bool,
) -> list[Diagnosis]:
    """Group findings, set supersedes/corroborates, keep one economic amount per group."""
    _ = config
    feat = _index(features)
    by_id: dict[str, list[Diagnosis]] = {}
    converted = [_from_finding(f) for f in existing_findings]
    existing_ids = {d.id for d in converted}
    merged = list(diagnoses)
    already = {d.id for d in diagnoses}
    keep_detector_ids = {
        "KITCHEN_SINK",
        "UNVERIFIED_COMPLETION",
        "COMPACTION_AMNESIA",
        "RUNAWAY_SESSION",
        "CONTEXT_ROT",
    }
    for d in converted:
        duplicate = any(x.id == d.id and x.spans == d.spans for x in merged)
        if not duplicate and (d.id not in already or d.id in keep_detector_ids):
            merged.append(d)

    for d in merged:
        by_id.setdefault(d.id, []).append(d)

    semantic_relation = any(t.edge_to_prev in _SEMANTIC_EDGES for t in tasks)

    # Semantic task relation supersedes KITCHEN_SINK.
    if semantic_relation:
        for d in by_id.get("KITCHEN_SINK", []):
            d.informational = True
            d.counts_as_agent_waste = False
            d.amount = None
        for primary in merged:
            if primary.id not in {"PHASE_SPEND", "KITCHEN_SINK"} and primary.counts_as_agent_waste:
                primary.superseded_ids = list(dict.fromkeys([*primary.superseded_ids, "KITCHEN_SINK"]))
                break
        else:
            # Still record supersession on the kitchen-sink row itself.
            for d in by_id.get("KITCHEN_SINK", []):
                d.evidence = {**d.evidence, "superseded_by": "semantic_task_relation"}

    # cites_verification in trailing atoms vetoes POST_COMPLETION_WORK / LOW_YIELD_TAIL.
    trailing_cites = False
    if episodes:
        tail = episodes[len(episodes) * 2 // 3 :] or episodes[-1:]
        for ep in tail:
            fs = feat.get(ep.episode_id)
            cites = _noul(fs, "episode.claim.cites_verification")
            if cites >= 0.7 or ep.facts.get("cites_verification"):
                trailing_cites = True
    if trailing_cites:
        merged = [d for d in merged if d.id not in {"POST_COMPLETION_WORK", "LOW_YIELD_TAIL"}]

    # DUPLICATE_WORK after compaction: already flagged in waste; corroborate COMPACTION_AMNESIA.
    for d in by_id.get("DUPLICATE_WORK", []):
        if d.evidence.get("downgraded") or d.evidence.get("compaction_exculpatory"):
            d.corroborates = list(dict.fromkeys([*d.corroborates, "COMPACTION_AMNESIA"]))
            d.informational = True  # downgraded: keep as supporting, not primary dollars
            if "COMPACTION_AMNESIA" not in existing_ids and not by_id.get("COMPACTION_AMNESIA"):
                merged.append(
                    Diagnosis(
                        id="COMPACTION_AMNESIA",
                        view=CostView.invoice,
                        amount=None,
                        pricing_status=PricingStatus.unknown,
                        evidence={"corroborated_by": "DUPLICATE_WORK"},
                        spans=list(d.spans),
                        group_id=d.group_id or "compaction",
                        informational=True,
                        counts_as_agent_waste=False,
                    )
                )
            d.group_id = d.group_id or "compaction"

    # RUNAWAY_SESSION stays a hard guard, grouped with any overlapping tail.
    runaway = [d for d in merged if d.id == "RUNAWAY_SESSION"]
    tails = [d for d in merged if d.id == "LOW_YIELD_TAIL"]
    if runaway and tails:
        gid = "runaway"
        for d in runaway + tails:
            d.group_id = gid
        runaway[0].superseded_ids = list(dict.fromkeys([*runaway[0].superseded_ids, "LOW_YIELD_TAIL"]))

    # Progress facts: CONTEXT_ROT with reason=suppressed is dropped; else kept.
    kept: list[Diagnosis] = []
    for d in merged:
        if d.id == "CONTEXT_ROT" and (d.reason == "suppressed" or d.evidence.get("suppressed")):
            continue
        kept.append(d)
    merged = kept

    # UNVERIFIED_COMPLETION → semantic composite, same id, tier=semantic — but
    # only when the composite is *supported*: ``declares_success`` was actually
    # answered on at least one atom. Unsupported (unknown/api_error/no features)
    # leaves the deterministic finding untouched; supported-but-false suppresses it.
    if semantic_on and any(d.id == "UNVERIFIED_COMPLETION" for d in merged):
        declares = 0.0
        verified = False
        supported = False
        for ep in episodes:
            fs = feat.get(ep.episode_id)
            if _answered(fs, "episode.claim.declares_success"):
                supported = True
                declares = max(declares, _noul(fs, "episode.claim.declares_success"))
            if ep.facts.get("verification") or ep.facts.get("verification_fact"):
                verified = True
        if supported:
            composite = declared_success_without_observed_verification(declares, verified)
            replaced: list[Diagnosis] = []
            for d in merged:
                if d.id != "UNVERIFIED_COMPLETION":
                    replaced.append(d)
                    continue
                if not composite:
                    continue  # semantic evidence contradicts the detector: suppress
                d.tier = "semantic"
                d.evidence = {
                    **d.evidence,
                    "composite": "declared_success_without_observed_verification",
                    "declares_success": declares,
                    "verification_fact": verified,
                }
                replaced.append(d)
            merged = replaced

    # CONTINUATION_BURDEN owns its turn: drop overlapping agent-waste dollars (§7.2 MECE).
    burden_spans = {s for d in merged if d.id == "CONTINUATION_BURDEN" for s in d.spans}
    if burden_spans:
        kept_mece: list[Diagnosis] = []
        for d in merged:
            if d.id == "CONTINUATION_BURDEN":
                kept_mece.append(d)
                continue
            if d.id in {"PHASE_SPEND", "EXTERNAL_BLOCK", "CONTEXT_CARRYOVER"} or d.informational:
                kept_mece.append(d)
                continue
            if d.counts_as_agent_waste and burden_spans.intersection(d.spans):
                d.amount = None
                d.counts_as_agent_waste = False
                d.informational = True
                d.evidence = {**d.evidence, "superseded_by": "CONTINUATION_BURDEN"}
                for b in merged:
                    if b.id == "CONTINUATION_BURDEN":
                        b.superseded_ids = list(dict.fromkeys([*b.superseded_ids, d.id]))
            kept_mece.append(d)
        merged = kept_mece

    # Rebuild by_id after filtering.
    groups: dict[str, list[Diagnosis]] = {}
    for d in merged:
        gid = d.group_id or d.id
        d.group_id = gid
        groups.setdefault(gid, []).append(d)

    # One economic amount per group: keep the first waste/monetary member's amount.
    for members in groups.values():
        primary: Diagnosis | None = None
        for d in members:
            if d.amount is not None and not d.informational:
                primary = d
                break
        if primary is None:
            for d in members:
                if d.amount is not None:
                    primary = d
                    break
        for d in members:
            if primary is not None and d is not primary and d.amount is not None:
                d.evidence = {
                    **d.evidence,
                    "group_amount_on": primary.id,
                    "dropped_amount": d.amount.as_dict() if isinstance(d.amount, Money) else d.amount,
                }
                d.amount = None
    return merged


__all__ = ["apply_precedence"]
