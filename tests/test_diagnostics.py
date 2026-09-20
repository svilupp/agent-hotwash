"""Diagnostics layer: cost views, MECE waste rules, report contract."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal

from agent_hotwash.aggregate import MonthlyRollup
from agent_hotwash.analytics import analyze
from agent_hotwash.config import Config, load_config
from agent_hotwash.detectors.registry import Finding, Severity, SpanRef
from agent_hotwash.diagnostics import diagnose, invoice_of, session_invoice, walk_monetary
from agent_hotwash.diagnostics.cost_views import CostView, build_cost_views, phase_spend_diagnoses
from agent_hotwash.events import (
    AgentKind,
    Event,
    EventKind,
    ModelCall,
    ModelConfig,
    PricingStatus,
    Provenance,
    Session,
    Trace,
    Turn,
    TurnStatus,
    Usage,
    UserInput,
)
from agent_hotwash.report.csv_writer import _BASE_COLUMNS, render_csv
from agent_hotwash.report.html import render_html
from agent_hotwash.report.json_writer import report_to_dict
from agent_hotwash.report.model import Report, ReportMeta, RunResult, StructureSection
from agent_hotwash.report.table import render_table
from agent_hotwash.semantic.results import FeatureSet, FeatureValue, Reason
from agent_hotwash.structure.episodes import Episode, Termination, Trigger
from agent_hotwash.structure.ledger import Ledger
from agent_hotwash.structure.tasks import EdgeKind, Task

_TS0 = datetime(2026, 9, 1, 12, 0, tzinfo=UTC)


def _exact_config(**over: object) -> Config:
    data = load_config().model_dump()
    data["pricing"] = {
        "test-model": {
            "input": 1.0,
            "output": 2.0,
            "cache_read": 0.1,
            "cache_write": 1.25,
            "as_of": "2026-01-01",
        },
        "gpt-5.6-luna": {
            "input": 1.0,
            "output": 2.0,
            "cache_read": 0.1,
            "cache_write": 1.25,
            "as_of": "2026-09-01",
        },
    }
    data["diagnostics"] = {**data["diagnostics"], "min_support": 1}
    data.update(over)
    return Config.model_validate(data)


def _usage(*, inp: int = 0, out: int = 0, cr: int = 0, cw: int = 0, reasoning: int = 0) -> Usage:
    return Usage(input=inp, output=out, cache_read=cr, cache_write=cw, reasoning_output=reasoning)


def _call(turn_id: str, rid: str, usage: Usage, *, start: int = 0, end: int = 0, minutes: int = 0) -> ModelCall:
    ts = _TS0 + timedelta(minutes=minutes)
    return ModelCall(
        response_id=rid,
        turn_id=turn_id,
        event_start=start,
        event_end=end,
        usage=usage,
        ts_start=ts,
        ts_end=ts + timedelta(seconds=5),
    )


def _turn(
    turn_id: str,
    session_id: str,
    calls: list[ModelCall],
    *,
    text: str = "do it",
    kind: Literal["user", "delegation", "injected", "none"] = "user",
    start: int = 0,
    end: int = 0,
    effort: str = "high",
    model: str = "test-model",
    compactions: int = 0,
    ctx: int | None = 200_000,
) -> Turn:
    return Turn(
        turn_id=turn_id,
        session_id=session_id,
        event_start=start,
        event_end=end,
        status=TurnStatus.completed,
        user_input=UserInput(text=text, kind=kind),
        model_config_active=ModelConfig(model=model, reasoning_effort=effort),
        model_calls=calls,
        compactions=compactions,
        context_window_tokens=ctx,
        ts_start=_TS0,
        ts_end=_TS0 + timedelta(minutes=1),
    )


def _session(events: list[Event], turns: list[Turn], session_id: str = "s0", model: str = "test-model") -> Session:
    for i, ev in enumerate(events):
        ev.idx = i
    return Session(
        session_id=session_id,
        agent=AgentKind.unknown,
        events=events,
        turns=turns,
        model=model,
    )


def _trace(session: Session, *, subagents: list[Session] | None = None, notes: list[str] | None = None) -> Trace:
    return Trace(
        trace_id="t0",
        agent=AgentKind.unknown,
        model=session.model,
        root=session,
        subagents=subagents or [],
        provenance=Provenance(
            source_format="claude_native",
            detector_confidence="high",
            root_path=Path("/tmp/x"),
            notes=notes or [],
            thread_linkage="partial" if notes else "full",
        ),
    )


def _ep(
    eid: str,
    *,
    usage: Usage,
    task_id: str = "s0:task0",
    turn_id: str = "t1",
    rids: list[str | None] | None = None,
    ops: list[str] | None = None,
    facts: dict | None = None,
    trigger: Trigger = "user_request",
    termination: Termination = "end_of_file",
    start: int = 0,
    end: int = 0,
) -> Episode:
    return Episode(
        episode_id=eid,
        task_id=task_id,
        turn_id=turn_id,
        response_ids=rids or ["r1"],
        event_start=start,
        event_end=end,
        ops=ops or [],
        usage=usage,
        trigger=trigger,
        termination=termination,
        facts=facts or {},
    )


def _task(session_id: str, turn: Turn, n: int = 0, edge: EdgeKind | None = None) -> Task:
    return Task(
        task_id=f"{session_id}:task{n}",
        session_id=session_id,
        turns=[turn],
        ledger=Ledger(request=turn.user_input.text),
        edge_to_prev=edge,
    )


def _fv(fid: str, value: object, confidence: float = 0.9, reason: Reason | None = None) -> FeatureValue:
    return FeatureValue(id=fid, value=value, confidence=confidence, reason=reason, source="jev")


def _fs(scope: str, oid: str, **values: object) -> FeatureSet:
    out: dict[str, FeatureValue] = {}
    for k, v in values.items():
        fid = k.replace("__", ".")
        if isinstance(v, FeatureValue):
            out[fid] = v
        else:
            out[fid] = _fv(fid, v)
    return FeatureSet(scope=scope, object_id=oid, values=out)


def _ids(diags: list, *wanted: str) -> set[str]:
    return {d.id for d in diags if d.id in wanted}


# ---------------------------------------------------------------------------
# Cost views
# ---------------------------------------------------------------------------


def test_invoice_of_never_adds_reasoning() -> None:
    cfg = _exact_config()
    entry, status = cfg.price_lookup("test-model")
    money = invoice_of(_usage(inp=1_000_000, out=1_000_000, reasoning=1_000_000), entry, status)
    # 1M input @1 + 1M output @2 = 3; reasoning would have added +2 if wrongly included
    assert money.amount == 3.0
    assert money.view is CostView.invoice
    assert money.pricing_status is PricingStatus.exact


def test_session_invoice_dedups_response_id() -> None:
    cfg = _exact_config()
    u = _usage(inp=1_000_000)
    t = _turn(
        "t1",
        "s0",
        [_call("t1", "same", u, start=0, end=0), _call("t1", "same", u, start=1, end=1)],
    )
    sess = _session([Event(kind=EventKind.assistant_msg, text="x")], [t])
    inv = session_invoice(sess, cfg)
    assert inv.amount == 1.0


def test_phase_spend_sum_equals_session_invoice() -> None:
    cfg = _exact_config()
    u1 = _usage(inp=1_000_000)
    u2 = _usage(inp=2_000_000)
    t = _turn("t1", "s0", [_call("t1", "r1", u1, start=0, end=0), _call("t1", "r2", u2, start=1, end=1)])
    sess = _session([], [t])
    eps = [
        _ep("s0:ep0", usage=u1, rids=["r1"]),
        _ep("s0:ep1", usage=u2, rids=["r2"], turn_id="t1"),
    ]
    diags = diagnose(_trace(sess), [_task("s0", t)], eps, [], [], cfg)
    phase = [d for d in diags if d.id == "PHASE_SPEND"]
    billed = session_invoice(sess, cfg)
    total = sum(d.amount.amount or 0.0 for d in phase if d.amount)
    assert billed.amount is not None
    assert abs(total - billed.amount) < 1e-6


def _amt(d: object) -> float:
    money = getattr(d, "amount", None)
    assert money is not None
    return float(money.amount or 0.0)


def _mixed_model_config() -> Config:
    data = load_config().model_dump()
    data["pricing"] = {
        "model-a": {"input": 1.0, "output": 0.0, "cache_read": 0.0, "cache_write": 0.0, "as_of": "2026-01-01"},
        "model-b": {"input": 10.0, "output": 0.0, "cache_read": 0.0, "cache_write": 0.0, "as_of": "2026-01-01"},
    }
    return Config.model_validate(data)


def test_phase_spend_prices_each_episode_at_its_own_model() -> None:
    """A/B/A three-turn thread: $1 / $10 / $1, not $1 / $1 / $10; total conserved."""
    cfg = _mixed_model_config()
    u = _usage(inp=1_000_000)
    turns = [
        _turn("t1", "s0", [_call("t1", "r1", u, start=0, end=0)], model="model-a", start=0, end=0),
        _turn("t2", "s0", [_call("t2", "r2", u, start=1, end=1, minutes=1)], model="model-b", start=1, end=1),
        _turn("t3", "s0", [_call("t3", "r3", u, start=2, end=2, minutes=2)], model="model-a", start=2, end=2),
    ]
    sess = _session([], turns, model="model-a")
    eps = [
        _ep("s0:ep0", usage=u, rids=["r1"], turn_id="t1"),
        _ep("s0:ep1", usage=u, rids=["r2"], turn_id="t2"),
        _ep("s0:ep2", usage=u, rids=["r3"], turn_id="t3"),
    ]
    rows = phase_spend_diagnoses(sess, cfg, eps)
    by_ep = {d.evidence["episode_id"]: d for d in rows if "episode_id" in d.evidence}
    assert [_amt(by_ep[e]) for e in ("s0:ep0", "s0:ep1", "s0:ep2")] == [1.0, 10.0, 1.0]
    assert by_ep["s0:ep1"].evidence["models"] == ["model-b"]
    assert not any(d.evidence.get("unallocated") for d in rows)
    billed = session_invoice(sess, cfg)
    assert billed.amount == 12.0
    assert abs(sum(_amt(d) for d in rows) - 12.0) < 1e-9


def test_phase_spend_unallocated_calls_get_their_own_row() -> None:
    """A call no atom owns is reported as ``unallocated``, not folded into the last atom."""
    cfg = _mixed_model_config()
    u = _usage(inp=1_000_000)
    turns = [
        _turn("t1", "s0", [_call("t1", "r1", u, start=0, end=0)], model="model-a", start=0, end=0),
        _turn("t2", "s0", [_call("t2", "r-orphan", u, start=1, end=1, minutes=1)], model="model-b", start=1, end=1),
    ]
    sess = _session([], turns, model="model-a")
    eps = [_ep("s0:ep0", usage=u, rids=["r1"], turn_id="t1")]
    rows = phase_spend_diagnoses(sess, cfg, eps)
    assert len(rows) == 2
    atom, orphan = rows
    assert _amt(atom) == 1.0
    assert orphan.evidence["unallocated"] is True
    assert orphan.evidence["response_ids"] == ["r-orphan"]
    assert _amt(orphan) == 10.0
    assert abs(sum(_amt(d) for d in rows) - (session_invoice(sess, cfg).amount or 0.0)) < 1e-9


def test_every_monetary_field_has_view_and_pricing_status() -> None:
    cfg = _exact_config()
    u = _usage(inp=1_000_000, out=10, reasoning=4)
    t = _turn("t1", "s0", [_call("t1", "r1", u)])
    sess = _session([], [t])
    ep = _ep("s0:ep0", usage=u, rids=["r1"], facts={"repeat_count": 3, "novel_output": 0.1})
    diags = diagnose(_trace(sess), [_task("s0", t)], [ep], [], [], cfg)
    views = build_cost_views(_trace(sess), cfg, episodes=[ep])
    for money in [*walk_monetary(views), *walk_monetary(diags)]:
        assert money.view in CostView
        assert money.pricing_status in PricingStatus


def test_tree_rollup_partial_missing_parent() -> None:
    cfg = _exact_config()
    u = _usage(inp=1_000_000)
    t = _turn("t1", "s0", [_call("t1", "r1", u)])
    sess = _session([], [t])
    views = build_cost_views(_trace(sess, notes=["parent not in input"]), cfg)
    assert views.rollup_status == "partial"
    assert views.invoice.view is CostView.invoice


def test_estimated_pricing_disables_monetary_waste() -> None:
    data = load_config().model_dump()
    data["pricing"] = {"test-model": {"input": 1.0, "output": 2.0, "cache_read": 0.1, "cache_write": 0.0}}
    data["diagnostics"] = {**data["diagnostics"], "min_support": 1}
    cfg = Config.model_validate(data)
    u = _usage(inp=1_000_000)
    t = _turn("t1", "s0", [_call("t1", "r1", u)])
    sess = _session([], [t])
    ep = _ep("s0:ep0", usage=u, facts={"repeat_count": 4, "novel_output": 0.05})
    diags = diagnose(_trace(sess), [_task("s0", t)], [ep], [], [], cfg)
    assert "DUPLICATE_WORK" not in {d.id for d in diags}
    phase = [d for d in diags if d.id == "PHASE_SPEND"]
    assert phase
    assert phase[0].amount is not None
    assert phase[0].amount.pricing_status is PricingStatus.estimated


# ---------------------------------------------------------------------------
# Waste rules — positive / negative / abstention
# ---------------------------------------------------------------------------


def _run_rule(ep: Episode, features: list[FeatureSet], *, findings: list | None = None, **turn_kw):
    cfg = _exact_config()
    t = _turn("t1", "s0", [_call("t1", "r1", ep.usage)], **turn_kw)
    sess = _session([], [t])
    diags = diagnose(_trace(sess), [_task("s0", t)], [ep], features, findings or [], cfg)
    return diags


def test_duplicate_work_positive_negative_abstain() -> None:
    u = _usage(inp=1_000_000)
    pos = _ep("s0:ep0", usage=u, facts={"repeat_count": 3, "novel_output": 0.1})
    pos_fs = [_fs("episode", "s0:ep0", episode__progress__novel_output=0.1)]
    assert "DUPLICATE_WORK" in _ids(_run_rule(pos, pos_fs), "DUPLICATE_WORK")

    neg = _ep("s0:ep0", usage=u, facts={"repeat_count": 0, "novel_output": 0.9})
    neg_fs = [_fs("episode", "s0:ep0", episode__progress__novel_output=0.9)]
    assert "DUPLICATE_WORK" not in _ids(_run_rule(neg, neg_fs), "DUPLICATE_WORK")

    band = [
        _fs(
            "episode",
            "s0:ep0",
            episode__progress__novel_output=_fv("episode.progress.novel_output", 0.1, confidence=0.5),
        )
    ]
    assert "DUPLICATE_WORK" not in _ids(_run_rule(pos, band), "DUPLICATE_WORK")


def test_ineffective_iteration_positive_negative() -> None:
    u = _usage(inp=500_000)
    pos = _ep("s0:ep0", usage=u, facts={"no_adapt_retry": True})
    assert "INEFFECTIVE_ITERATION" in _ids(_run_rule(pos, []), "INEFFECTIVE_ITERATION")
    neg = _ep("s0:ep0", usage=u, facts={"novel_output": 0.9})
    assert "INEFFECTIVE_ITERATION" not in _ids(_run_rule(neg, []), "INEFFECTIVE_ITERATION")


def test_irrelevant_work_positive_negative() -> None:
    u = _usage(inp=1_000_000)
    ep = _ep("s0:ep0", usage=u, facts={"artifact_overlap": 0}, trigger="prior_result")
    pos_fs = [
        _fs(
            "episode",
            "s0:ep0",
            episode__progress__targets_named_component=0.1,
            episode__phase__purpose="produce",
        )
    ]
    assert "IRRELEVANT_WORK" in _ids(_run_rule(ep, pos_fs), "IRRELEVANT_WORK")
    neg_fs = [
        _fs(
            "episode",
            "s0:ep0",
            episode__progress__targets_named_component=0.1,
            episode__phase__purpose="orient",
        )
    ]
    assert "IRRELEVANT_WORK" not in _ids(_run_rule(ep, neg_fs), "IRRELEVANT_WORK")


def test_post_completion_work_positive_and_veto() -> None:
    cfg = _exact_config()
    u = _usage(inp=1_000_000)
    done = _ep("s0:ep0", usage=u, rids=["r1"], facts={})
    later = _ep("s0:ep1", usage=u, rids=["r2"], trigger="prior_result")
    t = _turn("t1", "s0", [_call("t1", "r1", u), _call("t1", "r2", u, start=1, end=1, minutes=2)])
    sess = _session([], [t])
    feat = [
        _fs("episode", "s0:ep0", episode__claim__declares_success=0.9, episode__claim__cites_verification=0.9),
        _fs("episode", "s0:ep1", episode__claim__cites_verification=0.1, episode__claim__declares_success=0.1),
    ]
    diags = diagnose(_trace(sess), [_task("s0", t)], [done, later], feat, [], cfg)
    assert "POST_COMPLETION_WORK" in {d.id for d in diags}

    feat_veto = [
        _fs("episode", "s0:ep0", episode__claim__declares_success=0.9, episode__claim__cites_verification=0.9),
        _fs("episode", "s0:ep1", episode__claim__cites_verification=0.9),
    ]
    diags_v = diagnose(_trace(sess), [_task("s0", t)], [done, later], feat_veto, [], cfg)
    assert "POST_COMPLETION_WORK" not in {d.id for d in diags_v}


def test_excess_reasoning_tier_positive_negative() -> None:
    u = _usage(inp=10, out=1_000_000, reasoning=1_000_000)
    ep = _ep("s0:ep0", usage=u, facts={})
    requires = {
        "episode__reasoning__requires_diagnosis": 0.1,
        "episode__reasoning__requires_design_tradeoff": 0.1,
        "episode__reasoning__requires_long_context_synthesis": 0.1,
        "episode__reasoning__requires_domain_knowledge": 0.1,
    }
    pos_fs = [_fs("episode", "s0:ep0", episode__reasoning__demand="0_direct", **requires)]
    diags = _run_rule(ep, pos_fs, effort="max", model="gpt-5.6-luna")
    hit = [d for d in diags if d.id == "EXCESS_REASONING_TIER"]
    assert hit
    assert hit[0].amount is not None
    assert hit[0].amount.label and "avoidable amount unknown" in hit[0].amount.label
    assert hit[0].amount.view is CostView.invoice

    neg_fs = [_fs("episode", "s0:ep0", episode__reasoning__demand="3_systemic", **requires)]
    assert "EXCESS_REASONING_TIER" not in _ids(
        _run_rule(ep, neg_fs, effort="max", model="gpt-5.6-luna"), "EXCESS_REASONING_TIER"
    )


def test_coordination_and_external_block() -> None:
    u = _usage(inp=1_000_000)
    coord = _ep("s0:ep0", usage=u, ops=["agent.spawn", "agent.wait"], facts={"child_result_reuse": 0})
    assert "COORDINATION_OVERHEAD" in _ids(_run_rule(coord, []), "COORDINATION_OVERHEAD")
    reused = _ep("s0:ep0", usage=u, ops=["agent.spawn"], facts={"child_result_reuse": 0.9})
    assert "COORDINATION_OVERHEAD" not in _ids(_run_rule(reused, []), "COORDINATION_OVERHEAD")

    blocked = _ep("s0:ep0", usage=u, facts={"env_impediment": "sandbox_denied"})
    diags = _run_rule(blocked, [])
    ext = [d for d in diags if d.id == "EXTERNAL_BLOCK"]
    assert ext
    assert ext[0].counts_as_agent_waste is False


def test_low_yield_tail_no_dollars() -> None:
    u = _usage(inp=1_000_000)
    cfg = _exact_config()
    a = _ep("s0:ep0", usage=u, rids=["r1"], facts={"no_observable_progress": True})
    b = _ep("s0:ep1", usage=u, rids=["r2"], facts={"no_observable_progress": True})
    t = _turn("t1", "s0", [_call("t1", "r1", u), _call("t1", "r2", u, start=1, end=1)])
    feat = [
        _fs("episode", "s0:ep0", episode__outcome__kind="no_observable_progress"),
        _fs("episode", "s0:ep1", episode__outcome__kind="no_observable_progress"),
    ]
    diags = diagnose(_trace(_session([], [t])), [_task("s0", t)], [a, b], feat, [], cfg)
    tail = [d for d in diags if d.id == "LOW_YIELD_TAIL"]
    assert tail
    assert tail[0].amount is None
    assert tail[0].informational is True


def test_continuation_burden_vs_carryover() -> None:
    cfg = _exact_config()
    u_miss = _usage(inp=100_000, cr=0)
    t1 = _turn("t1", "s0", [_call("t1", "r1", _usage(inp=10), minutes=0)], text="first task")
    t2 = _turn(
        "t2",
        "s0",
        [_call("t2", "r2", u_miss, start=2, end=4, minutes=30)],
        text="totally different deliverable",
        start=2,
        end=4,
        compactions=1,
    )
    sess = _session([], [t1, t2])
    tasks = [_task("s0", t1, 0, None), _task("s0", t2, 1, "unrelated")]
    eps = [
        _ep("s0:ep0", usage=_usage(inp=10), rids=["r1"], turn_id="t1"),
        _ep("s0:ep1", usage=u_miss, rids=["r2"], turn_id="t2", task_id="s0:task1"),
    ]
    diags = diagnose(_trace(sess), tasks, eps, [], [], cfg)
    assert any(d.id == "CONTINUATION_BURDEN" for d in diags)
    burden = next(d for d in diags if d.id == "CONTINUATION_BURDEN")
    assert burden.amount is not None
    assert burden.view is CostView.counterfactual
    assert burden.amount.amount_low is not None and burden.amount.amount_high is not None
    assert burden.amount.assumptions

    # Related continuation → informational carryover, not burden.
    tasks_rel = [_task("s0", t1, 0, None), _task("s0", t2, 1, "continues")]
    diags_rel = diagnose(_trace(sess), tasks_rel, eps, [], [], cfg)
    assert "CONTINUATION_BURDEN" not in {d.id for d in diags_rel}
    assert any(d.id == "CONTEXT_CARRYOVER" for d in diags_rel)


def test_kitchen_sink_superseded_by_semantic_relation() -> None:
    cfg = _exact_config()
    u = _usage(inp=10)
    t1 = _turn("t1", "s0", [_call("t1", "r1", u)], text="implement parser")
    t2 = _turn("t2", "s0", [_call("t2", "r2", u, minutes=2)], text="now the exporter", start=2, end=3)
    sess = _session([], [t1, t2])
    tasks = [_task("s0", t1, 0, None), _task("s0", t2, 1, "unrelated")]
    finding = Finding(
        id="KITCHEN_SINK",
        kind="taxonomy",
        severity=Severity.low,
        confidence="high",
        session_id="s0",
        spans=[SpanRef(session_id="s0", event_idx=0)],
        evidence={},
        message="kitchen",
    )
    diags = diagnose(_trace(sess), tasks, [], [], [finding], cfg)
    ks = [d for d in diags if d.id == "KITCHEN_SINK"]
    assert ks
    assert any("KITCHEN_SINK" in d.superseded_ids or d.evidence.get("superseded_by") for d in diags)


def test_unverified_completion_replaced_when_semantic_on() -> None:
    cfg = _exact_config()
    u = _usage(inp=10)
    t = _turn("t1", "s0", [_call("t1", "r1", u)])
    sess = _session([], [t])
    ep = _ep("s0:ep0", usage=u, facts={"verification_fact": False})
    finding = Finding(
        id="UNVERIFIED_COMPLETION",
        kind="taxonomy",
        severity=Severity.medium,
        confidence="high",
        session_id="s0",
        spans=[SpanRef(session_id="s0", event_idx=0)],
        evidence={},
        message="unverified",
    )
    feat = [_fs("episode", "s0:ep0", episode__claim__declares_success=0.9)]
    diags = diagnose(_trace(sess), [_task("s0", t)], [ep], feat, [finding], cfg)
    uvc = [d for d in diags if d.id == "UNVERIFIED_COMPLETION"]
    assert uvc
    assert uvc[0].tier == "semantic"
    assert uvc[0].evidence["composite"] == "declared_success_without_observed_verification"

    # Unsupported composite (declares_success unknown / api_error): detector finding kept as-is.
    unknown = [
        _fs(
            "episode",
            "s0:ep0",
            episode__claim__declares_success=_fv("episode.claim.declares_success", None, reason="api_error"),
        )
    ]
    diags_u = diagnose(_trace(sess), [_task("s0", t)], [ep], unknown, [finding], cfg)
    uvc_u = [d for d in diags_u if d.id == "UNVERIFIED_COMPLETION"]
    assert uvc_u
    assert uvc_u[0].tier == "taxonomy"
    assert "composite" not in uvc_u[0].evidence

    # No features at all but semantic mode on: still the deterministic finding, untouched.
    cfg_on = _exact_config(semantic={**load_config().model_dump()["semantic"], "mode": "cached"})
    diags_n = diagnose(_trace(sess), [_task("s0", t)], [ep], [], [finding], cfg_on)
    uvc_n = [d for d in diags_n if d.id == "UNVERIFIED_COMPLETION"]
    assert uvc_n and uvc_n[0].tier == "taxonomy"

    # Supported and contradicted (verification observed): the detector finding is suppressed.
    verified_ep = _ep("s0:ep0", usage=u, facts={"verification": True})
    diags_v = diagnose(_trace(sess), [_task("s0", t)], [verified_ep], feat, [finding], cfg)
    assert "UNVERIFIED_COMPLETION" not in {d.id for d in diags_v}


def test_duplicate_work_fires_on_facts_from_real_segmentation() -> None:
    """Deterministic facts persisted on the atom (no JeV) are enough to drive DUPLICATE_WORK."""
    from agent_hotwash.structure.episodes import segment_episodes
    from agent_hotwash.structure.tasks import segment_tasks

    cfg = _exact_config()
    u = _usage(inp=1_000_000)

    def read(turn_id: str, rid: str, cid: str) -> Event:
        return Event(
            kind=EventKind.tool_call,
            op_kind="cmd.read",
            path="src/a.py",
            tool_args={"cmd": "cat src/a.py"},
            turn_id=turn_id,
            response_id=rid,
            call_id=cid,
        )

    events = [
        Event(kind=EventKind.user_msg, text="look at src/a.py", turn_id="t1"),
        read("t1", "r1", "c1"),
        Event(kind=EventKind.tool_result, call_id="c1", ok=True, output="x", turn_id="t1", response_id="r1"),
        Event(kind=EventKind.assistant_msg, text="read it", phase="final_answer", turn_id="t1"),
        Event(kind=EventKind.user_msg, text="again", turn_id="t2"),
        read("t2", "r2", "c2"),
        Event(kind=EventKind.tool_result, call_id="c2", ok=True, output="x", turn_id="t2", response_id="r2"),
        read("t2", "r2", "c3"),
        Event(kind=EventKind.tool_result, call_id="c3", ok=True, output="x", turn_id="t2", response_id="r2"),
    ]
    t1 = _turn("t1", "s0", [_call("t1", "r1", u, start=0, end=3)], start=0, end=3)
    t2 = _turn("t2", "s0", [_call("t2", "r2", u, start=4, end=8, minutes=1)], start=4, end=8)
    sess = _session(events, [t1, t2])
    tasks = segment_tasks(sess, cfg, semantic_mode="off")
    eps = segment_episodes(sess, tasks, cfg)
    second = next(ep for ep in eps if ep.turn_id == "t2")
    assert second.facts["repeat_count"] >= 2
    assert second.facts["paths_reread_unchanged"] >= 1
    feat = [_fs("episode", second.episode_id, episode__progress__novel_output=0.05)]
    diags = diagnose(_trace(sess), tasks, eps, feat, [], cfg)
    dup = [d for d in diags if d.id == "DUPLICATE_WORK"]
    assert dup
    assert dup[0].spans == [second.episode_id]


# ---------------------------------------------------------------------------
# Off-mode report contract
# ---------------------------------------------------------------------------


def test_off_mode_report_json_and_csv_contract(tf) -> None:
    root = tf.session([tf.user("please fix the bug"), tf.assistant("on it")])
    trace = tf.trace(root, trace_id="t0", instance_id="t0")
    analysis = analyze(trace, load_config())
    report = Report.build([RunResult(analysis=analysis, findings=[])], ReportMeta(tool_version="9.9.9"))
    data = report_to_dict(report)
    assert data["meta"]["schema_version"] == 2
    run0 = data["runs"][0]
    assert "structure" not in run0
    assert "features" not in run0
    assert "capabilities" not in run0
    assert "cost_views" not in run0
    assert "monthly" not in data
    header = render_csv(report).strip().splitlines()[0].split(",")
    assert header == list(_BASE_COLUMNS)


def test_table_task_card_when_structure_present(tf) -> None:
    cfg = _exact_config()
    u = _usage(inp=1_000_000)
    t = _turn("t1", "s0", [_call("t1", "r1", u)])
    sess = _session([], [t])
    analysis = analyze(_trace(sess), cfg)
    task = _task("s0", t)
    ep = _ep("s0:ep0", usage=u)
    views = build_cost_views(_trace(sess), cfg, episodes=[ep], tasks=[task])
    run = RunResult(
        analysis=analysis,
        findings=[],
        structure=StructureSection(tasks=[task], episodes=[ep]),
        features=[_fs("task", task.task_id, task__intent__change=0.9, task__scope__breadth="localized")],
        cost_views=views,
    )
    report = Report.build([run], ReportMeta(tool_version="0"))
    text = render_table(report)
    assert "Intent" in text
    assert "Shape" in text
    assert "Actual" in text
    assert "Trajectory" in text
    assert "Diagnosis" in text
    assert "invoice" in text
    # Existing per-run table still renders.
    assert "Per-run analysis" in text


def test_table_without_structure_unchanged(tf) -> None:
    root = tf.session([tf.user("go"), tf.assistant("ok")])
    analysis = analyze(tf.trace(root), load_config())
    report = Report.build([RunResult(analysis=analysis)], ReportMeta(tool_version="0"))
    text = render_table(report)
    assert "Per-run analysis" in text
    assert "Intent" not in text


def test_html_task_card_and_monthly_when_structure_present(tf) -> None:
    cfg = _exact_config()
    u = _usage(inp=1_000_000)
    t = _turn("t1", "s0", [_call("t1", "r1", u)])
    sess = _session([], [t])
    analysis = analyze(_trace(sess), cfg)
    task = _task("s0", t)
    ep0 = _ep("s0:ep0", usage=u, rids=["r1"])
    ep0.phase_activity = "inspect"
    ep0.phase_purpose = "orient"
    ep1 = _ep("s0:ep1", usage=u, rids=["r2"])
    ep1.phase_activity = "inspect"
    ep1.phase_purpose = "orient"
    views = build_cost_views(_trace(sess), cfg, episodes=[ep0, ep1], tasks=[task])
    run = RunResult(
        analysis=analysis,
        findings=[],
        structure=StructureSection(tasks=[task], episodes=[ep0, ep1]),
        features=[_fs("task", task.task_id, task__intent__change=0.9, task__scope__breadth="localized")],
        cost_views=views,
    )
    monthly = MonthlyRollup(
        timezone="UTC",
        ranked_by_task_count=["DUPLICATE_WORK"],
        ranked_by_invoice=["DUPLICATE_WORK"],
        ranked_by_counterfactual=[],
        overspend_statement="you mostly overspend on DUPLICATE_WORK (ranked by invoice dollars)",
    )
    report = Report.build([run], ReportMeta(tool_version="0"), monthly=monthly)
    table = render_table(report)
    assert "inspect x orient" in table
    assert "(x2)" in table
    assert "Monthly root-task rollup" in table
    assert "you mostly overspend on DUPLICATE_WORK" in table
    data = report_to_dict(report)
    assert data["monthly"]["overspend_statement"].startswith("you mostly overspend")
    doc = render_html(report)
    assert "Intent" in doc
    assert "Trajectory" in doc
    assert "inspect x orient (x2)" in doc
    assert "Monthly root-task rollup" in doc
    assert "https://" not in doc
