"""Bounded JeV interpretation of deterministically selected execution tails."""

from __future__ import annotations

import re
from collections import defaultdict
from typing import TYPE_CHECKING, Any

from agent_hotwash.events import EventKind, ToolCategory
from agent_hotwash.semantic.bank import load_bank
from agent_hotwash.semantic.pipeline import Annotator
from agent_hotwash.semantic.repetition import assess_repetition, repetition_state, supported
from agent_hotwash.semantic.results import FeatureSet, FeatureValue

if TYPE_CHECKING:
    from agent_hotwash.config import Config
    from agent_hotwash.diagnostics.tails import TailAnalysis, TailIncident
    from agent_hotwash.events import Trace
    from agent_hotwash.semantic.client import SystemOneAsker


def tail_state(trace: Trace, incident: TailIncident) -> dict[str, Any]:
    session = next(s for s in [trace.root, *trace.subagents] if s.session_id == incident.session_id)
    indices = set(incident.event_indices)
    if incident.kind == "delegation_repetition":
        indices = {incident.evidence[k] for k in ("prior_spawn_idx", "prior_return_idx", "new_spawn_idx")}
    # Information-acquisition incidents select model rounds and their calls;
    # include each unambiguous result so discovery dependencies remain visible.
    call_ids = {
        e.call_id
        for e in session.events
        if e.idx in indices
        and e.kind == EventKind.tool_call
        and e.call_id
        and (incident.kind != "delegation_repetition" or e.idx == incident.evidence["prior_spawn_idx"])
    }
    by_result: dict[str, list] = defaultdict(list)
    for ev in session.events:
        if ev.kind == EventKind.tool_result and ev.call_id in call_ids:
            by_result[ev.call_id].append(ev)
    for results in by_result.values():
        if len(results) == 1:
            indices.add(results[0].idx)
    lo, hi = min(indices), max(indices)
    attempts = []
    for ev in session.events:
        is_edit = lo < ev.idx < hi and ev.kind is EventKind.tool_call and ev.tool_category is ToolCategory.write
        if ev.idx not in indices and not is_edit:
            continue
        if ev.kind is EventKind.tool_call:
            attempts.append(
                {
                    "event_idx": ev.idx,
                    "call_id": ev.call_id,
                    "tool": ev.tool_name,
                    "source_record": ev.source.record_index if ev.source else None,
                    "response_id": ev.response_id,
                    "intervening_edit": is_edit,
                    "arguments": str(ev.tool_args)[:1500],
                }
            )
        elif ev.kind is EventKind.tool_result:
            attempts.append(
                {
                    "event_idx": ev.idx,
                    "call_id": ev.call_id,
                    "ok": ev.ok,
                    "diagnostic": (ev.diagnostic_excerpt or ev.error_text or ev.output or "")[:1500],
                    "output_clipped": ev.output_truncated,
                }
            )
    # Keep the beginning and end, and explicitly disclose sampling.
    requests = [
        e
        for e in session.events
        if e.idx <= lo and e.kind == EventKind.user_msg and (e.role_hint is None or str(e.role_hint) == "user")
    ]
    request = re.sub(r"<skill\b[^>]*>.*?</skill>", "", requests[-1].text or "", flags=re.S).strip() if requests else ""
    request = request if len(request) <= 1800 else request[:900] + "\n[excerpt gap]\n" + request[-900:]
    # Do not ask JeV to count model rounds or invent a batching opportunity.
    # Only exact, pre-named paths and their FIRST read before any edit qualify.
    named_reads = []
    seen_paths: set[str] = set()
    request_idx = requests[-1].idx if requests else -1
    for ev in session.events:
        if ev.idx <= request_idx or ev.idx > hi:
            continue
        if ev.kind == EventKind.user_msg and (ev.role_hint is None or str(ev.role_hint) == "user"):
            break
        if ev.kind != EventKind.tool_call:
            continue
        if ev.tool_category == ToolCategory.write:
            break
        if ev.tool_category != ToolCategory.read:
            continue
        path = ev.tool_args.get("path") or ev.tool_args.get("file_path")
        if not isinstance(path, str) or path in seen_paths:
            continue
        seen_paths.add(path)
        if not path or not re.search(r"(?<![\w./-])" + re.escape(path) + r"(?![\w/-]|\.[\w./-])", request):
            continue
        group = ev.response_id or (f"record:{ev.source.record_index}" if ev.source else None)
        if group and ev.idx in indices:
            named_reads.append({"path": path, "model_round": group, "event_idx": ev.idx})
    state = {
        "incident": {
            "kind": incident.kind,
            "request": request,
            "context": {
                "intervening_compactions": sum(
                    e.kind == EventKind.compaction for e in session.events if lo < e.idx < hi
                ),
                "intervening_write_calls": sum(
                    e.kind == EventKind.tool_call and e.tool_category == ToolCategory.write
                    for e in session.events
                    if lo < e.idx < hi
                ),
                "source_output_clipped": any(a.get("output_clipped") for a in attempts),
                "sampled": len(attempts) > 24,
                "external_changes_unobserved": True,
                "preknown_first_reads": named_reads,
                "preknown_distinct_rounds": len({r["model_round"] for r in named_reads}),
            },
            "attempts": attempts[:12] + attempts[-12:] if len(attempts) > 24 else attempts,
            "sampled": len(attempts) > 24,
        }
    }
    if incident.kind in {"delegation_repetition", "verification_repetition"}:
        return repetition_state(trace, incident, state)
    return state


def _supported(values: dict[str, FeatureValue], key: str) -> bool:
    value = values.get(key)
    return bool(
        value
        and not value.abstains
        and value.reason is None
        and isinstance(value.value, (int, float))
        and value.value >= 0.7
    )


def annotate_tails(
    trace: Trace,
    tails: TailAnalysis,
    config: Config,
    asker: SystemOneAsker,
    *,
    mode: str,
    allow_unredacted: bool = False,
) -> list[FeatureSet]:
    features = [f for f in load_bank() if f.scope == "tail"]
    routes = {
        "retry_attempts": {"tail.execution.same_blocker", "tail.execution.changed_approach"},
        "failure_chain": {"tail.execution.same_blocker", "tail.execution.changed_approach"},
        "inspection_rounds": {"tail.work.broad_inspection_requested", "tail.work.known_targets_batchable"},
        "status_probes": {"tail.work.blocking_wait_offered", "tail.work.explicit_monitoring"},
        "output_repetition": {"tail.work.repeat_needed"},
        "delegation_repetition": {
            "tail.review.same_question",
            "tail.review.reuses_findings",
            "tail.review.fresh_pass_reason",
        },
        "verification_repetition": {"tail.check.changed_inputs", "tail.check.explicit_repeat_reason"},
    }
    candidates = sorted(
        (r for r in tails.incidents if r.exceeds_threshold and r.kind in routes),
        key=lambda r: (-r.value / max(r.threshold, 1), r.id),
    )
    # Avoid paying twice for the same constituent failures under different labels.
    selected = []
    seen: set[tuple[str, int]] = set()
    for row in candidates if config.tails.max_semantic_incidents > 0 else []:
        coordinates = {(row.session_id, i) for i in row.event_indices}
        if coordinates <= seen:
            continue
        selected.append(row)
        seen.update(coordinates)
        if len(selected) == config.tails.max_semantic_incidents:
            break
    tails.coverage["semantic_candidates"] = len(candidates)
    tails.coverage["semantic_selected"] = len(selected)
    annotator = Annotator(asker, config, trace.capabilities, mode=mode, allow_unredacted=allow_unredacted)
    states = [tail_state(trace, row) for row in selected]
    for row, state in zip(selected, states, strict=True):
        row.evidence["semantic_state_sampled"] = state["incident"]["sampled"]
    answers = annotator.ask_many(
        [
            (
                state,
                [
                    f
                    for f in features
                    if f.id in routes[row.kind]
                    and (row.kind != "delegation_repetition" or f.id == "tail.review.same_question")
                    and (
                        f.id != "tail.work.known_targets_batchable"
                        or state["incident"]["context"]["preknown_distinct_rounds"] >= 3
                    )
                ],
            )
            for state, row in zip(states, selected, strict=True)
        ]
    )
    # Shared files frequently mean disjoint implementation tasks. Spend the
    # follow-up questions only when both requests are actually the same review.
    followups = [
        i
        for i, values in enumerate(answers)
        if selected[i].kind == "delegation_repetition"
        and supported(values, "tail.review.same_question")
        and states[i]["incident"]["context"]["review_requests_complete"]
    ]
    extra = (
        annotator.ask_many(
            [
                (
                    states[i],
                    [f for f in features if f.id in {"tail.review.reuses_findings", "tail.review.fresh_pass_reason"}],
                )
                for i in followups
            ]
        )
        if followups
        else []
    )
    for i, values in zip(followups, extra, strict=True):
        answers[i].update(values)
    out = []
    for row, values, state in zip(selected, answers, states, strict=True):
        out.append(FeatureSet(scope="tail", object_id=row.id, values=values))
        if row.kind in {"delegation_repetition", "verification_repetition"}:
            assess_repetition(row, values, state)
        elif row.kind in {"status_probes", "output_repetition"}:
            row.assessment = "unclear"
            row.assessment_reasons = ["A measured repeat does not establish redundant work."]
        same = values.get("tail.execution.same_blocker")
        changed = values.get("tail.execution.changed_approach")
        if same and not same.abstains and same.reason is None and same.value is not None and float(same.value) >= 0.7:
            row.label = "Repeated blocker supported by JeV"
            row.action = "Repair the recurring diagnostic before another attempt; inspect the linked results."
            if (
                changed
                and not changed.abstains
                and changed.reason is None
                and changed.value is not None
                and float(changed.value) >= 0.7
            ):
                row.label = "Persistent blocker despite adaptation"
        if row.evidence.get("intervening_edit_calls"):
            row.label = "Repeated verification with intervening edits"
            row.action = (
                "Inspect the repair cycle and isolate the failing fixture or prerequisite; "
                "repeated checks alone do not prove waste."
            )

        if _supported(values, "tail.work.broad_inspection_requested"):
            row.label = "Broad inspection explicitly requested"
            row.action = "Keep the requested audit scope; narrow query output and batch only independent reads."
        if _supported(values, "tail.work.known_targets_batchable"):
            row.label = "Pre-known inspection targets could be batched"
            row.action = (
                "Read the explicitly named targets together; preserve follow-ups that depend on earlier results."
            )
        if _supported(values, "tail.work.blocking_wait_offered"):
            row.label = "Repeated status probes with a visible blocking-wait alternative"
            if (
                supported(values, "tail.work.explicit_monitoring", positive=False)
                and row.evidence.get("unchanged_pending_progress_results", 0) >= 3
                and row.evidence.get("pending_only_model_rounds", 0) >= 3
            ):
                row.assessment = "supported_opportunity"
                row.assessment_reasons = [
                    "unchanged_pending_progress_results>=3",
                    "pending_only_model_rounds>=3",
                    "tail.work.blocking_wait_offered",
                    "negative:tail.work.explicit_monitoring",
                ]
                row.action = (
                    "Use the offered blocking wait or completion notification when pending progress is unchanged; "
                    "preserve checks that inform another action."
                )
        if _supported(values, "tail.work.explicit_monitoring"):
            row.label = "Repeated monitoring explicitly requested"
            row.action = "Honor the requested monitoring cadence; use completion notifications between checks."
            row.assessment = "justified_repeat"
            row.assessment_reasons = ["tail.work.explicit_monitoring"]
        if _supported(values, "tail.work.repeat_needed"):
            row.label = "Repeated output with a visible recheck reason"
            row.action = (
                "Keep rechecks required by state or context changes; reuse results only while they remain valid."
            )
            row.assessment = "justified_repeat"
            row.assessment_reasons = ["tail.work.repeat_needed"]
        row.evidence["semantic_feature_set"] = row.id
    return out
