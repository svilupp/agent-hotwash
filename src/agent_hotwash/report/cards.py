"""Shared task-card labels used by the table and HTML writers.

Atoms stay immutable; contiguous same-activity runs are display-only (WP4b).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from agent_hotwash.structure.episodes import group_display_runs

if TYPE_CHECKING:
    from agent_hotwash.report.model import RunResult
    from agent_hotwash.structure.episodes import Episode
    from agent_hotwash.structure.tasks import Task


def feat_values(run: RunResult, object_id: str) -> dict:
    for fs in run.features or []:
        if fs.object_id == object_id:
            return fs.values
    return {}


def noul_true(values: dict, feature_id: str) -> bool:
    fv = values.get(feature_id)
    if fv is None:
        return False
    val = getattr(fv, "value", fv)
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return float(val) >= 0.5
    return bool(val)


def choice_val(values: dict, feature_id: str) -> str | None:
    fv = values.get(feature_id)
    if fv is None:
        return None
    val = getattr(fv, "value", fv)
    if isinstance(val, str):
        return val
    if isinstance(val, dict):
        c = val.get("choice") or val.get("value")
        return str(c) if c is not None else None
    return str(val) if val is not None else None


def as_float(val: object) -> float | None:
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        return None
    return float(val)


def money_view(value: object) -> str:
    """Format a Money (or money-shaped dict) with view + pricing_status on the number."""
    if value is None:
        return "-"
    amount: object = getattr(value, "amount", None)
    low: object = getattr(value, "amount_low", None)
    high: object = getattr(value, "amount_high", None)
    view = getattr(value, "view", None)
    status = getattr(value, "pricing_status", None)
    if isinstance(value, dict):
        amount = value.get("amount")
        low = value.get("amount_low")
        high = value.get("amount_high")
        view = value.get("view")
        status = value.get("pricing_status")
    view_s = view.value if hasattr(view, "value") else (str(view) if view else "?")
    status_s = status.value if hasattr(status, "value") else (str(status) if status else "?")
    lo = as_float(low)
    hi = as_float(high)
    amt = as_float(amount)
    if lo is not None and hi is not None:
        return f"${lo:.4f}-${hi:.4f} {view_s} {status_s}"
    if amt is None:
        return f"- {view_s} {status_s}"
    return f"${amt:.4f} {view_s} {status_s}"


def intent_label(run: RunResult, task_id: str) -> str:
    values = feat_values(run, task_id)
    hits = [
        name
        for name in ("inquire", "change", "assess", "execute", "communicate")
        if noul_true(values, f"task.intent.{name}")
    ]
    return ", ".join(hits) or "-"


def shape_label(run: RunResult, task_id: str) -> str:
    values = feat_values(run, task_id)
    return str(choice_val(values, "task.scope.breadth") or "-")


def actual_label(task: Task) -> str:
    ledger = getattr(task, "ledger", None)
    status = getattr(ledger, "status", None) if ledger is not None else None
    return str(status or "-")


def resolved_phase(run: RunResult, ep: Episode) -> tuple[str | None, str | None]:
    """``(activity, purpose)`` from the answered FeatureSet, else the atom's label."""
    values = feat_values(run, ep.episode_id)
    activity = choice_val(values, "episode.phase.activity") or ep.phase_activity
    purpose = choice_val(values, "episode.phase.purpose") or ep.phase_purpose
    return activity, purpose


def trajectory_label(run: RunResult, task_id: str) -> str:
    """Contiguous same-activity runs (display grouping); atoms are not merged.

    Grouping keys on the *resolved* activity (FeatureSet answer first, then the
    atom's ``phase_activity``), so atoms whose label was never copied back still
    group correctly.
    """
    if run.structure is None:
        return "-"
    atoms = [ep for ep in run.structure.episodes if ep.task_id == task_id]
    pairs: list[str] = []
    for group in group_display_runs(atoms, key=lambda ep: resolved_phase(run, ep)[0]):
        activity, purpose = resolved_phase(run, group[0])
        label = f"{activity or '?'} x {purpose or '?'}"
        if len(group) > 1:
            label = f"{label} (x{len(group)})"
        pairs.append(label)
    return " → ".join(pairs) or "-"


def diagnosis_label(run: RunResult, task_id: str) -> str:
    views = run.cost_views
    if views is None:
        return "-"
    bits: list[str] = []
    for d in views.diagnoses:
        if d.id in {"PHASE_SPEND", "CONTEXT_CARRYOVER"}:
            continue
        spans = d.spans or []
        if spans and not any(task_id in s or s == task_id for s in spans):
            if d.evidence.get("task_id") not in {None, task_id}:
                continue
            if (
                d.group_id
                and task_id not in d.group_id
                and not any(task_id in s for s in spans)
                and d.evidence.get("task_id")
            ):
                continue
        money = money_view(d.amount) if d.amount is not None else ""
        bits.append(f"{d.id} {money}".strip() if money and money != "-" else d.id)
    invoice = views.invoice
    prefix = f"invoice {money_view(invoice)}"
    return (prefix + ("; " + "; ".join(bits) if bits else "")).strip()


__all__ = [
    "actual_label",
    "as_float",
    "choice_val",
    "diagnosis_label",
    "feat_values",
    "intent_label",
    "money_view",
    "noul_true",
    "resolved_phase",
    "shape_label",
    "trajectory_label",
]
