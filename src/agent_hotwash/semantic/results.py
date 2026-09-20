"""Feature values, sets, and derived formulas over digest facts (§6.3 / §6.6)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Reason = Literal[
    "insufficient_observability",
    "state_too_large",
    "api_error",
    "disabled",
    "low_support",
]
Source = Literal["jev", "fact", "derived"]


class FeatureValue(BaseModel):
    """One answered (or abstained) feature on a task/turn/episode."""

    model_config = ConfigDict(extra="ignore")

    id: str
    version: int = 1
    value: Any = None
    confidence: float | None = None
    reason: Reason | None = None
    source: Source = "jev"
    model: str | None = None
    question_hash: str | None = None


class FeatureSet(BaseModel):
    """All feature values for one object, plus answered coverage."""

    model_config = ConfigDict(extra="ignore")

    scope: str
    object_id: str
    values: dict[str, FeatureValue] = Field(default_factory=dict)
    coverage: float = 0.0

    @model_validator(mode="after")
    def _fill_coverage(self) -> FeatureSet:
        if not self.values:
            return self
        answered = sum(1 for v in self.values.values() if v.reason is None)
        self.coverage = answered / len(self.values)
        return self


def _as_mapping(atom: Any) -> dict[str, Any]:
    if isinstance(atom, dict):
        return atom
    facts = getattr(atom, "facts", None)
    if isinstance(facts, dict):
        out = dict(facts)
        if getattr(atom, "ops", None) is not None:
            out.setdefault("ops", list(atom.ops))
        if getattr(atom, "artifacts", None) is not None:
            out.setdefault("artifacts", list(atom.artifacts))
        novel = getattr(atom, "novel_output", None)
        if novel is not None:
            out.setdefault("novel_output", novel)
        return out
    return {}


def _facts(atom: Any) -> dict[str, Any]:
    m = _as_mapping(atom)
    inner = m.get("facts")
    if isinstance(inner, dict):
        merged = dict(inner)
        for k, v in m.items():
            if k != "facts":
                merged.setdefault(k, v)
        return merged
    return m


def _novel_output(atom: Any) -> float:
    facts = _facts(atom)
    val = facts.get("novel_output", 1.0)
    try:
        return float(val)
    except (TypeError, ValueError):
        return 1.0


def _artifact_changed(atom: Any) -> bool:
    facts = _facts(atom)
    if "artifact_change" in facts:
        return bool(facts["artifact_change"])
    ops = facts.get("ops") or []
    if any(op in {"file.edit", "file.write"} or (isinstance(op, str) and op.startswith("file.")) for op in ops):
        return True
    edits = facts.get("edits")
    return isinstance(edits, (int, float)) and edits > 0


def stuck_window(k: int, atoms: Sequence[Any]) -> bool:
    """≥k consecutive atoms with failure_signature_repeats>0, novel_output<.3, no artifact change."""
    if k <= 0:
        return False
    run = 0
    for atom in atoms:
        facts = _facts(atom)
        repeats = facts.get("failure_signature_repeats") or 0
        try:
            repeats_n = float(repeats)
        except (TypeError, ValueError):
            repeats_n = 0.0
        if repeats_n > 0 and _novel_output(atom) < 0.3 and not _artifact_changed(atom):
            run += 1
            if run >= k:
                return True
        else:
            run = 0
    return False


def _atom_paths(atom: Any) -> set[str]:
    facts = _facts(atom)
    paths = facts.get("paths") or facts.get("artifacts") or []
    return {str(p) for p in paths}


def _atom_ops(atom: Any) -> list[str]:
    facts = _facts(atom)
    ops = facts.get("ops") or []
    return [str(o) for o in ops]


def _is_edit_atom(atom: Any) -> bool:
    ops = _atom_ops(atom)
    return any(op in {"file.edit", "file.write"} for op in ops) or bool(_facts(atom).get("is_edit"))


def _is_failing_test_atom(atom: Any) -> bool:
    facts = _facts(atom)
    if (
        facts.get("tests_failed")
        or facts.get("test_failed")
        or (facts.get("class") == "test" and facts.get("ok") is False)
    ):
        return True
    ops = _atom_ops(atom)
    if facts.get("tests_failed"):
        return True
    return bool(facts.get("failing_test")) or ("cmd.exec" in ops and bool(facts.get("tests_failed")))


def _is_test_atom(atom: Any) -> bool:
    facts = _facts(atom)
    if facts.get("class") == "test" or facts.get("is_test"):
        return True
    return bool(facts.get("tests_run") or facts.get("tests_failed") or facts.get("failing_test"))


def thrashing_window(k: int, atoms: Sequence[Any]) -> bool:
    """≥k atoms alternating edit/test on the same path with failing tests."""
    if k <= 0 or len(atoms) < k:
        return False
    seq = list(atoms)
    for start in range(0, len(seq) - k + 1):
        window = seq[start : start + k]
        paths = _atom_paths(window[0])
        for atom in window[1:]:
            paths &= _atom_paths(atom)
        if not paths:
            continue
        failing = False
        ok_pattern = True
        for i, atom in enumerate(window):
            want_edit = i % 2 == 0
            if want_edit:
                if not _is_edit_atom(atom):
                    ok_pattern = False
                    break
            else:
                if not (_is_test_atom(atom) or _is_failing_test_atom(atom)):
                    ok_pattern = False
                    break
                if _is_failing_test_atom(atom) or _facts(atom).get("ok") is False:
                    failing = True
        if ok_pattern and failing:
            return True
    return False


def recovered(facts: dict[str, Any]) -> bool:
    """Failure then a later same-class op success."""
    if facts.get("recovered") or facts.get("failure_then_success"):
        return True
    outcomes = facts.get("op_outcomes") or []
    if not isinstance(outcomes, list):
        return False
    failed_classes: set[str] = set()
    for row in outcomes:
        if not isinstance(row, dict):
            continue
        cls = str(row.get("class") or row.get("kind") or "")
        ok = row.get("ok")
        if ok is False:
            failed_classes.add(cls)
        elif ok is True and cls in failed_classes:
            return True
    prior = facts.get("prior_failure_class")
    success = facts.get("success_class")
    return bool(prior) and prior == success


def _truthy(val: Any) -> bool:
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return float(val) >= 0.5
    return bool(val)


def declared_success_without_observed_verification(declares: float, verification_fact: bool) -> bool:
    """``declares_success ≥ .7`` and no observed verification fact."""
    return float(declares) >= 0.7 and not bool(verification_fact)


def investigation_then_change(inquire: Any, change: Any, explicit_ordering: Any) -> bool:
    """Derived: inquire ∧ change ∧ explicit_ordering."""
    return _truthy(inquire) and _truthy(change) and _truthy(explicit_ordering)


__all__ = [
    "FeatureSet",
    "FeatureValue",
    "declared_success_without_observed_verification",
    "investigation_then_change",
    "recovered",
    "stuck_window",
    "thrashing_window",
]
