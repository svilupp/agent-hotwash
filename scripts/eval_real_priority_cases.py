"""Locked prompt comparison on independently labeled real grouped cases.

Case labels and rationales stay outside the state sent to JeV. The decision
thresholds below were fixed from the earlier exploratory pass, before reading
this evaluation set. This script reports errors; it does not tune thresholds.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
from pathlib import Path

from agent_hotwash.semantic.client import SystemOneAsker

from prototype_priority_ranking import NOULS, SCORES, ask_methods


def noul_readiness(values: dict[str, float]) -> float:
    return (
        0.4 * values["specific_change"]
        + 0.35 * values["causal_support"]
        + 0.25 * values["small_trial"]
    ) * (1 - 0.4 * values["missing_evidence"]) * (1 - 0.5 * values["broader_than_evidence"])


def auc(rows: list[dict], score_key: str) -> float | None:
    positive = [row[score_key] for row in rows if row["label"] == "ready"]
    negative = [row[score_key] for row in rows if row["label"] == "investigate"]
    if not positive or not negative:
        return None
    return sum((p > n) + 0.5 * (p == n) for p in positive for n in negative) / (len(positive) * len(negative))


def wilson_95(successes: int, total: int) -> list[float] | None:
    if not total:
        return None
    z = 1.96
    rate = successes / total
    denominator = 1 + z * z / total
    midpoint = (rate + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(rate * (1 - rate) / total + z * z / (4 * total * total)) / denominator
    return [round(max(0, midpoint - radius), 3), round(min(1, midpoint + radius), 3)]


def confusion(rows: list[dict], prediction_key: str) -> dict:
    labeled = [row for row in rows if row["label"] in {"ready", "investigate"}]
    tp = sum(row["label"] == "ready" and row[prediction_key] for row in labeled)
    fp = sum(row["label"] == "investigate" and row[prediction_key] for row in labeled)
    tn = sum(row["label"] == "investigate" and not row[prediction_key] for row in labeled)
    fn = sum(row["label"] == "ready" and not row[prediction_key] for row in labeled)
    abstain_promoted = sum(row["label"] == "abstain" and row[prediction_key] for row in rows)
    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "abstain_promoted": abstain_promoted,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "false_promotion_rate": fp / (fp + tn) if fp + tn else None,
        "false_promotion_ci95": wilson_95(fp, fp + tn),
        "recall_ci95": wilson_95(tp, tp + fn),
    }


def paired_metrics(rows: list[dict]) -> dict:
    groups: dict[str, list[dict]] = {}
    for row in rows:
        if row.get("pair_id"):
            groups.setdefault(row["pair_id"], []).append(row)
    pairs = [
        (next(row for row in group if row["label"] == "ready"),
         next(row for row in group if row["label"] != "ready"))
        for group in groups.values()
        if len(group) == 2 and sum(row["label"] == "ready" for row in group) == 1
    ]
    if not pairs:
        return {"eligible_pairs": 0}
    return {
        "eligible_pairs": len(pairs),
        "noul_ready_above_other": sum(ready["noul_readiness"] > other["noul_readiness"] for ready, other in pairs),
        "score_ready_above_other": sum(ready["score_readiness"] > other["score_readiness"] for ready, other in pairs),
        "noul_both_gate_correct": sum(ready["noul_ready"] and not other["noul_ready"] for ready, other in pairs),
        "score_both_gate_correct": sum(ready["score_ready"] and not other["score_ready"] for ready, other in pairs),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("cases", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cache-dir", default=".cache/agent-hotwash/priority-real-eval-v1")
    args = parser.parse_args()
    if not os.environ.get("TYPESAFE_API_KEY"):
        for line in Path(".env").read_text().splitlines():
            if line.startswith("TYPESAFE_API_KEY="):
                os.environ["TYPESAFE_API_KEY"] = line.partition("=")[2].strip().strip("\"'")
                break
    payload = json.loads(args.cases.read_text())
    cases = payload["cases"]
    ids = [case["id"] for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("case IDs must be unique")
    for case in cases:
        if case["label"] not in {"ready", "investigate", "abstain"}:
            raise ValueError(f"invalid label for {case['id']}")
        if set(case["state"]) != {"evidence", "next_step", "verify", "limit"}:
            raise ValueError(f"wrong state shape for {case['id']}")
        if any(len(str(value)) > 1200 for value in case["state"].values()):
            raise ValueError(f"state field too long for {case['id']}")
    asker = SystemOneAsker("jev-1.13.0", args.cache_dir, mode="live")
    rows = []
    try:
        for case_index, case in enumerate(cases):
            answers, method_ms, method_order = ask_methods(asker, case["state"], case_index)
            noul = {key: float(answers[key]["noul"]) for key in NOULS}
            score = {
                key: {"level": int(answers[key]["score"]), "confidence": float(answers[key]["confidence"])}
                for key in SCORES
            }
            readiness = noul_readiness(noul)
            score_readiness = score["readiness"]["level"] / 3 * score["readiness"]["confidence"]
            rows.append({
                "id": case["id"], "pair_id": case.get("pair_id"),
                "label": case["label"], "rationale": case.get("rationale"),
                "source": case.get("source"), "intervention_type": case.get("intervention_type"),
                "trace_ids": case.get("trace_ids", []),
                "state": case["state"], "noul": noul, "score": score,
                "noul_readiness": round(readiness, 4), "score_readiness": round(score_readiness, 4),
                "noul_ready": readiness >= 0.4 and noul["causal_support"] >= 0.6 and noul["broader_than_evidence"] <= 0.7,
                "score_ready": score["readiness"]["level"] >= 2 and score["readiness"]["confidence"] >= 0.6,
                "method_ms": method_ms, "method_order": method_order,
            })
    finally:
        stats = asker.stats()
        asker.close()
    metrics = {
        "n": len(rows),
        "label_counts": {name: sum(row["label"] == name for row in rows) for name in ("ready", "investigate", "abstain")},
        "noul_gate": confusion(rows, "noul_ready"),
        "score_gate": confusion(rows, "score_ready"),
        "noul_auc": auc(rows, "noul_readiness"),
        "score_auc": auc(rows, "score_readiness"),
        "median_noul_ms": statistics.median(row["method_ms"]["noul"] for row in rows),
        "median_score_ms": statistics.median(row["method_ms"]["score"] for row in rows),
        "paired": paired_metrics(rows),
    }
    metrics["by_intervention_type"] = {
        kind: {
            "n": sum(row["intervention_type"] == kind for row in rows),
            "labels": {label: sum(row["intervention_type"] == kind and row["label"] == label for row in rows)
                       for label in ("ready", "investigate", "abstain")},
            "noul_gate": confusion([row for row in rows if row["intervention_type"] == kind], "noul_ready"),
            "score_gate": confusion([row for row in rows if row["intervention_type"] == kind], "score_ready"),
        }
        for kind in sorted({row["intervention_type"] for row in rows})
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"model": "jev-1.13.0", "locked_rules": {
        "noul": "readiness >= 0.4 and causal_support >= 0.6 and broader_than_evidence <= 0.7",
        "score": "readiness level >= 2 and confidence >= 0.6",
    }, "stats": stats, "metrics": metrics, "cases": rows}, indent=2))
    print(json.dumps(metrics, indent=2))
    print("wrote", args.out, "network requests", stats["requests"])


if __name__ == "__main__":
    main()
