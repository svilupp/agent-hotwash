"""Diagnostic of the revised atomic-Noul action review on saved labeled cases."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from agent_hotwash.report.jev_review import project_case_state, review_state
from agent_hotwash.report.priorities import ActionReviewContext
from agent_hotwash.semantic.client import SystemOneAsker

from eval_real_priority_cases import confusion


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("cases", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cache-dir", default=".cache/agent-hotwash/priority-review-v2")
    args = parser.parse_args()
    if not os.environ.get("TYPESAFE_API_KEY"):
        for line in Path(".env").read_text().splitlines():
            if line.startswith("TYPESAFE_API_KEY="):
                os.environ["TYPESAFE_API_KEY"] = line.partition("=")[2].strip().strip("\"'")
                break
    cases = json.loads(args.cases.read_text())["cases"]
    asker = SystemOneAsker("jev-1.13.0", args.cache_dir, mode="live")
    rows = []
    try:
        for case in cases:
            context = ActionReviewContext.model_validate(case["review_context"]) if case.get("review_context") else None
            review = review_state(project_case_state(case["state"], context), asker)
            rows.append({
                "id": case["id"], "pair_id": case.get("pair_id"), "label": case["label"],
                "decision": review.decision, "features": review.features,
                "promoted": review.decision != "review",
            })
    finally:
        stats = asker.stats()
        asker.close()
    metrics = confusion(rows, "promoted")
    pairs: dict[str, list[dict]] = {}
    for row in rows:
        if row["pair_id"]:
            pairs.setdefault(row["pair_id"], []).append(row)
    metrics["both_pair_gates_correct"] = sum(
        len(group) == 2 and any(row["label"] == "ready" and row["promoted"] for row in group)
        and all(row["label"] == "ready" or not row["promoted"] for row in group)
        for group in pairs.values()
    )
    metrics["pair_count"] = len(pairs)
    args.out.write_text(json.dumps({"model": asker.model, "metrics": metrics, "stats": stats, "cases": rows}, indent=2))
    print(json.dumps({"metrics": metrics, "stats": stats}, indent=2))


if __name__ == "__main__":
    main()
