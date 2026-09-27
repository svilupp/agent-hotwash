"""Show how editorial score weights change September within-lane ordering."""

from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    root = Path("reports/2026-09-26-september-run")
    brief = json.loads((root / "september-brief.json").read_text())
    investigations = [theme for theme in brief["themes"] if theme["status"] == "investigate"]
    detectors = [theme for theme in brief["themes"] if theme["kind"] == "detector"]

    def ranked(rows: list[dict], score):
        return [{"id": row["id"], "score": round(score(row), 2)} for row in sorted(rows, key=lambda r: -score(r))]

    scenarios = {
        "investigations_published": ranked(investigations, lambda row: row["score"]),
        "investigations_no_spend_points": ranked(
            investigations, lambda row: row["score"] - row["components"]["observed_spend"]
        ),
        "investigations_half_spend_points": ranked(
            investigations, lambda row: row["score"] - 0.5 * row["components"]["observed_spend"]
        ),
        "detectors_published": ranked(detectors, lambda row: row["score"]),
        "detectors_severity_50_reach_50": ranked(
            detectors,
            lambda row: 50 * row["components"]["affected_runs"] / 60
            + 50 * row["components"]["max_severity"] / 40,
        ),
    }
    out = root / "rank-sensitivity.json"
    out.write_text(json.dumps({"basis": "2026-09-26 saved brief, no rerun", "scenarios": scenarios}, indent=2))
    for name, rows in scenarios.items():
        print(name, [(row["id"], row["score"]) for row in rows[:5]])
    print("wrote", out)


if __name__ == "__main__":
    main()
