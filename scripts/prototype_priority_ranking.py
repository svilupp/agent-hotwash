"""Compare atomic Noul and explicit Score rubrics on grouped action themes.

Reads only the small trailing ``priorities`` array from a full report JSON, so
the 2 GB September snapshot does not need to be loaded into memory.
"""

from __future__ import annotations

import argparse
import json
import mmap
import os
import time
from pathlib import Path

from agent_hotwash.report.ranking import Judgment, RankedTheme, score_theme
from agent_hotwash.semantic.client import SystemOneAsker
from agent_hotwash.semantic.project import project_for_questions


NOULS = {
    "specific_change": "Does `next_step` propose a specific change to agent or tool behavior that can be tested, rather than only asking someone to inspect records?",
    "causal_support": "Does `evidence` directly support the particular behavioral change in `next_step`, rather than only showing a large count, charge, or elapsed time?",
    "small_trial": "Can the change in `next_step` be trialed on a small set of comparable workflows with an obvious rollback?",
    "missing_evidence": "Does `limit` identify missing causal, scope, or outcome evidence that must be resolved before the proposed change is justified?",
    "broader_than_evidence": "Does `next_step` impose a blanket requirement or change on more workflows or operations than `evidence` supports? A conditional trial on the observed case is not blanket scope.",
}

SCORES = {
    "readiness": {
        "type": "score",
        "instructions": {"question": "How ready is `next_step` as an intervention, considering `evidence` and `limit`? Judge causal support, not the size of the burden.", "inspect": ["next_step", "evidence", "limit"]},
        "criteria": [
            "Only an observation or detector pattern is present; no practical next step is stated.",
            "A concrete investigation or measurement is stated, but the evidence does not yet justify changing behavior.",
            "A specific change is proposed with a plausible link to the evidence, but the causal link needs a bounded trial.",
            "A specific behavior is directly supported by the trace evidence and can be tested while preserving task quality.",
        ],
    },
    "ease": {
        "type": "score",
        "instructions": {"question": "How easy is it to test `next_step` on comparable work, given `verify` and `limit`? Judge trial effort, not eventual impact.", "inspect": ["next_step", "verify", "limit"]},
        "criteria": [
            "Requires new instrumentation, missing evidence, or a broad design effort before a trial can begin.",
            "Requires focused investigation or a substantial tool or workflow change before a trial.",
            "Can be trialed in one bounded workflow with a clear comparison and limited implementation work.",
            "Can be trialed with a local prompt, configuration, or scheduling adjustment and a clear rollback.",
        ],
    },
}


def ask_methods(asker: SystemOneAsker, state: dict, case_index: int) -> tuple[dict, dict, str]:
    """Ask each primitive in a separate request, alternating call order."""
    noul_questions = {
        key: {
            "type": "noul",
            "instructions": {"question": prompt, "inspect": ["next_step", "evidence", "limit", "verify"]},
        }
        for key, prompt in NOULS.items()
    }
    methods = {"noul": noul_questions, "score": SCORES}
    order = ("noul", "score") if case_index % 2 == 0 else ("score", "noul")
    answers: dict = {}
    elapsed: dict = {}
    for method in order:
        questions = methods[method]
        if project_for_questions(state, questions) != state:
            raise ValueError(f"{method} questions must project every state field")
        started = time.monotonic()
        answers.update(asker.ask(state, questions))
        elapsed[method] = round((time.monotonic() - started) * 1000)
    return answers, elapsed, "→".join(order)


def load_trailing(path: Path, key: str, *, max_bytes: int = 1_000_000):
    with path.open("rb") as source, mmap.mmap(source.fileno(), 0, access=mmap.ACCESS_READ) as data:
        marker = f'"{key}"'.encode()
        start = data.rfind(marker)
        if start < 0:
            raise ValueError(f"report has no {key}")
        after = start + len(marker)
        sample = data[after : after + max_bytes].decode()
        return json.JSONDecoder().raw_decode(sample.lstrip(": \n"))[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default="jev-1.13.0")
    args = parser.parse_args()
    if not os.environ.get("TYPESAFE_API_KEY"):
        for line in Path(".env").read_text().splitlines():
            if line.startswith("TYPESAFE_API_KEY="):
                os.environ["TYPESAFE_API_KEY"] = line.partition("=")[2].strip().strip("\"'")
                break
    actions = load_trailing(args.report, "priorities")
    total_runs = load_trailing(args.report, "aggregate", max_bytes=6_000_000)["overall"]["n"]
    cost_reference = max((row["observed_cost"] for row in actions if row.get("observed_cost") is not None), default=None)
    asker = SystemOneAsker(args.model, ".cache/agent-hotwash/priority-isolated-v2", mode="live")
    results = []
    try:
        for case_index, action in enumerate(actions):
            state = {key: action[key] for key in ("evidence", "next_step", "verify", "limit")}
            started = time.monotonic()
            answers, method_ms, method_order = ask_methods(asker, state, case_index)
            noul = {key: float(answers[key]["noul"]) for key in NOULS}
            scores = {
                key: {"level": int(answers[key]["score"]), "confidence": float(answers[key]["confidence"])}
                for key in SCORES
            }
            # The level is zero-indexed. Confidence is a caution multiplier,
            # not an estimated probability of a correct answer.
            noul_readiness = (
                0.4 * noul["specific_change"]
                + 0.35 * noul["causal_support"]
                + 0.25 * noul["small_trial"]
            ) * (1 - 0.4 * noul["missing_evidence"]) * (1 - 0.5 * noul["broader_than_evidence"])
            score_readiness = scores["readiness"]["level"] / 3 * scores["readiness"]["confidence"]
            theme = RankedTheme(
                id=action["id"], kind="action", title=action["title"], status=action["status"],
                affected_runs=action["affected_runs"], incidents=action["incidents"],
                observed_cost=action.get("observed_cost"), evidence=action["evidence"],
                next_step=action["next_step"], verify=action["verify"], limit=action["limit"],
            )
            noul_total = score_theme(
                theme, total_runs, cost_reference=cost_reference,
                judgment=Judgment(actionability=noul_readiness, ease=noul["small_trial"], method="noul_v1"),
            ).score
            score_total = score_theme(
                theme, total_runs, cost_reference=cost_reference,
                judgment=Judgment(
                    actionability=scores["readiness"]["level"] / 3,
                    ease=scores["ease"]["level"] / 3,
                    confidence=min(scores["readiness"]["confidence"], scores["ease"]["confidence"]),
                    method="score_v1",
                ),
            ).score
            results.append(
                {
                    "id": action["id"],
                    "title": action["title"],
                    "status": action["status"],
                    "affected_runs": action["affected_runs"],
                    "incidents": action["incidents"],
                    "observed_cost": action.get("observed_cost"),
                    "state": state,
                    "noul": noul,
                    "score": scores,
                    "noul_readiness": round(noul_readiness, 4),
                    "score_readiness": round(score_readiness, 4),
                    "noul_total_score": noul_total,
                    "score_total_score": score_total,
                    "elapsed_ms": round((time.monotonic() - started) * 1000),
                    "method_ms": method_ms,
                    "method_order": method_order,
                }
            )
    finally:
        stats = asker.stats()
        asker.close()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"model": args.model, "stats": stats, "results": results}, indent=2))
    print(f"wrote {args.out} ({len(results)} grouped themes; {stats['requests']:.0f} network requests)")
    for method in ("noul_readiness", "score_readiness", "noul_total_score", "score_total_score"):
        print(method, [(row["id"], row[method]) for row in sorted(results, key=lambda r: -r[method])])


if __name__ == "__main__":
    main()
