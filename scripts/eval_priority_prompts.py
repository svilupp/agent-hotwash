"""Small boundary probe for JeV priority questions (synthetic cases only).

This is a prompt diagnostic, not a validated ranking benchmark. Run beside the
seven real September action groups, not pooled with them.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from agent_hotwash.semantic.client import SystemOneAsker

from prototype_priority_ranking import NOULS, SCORES, ask_methods

CASES = [
    {
        "id": "supported_status_polling",
        "change_ready": True,
        "evidence": "Three workflows produced 33 unchanged pending results and 157 model rounds that only asked for pending status.",
        "next_step": "Use the existing blocking wait when progress is unchanged; keep checks that enable other work.",
        "verify": "Compare pending-only rounds and p95 completion latency on matched delegations.",
        "limit": "Confirm on comparable tasks before rollout.",
    },
    {
        "id": "slow_call_timeout_leap",
        "change_ready": False,
        "evidence": "2,367 slow tool calls; the longest call-to-result interval was 16 hours. The operation and external wait are unclassified.",
        "next_step": "Cut every tool timeout to 30 seconds.",
        "verify": "Compare average call duration.",
        "limit": "Duration is not CPU time; the call may be doing required work or waiting externally.",
    },
    {
        "id": "cost_cap_leap",
        "change_ready": False,
        "evidence": "Seventeen expensive runs cost $1,821 in total, but accepted scope and outcome are unknown.",
        "next_step": "Cap every coding session at $20.",
        "verify": "Compare spend per run.",
        "limit": "Some runs may be large requested work; completion quality is unmeasured.",
    },
    {
        "id": "cache_investigation",
        "change_ready": False,
        "evidence": "Forty-five groups crossed a cache-write threshold with $130.91 in estimated write charges.",
        "next_step": "Inspect prompt-prefix fingerprints and expiry before changing cache policy.",
        "verify": "Compare write tokens and input charge per completed task.",
        "limit": "Prefix identity and expiry are unobserved; charge is not proven waste.",
    },
    {
        "id": "credential_rotation_leap",
        "change_ready": False,
        "evidence": "Seventeen outbound strings matched a credential-shaped pattern in eleven runs.",
        "next_step": "Rotate all matching values immediately.",
        "verify": "Count rotations.",
        "limit": "Pattern matches do not prove values are valid or exposed to an unauthorized recipient.",
    },
    {
        "id": "verification_after_edit",
        "change_ready": True,
        "evidence": "The agent declared completion after editing and ran no build or test after its last edit.",
        "next_step": "Require a relevant check after the final edit, or state explicitly why verification could not run.",
        "verify": "Measure final-edit-to-check coverage and accepted task outcomes.",
        "limit": "Some changes cannot be exercised locally; preserve an explicit unavailable path.",
    },
    {
        "id": "identical_failed_retry",
        "change_ready": True,
        "evidence": "A command returned the same diagnostic four times; the next calls reused identical arguments with no intervening state change.",
        "next_step": "After a repeated identical failure, require a changed hypothesis or stop-and-report path before another call.",
        "verify": "Compare repeat calls per resolved blocker and recovery completion.",
        "limit": "Do not suppress retries after state changes.",
    },
    {
        "id": "unknown_child_completion",
        "change_ready": True,
        "evidence": "Most delegation records have no exact child completion or parent-consumption marker.",
        "next_step": "Record explicit child completion, cancellation, and parent consumption in the trace.",
        "verify": "Reduce the share with unknown lifecycle state.",
        "limit": "Missing joins do not prove a child hung; this is an observability change.",
    },
    {
        "id": "large_count_no_intervention",
        "change_ready": False,
        "evidence": "A detector fired 3,786 times, but its underlying operations and outcomes are not classified.",
        "next_step": "Inspect a stratified sample and identify distinct causes.",
        "verify": "Measure share of reviewed samples with a common cause.",
        "limit": "A large count can mix expected and harmful behavior.",
    },
    {
        "id": "edit_without_read_overreach",
        "change_ready": False,
        "evidence": "The agent edited files without a recorded read; the user may have supplied the exact patch context.",
        "next_step": "Force a full-file read before every edit.",
        "verify": "Count reads before edits.",
        "limit": "The read could be redundant or costly; the sample lacks outcome checks.",
    },
]


def main() -> None:
    if not os.environ.get("TYPESAFE_API_KEY"):
        for line in Path(".env").read_text().splitlines():
            if line.startswith("TYPESAFE_API_KEY="):
                os.environ["TYPESAFE_API_KEY"] = line.partition("=")[2].strip().strip("\"'")
                break
    asker = SystemOneAsker("jev-1.13.0", ".cache/agent-hotwash/priority-isolated-v2", mode="live")
    rows = []
    try:
        for case_index, case in enumerate(CASES):
            state = {key: case[key] for key in ("evidence", "next_step", "verify", "limit")}
            answers, method_ms, method_order = ask_methods(asker, state, case_index)
            noul = {key: float(answers[key]["noul"]) for key in NOULS}
            scores = {
                key: {"level": int(answers[key]["score"]), "confidence": float(answers[key]["confidence"])}
                for key in SCORES
            }
            readiness = (
                0.4 * noul["specific_change"]
                + 0.35 * noul["causal_support"]
                + 0.25 * noul["small_trial"]
            ) * (1 - 0.4 * noul["missing_evidence"]) * (1 - 0.5 * noul["broader_than_evidence"])
            rows.append({
                "id": case["id"], "origin": "synthetic", "change_ready_label": case["change_ready"],
                "state": state, "noul": noul, "score": scores, "noul_readiness": round(readiness, 4),
                "method_ms": method_ms, "method_order": method_order,
            })
    finally:
        stats = asker.stats()
        asker.close()
    out = Path("reports/2026-09-26-september-run/priority-boundary-probe.json")
    out.write_text(json.dumps({"model": "jev-1.13.0", "stats": stats, "cases": rows}, indent=2))
    for row in rows:
        score = row["score"]["readiness"]
        print(row["id"], row["change_ready_label"], row["noul_readiness"], score)
    print("wrote", out, "requests", stats["requests"])


if __name__ == "__main__":
    main()
