"""Stream a full JSON report to add spend and failure-action views to a brief.

Run with ``uv run --with ijson python scripts/enrich_brief_overview.py FULL BRIEF OUT``.
The 2 GB September report is read one run at a time; no trace payloads are
copied into the portable brief.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import ijson

from agent_hotwash.config import load_config
from agent_hotwash.report.error_actions import ErrorActionAccumulator
from agent_hotwash.report.highlights import HighlightsData
from agent_hotwash.report.overview import SpendOverviewBuilder


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    parser.add_argument("brief", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    brief = HighlightsData.model_validate_json(args.brief.read_text())
    pricing = load_config()
    spend = SpendOverviewBuilder(pricing=pricing)
    errors = ErrorActionAccumulator()
    count = 0
    with args.report.open("rb") as source:
        for run in ijson.items(source, "runs.item"):
            spend.add_run(run)
            errors.add_run(run)
            count += 1
    if count != brief.traces:
        raise ValueError(f"report has {count} runs; brief expects {brief.traces}")
    updated = brief.model_copy(update={"spend_overview": spend.build(), "error_actions": errors.summary()})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(updated.model_dump_json(indent=2))
    print(f"wrote {args.out}: {count} runs, {updated.spend_overview.calls if updated.spend_overview else 0} calls")


if __name__ == "__main__":
    main()
