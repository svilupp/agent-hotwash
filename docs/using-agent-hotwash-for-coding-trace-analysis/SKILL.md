---
name: using-agent-hotwash-for-coding-trace-analysis
description: Analyze Claude, Codex, Pi, and code-bench traces with agent-hotwash to review execution patterns and propose evidence-linked guidance improvements.
---

# Review coding-agent traces

Use `agent-hotwash` to measure a recorded run and inspect its findings. Keep
recommendations tied to visible evidence and the user's requested scope.

## Run

Start with local analysis:

```bash
uvx agent-hotwash analyze /path/to/traces --semantic off --format html --out report.html
```

Inside a checkout, use `uv run` instead of `uvx`. Inputs may be files or
directories; formats are detected automatically. Preserve `report-evidence/`
alongside the HTML when sharing.

For guidance proposals:

```bash
uvx agent-hotwash analyze /path/to/traces --analysis memory \
  --guidance /path/to/CLAUDE.md --semantic cached --format html --out memory.html
```

`cached` requires saved responses. Live diagnostics need `TYPESAFE_API_KEY`;
live memory audits also need `OPENAI_API_KEY`. Confirm provider use is within
the user's authorized scope before selecting `--semantic live`.

## Review

1. Check source coverage and degraded metrics. Missing measurements are unknown.
2. Read the highest-ranked action cards and follow their citations.
3. Inspect the source span before accepting a detector or model judgment.
4. Separate measured burden from evidence of an avoidable problem.
5. Report concrete changes, supporting evidence, and remaining uncertainty.

Run `detectors` to inspect unfamiliar findings. Use `config-show` and
`--config FILE` to inspect or adjust thresholds. For memory proposals, compare
explicit guidance files and review drafts before changing durable instructions.

See [reports](../reports.md), [memory audits](../memory-audit.md), and
[design](../DESIGN.md) for the maintained reference. Use `analyze --help` for
CLI options.
