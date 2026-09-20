# agent-hotwash

Analyzes coding-agent traces (Claude, Codex, pi, code-bench) to surface bad
patterns and improvement opportunities via analytics and detectors. Point it
at what an agent did to learn how it could have done better, with reports in
table, JSON, CSV, or HTML for humans or CI.

## Quickstart

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12.

```bash
uv sync          # create the venv and install deps (incl. dev tools)
uv run agent-hotwash --help
```

Point `analyze` at any trace file or directory. It auto-detects the format
(code-bench run/experiment dirs, native Claude project dirs, native Codex
rollouts, native pi session dirs), runs analytics + detectors, and renders a
report:

```bash
uv run agent-hotwash analyze <path>...
```

See the [README](https://github.com/svilupp/agent-hotwash#readme) for the full
CLI reference, and [Design](DESIGN.md) for architecture notes.
