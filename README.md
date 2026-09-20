# agent-hotwash

[![PyPI version](https://img.shields.io/pypi/v/agent-hotwash.svg)](https://pypi.org/project/agent-hotwash/)
[![Python versions](https://img.shields.io/pypi/pyversions/agent-hotwash.svg)](https://pypi.org/project/agent-hotwash/)
[![CI status](https://github.com/svilupp/agent-hotwash/workflows/CI/badge.svg)](https://github.com/svilupp/agent-hotwash/actions)
[![Coverage](https://img.shields.io/codecov/c/github/svilupp/agent-hotwash)](https://codecov.io/gh/svilupp/agent-hotwash)
[![License](https://img.shields.io/pypi/l/agent-hotwash.svg)](https://github.com/svilupp/agent-hotwash/blob/main/LICENSE)

Analyzes coding-agent traces (Claude, Codex, pi, code-bench) to surface bad patterns and improvement opportunities via analytics and detectors.
Point it at what an agent did to learn how it could have done better, with reports in table, JSON, CSV, or HTML for humans or CI.

## Quickstart

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12.

```bash
uv sync          # create the venv and install deps (incl. dev tools)
make check       # run the full hygiene gate (format, lint, typecheck, tests)
uv run agent-hotwash --help
```

## Usage

Point `analyze` at any trace file or directory. It auto-detects the format
(code-bench run/experiment dirs, native Claude project dirs, native Codex
rollouts, native pi session dirs), runs analytics + detectors, and renders a
report.

```bash
uv run agent-hotwash --help
uv run agent-hotwash analyze <path>...                 # analyze one or more traces
uv run agent-hotwash analyze <path> --format table     # human table (default on a TTY)
uv run agent-hotwash analyze <path> --format json      # machine JSON (default when piped)
uv run agent-hotwash analyze <path> --format html --out report.html
uv run agent-hotwash analyze <path> --format csv --out rows.csv
uv run agent-hotwash analyze ~/.codex/sessions/2026/09/18 --format table
uv run agent-hotwash analyze ~/.codex/sessions/2026/09 --format json --out month.json   # whole month, all CPUs
uv run agent-hotwash analyze <rollout> --semantic cached --format table
```

Common flags: `--config FILE` (user TOML merged over defaults), `--no-detectors`
(analytics only), `--fail-on high` (CI gate: non-zero exit if a finding at/above
the given severity is present), `--semantic off|cached|live` (default `off`;
`live` needs `TYPESAFE_API_KEY` and redacts digests unless `--allow-unredacted`),
`--jobs N` (worker processes; default `0` = one per CPU, capped at the number
of traces; `1` runs in-process). `cached` reads `~/.cache/agent-hotwash/systemone`
(not the old `jev` cache); warm it with one `live` run.

Throughput: a Codex directory is indexed once (whole tree, so parent/child
threads on different days still link), grouped into thread trees, and each
tree is loaded and analyzed in its own worker, largest first. A trace that
fails to parse is reported on stderr and skipped; the report still covers the
rest and the exit code is 1. In `--semantic live`, the JeV budget in
`[semantic]` (`requests_per_second`, `burst`, `max_concurrency`,
`max_retries`, `timeout_s`) is global per invocation — one shared token
bucket across all workers — and 429/5xx responses are retried with capped,
jittered backoff that honours `Retry-After`.

Formats: `table` (rich, degrades to plain text off a TTY), `json`, `csv` (one
row per run + a column per finding id), `html` (single self-contained file, no
external assets). `--out` takes a file or a directory (writes `report.<ext>`);
omit it to stream to stdout. When semantic mode is on, table/HTML include a
task card (Intent / Shape / Actual / Trajectory / Diagnosis) and a monthly
root-task rollup. Those JSON sections are omitted in `--semantic off`; CSV
columns stay the same.

Other commands:

```bash
uv run agent-hotwash threads PATH --format json   # Codex thread trees (json or table)
uv run agent-hotwash detectors --format json      # list registered detectors
uv run agent-hotwash config-show                  # dump the effective merged config
uv run agent-hotwash label PATH --store labels.jsonl --annotator you
uv run agent-hotwash eval --store labels.jsonl
uv run agent-hotwash version
```

**Output contract (AI-friendly):** structured data goes to **stdout**, all
human/progress messages to **stderr**. Exit codes: `0` success, `1` error,
`2` no analyzable traces found (or `--fail-on` tripped). `cached` mode exits
`1` on a cache miss; `live` exits `1` if `TYPESAFE_API_KEY` is unset.

Architecture: [docs/DESIGN.md](docs/DESIGN.md). Semantic diagnoses are
experimental until the labelling eval in that doc.

## Make targets

Run `make help` for the full list. The static-check targets are **quiet on
success** — they print one `<Name>: OK` line and only dump full tool output on
failure, to keep transcripts short for AI agents.

| Target       | What it does                                                   |
| ------------ | ------------------------------------------------------------- |
| `install` / `sync` | Sync the venv from `uv.lock` (incl. dev group).         |
| `format` / `fmt`   | Format `src` + `tests` with ruff.                       |
| `lint`       | Lint with ruff (silent unless it fails).                       |
| `format-check` | Check formatting with ruff (silent unless it fails).        |
| `typecheck`  | Type-check with [ty](https://github.com/astral-sh/ty).        |
| `bank-check` | Lint native System One feature TOML (`systemoneprompts check`). |
| `test`       | Run the pytest suite (verbose).                               |
| `check`      | Full gate: format-check + lint + typecheck + bank-check + tests, parallel. |
| `release`    | `make release BUMP=minor` — gate, bump, tag, build dists.      |
| `clean`      | Remove caches and build artifacts.                            |

## Layout

```
src/agent_hotwash/   package (CLI entry point in cli.py)
tests/               pytest suite
```

## Tooling

- **[uv](https://docs.astral.sh/uv/)** — env and dependency management.
- **[ruff](https://docs.astral.sh/ruff/)** — lint + format.
- **[ty](https://github.com/astral-sh/ty)** — type checking.
- **pytest** — tests.
