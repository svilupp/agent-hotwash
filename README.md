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
uv run agent-hotwash analyze <path>...                 # analyze one or more traces
uv run agent-hotwash analyze <path> --format table     # human table (default on a TTY)
uv run agent-hotwash analyze <path> --format json      # machine JSON (default when piped)
uv run agent-hotwash analyze <path> --format html --out report.html
uv run agent-hotwash analyze <path> --format csv --out rows.csv
```

Common flags: `--config FILE` (user TOML merged over defaults), `--no-detectors`
(analytics only), `--fail-on high` (CI gate: non-zero exit if a finding at/above
the given severity is present).

Formats: `table` (rich, degrades to plain text off a TTY), `json`, `csv` (one
row per run + a column per finding id), `html` (single self-contained file, no
external assets). `--out` takes a file or a directory (writes `report.<ext>`);
omit it to stream to stdout.

Other commands:

```bash
uv run agent-hotwash detectors --format json   # list registered detectors
uv run agent-hotwash config-show               # dump the effective merged config
uv run agent-hotwash version
```

**Output contract (AI-friendly):** structured data goes to **stdout**, all
human/progress messages to **stderr**. Exit codes: `0` success, `1` error,
`2` no analyzable traces found (or `--fail-on` tripped).

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
| `test`       | Run the pytest suite (verbose).                               |
| `check`      | Full gate: format-check + lint + typecheck + tests, parallel. |
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
