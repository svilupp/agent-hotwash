# agent-hotwash

Analyze Claude, Codex, Pi, and code-bench traces for repeated work, execution
costs, and opportunities to improve agent guidance. Reports link findings to
source evidence.

## Quickstart

Requires Python 3.12 or 3.13 and [uv](https://docs.astral.sh/uv/).

```bash
uvx agent-hotwash analyze /path/to/traces --semantic off --format html --out report.html
```

Open `report.html` and keep its sibling `report-evidence/` directory when sharing.
`--semantic off` runs local analytics and detectors without provider calls.
The default, `live`, requires `TYPESAFE_API_KEY`; `cached` reuses saved responses
and fails on a cache miss.

## Usage

```bash
uvx agent-hotwash analyze /path/to/traces --semantic off --format json --out report.json
uvx agent-hotwash analyze /path/to/traces --semantic live --format brief-json --out brief.json
uvx agent-hotwash render-brief brief.json --out brief.html
uvx agent-hotwash threads /path/to/sessions --format json
uvx agent-hotwash detectors
uvx agent-hotwash config-show
```

Pass multiple files or directories to `analyze`. Source formats are detected
automatically. Codex and Pi child sessions are grouped into thread trees.

| Option | Purpose |
| --- | --- |
| `--analysis diagnostics\|memory\|all` | Diagnostics (default), memory audit, or both |
| `--semantic off\|cached\|live` | Local measurements, cached judgments, or provider calls |
| `--format table\|json\|brief-json\|csv\|html` | Diagnostics output; default is table on a terminal, JSON when piped |
| `--out PATH` | Output file or existing directory; omit for stdout |
| `--config FILE` | TOML overrides merged over defaults |
| `--jobs N` | Worker processes; `0` uses available CPUs, capped by trace count |
| `--no-detectors` | Analytics only |
| `--fail-on LEVEL` | Exit 2 for findings at or above info, low, medium, or high |

Use `analyze --help` for selection filters and all options. Structured output
goes to stdout; progress and errors go to stderr. Exit codes are `0` for success,
`1` for errors, and `2` for no analyzable traces or a tripped finding threshold.

## Memory audits

```bash
uvx agent-hotwash analyze /path/to/traces --analysis memory --semantic off --format table
uvx agent-hotwash analyze /path/to/traces --analysis all \
  --guidance /path/to/CLAUDE.md --format html --out review.html
```

Memory audits prepare file-read analytics and command datasets, then optionally
propose guidance changes from corrections and repeated discoveries. Live audits
need both `TYPESAFE_API_KEY` and `OPENAI_API_KEY`. `all` writes diagnostics to
`review.html` and a separate audit to `review.memory.html`.

See [memory audits](docs/memory-audit.md) for configuration, outputs, and replay.
Review proposals before changing guidance.

## Reports and configuration

Start with the report's action cards, then follow citations to the trace.
Missing evidence remains unknown. Cost, repetition, and duration alone do not
establish waste or savings; semantic judgments remain experimental.

See [report interpretation](docs/reports.md) for formats and evidence rules.
Thresholds, model prices, and provider budgets live in
[config/defaults.toml](config/defaults.toml). Override selected keys with
`--config`; inspect the merged result with `config-show`.

## Development

```bash
make install
make check
uv run agent-hotwash --help
```

`make check` runs formatting, lint, type checks, feature-bank validation, and
deterministic tests. Each leg prints one result line; failures include details.
`make format` applies formatting. `make test-live` requires `TYPESAFE_API_KEY`.
Use `make help` for individual targets.

Architecture: [design](docs/design.md). Release history: [changelog](changelog.md).

To release, prepare matching versions in `pyproject.toml` and
`src/agent_hotwash/__init__.py`, then commit the reviewed changes. From a clean
checkout, `make release` checks, builds, and tags that version. `make publish`
requires the tag at HEAD and uploads to PyPI using configured uv credentials.
Neither target bumps the version.
