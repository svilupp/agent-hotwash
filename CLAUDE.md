# CLAUDE.md

`agent-hotwash` analyzes coding-agent traces (code-bench runs, native Claude
project dirs, native Codex rollouts) to surface improvement opportunities and
bad patterns. Python 3.12, managed with `uv`.

See `README.md` for usage/CLI and `docs/DESIGN.md` for architecture — don't
duplicate them here.

## Layout

- `src/agent_hotwash/cli.py` — CLI entrypoint (`agent-hotwash`).
- `src/agent_hotwash/sources/` — trace loaders + format autodetect (`detect.py`).
- `src/agent_hotwash/primitives/` — shared trace primitives (windows, file state, errors, commands).
- `src/agent_hotwash/analytics.py`, `aggregate.py`, `events.py` — analytics pipeline.
- `src/agent_hotwash/detectors/` — pattern detectors + registry/taxonomy.
- `src/agent_hotwash/report/` — output renderers (table/json/csv/html).
- `src/agent_hotwash/config.py`, `config/defaults.toml` — config + defaults.
- `tests/` — pytest suite. `docs/` — design notes and research.

## Checks — use the Makefile

Always run checks via the Makefile targets. They are **quiet on success** (one
`<Name>: OK` line, full tool output only on failure) to keep transcripts short.

- `make check` — full gate: format-check, lint, typecheck, tests (parallel). Use before finishing work.
- `make lint` / `make format-check` / `make typecheck` / `make test` — individual legs.
- `make format` — apply ruff formatting (mutates files).
- `make install` — sync the venv from `uv.lock`. `make help` — list all targets.

Prefer these over calling `uv run ruff`/`pytest`/`ty` directly.
