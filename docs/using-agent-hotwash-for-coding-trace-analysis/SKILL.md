---
name: using-agent-hotwash-for-coding-trace-analysis
description: Analyze coding-agent traces and transcripts with the agent-hotwash CLI to surface bad patterns, inefficiencies, and improvement opportunities. Use when reviewing what a coding agent did — after a Claude/Codex/pi session, when auditing code-bench benchmark runs, or when running a hotwash/retro on agent behavior — and you want structured findings (smells, taxonomy detectors, cost/timing analytics) in table, JSON, CSV, or HTML.
---

# Analyzing coding-agent traces with agent-hotwash

`agent-hotwash` reads a coding agent's trace, computes deterministic analytics
(tokens, cost, timing, tool/file ratios, outcome), runs pattern **detectors**,
and renders a report. Point it at what an agent did to learn how it could have
done better.

## Running it

Prefer `uvx` so it works on any machine without an install:

```bash
uvx agent-hotwash analyze <path>...
```

Inside a checkout of the repo, use the local copy instead:

```bash
uv run agent-hotwash analyze <path>...      # from the repo root
uvx --from . agent-hotwash analyze <path>   # local source without uv sync
```

## Input formats (auto-detected)

Pass a file or directory to `analyze`; the format is fingerprinted by structure,
not filename. Supported:

- **code-bench** — a run dir (`run.json` + stdout stream), an instance dir, an
  experiment dir, or a runs root (container dirs are expanded recursively).
  Dirs suffixed `.quarantine` / `.interrupted` / `.crashed*` are skipped.
- **native Claude** — a project dir of `<session>.jsonl` files (records with
  `uuid`/`parentUuid`), or a single such session file.
- **native Codex** — a `rollout-*.jsonl` file, or a sessions/date tree of them.
- **native pi** — a project dir of `<ISO-ts>_<uuid>.jsonl` session files, or one
  such file.

You do not choose the format; just give the path. Multiple paths are allowed.

## Commands

```bash
uvx agent-hotwash analyze <path>...          # detect, analyze, detect patterns, render
uvx agent-hotwash detectors                  # list registered detectors (discovery)
uvx agent-hotwash config-show                # dump the effective merged config as JSON
uvx agent-hotwash version                    # print version
```

### `analyze` flags

- `--format, -f {table,json,csv,html}` — output format. Default: `table` on a
  TTY, `json` when piped.
- `--out, -o PATH` — write to a file or directory (writes `report.<ext>` in a
  dir). Omit to stream to stdout.
- `--config, -c FILE` — user TOML deep-merged over the defaults.
- `--no-detectors` — analytics + aggregate only, skip pattern detectors.
- `--fail-on {info,low,medium,high}` — exit non-zero (code 2) if any finding at
  or above this severity is present (CI gate).

`detectors` and `config-show` also take `-f`/`-c` respectively (see `--help`).

## Output formats — when to use each

- `json` — machine consumption; full structured report. Best for feeding results
  back to an agent or another script.
- `csv` — one row per run, one column per finding id. Best for spreadsheets or
  comparing many runs.
- `table` — quick human look in the terminal (degrades to plain text off a TTY).
- `html` — self-contained single file (no external assets) for sharing with
  people.

**Output contract:** structured data goes to **stdout**, all progress/diagnostic
messages to **stderr**. Exit codes: `0` success, `1` error, `2` no analyzable
traces found (or `--fail-on` tripped).

## Concepts

- **Analytics** — deterministic per-trace metrics: turn/tool/error counts, token
  totals and USD cost, wall-clock and active/idle timing, file-operation ratios,
  and an inferred outcome. Metrics that can't be measured (missing timestamps or
  token usage) are `None`, not `0`, and listed in `degraded` — so "no idle time"
  is distinguishable from "couldn't measure it".
- **Detectors** — pure `(session, config) -> findings` checks. Two kinds:
  **smells** (single-threshold heuristics like `bloated_opener`, `thin_prompt`,
  `cold_start_reads`) and **taxonomy** patterns (richer behaviors like
  `edit_thrash`, `correction_loop`, `retry_storm`, `context_rot`,
  `unverified_completion`, `credential_leak`, `runaway_session`). Each finding
  carries an id, kind, tier, severity (info/low/medium/high), confidence, and a
  span pointing back into the event stream.
- **Taxonomy** — the catalog of detectable patterns; each has knobs under
  `[taxonomy.<id>]` in the config. Run `agent-hotwash detectors` to see the full
  registered set before interpreting a report.

## Configuration

All thresholds, lexicons, and pricing live in `config/defaults.toml`. Override
only the keys you want in a user TOML and pass `--config`; it deep-merges over
the defaults. Use `config-show` to inspect the effective merged config. Common
overrides: `[smells]` thresholds, `[detectors].disabled`/`enabled`,
`[detectors.severity]` per-id bumps, `[taxonomy.<id>]` knobs, `[pricing.*]`.

## Interpreting results — best practices

- Run `detectors` first when a finding id is unfamiliar; the `doc` field explains
  what fired.
- Treat `severity` and `confidence` together — a low-confidence `info` smell is a
  hint, not a verdict. Read the finding's span in the trace before acting.
- Check `degraded` metrics: a source that carries no timestamps or token usage
  will null out timing/cost, so absent numbers may mean unmeasured, not zero.
- For CI or batch triage, use `--format json` (or `csv`) with `--fail-on medium`
  to gate on the patterns you care about.
- Tune thresholds via `--config` rather than dismissing findings — most smells
  are one threshold away from matching your team's norms.
