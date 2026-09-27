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
uv run agent-hotwash analyze <path> --format html --out report.html   # report.html + report-evidence/
uv run agent-hotwash analyze <path> --format brief-json --out brief.json
uv run agent-hotwash render-brief brief.json --out brief.html
uv run agent-hotwash analyze <path> --format html --full-html --out evidence.html
uv run agent-hotwash analyze <path> --format csv --out rows.csv
uv run agent-hotwash analyze ~/.codex/sessions/2026/09/18 --format table
uv run agent-hotwash analyze ~/.codex/sessions/2026/09 --format json --out month.json   # whole month, all CPUs
uv run agent-hotwash analyze <rollout> --semantic cached --format table
```

Common flags: `--config FILE` (user TOML merged over defaults), `--no-detectors`
(analytics only), `--fail-on high` (CI gate: non-zero exit if a finding at/above
the given severity is present), `--semantic off|cached|live` (default `live`;
exits 1 immediately if `TYPESAFE_API_KEY` is unset — pass `--semantic off` or
export the key), `--jobs N` (worker processes; default `0` = one per CPU,
capped at the number of traces; `1` runs in-process). `cached` reads
`~/.cache/agent-hotwash/systemone` (not the old `jev` cache); warm it with one
`live` run. Pytest / `make check` / GitHub Actions set `AGENT_HOTWASH_SEMANTIC=off`
so the suite does not need a key.

Throughput: a Codex directory is indexed once (whole tree, so parent/child
threads on different days still link), grouped into thread trees, and each
tree is loaded and analyzed in its own worker, largest first. A trace that
fails to parse is reported on stderr and skipped; the report still covers the
rest and the exit code is 1. In `--semantic live`, the JeV budget in
`[semantic]` (`requests_per_second`, `burst`, `max_concurrency`,
`max_retries`, `timeout_s`) is global per invocation — one shared token
bucket across all workers — and 429/5xx responses are retried with capped,
jittered backoff that honours `Retry-After`.

Formats: `table` (rich, degrades to plain text off a TTY), `json` (all records),
`brief-json` (ranked themes and top-level metrics), `csv` (one row per run + a
column per finding id), `html` (ranked, linked report). With `--out report.html`,
HTML writes a compact front page and `report-evidence/`: a run index paged at
100 rows and one evidence page per run. Keep the file and directory together
when sharing the offline report. The front page shows at most five action cards
and five detector cards, with lower-ranked themes in compact indexes. HTML sent
to stdout and HTML produced by `render-brief` are compact single pages without
run evidence. `--full-html` retains the legacy single-file evidence view; it can
be large. `--out` takes a file or an existing directory (writes `report.<ext>`
there). Per-run evidence pages include task cards when semantic mode is on. Those
JSON sections are omitted in `--semantic off`; CSV uses the same columns in
both semantic modes.

Schema 4 added `analysis.handovers` and `analysis.cache_waits` to each JSON run,
plus handover counts in `aggregate`. Schema 6 adds `ranked_themes` without removing
per-run evidence. Each handover ID comes from the observed
spawn coordinate; continuations and steering are separate events. Payload
lengths exist only for visible plaintext. Codex encrypted messages have null
lengths. The CSV keeps one row per trace and adds handover IDs, statuses, and
visibility counts; use JSON or `--full-html` for the event ledger. Existing schema 3
failed-result fields remain available, with owner, disposition, evidence status,
and incident ID added. A Pi child with only a display prefix is listed as a
link candidate rather than treated as an exact session match.
See [handover metrics](docs/HANDOVER_METRICS.md) for field definitions,
denominators, null rules, and the HTML cache-wait table.

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
`1` on a cache miss; default `live` exits `1` before analysis if
`TYPESAFE_API_KEY` is unset.

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
| `test`       | Deterministic pytest suite (no TypeSafe key).                  |
| `test-live`  | Live TypeSafe smoke (`TYPESAFE_API_KEY` required).             |
| `check`      | Full gate: format-check + lint + typecheck + bank-check + tests, parallel. |
| `release`    | Check, build, and tag the committed version.                 |
| `publish`    | Check the clean tagged commit, then build and upload to PyPI. |
| `clean`      | Remove caches and build artifacts.                            |

See [Releasing](docs/RELEASING.md) for the merge, tag, and publish sequence.
`make release` tags the prepared commit; `make publish` requires that tag at
the clean checkout's HEAD. Neither target bumps the version.

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

### Execution extremes and expensive runs

Start with **Actions to examine** in the compact HTML or **Actions to review** in the
terminal. Each item has measured evidence, a next step, and a verification
criterion. Supported changes appear before investigations and visibility
fixes. Action themes score affected runs, observed charge relative to this
report's largest action charge, actionability, and ease of testing. Detector
themes score affected runs and maximum severity. The two 0-100 scales are
separate reading aids, not savings estimates or calibrated risk. `brief-json` retains each score component; full JSON also
retains the action queue in `priorities` and grouped ranking in `ranked_themes`.
CSV retains its per-trace layout.
See [action queue rules and validation](docs/ACTIONABLE_REPORTS.md).

Classifier coverage separates received answers, uncertain answers, failed
requests, and unavailable evidence in HTML, terminal, and JSON
(`classifier_coverage`). A failed request cannot support a change recommendation.
Authentication and billing errors fail the review; other API failures remain
explicitly visible in partial reports.

In the linked HTML report, example trace IDs open the specific incident when
its anchor is present, or the run evidence page otherwise.
The legacy full HTML view links directly to incident details, including cases
outside the default top 20. Slow-call details show redacted invocation and result excerpts. Cache
details show individual transitions and charges; full source and classifier
records remain available behind a second disclosure. The report is self-contained
and works offline; bundled JavaScript opens linked evidence automatically.

Reports retain execution extremes even below alert thresholds: long tool calls,
delegation lifetimes, repeated attempts, large outputs, and model-token use.
Threshold crossings emit `TAIL_*` **observation** findings. An unfinished child
is a censored duration, not a proven hang; overlapping durations are never
added into a savings estimate.

The full HTML report includes the most expensive 5% within each cost basis (including
cutoff ties), a model work-mix table with input/output/cache charge components,
and an execution ledger with source coordinates, suggested actions, and JeV
evidence. Parent and child files are assembled into workflows before ranking.
Expense review checks requested scope and returned verification; an appropriate
result does not establish that the run used the cheapest approach.

### How deterministic detection and JeV work together

Code selects and measures candidates. Small `systemoneprompts` feature definitions
ask JeV about specific visible facts; Python combines their answers into an
assessment. The classifier never estimates saved dollars or decides that a long
run was wasteful from its cost alone.

| Pattern | Deterministic evidence | JeV's bounded question |
| --- | --- | --- |
| Recovery loops | Matched attempts and diagnostic chains | Same blocker? Changed approach? |
| Serial inspection | Tool calls joined to model rounds by response ID or source record | Broad audit requested? Named independent targets batchable? |
| Polling | Nonblocking calls, unchanged pending progress for the same arguments, and pending-only model rounds | Blocking wait offered? Repeated monitoring requested? |
| Repeated output | Same invocation and full captured-output hash within one user request | Visible reason to reacquire this result? |
| Repeated review delegation | Review-candidate launches share explicit file paths; a prior return is visible in the parent before the new launch | Same review question? Prior findings reused? Fresh pass justified? |
| Repeated checks | Identical check commands and arguments, paired with non-error tool returns within one user request | Inputs changed before every repeat? All repeats explicitly required? |
| Cache creation/replay | Reliable per-response cache token observations | Measurement only; cache invalidation and avoidable spend are not inferred. |
| Repeated full cache writes | Adjacent responses in one request/model move from a cache hit to zero cache read and a large write | Measurement only; missing usage and recorded context changes break the sequence. |

Review delegation uses a cheap role/text prefilter. JeV first checks whether the
two requests ask the same review question; only a supported positive with both
requests fully visible triggers the two follow-up questions. Implementation workers, disjoint batches, and
different review criteria must not become redundant-review claims. The compared
pair is the nearest earlier returned candidate with overlapping paths, not every
possible pair. Missing paths, unnamed artifacts, and unobserved parent returns
limit coverage.

Each repeated-work incident has an `assessment` and `assessment_reasons` in JSON
and HTML:

- `observation`: measured but not assessed, including semantic-off runs and
  candidates outside the review budget.
- `supported_opportunity`: all required evidence supports an intervention.
  Carrying prior findings forward requires the same review question, strong
  negative answers for reuse and a fresh-pass reason, and complete bounded local
  evidence. It does **not** recommend skipping the review. Polling requires at
  least three unchanged pending-progress repeats and three pending-only model rounds,
  an offered alternative, and no explicit monitoring request in the reviewed request.
- `justified_repeat` or `reuse_visible`: a repeat reason or reuse of prior findings
  is supported. Review assessments describe the compared pair; check assessments
  must cover every repeated invocation in the incident.
- `unclear`: a reviewed repeat remains unresolved. Clipping, sampling, compaction,
  API errors, missing features, and middle-band classifier answers cannot support
  an opportunity based on absent reasons.

Non-error tool returns are not proof that tests passed. Check incidents expose
the counts of explicit zero exits and unknown exits, and exclude known nonzero
exits and visible background launches. Exact command identity does not establish
unchanged dependencies or external state. Cache charges and check round-trip
durations remain observed burdens, not avoidable costs.

Polling retains both exact-output repeats and progress repeats. For the recognized
Pi running-status header only, progress matching ignores elapsed seconds while
preserving tool count, tokens, context percentage, target, and all other text.
Clipped outputs cannot use this normalization. Changing progress never becomes
an unchanged response merely because the agent is still running. Check selection
excludes formatter-only commands and Make targets that do not name a known check
or build, such as `make recipes` and `make generate`.

The HTML report splits coordination charges into spawn, resume, steer, wait,
poll, control, and mixed operations. Each response contributes once; a response
requesting several operation types stays in the mixed row. JSON keeps the split
in `runs[].tails.model_activity`. These are charges on responses requesting
coordination tools; they do not measure all orchestration thinking.

Parent cache-wait comparisons use the response issuing the wait and the immediate
response after its paired result. They expose input-side charge components and
model/duration cohorts, separated by explicit blocking, nonblocking, or unspecified
wait mode. `cache_wait_cohorts` deduplicates shared responses within each cohort;
charges across cohorts can overlap. Duration bands do not establish cache expiry.

`TAIL_CACHE_REBUILDS` flags at least two observed cache-hit-to-full-write
transitions in one request. Each write must reach 10,000 tokens by default.
The ledger shows source coordinates, response gaps, and cache-write charges.
Pi preserves response models, compactions, context edits, and empty failed
attempts so these comparisons cannot silently cross missing evidence.

The older `COORDINATION_OVERHEAD` diagnostic requires explicit reuse evidence
and is informational: missing reuse data is unknown. None of these cache or
coordination charges is treated as proven avoidable spend.

The new atomic definitions live in
[`review.toml`](src/agent_hotwash/semantic/features/review.toml) and
[`check.toml`](src/agent_hotwash/semantic/features/check.toml), alongside
[`work.toml`](src/agent_hotwash/semantic/features/work.toml). Answers retain feature
versions, question/state hashes, model, and abstention reasons. Positive and
negative support thresholds default to 0.7 and 0.3; missing or uncertain answers
do not count as negatives. Payloads use bounded excerpts and the existing
redaction/cache path.

```bash
# Local measurements only; no JeV requests.
uv run agent-hotwash analyze <traces> --semantic off --format html --full-html --out tails.html

# Set TYPESAFE_API_KEY in the environment before live review.
uv run agent-hotwash analyze <traces> --semantic live --format html --full-html --out reviewed.html

# Reuse exact cached questions and states without sending new requests.
uv run agent-hotwash analyze <traces> --semantic cached --format json --out reviewed.json
```

Override selection and alert thresholds with `--config FILE`:

```toml
[diagnostics]
cache_wait_bands_seconds = [300, 3600] # descriptive bands; finite, positive, increasing
cache_rewrite_min_write_tokens = 10000 # minimum write for wait and rebuild probes

[tails]
cache_rebuilds = 2.0         # cache-hit-to-full-write transitions within one request
delegation_repetition = 2.0   # earlier completed candidate delegations sharing targets
verification_repetition = 2.0 # repeat invocations after the first
status_probes = 10.0
inspection_rounds = 20.0
retained_per_cohort = 10      # top observations retained even below threshold
max_semantic_incidents = 12  # per trace, ranked by value / threshold
max_expense_reviews = 50
```

Zero `max_semantic_incidents` disables tail classification while preserving
measurements. Semantic budgets can leave candidates unreviewed; the ledger
reports selected/candidate counts. A cached-mode miss is reported rather than
silently sending a live request.

See [execution-tail evidence rules](docs/EXECUTION_TAIL_ANALYSIS.md),
[the repeated-work study](docs/REPEATED_WORK_ANALYSIS.md),
[cache and coordination checks](docs/CACHE_COORDINATION_ANALYSIS.md), and
[hybrid detector validation](docs/HYBRID_REPETITION_VALIDATION.md) for sampling,
counterexamples, measured results, and coverage limits.
