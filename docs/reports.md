# Reports

Use diagnostics to identify actions worth reviewing, then inspect their source
evidence. Scores rank items within the current report; they are not estimates
of savings or calibrated risk.

## Output formats

| Format | Contents |
| --- | --- |
| `table` | Terminal summary; plain text outside a terminal |
| `json` | Full per-run records, aggregates, priorities, and ranked themes |
| `brief-json` | Ranked themes and summary metrics for `render-brief` |
| `csv` | One row per trace with fixed columns across semantic modes |
| `html` | Compact front page with linked run evidence when saved with `--out` |

Saved HTML includes a sibling evidence directory, with a run index paged at 100
rows and one page per run. Keep both together. HTML on stdout and `render-brief`
produce compact single pages. `--full-html` produces the larger single-file
view. Memory reports have a separate [output contract](memory-audit.md).

## Actions and expensive runs

Start with **Actions to examine** in the compact HTML or **Actions to review** in the
terminal. Each item has measured evidence, a next step, and a verification
criterion. Supported changes appear before investigations and visibility
fixes. Action themes score affected runs, observed charge relative to this
report's largest action charge, actionability, and ease of testing. Detector
themes score affected runs and maximum severity. The two 0-100 scales are
separate reading aids, not savings estimates or calibrated risk. `brief-json` retains each score component; full JSON also
retains the action queue in `priorities` and grouped ranking in `ranked_themes`.
CSV retains its per-trace layout.

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
Threshold crossings emit `TAIL_*` observation findings. An unfinished child
is a censored duration, not a proven hang; overlapping durations are never
added into a savings estimate.

The full HTML report includes the most expensive 5% within each cost basis (including
cutoff ties), a model work-mix table with input/output/cache charge components,
and an execution ledger with source coordinates, suggested actions, and JeV
evidence. Parent and child files are assembled into workflows before ranking.
Expense review checks requested scope and returned verification; an appropriate
result does not establish that the run used the cheapest approach.

## Evidence and judgments

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

The atomic definitions live in
[`review.toml`](https://github.com/svilupp/agent-hotwash/blob/main/src/agent_hotwash/semantic/features/review.toml) and
[`check.toml`](https://github.com/svilupp/agent-hotwash/blob/main/src/agent_hotwash/semantic/features/check.toml), alongside
[`work.toml`](https://github.com/svilupp/agent-hotwash/blob/main/src/agent_hotwash/semantic/features/work.toml). Answers retain feature
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


## Offline report enrichment and evaluation

Use `scripts/enrich_brief_overview.py` to add spend and failure summaries to an
existing brief without loading a large full report into memory:

```bash
uv run --with ijson python scripts/enrich_brief_overview.py full.json brief.json enriched.json
```

The evaluation CLIs accept user-supplied reports or labeled cases:

```bash
uv run python scripts/eval_real_priority_cases.py --help
uv run python scripts/eval_jev_review_v2.py --help
uv run python scripts/prototype_priority_ranking.py --help
```

These experimental comparisons make live provider calls when run on inputs;
keep datasets and outputs local. They do not establish production accuracy.
