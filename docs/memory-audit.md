# Skill and memory audit

Memory analysis finds user corrections, repeated discoveries, and repeated reading in coding-agent traces. It produces evidence-linked proposals for instructions, skills, documentation, navigation pointers, or tooling.

## Run a memory audit

```bash
uv run agent-hotwash analyze /path/to/sessions --analysis memory \
  --guidance /path/to/CLAUDE.md \
  --guidance /path/to/skills/example/SKILL.md \
  --format html --out reports/memory.html
```

Open `reports/memory.html` to review proposals, citations, guidance assessments,
and file-read analytics. This mode runs only the memory audit.

To create both reports:

```bash
uv run agent-hotwash analyze /path/to/sessions --analysis all \
  --guidance /path/to/CLAUDE.md --format html --out reports/review.html
```

This writes diagnostics to `reports/review.html` and the separate memory audit to
`reports/review.memory.html`. Memory proposals have their own report and are not
included in the diagnostics report's action queue or theme ranking.

## Execution modes

| `--semantic` | Behavior |
| --- | --- |
| `off` | Prepare evidence, file-read analytics, and command exports without model calls |
| `cached` | Reuse saved responses; a cache miss exits 1 |
| `live` (default) | Permit JeV screening, OpenAI synthesis, and guidance comparison |

Live audits need `TYPESAFE_API_KEY` and `OPENAI_API_KEY`. Export them or put them
in the current directory's ignored `.env`; exported values take precedence.
Repeat a live command with `--semantic cached` to replay it.

## Inputs and output

`PATH...`, `--config`, and session selection filters use the existing `analyze`
conventions. Repeat `--guidance FILE` to compare drafts against explicit local
files. Guidance requires `--analysis memory` or `all`; omitting guidance leaves
coverage unchecked. Model and extraction limits use `[memory]` configuration.

| Mode | Output |
| --- | --- |
| `all --out review.html` | Diagnostics report plus `review.memory.html` beside it. |
| `all --out reports/` with an existing directory | Diagnostics report plus `reports/memory.html`. |
| `all` without `--out` | Diagnostics on stdout, memory HTML in the local audit cache, and its path on stderr. |
| `memory --format html --out memory.html` | Only the memory HTML report. |
| `memory --format json` without `--out` | Only memory JSON on stdout. |
| `memory --format table` | Human-readable memory Markdown. |
| `memory --out reports/` with an existing directory | `memory.html`, `memory.json`, or `memory.md`, according to `--format`. |

Memory mode supports `html`, `json`, and `table`. It rejects `csv` and
`brief-json`. The default follows `analyze`: `table` on a terminal and `json` when
piped. With `--analysis all`, the separate memory report is always HTML,
regardless of the diagnostics report's format.

The audit stores resumable artifacts under
`.cache/agent-hotwash/memory/<corpus-and-filters-hash>/`. Repeating the same input
selection reuses unchanged stages. Changing selected guidance reuses unchanged
screening and proposals. Pipeline stages are internal library operations; the
public workflow uses `analyze --analysis memory` or `all`.

## Configuration

Pass `--config FILE` to override `[memory]` settings. Use `config-show` for all
resolved defaults. `semantic.max_concurrency` controls screening and synthesis
workers; set it to `1` for serial execution.

```toml
[memory]
max_candidates = 64
retain_threshold = 0.5
```

Request budgets, guidance limits, chunk overlap, and comparison thresholds are
validated before use. The candidate limit caps proposals per provider response;
it does not measure corpus coverage.

## Evidence and proposals

The audit retains visible assistant text and reasoning. It records unavailable encrypted reasoning rather than inferring its contents. Tool results longer than 2,000 characters retain their first 1,000 and last 1,000 characters with a truncation marker. Credentials detected by the audit redactor are removed before evidence is persisted or sent.

Shared source adapters perform native decoding and read classification. The audit projects canonical evidence, deduplicates occurrences, sizes requests, and groups inputs. JeV screens many feature questions in each bounded request. OpenAI groups retained evidence into themes and proposes changes. Repeated-discovery and repeated-reading proposals require evidence from independent families. Parent/child observations belong to one family. Ambiguous lineage cannot establish recurrence. User corrections may support a task-scoped proposal on their own.

Guidance comparison reads only explicitly selected files. Its coverage and conflict judgments remain advisory. A missing match does not establish novelty across every installed skill or repository instruction. Applicability and evidence citations remain available for human review. The command never installs generated guidance.

## Large corpora

Screening covers every eligible record with bounded requests and a shared rate
limiter. Concurrent identical cache misses share one request. Synthesis batches
and merge groups use the same concurrency limit. Transient errors retry under
`semantic.max_retries`, honoring `Retry-After`.

When drafts exceed the merge budget, hierarchical consolidation retains citation
metadata and independent-family identities. The final pass validates citations
and recurrence against original records. `synthesis_evidence.consolidation`
records compaction; `prepared.json` retains the evidence.

Extraction blocks if a candidate cannot fit, merging stops shrinking, or 32
intermediate levels are exhausted. Completed calls remain cached; repeat the
same command to resume.

## Artifacts

| File | Contents |
| --- | --- |
| `report.md` | Proposals, assessments, citations, and frequently read files. |
| `manifest.json` | Current run identity, effective policy, stage states, hashes, and cache/provider usage. |
| `prepared.json` | Redacted evidence, source coordinates, truncation metadata, and read analytics. |
| `memory.json` | Execution status, proposals, assessments, and provider-call metadata. |
| `features.json` | Screening probabilities and retention decisions after extraction. |
| `guidance.json` | Snapshots and hashes of explicitly selected guidance. |
| `calls/` | Request-keyed provider responses for reproducible replay. |

File analytics count canonical read artifacts. Counts indicate navigation patterns, not automatically missing memory. Relative path labels such as `CLAUDE.md` or `AGENTS.md` may combine different project files; absolute paths identify files more precisely. Source snapshots and repeated events are deduplicated where identity is available. Unknown source coordinates remain unavailable rather than becoming inferred line citations.

Artifacts use private file permissions and still contain session content. JSON and Markdown are local review outputs. Model judgments are exploratory and require review before changing durable guidance.

## Command datasets and hooks

Every memory run, including `--semantic off`, exports recognized submitted
shell statements after redaction. `off` prepares deterministic analytics and
datasets; it does not screen evidence or generate memory proposals. Native Claude, Codex, Pi, and code-bench sources
share the canonical evidence adapter. Literal shell calls inside Codex
`functions.exec` JavaScript are extracted without executing the JavaScript.
Dynamic JavaScript arguments, shell substitutions, and heredocs produce
diagnostics where detectable. Each row retains the submitted `shell_command`
payload alongside its extracted `command` statement. The extractor does not
execute commands or prove runtime execution. Conditional branches may never
run. Counts do not establish command success, file consumption, or usefulness.
The full tool-call denominator remains in `prepared.json`.

| Artifact | Contents |
| --- | --- |
| `command-hooks.json` | All calls, hook matches, summaries, and parsing diagnostics. |
| `command-calls.jsonl`, `command-calls.csv` | Complete recognized command dataset with evidence IDs and source references. |
| `command-matches.jsonl`, `command-matches.csv` | Matched hook names, extracted values, and named captures linked by call ID. |

The manifest records SHA-256 hashes for these private exports. Saved-audit
loading verifies them. JSONL preserves nested fields; CSV stores nested fields
as JSON cells. Reports show hook counts and the top 20 values. The full dataset
has no report display limit. Hook summaries are local analytics and do not add
extra text to model requests.

Defaults count `agent-playbooks read` / `load` / `show` playbook names,
submitted `mem index` statements, and `mem search` / `deepsearch` queries. Playbook names separated by commas
or supplied as separate arguments produce individual values. Search queries
are observed text; deciding themes requires review.

Define custom hooks in the existing TOML configuration. An explicit
`memory.command_hooks` list replaces the default hooks.

```toml
[[memory.command_hooks]]
name = "selected_playbooks"
patterns = ['agent-playbooks\s+(?:read|load|show)\s']
extract_patterns = ['(?P<playbook>poteto-mode|supervisor-mode)']
split_values = ","
```

`patterns` are alternative regexes that select a command. Every matching
`extract_patterns` regex contributes extracted values. Named captures remain
in the match dataset. Each named capture, or each positional capture when no
named captures exist, contributes a value. With no capture groups, the full
regex match contributes a value. Overlapping extraction patterns can count a
value more than once.
Use a TOML array for several regexes; commas inside a regex remain literal.
`split_values` splits captured values when a capture contains a comma-separated
list. Invalid regexes and duplicate hook names fail configuration validation.
Set `command_hooks = []` under `[memory]` to disable hooks while keeping the
command dataset.


Select custom hooks with `--config /path/to/hooks.toml`. Add `--since` and
`--until` to bound the corpus. Preserve the audit manifest when comparing runs.

## Evaluation

`python -m agent_hotwash.memory.evaluation --help` describes scoring and mapping
externally supplied labels. Repository fixtures are synthetic evaluator
regressions, not evidence of classifier accuracy.
