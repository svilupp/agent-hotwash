# Design

`agent-hotwash` turns coding-agent traces into a **harness-independent
semantic representation of work**, then analytics, detectors, and (optionally)
JeV-labelled diagnoses. Codex is the first decoder; Pi is the second. Nothing
under `structure/`, `semantic/`, or `diagnostics/` imports a harness format.

This document describes the architecture as implemented. Work-package history
and eval gates live in `PLAN.md`. Diagnoses are **scaffolding** until the WP9
labelling eval graduates a subset; do not treat them as a production set.

## Pipeline

```text
Codex / Pi / Claude / code-bench     sources/<harness>.py
        ↓
canonical trace facts                events.py + canonical.py
        ↓
tasks + atomic episodes              structure/
        ↓
optional JeV feature bank            semantic/   (--semantic off|cached|live)
        ↓
cost-view diagnostics                diagnostics/
        ↓
reports + monthly root-task rollup   report/ + aggregate.py
```

Hard rules:

- JeV answers **literal, single, observable** judgments from a capped digest
  that already contains deterministic facts. Counting, arithmetic, dates,
  thresholds, and composition stay in code.
- Every dollar figure names its **cost view** (invoice / origin-attributed /
  counterfactual) and **pricing status** (exact / estimated / unknown).
- Monetary waste diagnoses fire only when `pricing_status=exact` (a named
  price row with `as_of`). Prefix/default lookup is estimated only.

## Canonical layer

Decoders emit `Session` / `Trace` plus:

- `Turn` — lifecycle (`open|completed|aborted`), user-input kind
  (`user|delegation|injected|none`), model config revisions, model calls.
- `ModelCall` — one `response_id`, bracketed by its terminating usage record
  (unterminated trailing call: `response_id=None`).
- `Op` / `ArtifactInteraction` — canonical `op_kind` (`cmd.read`, `file.edit`,
  `mcp.<server>.<tool>`, …), `tool_category`, classifications, artifacts.
- `Usage` — **normative**: `input` is uncached (`input_tokens − cached`);
  `cache_read` / `cache_write` / `output` billed separately; `reasoning_output`
  is an informational **subset** of output and is never added into cost.
  Billing uses per-response `usage` deduplicated by `(thread_id, response_id)`.
- `Capabilities` — `declared` (what the decoder can expose for this
  format/version) vs `observed` (what this session carried). Tree aggregation
  uses the lattice `false < partial < true` with min.
- `ThreadLink` / `SourceRef` — spawn/fork/created evidence, source coordinates.

Codex decoder is `item_completed`-first (CLI 0.150–0.155) and split in three
modules: `sources/codex_items.py` (pure item → `Event` mappers, path
resolution, inter-agent messages), `sources/codex_legacy.py` (0.142 path, also
the fallback when a newer file has no `item_completed`), and
`sources/codex_native.py` (the streaming `_decode_v2` loop: turns, model calls,
replay gating, `load_rollout`, directory indexing). Rules learnt from real
0.15x rollouts:

- Replay prefixes are gated **only** by `subagent_history_start_ordinal` (or a
  second `session_meta`) against the envelope `ordinal`. A fork's
  `history_base.end_ordinal_exclusive` indexes the *parent's* file and is
  linkage evidence only.
- A `compacted` record and its `ContextCompaction` item are one compaction
  when they are close in the stream *and* in envelope time (the item's
  `started_at_ms` is minutes earlier — when compaction began — so it is not
  used); the parent's compactions inside a replay prefix are kept as `meta`
  context.
- `SubAgentActivity` is a lifecycle notification (`meta`), and a
  `CollabAgentToolCall` item enriches the `function_call` it mirrors instead
  of duplicating it. Inter-agent `agent_message` records become the child's
  delegated `user_msg` (from the parent) or an `agent.message` `tool_result`
  (a child reporting back).
- `CommandExecution` read/search paths are resolved against the decoded
  `file://` cwd, and `parsed_cmd.type == "unknown"` compound lines are
  scanned for read/search targets, so reads and `FileChange` edits share one
  absolute key. A `FileChange` that only adds files is `file.write`, only
  deletes is `file.delete`; line counts sum over every changed file.
- Files without any `token_usage_record` (0.150-alpha) take per-response usage
  from `token_count.last_token_usage` (a lower bound; noted in provenance).
- Foreign `thread_id` usage records are dropped. Wrapper `exec` children share a
  `group_id`. Detector windows exclude `meta` wrapper events via
  `logical_events(session)`.

A Codex directory becomes **connected components** (`sources/codex_forest.py`):
the whole input tree is indexed cheaply (`index_session_meta`), linked by
spawn / fork / created evidence — `agent_created_thread` children name their
parent only inside the `<codex_delegation>` echoed by the `create_thread`
output; `send_message_to_thread` outputs carry the *sender's* id and are
never parentage — and loaded lazily per component. Children whose parent is
outside the input become roots with `provenance.notes=["parent not in input"]`;
that, a duplicated thread id, an ambiguous parent or a dropped cycle edge all
set `thread_linkage=partial` and `degraded=["thread_linkage"]`.
`agent-hotwash threads PATH` lists the graph.

Pi project dirs group the same way on `parentSession` (absolute path of the
spawner). Children whose files were never persisted (in-memory pi-subagents
before `rememberAgents`) are reconstructed as estimated subagent sessions from
`subagent-notification` `totalTokens`, joined to persisted children via
`session_info.name` (`{type}#{id-prefix}`) so spend is not double-counted.
Estimated stubs skip JeV and detectors; `pricing_status` is `estimated`.

`sources/detect.py` turns paths into picklable `WorkUnit`s; `runner.py` runs
them in a process pool (`--jobs`, largest units first) and merges results. In
live semantic mode the JeV request budget (`semantic.requests_per_second`,
`burst`, `max_concurrency`) is global: one token-bucket `RateLimiter` lives in
shared memory (`RateLimiter.shared`) and every worker draws from it, so both
the rate and the burst are true global ceilings; the client retries 429/5xx
with capped, jittered backoff honouring `Retry-After` (delta or HTTP-date). Every report row carries
`analysis.provenance` (format, harness version, files, linkage, decoder notes).

Pi declares `reasoning_effort=false`, `thread_linkage=true`, emits `model_change`
as meta, and builds turns through the generic `canonical.build_turns` heuristic.

## Structure

Task candidates are substantive user turns and delegation prompts. With
`--semantic off` there is **one task per session**. With semantic on, a
relationship question (identity Choice + three Nouls over the ledger) maps to
an edge (`continues|corrects|depends_on|sibling_same_area|unrelated`). Intra-turn
splitting is out of scope.

`semantic.pipeline.annotate_trace` segments **every session of the tree**
(root first, parents before children). A delegated child's tasks bind to the
parent atom that started its thread (`Task.parent_task` / `spawn_episode_id`)
using `Episode.spawned_thread_ids`: ids named by `agent.spawn` args, echoed by
an MCP `create_thread` *result*, or — on 0.150 — first mentioned by an
`agent.activity` meta event. Children whose id never appears in the parent
file stay unbound.

Episodes are **atomic and immutable** at model-call boundaries (spend exists
only per response). Reporting may group contiguous same-`phase.activity` runs
for display; atoms, their spend, and feature provenance never change.

JeV sees a **capped digest** (schema v3): `task` plus `episode` with
structured `messages[]` (`kind`, `text`, optional `phase`), `ops[]`
(`kind`, `cmd`, `paths`, `exit`, …), `episode.instruction` (operative user
step), and pre-counted `episode.counts`. Prompts point at exact keys such as
`` `episode.instruction` ``, `` `episode.counts.by_family` ``,
`` `episode.messages[].text` ``, and `` `task.request` ``. The wire payload
is `project_for_questions` of those inspect/compare paths — the same
projection `label` drafts and protocol packets use — not the raw digest.
Fact flags and winner counts (`majority_family`, `env_impediment`,
`artifact_change`) stay off the wire unless a question names them. Raw tool
output is never sent. Redaction runs after digest, before project. Counting
stays in code; precomputed winners are not the JeV label.

## Semantic layer (JeV / System One)

Default `--semantic live`. The CLI exits 1 before analysis if `TYPESAFE_API_KEY`
is unset (export the key, or pass `--semantic off`). `cached` reads
`~/.cache/agent-hotwash/systemone/r<redaction>` and exits 1 on a miss (the old
`~/.cache/agent-hotwash/jev` directory is not readable; warm `cached` with one
`live` run). `live` refuses unredacted sends unless `--allow-unredacted`.
Pytest, `make check`, and GitHub Actions set `AGENT_HOTWASH_SEMANTIC=off` so
the suite does not need a key; `--semantic` still wins when passed.

The milestone bank is native System One definitions
(`semantic/features/{task,episode,turn}.toml`): `[questions]` plus
`[data.features]` overlay (`version`, capability `requires`) and
`[data.routing]` for intent subtypes. `[requires]` is the digest contract;
`systemoneprompts check --strict` and `make bank-check` guard it. Features
gate on **declared** capabilities; unmet requirements become
`unknown/insufficient_observability`. Combined rules are their own
classifiers and must be validated as such.

Transport stack (outermost first): upstream per-question cache → our
cross-process rate limiter → HTTP. Cache key is `{model, state, question}`
(redaction version is a cache subdirectory). Score `FeatureValue.value` is
the expected-value float; the native answer lives on `FeatureValue.answer`.

## Diagnostics

Three cost views:

| view | meaning |
|---|---|
| invoice | billed charge of each response, assigned where the call occurred |
| origin-attributed | later cache re-billing charged to the episode that introduced the content (always estimated) |
| counterfactual | modelled alternative; milestone uses only `CONTINUATION_BURDEN` |

MECE waste partition (one primary cause per span): `DUPLICATE_WORK`,
`INEFFECTIVE_ITERATION`, `IRRELEVANT_WORK`, `POST_COMPLETION_WORK`,
`CONTINUATION_BURDEN`, `EXCESS_REASONING_TIER`, `COORDINATION_OVERHEAD`,
`EXTERNAL_BLOCK`. Cross-cutting indicators (`LOW_YIELD_TAIL`, `RUNAWAY_SESSION`,
`CONTEXT_ROT`, `PHASE_SPEND`) never own dollars.

Tree rollup = root invoice + descendants' incremental invoice, with a
per-thread breakdown. Fork carryover input is a separate informational line.

Dated Standard short-context rows (verified 2026-09-19 against
[OpenAI API pricing](https://developers.openai.com/api/docs/pricing)):

| model | input | cache_read | cache_write | output | `as_of` |
|---|---:|---:|---:|---:|---|
| `gpt-5.6-luna` | 0.20 | 0.02 | 0.25 | 1.20 | 2026-09-19 |
| `gpt-5.6-sol` | 4.00 | 0.40 | 5.00 | 20.00 | 2026-09-19 |
| `gpt-6-astra` | 10.00 | 1.00 | 12.50 | 50.00 | 2026-09-19 |

Units are USD per million tokens. `gpt-5.6-sol` is OpenAI's promotional
Standard rate (stated available at least through 2026-11-21). Long-context
(>272K input), Batch/Flex, and Fast modes are not modelled.

Dated first-party Claude API Standard rows (verified 2026-09-20 against
[Anthropic API pricing](https://platform.claude.com/docs/en/about-claude/pricing)).
`cache_write` is the 5-minute cache-write class. Fable/Mythos 5.1 cache reads
are 0.025× input; every other listed Claude model uses 0.1×. Sonnet 5's $2/$10
is the standard price (the scheduled 2026-09-01 rise to $3/$15 did not occur).

| model | input | cache_read | cache_write | output | `as_of` |
|---|---:|---:|---:|---:|---|
| `claude-fable-5-1` / `claude-mythos-5-1` | 10.00 | 0.25 | 12.50 | 50.00 | 2026-09-20 |
| `claude-fable-5` / `claude-mythos-5` | 10.00 | 1.00 | 12.50 | 50.00 | 2026-09-20 |
| `claude-opus-5` and Opus 4.5–4.8 | 5.00 | 0.50 | 6.25 | 25.00 | 2026-09-20 |
| `claude-sonnet-5` | 2.00 | 0.20 | 2.50 | 10.00 | 2026-09-20 |
| Sonnet 4 / 4.5 / 4.6 | 3.00 | 0.30 | 3.75 | 15.00 | 2026-09-20 |
| `claude-haiku-4-5` | 1.00 | 0.10 | 1.25 | 5.00 | 2026-09-20 |
| Opus 4 / 4.1 (retired) | 15.00 | 1.50 | 18.75 | 75.00 | 2026-09-20 |
| Haiku 3.5 (retired) | 0.80 | 0.08 | 1.00 | 4.00 | 2026-09-20 |

Batch (50% off), Fast mode ($10/$50 on Opus 5 and 4.8), US-only inference
(1.1×), and 1-hour cache writes are not modelled. Prefix lookup uses the
longest matching key.

## Reports

- `--semantic off`: JSON/CSV/table/HTML match the pre-semantic contract. CSV
  columns stay fixed. New JSON keys (`runs[].structure|features|capabilities|cost_views`,
  top-level `monthly`) appear **only** when semantic ≠ off.
- Table and HTML task card: Intent / Shape / Actual / Trajectory (grouped
  same-activity runs) / Diagnosis. Every money figure shows view + pricing
  status.
- Monthly rollup (semantic on): unit = root task in a thread tree; global
  `(thread_id, response_id)` dedup; three ranking columns (task count, invoice
  dollars, counterfactual range) never mixed. The overspend sentence states
  which measure it ranked by.

## Labelling (WP9 — human)

Diagnoses stay experimental until a diagnosis-level **human** eval. Tooling:

```bash
uv run agent-hotwash label PATH --store labels.jsonl --annotator you --split dev
uv run agent-hotwash eval --store labels.jsonl
```

Pilot: ~10 tasks / 30 episodes, **double-labelled** (two annotators, same
store, different `--annotator`). `eval` reports per-feature agreement and
positive rates. Then set quotas (≥15 positive, ≥15 negative, ≥5 ambiguous
real items). Split by **root thread tree**, 60/40; held-out is evaluated
**once** (`--held-out` burns a marker). Any criterion change after exposure
requires a new held-out.

Proposed first graduation set: intent super-families, `task.scope.breadth`,
`episode.phase.activity`, `episode.phase.purpose`, `episode.reasoning.demand`,
`episode.claim.*`, `episode.outcome.kind`, `turn.relationship.*`.

Gates (proposal, `PLAN.md` §9.8): Noul F1 ≥ 0.85 on real held-out with n ≥ 20
per class; Choice macro-F1 ≥ 0.80; diagnosis precision ≥ 0.85 with support ≥ 20.
Below gate → `experimental`, excluded from diagnostics.

### Protocol probes (not graduation)

Four held-out 20-task independent-vs-JeV-protocol probes (Codex 2026-09,
seeds 20260919–22, not Typesafe live JeV) tuned inspect paths. They do **not**
graduate features or diagnoses.

Stop: counts and fact flags are evidence, not labels. Inspect paths must not
name `majority_family` or incidental `env_impediment` as the answer; those
keys are also stripped from the JeV/label payload unless named. Do not add
if-then on count fields. Keep the 8-way purpose Choice. Pin task questions to
`task.request` (omit episode ops). Purpose and outcome judge
`episode.instruction`. `env_impediment` is set only from a failed op /
`error_text`, never from success or assistant wording.

Probe 3 overfit: purpose 67.5% (`*→recover` from `env_impediment`); outcome
61.1% (`*→blocked_external`); activity 100% because the question mapped
`majority_family`.

Probe 4 after dropping those inspect keys: 93.7% item agreement (n=255),
Noul ECE 0.052. Purpose 84% with recover 0. Outcome 96% with
`blocked_external` 0. Activity 100% on this draw because `by_family`
histograms were unambiguous (26 inspect / 8 run / 2 communicate), not
because the question named the winner field.

Residuals, not for more if-then: purpose **orient↔investigate**; JeV
**execute** over-call; JeV skips purpose/outcome when inspect includes empty
`task.request` (delegated / `agent_created_thread` often have no
`ledger.request`). Independent still labels those episodes; they drop from
agreement n. Human WP9 remains open. Live System One is `make test-live`
(`tests/live/`, needs `TYPESAFE_API_KEY`). `make test` / `make check` are
deterministic and do not collect that module.

## Out of scope (milestone)

Alignment, read-level and delegation-level feature families; intra-turn JeV
task splitting; sending uncapped tool output; seeding this file was delayed
until the pipeline landed. Expensive-tier questions (`advances_task`,
`would_be_cheaper_as_fresh_session`, …) stay out of the bank.
