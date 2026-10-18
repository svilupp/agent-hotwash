# Design

`agent-hotwash` turns coding-agent traces into a shared
representation of work, then analytics, detectors, and (optionally)
JeV-labelled diagnoses. Nothing
under `structure/`, `semantic/`, or `diagnostics/` imports a harness format.

This document describes the architecture as implemented. Usage and report interpretation are linked from the [documentation index](index.md). Diagnoses remain experimental until a
labelled evaluation supports their use as production findings.

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

- `Turn` - lifecycle (`open|completed|aborted`), user-input kind
  (`user|delegation|injected|none`), model config revisions, model calls.
- `ModelCall` - one `response_id`, bracketed by its terminating usage record
  (unterminated trailing call: `response_id=None`).
- `Op` / `ArtifactInteraction` - canonical `op_kind` (`cmd.read`, `file.edit`,
  `mcp.<server>.<tool>`, ...), `tool_category`, classifications, artifacts.
- `Usage` - **normative**: `input` is uncached (`input_tokens − cached`);
  `cache_read` / `cache_write` / `output` billed separately; `reasoning_output`
  is an informational **subset** of output and is never added into cost.
  Billing uses per-response `usage` deduplicated by `(thread_id, response_id)`.
- `Capabilities` - `declared` (what the decoder can expose for this
  format/version) vs `observed` (what this session carried). Tree aggregation
  uses the lattice `false < partial < true` with min.
- `ThreadLink` / `SourceRef` - spawn/fork/created evidence, source coordinates.

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
  `started_at_ms` is minutes earlier - when compaction began - so it is not
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
spawn / fork / created evidence - `agent_created_thread` children name their
parent only inside the `<codex_delegation>` echoed by the `create_thread`
output; `send_message_to_thread` outputs carry the *sender's* id and are
never parentage - and loaded lazily per component. Children whose parent is
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
an MCP `create_thread` *result*, or - on 0.150 - first mentioned by an
`agent.activity` meta event. Children whose id never appears in the parent
file stay unbound.

Episodes are **atomic and immutable** at model-call boundaries (spend exists
only per response). Reporting may group contiguous same-`phase.activity` runs
for display; atoms, their spend, and feature provenance never change.

JeV sees a **capped digest** (schema v3): `task` plus `episode` with
structured `messages[]` (`kind`, `text`, optional `phase`), `ops[]`
(`kind`, `cmd`, `paths`, `exit`, ...), `episode.instruction` (operative user
step), and pre-counted `episode.counts`. Prompts point at exact keys such as
`` `episode.instruction` ``, `` `episode.counts.by_family` ``,
`` `episode.messages[].text` ``, and `` `task.request` ``. The wire payload
is `project_for_questions` of those inspect/compare paths - the same
projection `label` drafts and protocol packets use - not the raw digest.
Fact flags and winner counts (`majority_family`, `env_impediment`,
`artifact_change`) stay off the wire unless a question names them. Raw tool
output is never sent. Redaction runs after digest, before project. Counting
stays in code; precomputed winners are not the JeV label.

## Semantic layer (JeV / System One)

The feature bank is native System One definitions
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
| counterfactual | modelled alternative; currently uses `CONTINUATION_BURDEN` |

MECE waste partition (one primary cause per span): `DUPLICATE_WORK`,
`INEFFECTIVE_ITERATION`, `IRRELEVANT_WORK`, `POST_COMPLETION_WORK`,
`CONTINUATION_BURDEN`, `EXCESS_REASONING_TIER`, `COORDINATION_OVERHEAD`,
`EXTERNAL_BLOCK`. Cross-cutting indicators (`LOW_YIELD_TAIL`, `RUNAWAY_SESSION`,
`CONTEXT_ROT`, `PHASE_SPEND`) never own dollars.

Tree rollup = root invoice + descendants' incremental invoice, with a
per-thread breakdown. Fork carryover input is a separate informational line.

Pricing rows and their verification dates live in `config/defaults.toml`, mirrored
in the package defaults. Prefix matches remain estimated; dated exact-model
matches can support monetary diagnoses. Special billing modes are not inferred.

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

## Evaluation

`label` creates or resumes a JSONL annotation store; `eval` reports agreement and
positive rates. Split by root thread tree to prevent related sessions leaking
between development and held-out sets. A held-out evaluation writes a marker;
changing criteria after exposure requires a new held-out set.

```bash
uv run agent-hotwash label /path/to/traces --store labels.jsonl --annotator reviewer --split dev
uv run agent-hotwash eval --store labels.jsonl
```

Synthetic regression tests verify contracts, not classifier accuracy. Diagnoses
remain experimental pending human evaluation. `make test-live` exercises the
TypeSafe transport separately from the deterministic suite.

## Handover ledger (schema 4)

`primitives/handovers.py` creates one record for each observed new spawn. It
keeps provider agent identity distinct from child session identity and records
source coordinates for spawn, steering, waits, notifications, and final replies.
Pi `session_info` contains an eight-character display prefix; this is shown as
a candidate and never used as an exact join. Request and reply sizes use visible
plaintext only. Encrypted Codex handovers have unknown size and quality.

`diagnostics/handover_cache.py` records parent model-call usage on either side
of a joined wait. A cache rewrite flag is descriptive and requires comparable
usage, a zero next cache read, and a configurable minimum next cache write.
It does not attribute the transition to the child or claim counterfactual cost.

The handover JeV bank asks literal contract and return questions only when
their input text is visible. Its positive and negative thresholds are 0.8 and
0.2. Missing source evidence produces an abstention. The HTML report shows an
overview with visibility denominators and a source-linked per-spawn ledger;
JSON preserves the full record and CSV carries per-trace summary columns.

Nested delegations are separate rows with explicit depth. Shareable excerpts
are redacted and capped at 600 characters; local analysis measures the full
visible payload before clipping. The first release does not assume a provider
cache horizon because the source does not expose cache keys. The configurable
large-write threshold is descriptive, and its raw usage rows remain available
in JSON and HTML. Encrypted Codex requests need visible plaintext before
request-quality questions can run.

## Memory and skill audit

Memory analysis branches from shared source discovery and canonical events.
Adapters own raw decoding, captured output boundaries, source coordinates,
authorship, and read classification. `memory.inputs` projects that evidence
into the versioned contracts in `memory.models`. Family groups distinguish
observed repetition from independent support.

`memory.pipeline` orchestrates preparation, literal screening, proposal
synthesis, selected-guidance comparison, and rendering. Native semantic
questions reuse the bank, JeV client, cache, redaction, and rate limiter.
Synthesis has a bounded provider boundary with explicit paid-call permission.
Memory policy lives in `config.MemoryConfig` and mirrored package defaults.

Memory reports stay separate from diagnostic priorities. See [memory audits](memory-audit.md)
for CLI usage, cache artifacts, guidance comparison, and command exports.
