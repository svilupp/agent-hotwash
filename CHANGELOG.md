# Changelog

Semver. Each release gets a short, user-facing note: what changed for someone *using* the platform (operators, API consumers, deployers), not internal refactors. Keep entries minimal — one line where possible, grouped under `Added` / `Changed` / `Fixed` / `Removed` only when needed.

## [Unreleased]

### Added

- Codex decoder v2 (`item_completed`-first for CLI 0.150–0.155), canonical turns/ops/capabilities, and `agent-hotwash threads`.
- Optional JeV semantic layer (`--semantic off|cached|live`), feature bank, `label` / `eval` CLIs.
- Cost-view diagnostics (invoice / origin-attributed / counterfactual) and monthly root-task rollup.
- Dated Standard-tier prices for `gpt-5.6-luna`, `gpt-5.6-sol`, and `gpt-6-astra` (`as_of` 2026-09-19).
- Architecture notes in `docs/DESIGN.md`.

- `--jobs N` parallel analysis with one shared JeV rate budget across workers; Codex thread trees are linked across day directories (spawn / fork / `create_thread` delegation).

### Changed

- Semantic JSON sections (`structure`, `features`, `capabilities`, `cost_views`, `monthly`) appear only when `--semantic` is not `off`; CSV columns are unchanged in off.
- JSON `meta.schema_version` is now `2`: every run carries `analysis.provenance` (format, harness version, files, linkage, decoder notes) and per-session `tokens_by_model`; `structure.episodes[].invoice_cost` was removed (dollar figures live only on tagged cost views).
- `--semantic` defaults to `[semantic] mode` from config; `--allow-unredacted` only lifts the live-mode refusal when `redact = false` is configured (it no longer disables redaction itself).
- JeV digest schema is now `3`: episode `messages` are `{kind, text, phase?}` objects (last 6 kept); `episode.counts` pre-counts ops (`majority_family`, `test_after_edit`, `n_final_answer`, …); questions send `{question, inspect, focus}` with backticked JSON paths (`task.request`, `episode.counts.by_family`, `episode.messages[].text`, `messages[0].text`). Activity/purpose/outcome inspect histograms and messages, not `majority_family` or `env_impediment`.

### Fixed

- `env_impediment` is taken only from a failed op or `error_text`, not incidental wording in successful output or assistant text.

## [0.1.0] — 2026-07-05

### Added
- Initial Release
