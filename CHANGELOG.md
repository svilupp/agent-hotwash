# Changelog

Semver. Each release gets a short, user-facing note: what changed for someone *using* the platform (operators, API consumers, deployers), not internal refactors. Keep entries minimal — one line where possible, grouped under `Added` / `Changed` / `Fixed` / `Removed` only when needed.

## [Unreleased]

## [0.2.0] — 2026-09-20

Optional JeV semantic layer (`--semantic off|cached|live`), Codex thread trees, cost views, and `label` / `eval`.

### Added

- `agent-hotwash threads`, `--jobs N`, and dated Standard-tier prices for `gpt-5.6-luna`, `gpt-5.6-sol`, and `gpt-6-astra`.
- `eval` overall agreement plus a confident band (confidence outside 0.3–0.7) and Choice confusion.

### Changed

- Semantic JSON sections appear only when `--semantic` is not `off`; CSV columns stay the same in off. JSON `meta.schema_version` is `2`.
- JeV digest schema `3`, native `systemoneprompts` client, cache under `~/.cache/agent-hotwash/systemone`.
- Purpose and outcome are judged against `episode.instruction` (the latest user step), not the root request. Live JeV, `label` drafts, and protocol packets share the same inspect-projected state.

### Fixed

- `env_impediment` is taken only from a failed op or `error_text`.
- Typesafe client forwards only `TYPESAFE_*` environment variables.

## [0.1.0] — 2026-07-05

### Added
- Initial Release
