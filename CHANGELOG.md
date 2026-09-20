# Changelog

Semver. Each release gets a short, user-facing note: what changed for someone *using* the platform (operators, API consumers, deployers), not internal refactors. Keep entries minimal — one line where possible, grouped under `Added` / `Changed` / `Fixed` / `Removed` only when needed.

## [Unreleased]

## [0.2.0] — 2026-09-20

Optional JeV semantic layer (`--semantic off|cached|live`), Codex thread trees, cost views, and `label` / `eval`.

### Added

- `threads`, `--jobs N`, and dated prices for current Claude models plus `gpt-5.6-luna`, `gpt-5.6-sol`, and `gpt-6-astra`.
- `eval` overall agreement, a confident band, and Choice confusion.

### Changed

- `analyze` defaults to live JeV and exits 1 without `TYPESAFE_API_KEY`.
- Semantic JSON sections appear only when `--semantic` is not `off`; `meta.schema_version` is `2`.
- `systemoneprompts` is resolved from PyPI; JeV cache is `~/.cache/agent-hotwash/systemone`.
- Purpose and outcome are judged against `episode.instruction`.

### Fixed

- `env_impediment` is taken only from a failed op or `error_text`.
- Typesafe client forwards only `TYPESAFE_*` environment variables.

## [0.1.0] — 2026-07-05

### Added
- Initial Release
