# Changelog

Semver. Each release gets a short, user-facing note: what changed for someone *using* the platform (operators, API consumers, deployers), not internal refactors. Keep entries minimal — one line where possible, grouped under `Added` / `Changed` / `Fixed` / `Removed` only when needed.

## [0.4.0]

### Added

- Execution analysis highlights slow tools, repeated retries and reviews, status polling, open delegations, and cache rebuilds with links to trace evidence.
- Reviews of the most expensive runs compare observed work and verification with the requested scope.

### Changed

- Saved HTML has a ranked front page and paged, linked run evidence; `--full-html` retains the legacy single-file view. Reports include an action queue and classifier coverage.
- JSON schema 6 adds handover, cache, and execution details; CSV includes handover and failure fields.

### Fixed

- Improved child-session linkage and cache/coordination cost accounting. Installed packages include their default configuration. Live review stops on authentication or billing errors.

## [0.3.0]

### Added

- Improved the native Pi adapter with parent/child session rollups and estimated usage for missing subagent transcripts.

### Changed

- Increased the default live JeV throughput to 20 requests/second, with a 20-request burst and 12 concurrent requests.

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
