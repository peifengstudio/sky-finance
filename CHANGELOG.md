# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] — 2026-05-16

### Added

- **`/health` Ollama model check** — the health endpoint now parses the `/api/tags`
  response and verifies that all required models (`qwen2.5:3b-instruct`,
  `nomic-embed-text`) are present on the local Ollama instance. The `ollama` check
  now returns `missing_models: [...]` in its payload; the status is `"degraded"` (HTTP
  503) when any model is absent, making onboarding failures immediately actionable
  instead of silently passing. Only models relevant to the active configuration are
  checked — switching `EMBEDDING_BACKEND=openai` removes `nomic-embed-text` from
  the required set.

- **Unit tests for previously untested modules** — new test files cover
  `ingestion/yfinance_fetcher`, `notifications/slack`, `evaluation/judge`,
  `evaluation/retrieval`, and `strategies/seed`, bringing overall line coverage from
  66 % to 81 % and resolving the failing `make test` coverage gate.

### Fixed

- **Python 3 syntax in `yfinance_fetcher.py`** — `except TypeError, ValueError:`
  (Python 2 syntax) corrected to `except (TypeError, ValueError):`, which prevented
  the module from importing on Python 3.14.

### Changed

- **Coverage omit list narrowed** — `notifications/slack.py`, `evaluation/judge.py`,
  and `evaluation/retrieval.py` are now included in coverage measurement. Only
  infrastructure entry points that require live services (Celery tasks, DB
  migrations, CLI runners) remain excluded.

- **Formatting toolchain migrated from black to ruff** — `ruff format` replaces
  `black` for a single-tool lint-and-format workflow. No code style changes; the
  migration is purely tooling.
