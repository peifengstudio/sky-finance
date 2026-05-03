# Agent Instructions — sky-finance

## Rules

1. **All project documentation must be written in English.** This applies to code comments, commit messages, config files, and all markdown documents.
2. Never commit secrets (API keys, tokens, passwords). Use `.env` and `.env.example`.
3. Python source lives under `src/sky_finance/`. Use absolute imports (`from sky_finance.ingestion import ...`).
4. Before modifying any file, read it first.
5. Do not implement code unless explicitly asked. Prefer editing existing files over creating new ones.
6. Keep Docker and application config in their respective directories (`docker/`, `config/`).
7. Each stock is configured via its own file at `config/stocks/<TICKER>.toml`. The filename **is** the ticker symbol (e.g., `AAPL.toml`, `7203.T.toml`). Never add tickers to a shared list; always create a new per-stock file instead. Private overrides (real buy prices, share counts, personal notes) go in `config/stocks/local/<TICKER>.toml` — this directory is gitignored. Local files are deep-merged on top of shared files at load time; only specify the keys you want to override. Always load configs via `sky_finance.config.load_stock_config` / `list_stock_configs` — never read TOML files directly in application code.
8. When adding a dependency, update `pyproject.toml` and document why in the PR/commit message.
9. **All schema changes must go through Alembic.** Never edit tables by hand or via raw SQL files. Workflow: `uv run alembic revision -m "description"` → edit the generated file in `alembic/versions/` → `uv run alembic upgrade head`. The `docker/postgres/init.sql` file is reserved for Docker-only bootstrap (currently just `CREATE EXTENSION vector`) and must not contain table definitions.
10. **Keep README in sync with the actual run commands.** Any time a dev or prod command changes — new process, renamed script, different flags, tool swap — update the "Running the Application" section of `README.md` in the same commit.
11. **Every process must write structured logs.** Rules:
    - All modules use `logging.getLogger(__name__)`. No `print()` in production code.
    - Call `setup_logging()` from `sky_finance.logging_config` at every process entry point (Celery worker/beat, web server, CLI scripts).
    - Log files go in `logs/` at the project root (gitignored). Rotating 10 MB × 5 files.
    - Follow log-level semantics: `DEBUG` internal state, `INFO` normal operations, `WARNING` retryable / degraded, `ERROR` unrecoverable failures.
    - Never silence the `sky_finance.*` logger hierarchy.

## Project Overview

**sky-finance** is a local-first financial intelligence platform that:
- Fetches market data (yfinance) and news (Google RSS)
- Cleans data via a Python pipeline assisted by a local 4–7B LLM
- Stores embeddings in PostgreSQL + pgvector (Docker)
- Exposes a web dashboard for stock watchlist management (US + Japan equities)
- Runs RAG-powered strategy analysis using multi-provider LLMs (OpenAI, Anthropic, Ollama)
- Delivers results via Slack notifications
- Orchestrates everything with a built-in scheduler

## Architecture

```
src/sky_finance/
├── ingestion/          # yfinance + Google RSS fetchers
├── pipeline/           # cleaning + local LLM summarisation + embedding
├── storage/            # pgvector read/write
├── dashboard/          # FastAPI + Jinja2 + HTMX web UI
├── strategies/         # RAG retrieval + multi-provider model analysis
├── evaluation/         # LLM-as-a-judge RAG evaluation (sky-eval CLI)
├── notifications/      # Slack delivery
└── scheduler/          # Celery app + beat schedule
```

## Environment

- Runtime managed by **mise** (`mise.toml`)
- Python **3.14**
- PostgreSQL + pgvector via **Docker Compose** (`docker/docker-compose.yml`)
- Secrets loaded from `.env` (never committed)
- Config loaded from `config/settings.toml` (global) and `config/stocks/<TICKER>.toml` (per-stock)

## Development Setup

```bash
mise install                              # install Python 3.14 + uv
make setup                                # install dependencies, start infra, apply migrations
cp .env.example .env                      # fill in secrets
make models                               # pull local Ollama models
make dev                                  # start worker + beat + flower + web
```

## Key External Services

| Service | Purpose | Config key |
|---------|---------|-----------|
| OpenAI / Anthropic API | Advanced financial analysis | `OPENAI_API_KEY`, `ANTHROPIC_API_KEY` |
| Slack | Notifications | `SLACK_BOT_TOKEN`, `SLACK_CHANNEL` |
| Local LLM (Ollama) | Data cleaning / summarisation | `OLLAMA_BASE_URL` |
| PostgreSQL + pgvector | Vector + relational storage | `DATABASE_URL` |
