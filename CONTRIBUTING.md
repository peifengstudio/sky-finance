# Contributing to sky-finance

Thank you for your interest in contributing! This guide covers everything you need to get started.

## Table of Contents

- [Development Setup](#development-setup)
- [Project Structure](#project-structure)
- [Making Changes](#making-changes)
- [Code Standards](#code-standards)
- [Testing](#testing)
- [Adding a Stock](#adding-a-stock)
- [Database Schema Changes](#database-schema-changes)
- [Submitting a Pull Request](#submitting-a-pull-request)

---

## Development Setup

**Prerequisites:** [mise](https://mise.jdx.dev/), Docker

```bash
# 1. Clone and enter the repo
git clone https://github.com/<your-fork>/sky-finance.git
cd sky-finance

# 2. Install Python 3.14 + uv
mise install

# 3. Create the virtualenv and install all dependencies (including dev tools)
uv sync --extra dev

# 4. Copy the env template and fill in your secrets
cp .env.example .env

# 5. Start PostgreSQL + Redis
docker compose -f docker/docker-compose.yml up -d

# 6. Run database migrations
uv run alembic upgrade head

# 7. Start all processes (worker, beat, flower, web)
uv run honcho start
```

The dashboard is at `http://localhost:8000`. Flower (Celery monitoring) is at `http://localhost:5555`.

---

## Project Structure

```
src/sky_finance/
├── ingestion/      # yfinance + Google RSS fetchers
├── pipeline/       # LLM cleaning, summarisation, embedding
├── storage/        # PostgreSQL + pgvector read/write
├── strategies/     # RAG query + multi-provider LLM analysis
├── notifications/  # Slack delivery
├── scheduler/      # Celery app, beat schedule, queue routing
└── dashboard/      # FastAPI + Jinja2 web UI
```

Use absolute imports everywhere: `from sky_finance.ingestion import ...`

---

## Making Changes

1. **Fork** the repository and create a feature branch from `main`.
2. **Read every file you intend to modify** before making changes.
3. **Do not add features or refactor** beyond the scope of your change.
4. **Prefer editing existing files** over creating new ones.
5. Keep all documentation and comments in **English**.

---

## Code Standards

We enforce formatting and lint on every PR. Run these locally before pushing:

```bash
# Format
uv run ruff format src tests

# Lint
uv run ruff check src tests

# Type check
uv run mypy src
```

**Style rules (enforced by ruff):**
- `E`, `F` — pyflakes / pycodestyle basics
- `I` — import sorting (isort-compatible)
- `UP` — pyupgrade (modern Python syntax)

**Never use `print()` in production code.** Every module must use `logging.getLogger(__name__)` and call `setup_logging()` from `sky_finance.logging_config` at each process entry point.

---

## Testing

```bash
# Run the full test suite with coverage
uv run pytest

# Run a single file
uv run pytest tests/test_pipeline_cleaner.py -v

# Run without coverage (faster)
uv run pytest --no-cov
```

- External services (Ollama, OpenAI, Slack, DB) must be **mocked** in unit tests.
- Tests live in `tests/` and mirror the module they cover (e.g. `test_pipeline_cleaner.py` → `pipeline/cleaner.py`).
- All new features require tests. Bug fixes should include a regression test.

---

## Adding a Stock

Never add tickers to a shared list. Each stock gets its own config file:

```bash
# Create config/stocks/TSLA.toml (filename = ticker symbol)
```

```toml
ticker   = "TSLA"
name     = "Tesla, Inc."
market   = "us"
currency = "USD"
enabled  = true

[ingestion]
l1_keywords = ["Tesla earnings", "Tesla deliveries"]
l2_topics   = ["EV market", "autonomous driving"]
l3_macro    = ["Fed rate cut", "energy policy"]
```

Private overrides (buy price, share count, personal notes) go in `config/stocks/local/TSLA.toml`, which is gitignored and deep-merged at load time.

---

## Database Schema Changes

**All schema changes must go through Alembic.** Never edit tables by hand.

```bash
# 1. Generate a new migration
uv run alembic revision -m "add_foo_column_to_bar"

# 2. Edit the generated file in alembic/versions/
# 3. Apply the migration
uv run alembic upgrade head
```

Do not add table definitions to `docker/postgres/init.sql` — that file only bootstraps the pgvector extension.

---

## Submitting a Pull Request

1. Ensure `ruff`, `mypy`, and `pytest` all pass locally.
2. Update `README.md` if any run command changed.
3. If you changed the database schema, include the Alembic migration.
4. Open a PR against `main` and fill in the pull request template.
5. Keep PRs focused — one logical change per PR.

PRs are reviewed for correctness, test coverage, and adherence to the code standards above. Thank you for contributing!
