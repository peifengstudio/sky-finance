# sky-finance Makefile
# Wraps the most common dev workflows so you don't have to memorise the full
# uv / docker compose / alembic command syntax.
#
# All commands assume you have already run `mise install` to get Python + uv.
# Run `make help` to see a short description of every target.

.DEFAULT_GOAL := help
.PHONY: help install models up down migrate seed-strategies dev worker beat flower web \
        test lint fmt check ci clean validate-stocks

# ── colour helpers ─────────────────────────────────────────────────────────────
BOLD  := $(shell tput bold 2>/dev/null)
RESET := $(shell tput sgr0 2>/dev/null)

# ── infrastructure ─────────────────────────────────────────────────────────────

up:			## Start PostgreSQL + Redis (detached)
	docker compose -f docker/docker-compose.yml up -d

down:			## Stop and remove infrastructure containers
	docker compose -f docker/docker-compose.yml down

# ── first-time setup ───────────────────────────────────────────────────────────

install:		## Install all Python dependencies (including dev extras)
	uv sync --extra dev

models:			## Pull required Ollama models
	ollama pull qwen2.5:3b-instruct
	ollama pull qwen2.5:14b-instruct
	ollama pull nomic-embed-text

migrate:		## Apply all pending Alembic migrations
	uv run alembic upgrade head

seed-strategies:	## Upsert default strategies from config/strategies/*.toml into the DB
	uv run python -m sky_finance.strategies.seed

setup: install up migrate seed-strategies	## Full first-time setup (install + infra + migrate + seed)
	@echo ""
	@echo "$(BOLD)Setup complete.$(RESET)"
	@echo "  1. Copy and fill in secrets:  cp .env.example .env"
	@echo "  2. Pull local models:         make models"
	@echo "  3. Start the app:             make dev"

# ── application processes ──────────────────────────────────────────────────────

dev:			## Start all four processes via honcho (worker + beat + flower + web)
	uv run honcho start

worker:			## Start Celery worker only
	uv run honcho start worker

beat:			## Start Celery beat scheduler only
	uv run honcho start beat

flower:			## Start Flower monitoring UI only  →  http://localhost:5555
	uv run honcho start flower

web:			## Start FastAPI dashboard only  →  http://localhost:8000
	uv run honcho start web

# ── database ───────────────────────────────────────────────────────────────────

migration:		## Create a new migration  (usage: make migration msg="add foo column")
	uv run alembic revision -m "$(msg)"

db-current:		## Show the currently applied Alembic revision
	uv run alembic current

db-history:		## Show the full Alembic migration history
	uv run alembic history

db-rollback:		## Roll back one migration
	uv run alembic downgrade -1

# ── quality ────────────────────────────────────────────────────────────────────

test:			## Run the full test suite with coverage
	uv run pytest

test-fast:		## Run tests without coverage (faster feedback loop)
	uv run pytest --no-cov

lint:			## Run ruff (lint) + mypy (type-check)
	uv run ruff check src tests
	uv run mypy src

fmt:			## Auto-format with ruff format + ruff --fix
	uv run ruff format src tests
	uv run ruff check --fix src tests

validate-stocks:	## Validate all stock TOML configs and print a structured report
	uv run sky-validate-stocks

check: fmt lint test	## Format, lint, and test in one shot (pre-PR check)

ci: lint test		## Lint + test without auto-formatting (for CI pipelines)

# ── housekeeping ───────────────────────────────────────────────────────────────

clean:			## Remove build artefacts, caches, and coverage reports
	rm -rf htmlcov .coverage .pytest_cache .ruff_cache .mypy_cache
	find . -type d -name __pycache__ -exec rm -rf {} +

# ── help ───────────────────────────────────────────────────────────────────────

help:			## Show this help message
	@echo "$(BOLD)sky-finance$(RESET) — available targets:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*##' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*##"}; {printf "  $(BOLD)%-16s$(RESET) %s\n", $$1, $$2}'
	@echo ""
