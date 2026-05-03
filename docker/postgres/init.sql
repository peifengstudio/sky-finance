-- Docker container initialisation — runs once on first container start.
-- Schema is managed by Alembic (alembic/versions/).
-- Run `uv run alembic upgrade head` after the container is healthy.

CREATE EXTENSION IF NOT EXISTS vector;
