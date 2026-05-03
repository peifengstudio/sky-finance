"""Initial schema — all tables

Revision ID: 001
Revises: —
Create Date: 2026-04-17
"""

from collections.abc import Sequence

from alembic import op

revision: str = "001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # raw_data — yfinance payloads stored as JSONB
    op.execute("""
        CREATE TABLE IF NOT EXISTS raw_data (
            id          BIGSERIAL       PRIMARY KEY,
            ticker      TEXT            NOT NULL,
            source      TEXT            NOT NULL,
            fetched_at  TIMESTAMPTZ     NOT NULL DEFAULT now(),
            payload     JSONB           NOT NULL
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS raw_data_ticker_idx     ON raw_data (ticker)")
    op.execute("CREATE INDEX IF NOT EXISTS raw_data_fetched_at_idx ON raw_data (fetched_at DESC)")

    # news_raw — Google RSS articles
    op.execute("""
        CREATE TABLE IF NOT EXISTS news_raw (
            id           BIGSERIAL   PRIMARY KEY,
            ticker       TEXT        NOT NULL,
            title        TEXT        NOT NULL,
            url          TEXT        NOT NULL UNIQUE,
            published_at TIMESTAMPTZ,
            fetched_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
            content      TEXT,
            source_name  TEXT
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS news_raw_ticker_idx       ON news_raw (ticker)")
    op.execute(
        "CREATE INDEX IF NOT EXISTS news_raw_published_at_idx ON news_raw (published_at DESC)"
    )

    # documents — cleaned + LLM-summarised records
    op.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id          BIGSERIAL   PRIMARY KEY,
            source_type TEXT        NOT NULL,
            source_id   BIGINT,
            ticker      TEXT        NOT NULL,
            title       TEXT,
            body        TEXT        NOT NULL,
            sentiment   TEXT,
            key_facts   JSONB,
            created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS documents_ticker_idx     ON documents (ticker)")
    op.execute("CREATE INDEX IF NOT EXISTS documents_created_at_idx ON documents (created_at DESC)")

    # embeddings — pgvector (768 dims = nomic-embed-text)
    op.execute("""
        CREATE TABLE IF NOT EXISTS embeddings (
            id           BIGSERIAL    PRIMARY KEY,
            document_id  BIGINT       NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
            ticker       TEXT         NOT NULL,
            source_type  TEXT         NOT NULL,
            created_at   TIMESTAMPTZ  NOT NULL DEFAULT now(),
            embedding    VECTOR(768)  NOT NULL
        )
    """)
    op.execute("""
        CREATE INDEX IF NOT EXISTS embeddings_hnsw_idx
            ON embeddings
            USING hnsw (embedding vector_cosine_ops)
            WITH (m = 16, ef_construction = 64)
    """)
    op.execute("CREATE INDEX IF NOT EXISTS embeddings_ticker_idx ON embeddings (ticker)")

    # strategy_results
    op.execute("""
        CREATE TABLE IF NOT EXISTS strategy_results (
            id        BIGSERIAL   PRIMARY KEY,
            strategy  TEXT        NOT NULL,
            tickers   TEXT[]      NOT NULL,
            report    TEXT        NOT NULL,
            model     TEXT        NOT NULL,
            ran_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
            metadata  JSONB
        )
    """)
    op.execute(
        "CREATE INDEX IF NOT EXISTS strategy_results_ran_at_idx ON strategy_results (ran_at DESC)"
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS strategy_results CASCADE")
    op.execute("DROP TABLE IF EXISTS embeddings CASCADE")
    op.execute("DROP TABLE IF EXISTS documents CASCADE")
    op.execute("DROP TABLE IF EXISTS news_raw CASCADE")
    op.execute("DROP TABLE IF EXISTS raw_data CASCADE")
