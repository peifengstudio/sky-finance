"""Add hybrid retrieval support: body_tsv on documents, retrieval_mode on strategies

Revision ID: 007
Revises: 006
Create Date: 2026-05-02

Changes
-------
documents
  - body_tsv TSVECTOR GENERATED ALWAYS AS STORED — auto-updated full-text search
    vector combining title and body; used by the BM25 leg of hybrid retrieval.
  - GIN index on body_tsv for fast keyword lookup.

strategies
  - retrieval_mode TEXT DEFAULT 'hybrid' — 'hybrid' (BM25 + vector + RRF) or
    'vector' (legacy cosine-only).
"""

from collections.abc import Sequence

from alembic import op

revision: str = "007"
down_revision: str | None = "006"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Generated tsvector column — PostgreSQL 12+ feature, available in pg17.
    # Automatically updated on every INSERT/UPDATE; no trigger needed.
    op.execute("""
        ALTER TABLE documents
        ADD COLUMN IF NOT EXISTS body_tsv tsvector
        GENERATED ALWAYS AS (
            to_tsvector('english', coalesce(title, '') || ' ' || body)
        ) STORED
    """)
    op.execute("""
        CREATE INDEX IF NOT EXISTS documents_body_tsv_gin
        ON documents USING gin(body_tsv)
    """)

    op.execute("""
        ALTER TABLE strategies
        ADD COLUMN IF NOT EXISTS retrieval_mode TEXT NOT NULL DEFAULT 'hybrid'
    """)


def downgrade() -> None:
    op.execute("ALTER TABLE strategies DROP COLUMN IF EXISTS retrieval_mode")
    op.execute("DROP INDEX IF EXISTS documents_body_tsv_gin")
    op.execute("ALTER TABLE documents DROP COLUMN IF EXISTS body_tsv")
