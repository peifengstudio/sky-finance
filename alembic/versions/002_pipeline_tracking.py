"""Add processed_at columns for pipeline tracking

Revision ID: 002
Revises: 001
Create Date: 2026-04-18
"""

from collections.abc import Sequence

from alembic import op

revision: str = "002"
down_revision: str | None = "001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Track whether each raw record has been processed by the pipeline.
    # NULL  = unprocessed (pending)
    # value = timestamp when pipeline completed successfully
    op.execute("ALTER TABLE raw_data ADD COLUMN IF NOT EXISTS processed_at TIMESTAMPTZ")
    op.execute("ALTER TABLE news_raw  ADD COLUMN IF NOT EXISTS processed_at TIMESTAMPTZ")

    # Partial indexes for efficient "fetch unprocessed" queries
    op.execute("""
        CREATE INDEX IF NOT EXISTS raw_data_unprocessed_idx
            ON raw_data (fetched_at ASC)
            WHERE processed_at IS NULL
    """)
    op.execute("""
        CREATE INDEX IF NOT EXISTS news_raw_unprocessed_idx
            ON news_raw (fetched_at ASC)
            WHERE processed_at IS NULL
    """)


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS news_raw_unprocessed_idx")
    op.execute("DROP INDEX IF EXISTS raw_data_unprocessed_idx")
    op.execute("ALTER TABLE news_raw  DROP COLUMN IF EXISTS processed_at")
    op.execute("ALTER TABLE raw_data  DROP COLUMN IF EXISTS processed_at")
