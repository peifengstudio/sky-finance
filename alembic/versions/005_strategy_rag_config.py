"""Add per-strategy RAG config (threshold + per-sentiment top_k)

Revision ID: 005
Revises: 004
Create Date: 2026-04-21
"""

from collections.abc import Sequence

from alembic import op

revision: str = "005"
down_revision: str | None = "004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("ALTER TABLE strategies ADD COLUMN IF NOT EXISTS rag_threshold FLOAT DEFAULT 0.55")
    op.execute("ALTER TABLE strategies ADD COLUMN IF NOT EXISTS rag_top_k_positive INT DEFAULT 20")
    op.execute("ALTER TABLE strategies ADD COLUMN IF NOT EXISTS rag_top_k_neutral  INT DEFAULT 20")
    op.execute("ALTER TABLE strategies ADD COLUMN IF NOT EXISTS rag_top_k_negative INT DEFAULT 20")


def downgrade() -> None:
    op.execute("ALTER TABLE strategies DROP COLUMN IF EXISTS rag_top_k_negative")
    op.execute("ALTER TABLE strategies DROP COLUMN IF EXISTS rag_top_k_neutral")
    op.execute("ALTER TABLE strategies DROP COLUMN IF EXISTS rag_top_k_positive")
    op.execute("ALTER TABLE strategies DROP COLUMN IF EXISTS rag_threshold")
