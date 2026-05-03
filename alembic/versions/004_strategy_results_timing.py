"""Add started_at and duration_seconds to strategy_results

Revision ID: 004
Revises: 003
Create Date: 2026-04-20
"""

from collections.abc import Sequence

from alembic import op

revision: str = "004"
down_revision: str | None = "003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("ALTER TABLE strategy_results ADD COLUMN IF NOT EXISTS started_at TIMESTAMPTZ")
    op.execute("ALTER TABLE strategy_results ADD COLUMN IF NOT EXISTS duration_seconds FLOAT")


def downgrade() -> None:
    op.execute("ALTER TABLE strategy_results DROP COLUMN IF EXISTS duration_seconds")
    op.execute("ALTER TABLE strategy_results DROP COLUMN IF EXISTS started_at")
