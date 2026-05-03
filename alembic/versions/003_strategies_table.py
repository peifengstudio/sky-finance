"""Add strategies table and link strategy_results

Revision ID: 003
Revises: 002
Create Date: 2026-04-20
"""

from collections.abc import Sequence

from alembic import op

revision: str = "003"
down_revision: str | None = "002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE strategies (
            id                  BIGSERIAL       PRIMARY KEY,
            name                TEXT            NOT NULL UNIQUE,
            description         TEXT            NOT NULL DEFAULT '',
            scope               TEXT            NOT NULL DEFAULT 'global',
            scope_value         TEXT,
            rag_query_template  TEXT            NOT NULL DEFAULT '',
            prompt_template     TEXT            NOT NULL DEFAULT '',
            model_tier          TEXT            NOT NULL DEFAULT 'local',
            schedule            TEXT,
            enabled             BOOLEAN         NOT NULL DEFAULT true,
            created_at          TIMESTAMPTZ     NOT NULL DEFAULT now(),
            updated_at          TIMESTAMPTZ     NOT NULL DEFAULT now()
        )
    """)
    op.execute("CREATE INDEX strategies_enabled_idx ON strategies (enabled)")

    # Link results to their defining strategy (nullable — old rows stay valid)
    op.execute("""
        ALTER TABLE strategy_results
            ADD COLUMN strategy_id BIGINT REFERENCES strategies(id) ON DELETE SET NULL
    """)
    op.execute("CREATE INDEX strategy_results_strategy_id_idx ON strategy_results (strategy_id)")


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS strategy_results_strategy_id_idx")
    op.execute("ALTER TABLE strategy_results DROP COLUMN IF EXISTS strategy_id")
    op.execute("DROP TABLE IF EXISTS strategies CASCADE")
