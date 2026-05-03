"""add eval_results table

Revision ID: 006
Revises: 005
"""

from alembic import op

revision: str = "006"
down_revision: str | None = "005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE eval_results (
            id                SERIAL PRIMARY KEY,

            -- Source strategy (nullable: strategy may be deleted later)
            strategy_id       INTEGER REFERENCES strategies(id) ON DELETE SET NULL,
            strategy_name     TEXT NOT NULL,

            -- What was evaluated
            ticker            VARCHAR(20)  NOT NULL,
            query             TEXT         NOT NULL,

            -- Retrieval stats
            bucketed_n_chunks INTEGER      NOT NULL DEFAULT 0,
            plain_n_chunks    INTEGER      NOT NULL DEFAULT 0,

            -- Generated reports
            bucketed_report   TEXT         NOT NULL DEFAULT '',
            plain_report      TEXT         NOT NULL DEFAULT '',

            -- Judge aggregate scores (mean of the three dimensions, 0–10)
            bucketed_score    FLOAT        NOT NULL DEFAULT 0,
            plain_score       FLOAT        NOT NULL DEFAULT 0,

            -- Per-dimension scores from the judge  {faithfulness, coverage, actionability}
            bucketed_scores   JSONB        NOT NULL DEFAULT '{}',
            plain_scores      JSONB        NOT NULL DEFAULT '{}',

            -- Verdict
            winner            VARCHAR(20)  NOT NULL DEFAULT 'tie',
            judge_reasoning   TEXT         NOT NULL DEFAULT '',
            judge_model       VARCHAR(100) NOT NULL DEFAULT '',

            ran_at            TIMESTAMPTZ  NOT NULL DEFAULT NOW()
        );

        CREATE INDEX ix_eval_results_strategy_id ON eval_results(strategy_id);
        CREATE INDEX ix_eval_results_ticker       ON eval_results(ticker);
        CREATE INDEX ix_eval_results_ran_at       ON eval_results(ran_at DESC);
        """)


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS eval_results CASCADE")
