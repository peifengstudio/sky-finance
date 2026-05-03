"""
Alembic migration environment.

DATABASE_URL is read from the environment (via .env) so the same migrations
work in local dev, CI, and production without touching alembic.ini.

Usage:
    uv run alembic upgrade head       # apply all pending migrations
    uv run alembic downgrade -1       # roll back one step
    uv run alembic revision -m "..."  # create a new empty migration
    uv run alembic history            # list all migrations
    uv run alembic current            # show applied version
"""

import os
from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

# Load .env so DATABASE_URL is available when running alembic CLI directly
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Alembic config object (reads alembic.ini)
# ---------------------------------------------------------------------------
config = context.config

# Override the URL from the environment — takes precedence over alembic.ini
_db_url = os.environ.get(
    "DATABASE_URL",
    "postgresql+psycopg://skyfinance:skyfinance@localhost:5432/skyfinance",
)
# Alembic expects postgresql+psycopg:// for psycopg3; normalise if bare URL supplied
if _db_url.startswith("postgresql://"):
    _db_url = _db_url.replace("postgresql://", "postgresql+psycopg://", 1)

config.set_main_option("sqlalchemy.url", _db_url)

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# We don't use SQLAlchemy ORM metadata — migrations are written as raw SQL.
target_metadata = None


# ---------------------------------------------------------------------------
# Migration runners
# ---------------------------------------------------------------------------


def run_migrations_offline() -> None:
    """Emit SQL to stdout without connecting (useful for review / CI dry-run)."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Connect to the database and apply migrations."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
