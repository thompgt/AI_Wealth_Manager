"""Alembic environment.

The database URL is read from the application settings rather than
`alembic.ini`, so migrations always target the same database the app does.
Duplicating the URL in two places is how a migration ends up applied to the
wrong environment.
"""

import os
import sys
from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

# `alembic` is normally invoked from the repo root, but not always (CI, a
# container entrypoint). Put the project root on the path explicitly rather
# than depending on the caller's working directory.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import settings  # noqa: E402
from db import Base  # noqa: E402

# Importing db registers every model on Base.metadata; without it autogenerate
# would see an empty schema and cheerfully emit a migration dropping
# everything.
import db  # noqa: F401,E402

config = context.config

resolved_url = settings.NEON_DATABASE_URL
if resolved_url.startswith("postgres://"):
    resolved_url = resolved_url.replace("postgres://", "postgresql://", 1)
# ConfigParser treats `%` as interpolation syntax, and Postgres passwords
# routinely contain percent-encoded characters.
config.set_main_option("sqlalchemy.url", resolved_url.replace("%", "%%"))

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata

# SQLite cannot ALTER most things in place. Batch mode rewrites the table
# instead, which is what makes one migration script run on both the local
# SQLite database and Postgres.
render_as_batch = resolved_url.startswith("sqlite")


def run_migrations_offline() -> None:
    context.configure(
        url=resolved_url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        render_as_batch=render_as_batch,
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            render_as_batch=render_as_batch,
            # Without this a column widened from String(50) to String(200) is
            # silently skipped by autogenerate.
            compare_type=True,
            compare_server_default=True,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
