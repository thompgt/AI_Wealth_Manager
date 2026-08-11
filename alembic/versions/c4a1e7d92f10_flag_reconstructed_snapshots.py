"""Flag reconstructed snapshots

Backfilled valuations used to be marked with a `_reconstructed` key inside the
`holdings_snapshot` JSON, which no query filtered on, so they fed the reported
return as if they were measured. Promote it to a real column.

Existing rows carrying the old JSON marker are flagged; everything else is
treated as measured, which is the safe direction — a genuine snapshot wrongly
excluded understates history, while a reconstructed one wrongly included
overstates performance.

Revision ID: c4a1e7d92f10
Revises: bffddbb4e18d
Create Date: 2026-08-11 09:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'c4a1e7d92f10'
down_revision: Union[str, Sequence[str], None] = 'bffddbb4e18d'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    with op.batch_alter_table('portfolio_snapshots', schema=None) as batch_op:
        batch_op.add_column(
            sa.Column(
                'is_reconstructed',
                sa.Boolean(),
                nullable=False,
                server_default=sa.false(),
            )
        )
        batch_op.create_index(
            batch_op.f('ix_portfolio_snapshots_is_reconstructed'),
            ['is_reconstructed'],
            unique=False,
        )

    # Backfill the flag from the legacy JSON marker. Compared as text so this
    # works on both SQLite and Postgres without a JSON operator per dialect.
    op.execute(
        "UPDATE portfolio_snapshots SET is_reconstructed = true "
        "WHERE CAST(holdings_snapshot AS VARCHAR(2000)) LIKE '%_reconstructed%'"
    )


def downgrade() -> None:
    """Downgrade schema."""
    with op.batch_alter_table('portfolio_snapshots', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_portfolio_snapshots_is_reconstructed'))
        batch_op.drop_column('is_reconstructed')
