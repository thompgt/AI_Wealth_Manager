"""Client marginal tax rates on the investment policy

The tax estimate is not merely informational -- the rebalancer withholds any
trim whose estimated tax exceeds 25% of the trade -- so it is a hard gate, and
gating every client at 37%/20% over-estimates a 22%-bracket client by about 70%
and suppresses concentration fixes they should be making.

The rates live on the versioned policy rather than the client record so the
rate that produced a decision is reproducible from the version number the rest
of the decision cites. NULL means "not on file", which the estimate answers
with top-bracket rates plus a disclosure, rather than silently.

Revision ID: d7b2c9a41e35
Revises: c4a1e7d92f10
Create Date: 2026-08-11 09:30:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'd7b2c9a41e35'
down_revision: Union[str, Sequence[str], None] = 'c4a1e7d92f10'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    with op.batch_alter_table('investment_policies', schema=None) as batch_op:
        batch_op.add_column(sa.Column('marginal_tax_rate', sa.Float(), nullable=True))
        batch_op.add_column(sa.Column('capital_gains_tax_rate', sa.Float(), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    with op.batch_alter_table('investment_policies', schema=None) as batch_op:
        batch_op.drop_column('capital_gains_tax_rate')
        batch_op.drop_column('marginal_tax_rate')
