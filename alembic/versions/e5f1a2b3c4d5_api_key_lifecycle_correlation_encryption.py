"""Add API key rotation, correlation ID, and encrypted PII fields

Revision ID: e5f1a2b3c4d5
Revises: d7b2c9a41e35
Create Date: 2026-09-18 09:25:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from services.encryption import EncryptedDateTime, EncryptedString


# revision identifiers, used by Alembic.
revision: str = 'e5f1a2b3c4d5'
down_revision: Union[str, Sequence[str], None] = 'd7b2c9a41e35'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # 1. API key rotation and revocation tracking
    with op.batch_alter_table('api_keys', schema=None) as batch_op:
        batch_op.add_column(sa.Column('revocation_reason', sa.String(length=200), nullable=True))
        batch_op.add_column(sa.Column('superseded_by_id', sa.Integer(), nullable=True))
        batch_op.create_foreign_key(
            'fk_api_keys_superseded_by_id',
            'api_keys',
            ['superseded_by_id'],
            ['id'],
            ondelete='SET NULL',
        )

    # 2. Correlation ID propagation on jobs
    with op.batch_alter_table('jobs', schema=None) as batch_op:
        batch_op.add_column(sa.Column('correlation_id', sa.String(length=64), nullable=True))
        batch_op.create_index('ix_jobs_correlation_id', ['correlation_id'], unique=False)

    # 3. Transparent PII encryption at rest
    with op.batch_alter_table('client_profiles', schema=None) as batch_op:
        batch_op.alter_column(
            'email',
            type_=EncryptedString(),
            existing_type=sa.String(length=320),
            existing_nullable=True,
        )
        batch_op.alter_column(
            'phone',
            type_=EncryptedString(),
            existing_type=sa.String(length=50),
            existing_nullable=True,
        )
        batch_op.alter_column(
            'date_of_birth',
            type_=EncryptedDateTime(),
            existing_type=sa.DateTime(),
            existing_nullable=True,
            postgresql_using='date_of_birth::text',
        )


def downgrade() -> None:
    """Downgrade schema."""
    with op.batch_alter_table('client_profiles', schema=None) as batch_op:
        batch_op.alter_column(
            'date_of_birth',
            type_=sa.DateTime(),
            existing_type=EncryptedDateTime(),
            existing_nullable=True,
            postgresql_using='date_of_birth::timestamp without time zone',
        )
        batch_op.alter_column(
            'phone',
            type_=sa.String(length=50),
            existing_type=EncryptedString(),
            existing_nullable=True,
        )
        batch_op.alter_column(
            'email',
            type_=sa.String(length=320),
            existing_type=EncryptedString(),
            existing_nullable=True,
        )

    with op.batch_alter_table('jobs', schema=None) as batch_op:
        batch_op.drop_index('ix_jobs_correlation_id')
        batch_op.drop_column('correlation_id')

    with op.batch_alter_table('api_keys', schema=None) as batch_op:
        batch_op.drop_constraint('fk_api_keys_superseded_by_id', type_='foreignkey')
        batch_op.drop_column('superseded_by_id')
        batch_op.drop_column('revocation_reason')
