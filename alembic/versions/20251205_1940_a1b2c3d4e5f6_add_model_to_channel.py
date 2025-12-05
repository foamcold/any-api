"""add model to channel

Revision ID: a1b2c3d4e5f6
Revises: 79c087902f1c
Create Date: 2025-12-05 19:40:15.123456

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'a1b2c3d4e5f6'
down_revision: Union[str, None] = '79c087902f1c'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('channels', sa.Column('model', sa.String(), nullable=False, server_default='auto'))


def downgrade() -> None:
    op.drop_column('channels', 'model')