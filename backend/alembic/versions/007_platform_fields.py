"""Add additional platform fields to user profiles

Revision ID: 007_add_additional_platform_fields
Revises: 006_add_enhanced_profile_analysis_tables
Create Date: 2024-12-19 12:30:00.000000

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '007_platform_fields'
down_revision = '006_enhanced_profile_tables'
branch_labels = None
depends_on = None


def upgrade():
    """Add additional platform fields to user_profiles table"""
    
    # Add new platform fields to user_profiles table
    op.add_column('user_profiles', sa.Column('atcoder_username', sa.String(100), nullable=True))
    op.add_column('user_profiles', sa.Column('hackerrank_username', sa.String(100), nullable=True))
    op.add_column('user_profiles', sa.Column('kaggle_username', sa.String(100), nullable=True))


def downgrade():
    """Remove additional platform fields from user_profiles table"""
    
    # Remove the added platform fields
    op.drop_column('user_profiles', 'kaggle_username')
    op.drop_column('user_profiles', 'hackerrank_username')
    op.drop_column('user_profiles', 'atcoder_username')