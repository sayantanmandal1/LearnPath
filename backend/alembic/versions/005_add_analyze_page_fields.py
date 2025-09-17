"""Add analyze page fields to user profiles

Revision ID: 005_add_analyze_page_fields
Revises: 004_add_security_privacy_tables
Create Date: 2024-12-19 10:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '005_add_analyze_page_fields'
down_revision = '004_add_security_privacy_tables'
branch_labels = None
depends_on = None


def upgrade():
    """Add new fields to user_profiles table for analyze page support"""
    
    # Add new columns to user_profiles table
    op.add_column('user_profiles', sa.Column('industry', sa.String(255), nullable=True))
    op.add_column('user_profiles', sa.Column('desired_role', sa.String(255), nullable=True))
    op.add_column('user_profiles', sa.Column('career_goals', sa.Text(), nullable=True))
    op.add_column('user_profiles', sa.Column('timeframe', sa.String(100), nullable=True))
    op.add_column('user_profiles', sa.Column('salary_expectation', sa.String(100), nullable=True))
    op.add_column('user_profiles', sa.Column('education', sa.String(500), nullable=True))
    op.add_column('user_profiles', sa.Column('certifications', sa.Text(), nullable=True))
    op.add_column('user_profiles', sa.Column('languages', sa.String(500), nullable=True))
    op.add_column('user_profiles', sa.Column('work_type', sa.String(100), nullable=True))
    op.add_column('user_profiles', sa.Column('company_size', sa.String(100), nullable=True))
    op.add_column('user_profiles', sa.Column('work_culture', sa.Text(), nullable=True))
    op.add_column('user_profiles', sa.Column('benefits', sa.JSON(), nullable=True))
    op.add_column('user_profiles', sa.Column('profile_score', sa.Float(), nullable=True))
    op.add_column('user_profiles', sa.Column('completeness_score', sa.Float(), nullable=True))


def downgrade():
    """Remove analyze page fields from user_profiles table"""
    
    # Remove the added columns
    op.drop_column('user_profiles', 'completeness_score')
    op.drop_column('user_profiles', 'profile_score')
    op.drop_column('user_profiles', 'benefits')
    op.drop_column('user_profiles', 'work_culture')
    op.drop_column('user_profiles', 'company_size')
    op.drop_column('user_profiles', 'work_type')
    op.drop_column('user_profiles', 'languages')
    op.drop_column('user_profiles', 'certifications')
    op.drop_column('user_profiles', 'education')
    op.drop_column('user_profiles', 'salary_expectation')
    op.drop_column('user_profiles', 'timeframe')
    op.drop_column('user_profiles', 'career_goals')
    op.drop_column('user_profiles', 'desired_role')
    op.drop_column('user_profiles', 'industry')