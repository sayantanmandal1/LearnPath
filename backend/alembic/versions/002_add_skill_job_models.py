"""Add skill and job models

Revision ID: 002
Revises: 001
Create Date: 2024-01-02 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '002'
down_revision = '001'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create skill_categories table
    op.create_table('skill_categories',
        sa.Column('id', postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column('name', sa.String(length=100), nullable=False),
        sa.Column('parent_id', postgresql.UUID(as_uuid=False), nullable=True),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('display_order', sa.Integer(), nullable=False),
        sa.Column('is_active', sa.Boolean(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['parent_id'], ['skill_categories.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_skill_categories_id'), 'skill_categories', ['id'], unique=False)
    op.create_index(op.f('ix_skill_categories_name'), 'skill_categories', ['name'], unique=True)
    op.create_index(op.f('ix_skill_categories_parent_id'), 'skill_categories', ['parent_id'], unique=False)

    # Create skills table
    op.create_table('skills',
        sa.Column('id', postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column('name', sa.String(length=255), nullable=False),
        sa.Column('category', sa.String(length=100), nullable=False),
        sa.Column('subcategory', sa.String(length=100), nullable=True),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('aliases', sa.Text(), nullable=True),
        sa.Column('market_demand', sa.Float(), nullable=True),
        sa.Column('average_salary_impact', sa.Float(), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_skills_id'), 'skills', ['id'], unique=False)
    op.create_index(op.f('ix_skills_name'), 'skills', ['name'], unique=True)
    op.create_index(op.f('ix_skills_category'), 'skills', ['category'], unique=False)

    # Create user_skills table
    op.create_table('user_skills',
        sa.Column('id', postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column('user_id', postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column('skill_id', postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column('confidence_score', sa.Float(), nullable=False),
        sa.Column('proficiency_level', sa.String(length=50), nullable=False),
        sa.Column('source', sa.String(length=100), nullable=False),
        sa.Column('evidence', sa.Text(), nullable=True),
        sa.Column('years_experience', sa.Float(), nullable=True),
        sa.Column('last_used', sa.DateTime(timezone=True), nullable=True),
        sa.Column('is_verified', sa.Boolean(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['skill_id'], ['skills.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_user_skills_id'), 'user_skills', ['id'], unique=False)
    op.create_index(op.f('ix_user_skills_user_id'), 'user_skills', ['user_id'], unique=False)
    op.create_index(op.f('ix_user_skills_skill_id'), 'user_skills', ['skill_id'], unique=False)

    # Create companies table
    op.create_table('companies',
        sa.Column('id', postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column('name', sa.String(length=255), nullable=False),
        sa.Column('domain', sa.String(length=255), nullable=True),
        sa.Column('industry', sa.String(length=100), nullable=True),
        sa.Column('size', sa.String(length=50), nullable=True),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('headquarters', sa.String(length=255), nullable=True),
        sa.Column('founded_year', sa.Integer(), nullable=True),
        sa.Column('glassdoor_rating', sa.Float(), nullable=True),
        sa.Column('employee_count', sa.Integer(), nullable=True),
        sa.Column('tech_stack', sa.JSON(), nullable=True),
        sa.Column('culture_keywords', sa.JSON(), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_companies_id'), 'companies', ['id'], unique=False)
    op.create_index(op.f('ix_companies_name'), 'companies', ['name'], unique=True)
    op.create_index(op.f('ix_companies_domain'), 'companies', ['domain'], unique=False)
    op.create_index(op.f('ix_companies_industry'), 'companies', ['industry'], unique=False)

    # Create job_postings table
    op.create_table('job_postings',
        sa.Column('id', postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column('external_id', sa.String(length=255), nullable=True),
        sa.Column('title', sa.String(length=255), nullable=False),
        sa.Column('company', sa.String(length=255), nullable=False),
        sa.Column('location', sa.String(length=255), nullable=True),
        sa.Column('remote_type', sa.String(length=50), nullable=True),
        sa.Column('employment_type', sa.String(length=50), nullable=True),
        sa.Column('experience_level', sa.String(length=50), nullable=True),
        sa.Column('description', sa.Text(), nullable=False),
        sa.Column('requirements', sa.Text(), nullable=True),
        sa.Column('salary_min', sa.Integer(), nullable=True),
        sa.Column('salary_max', sa.Integer(), nullable=True),
        sa.Column('salary_currency', sa.String(length=10), nullable=True),
        sa.Column('salary_period', sa.String(length=20), nullable=True),
        sa.Column('source', sa.String(length=100), nullable=False),
        sa.Column('source_url', sa.String(length=1000), nullable=True),
        sa.Column('posted_date', sa.DateTime(timezone=True), nullable=True),
        sa.Column('expires_date', sa.DateTime(timezone=True), nullable=True),
        sa.Column('processed_skills', sa.JSON(), nullable=True),
        sa.Column('market_analysis', sa.JSON(), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=False),
        sa.Column('is_processed', sa.Boolean(), nullable=False),
        sa.Column('quality_score', sa.Float(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_job_postings_id'), 'job_postings', ['id'], unique=False)
    op.create_index(op.f('ix_job_postings_external_id'), 'job_postings', ['external_id'], unique=False)
    op.create_index(op.f('ix_job_postings_title'), 'job_postings', ['title'], unique=False)
    op.create_index(op.f('ix_job_postings_company'), 'job_postings', ['company'], unique=False)
    op.create_index(op.f('ix_job_postings_location'), 'job_postings', ['location'], unique=False)
    op.create_index(op.f('ix_job_postings_source'), 'job_postings', ['source'], unique=False)
    op.create_index(op.f('ix_job_postings_posted_date'), 'job_postings', ['posted_date'], unique=False)

    # Create job_skills table
    op.create_table('job_skills',
        sa.Column('id', postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column('job_posting_id', postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column('skill_id', postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column('importance', sa.String(length=50), nullable=False),
        sa.Column('confidence_score', sa.Float(), nullable=False),
        sa.Column('years_required', sa.Integer(), nullable=True),
        sa.Column('proficiency_level', sa.String(length=50), nullable=True),
        sa.Column('context', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['job_posting_id'], ['job_postings.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['skill_id'], ['skills.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_job_skills_id'), 'job_skills', ['id'], unique=False)
    op.create_index(op.f('ix_job_skills_job_posting_id'), 'job_skills', ['job_posting_id'], unique=False)
    op.create_index(op.f('ix_job_skills_skill_id'), 'job_skills', ['skill_id'], unique=False)


def downgrade() -> None:
    op.drop_index(op.f('ix_job_skills_skill_id'), table_name='job_skills')
    op.drop_index(op.f('ix_job_skills_job_posting_id'), table_name='job_skills')
    op.drop_index(op.f('ix_job_skills_id'), table_name='job_skills')
    op.drop_table('job_skills')
    
    op.drop_index(op.f('ix_job_postings_posted_date'), table_name='job_postings')
    op.drop_index(op.f('ix_job_postings_source'), table_name='job_postings')
    op.drop_index(op.f('ix_job_postings_location'), table_name='job_postings')
    op.drop_index(op.f('ix_job_postings_company'), table_name='job_postings')
    op.drop_index(op.f('ix_job_postings_title'), table_name='job_postings')
    op.drop_index(op.f('ix_job_postings_external_id'), table_name='job_postings')
    op.drop_index(op.f('ix_job_postings_id'), table_name='job_postings')
    op.drop_table('job_postings')
    
    op.drop_index(op.f('ix_companies_industry'), table_name='companies')
    op.drop_index(op.f('ix_companies_domain'), table_name='companies')
    op.drop_index(op.f('ix_companies_name'), table_name='companies')
    op.drop_index(op.f('ix_companies_id'), table_name='companies')
    op.drop_table('companies')
    
    op.drop_index(op.f('ix_user_skills_skill_id'), table_name='user_skills')
    op.drop_index(op.f('ix_user_skills_user_id'), table_name='user_skills')
    op.drop_index(op.f('ix_user_skills_id'), table_name='user_skills')
    op.drop_table('user_skills')
    
    op.drop_index(op.f('ix_skills_category'), table_name='skills')
    op.drop_index(op.f('ix_skills_name'), table_name='skills')
    op.drop_index(op.f('ix_skills_id'), table_name='skills')
    op.drop_table('skills')
    
    op.drop_index(op.f('ix_skill_categories_parent_id'), table_name='skill_categories')
    op.drop_index(op.f('ix_skill_categories_name'), table_name='skill_categories')
    op.drop_index(op.f('ix_skill_categories_id'), table_name='skill_categories')
    op.drop_table('skill_categories')