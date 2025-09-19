"""Add job application tracking tables

Revision ID: 008_add_job_application_tables
Revises: 007_platform_fields
Create Date: 2024-01-01 12:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '008_add_job_application_tables'
down_revision = '007_platform_fields'
branch_labels = None
depends_on = None


def upgrade():
    # Create job_applications table
    op.create_table('job_applications',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('job_posting_id', sa.String(length=255), nullable=False),
        sa.Column('job_title', sa.String(length=255), nullable=False),
        sa.Column('company_name', sa.String(length=255), nullable=False),
        sa.Column('job_url', sa.Text(), nullable=True),
        sa.Column('status', sa.String(length=50), nullable=False),
        sa.Column('applied_date', sa.DateTime(timezone=True), nullable=True),
        sa.Column('last_updated', sa.DateTime(timezone=True), nullable=False),
        sa.Column('match_score', sa.Float(), nullable=True),
        sa.Column('skill_matches', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('skill_gaps', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('application_method', sa.String(length=100), nullable=True),
        sa.Column('cover_letter', sa.Text(), nullable=True),
        sa.Column('notes', sa.Text(), nullable=True),
        sa.Column('interview_scheduled', sa.Boolean(), nullable=True),
        sa.Column('interview_date', sa.DateTime(timezone=True), nullable=True),
        sa.Column('interview_notes', sa.Text(), nullable=True),
        sa.Column('feedback_received', sa.Boolean(), nullable=True),
        sa.Column('feedback_text', sa.Text(), nullable=True),
        sa.Column('rejection_reason', sa.String(length=255), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_job_applications_user_id'), 'job_applications', ['user_id'], unique=False)
    op.create_index(op.f('ix_job_applications_job_posting_id'), 'job_applications', ['job_posting_id'], unique=False)
    op.create_index(op.f('ix_job_applications_status'), 'job_applications', ['status'], unique=False)
    op.create_index(op.f('ix_job_applications_created_at'), 'job_applications', ['created_at'], unique=False)
    
    # Create unique constraint for user_id + job_posting_id
    op.create_index('ix_job_applications_user_job_unique', 'job_applications', ['user_id', 'job_posting_id'], unique=True)

    # Create job_application_feedback table
    op.create_table('job_application_feedback',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('application_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('feedback_type', sa.String(length=50), nullable=False),
        sa.Column('rating', sa.Integer(), nullable=True),
        sa.Column('feedback_text', sa.Text(), nullable=True),
        sa.Column('match_accuracy_rating', sa.Integer(), nullable=True),
        sa.Column('recommendation_helpfulness', sa.Integer(), nullable=True),
        sa.Column('gap_analysis_accuracy', sa.Integer(), nullable=True),
        sa.Column('suggested_improvements', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(['application_id'], ['job_applications.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_job_application_feedback_application_id'), 'job_application_feedback', ['application_id'], unique=False)
    op.create_index(op.f('ix_job_application_feedback_feedback_type'), 'job_application_feedback', ['feedback_type'], unique=False)

    # Create job_recommendation_feedback table
    op.create_table('job_recommendation_feedback',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('job_posting_id', sa.String(length=255), nullable=False),
        sa.Column('recommendation_shown', sa.Boolean(), nullable=True),
        sa.Column('user_interested', sa.Boolean(), nullable=True),
        sa.Column('user_applied', sa.Boolean(), nullable=True),
        sa.Column('match_score_feedback', sa.String(length=50), nullable=True),
        sa.Column('skill_match_feedback', sa.String(length=50), nullable=True),
        sa.Column('location_feedback', sa.String(length=50), nullable=True),
        sa.Column('feedback_text', sa.Text(), nullable=True),
        sa.Column('improvement_suggestions', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_job_recommendation_feedback_user_id'), 'job_recommendation_feedback', ['user_id'], unique=False)
    op.create_index(op.f('ix_job_recommendation_feedback_job_posting_id'), 'job_recommendation_feedback', ['job_posting_id'], unique=False)


def downgrade():
    # Drop job_recommendation_feedback table
    op.drop_index(op.f('ix_job_recommendation_feedback_job_posting_id'), table_name='job_recommendation_feedback')
    op.drop_index(op.f('ix_job_recommendation_feedback_user_id'), table_name='job_recommendation_feedback')
    op.drop_table('job_recommendation_feedback')
    
    # Drop job_application_feedback table
    op.drop_index(op.f('ix_job_application_feedback_feedback_type'), table_name='job_application_feedback')
    op.drop_index(op.f('ix_job_application_feedback_application_id'), table_name='job_application_feedback')
    op.drop_table('job_application_feedback')
    
    # Drop job_applications table
    op.drop_index('ix_job_applications_user_job_unique', table_name='job_applications')
    op.drop_index(op.f('ix_job_applications_created_at'), table_name='job_applications')
    op.drop_index(op.f('ix_job_applications_status'), table_name='job_applications')
    op.drop_index(op.f('ix_job_applications_job_posting_id'), table_name='job_applications')
    op.drop_index(op.f('ix_job_applications_user_id'), table_name='job_applications')
    op.drop_table('job_applications')