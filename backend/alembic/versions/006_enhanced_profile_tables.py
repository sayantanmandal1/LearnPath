"""Add enhanced profile analysis tables

Revision ID: 006_add_enhanced_profile_analysis_tables
Revises: 005_add_analyze_page_fields
Create Date: 2024-12-19 12:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '006_enhanced_profile_tables'
down_revision = '005_add_analyze_page_fields'
branch_labels = None
depends_on = None


def upgrade():
    """Add enhanced profile analysis tables"""
    
    # Create resume_data table
    op.create_table('resume_data',
        sa.Column('id', postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column('user_id', postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column('original_filename', sa.String(length=255), nullable=False),
        sa.Column('file_path', sa.String(length=500), nullable=False),
        sa.Column('file_size', sa.Integer(), nullable=False),
        sa.Column('file_type', sa.String(length=50), nullable=False),
        sa.Column('processing_status', sa.Enum('PENDING', 'PROCESSING', 'COMPLETED', 'FAILED', 'MANUAL_ENTRY', name='processingstatus'), nullable=False),
        sa.Column('extracted_text', sa.Text(), nullable=True),
        sa.Column('extraction_confidence', sa.Float(), nullable=True),
        sa.Column('parsed_sections', sa.JSON(), nullable=True),
        sa.Column('contact_info', sa.JSON(), nullable=True),
        sa.Column('work_experience', sa.JSON(), nullable=True),
        sa.Column('education_data', sa.JSON(), nullable=True),
        sa.Column('skills_extracted', sa.JSON(), nullable=True),
        sa.Column('certifications_data', sa.JSON(), nullable=True),
        sa.Column('processing_started_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('processing_completed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('gemini_request_id', sa.String(length=255), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_resume_data_id'), 'resume_data', ['id'], unique=False)
    op.create_index(op.f('ix_resume_data_user_id'), 'resume_data', ['user_id'], unique=False)
    
    # Create platform_accounts table
    op.create_table('platform_accounts',
        sa.Column('id', postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column('user_id', postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column('platform', sa.Enum('GITHUB', 'LEETCODE', 'LINKEDIN', 'CODEFORCES', 'ATCODER', 'HACKERRANK', 'KAGGLE', name='platformtype'), nullable=False),
        sa.Column('username', sa.String(length=255), nullable=False),
        sa.Column('profile_url', sa.String(length=500), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=False),
        sa.Column('is_verified', sa.Boolean(), nullable=False),
        sa.Column('scraping_status', sa.Enum('PENDING', 'IN_PROGRESS', 'COMPLETED', 'FAILED', 'RATE_LIMITED', 'UNAUTHORIZED', 'NOT_FOUND', name='scrapingstatus'), nullable=False),
        sa.Column('last_scraped_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('next_scrape_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('scrape_frequency_hours', sa.Integer(), nullable=False),
        sa.Column('raw_data', sa.JSON(), nullable=True),
        sa.Column('processed_data', sa.JSON(), nullable=True),
        sa.Column('skills_data', sa.JSON(), nullable=True),
        sa.Column('achievements_data', sa.JSON(), nullable=True),
        sa.Column('statistics', sa.JSON(), nullable=True),
        sa.Column('last_error', sa.Text(), nullable=True),
        sa.Column('error_count', sa.Integer(), nullable=False),
        sa.Column('rate_limit_reset_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('requests_remaining', sa.Integer(), nullable=True),
        sa.Column('data_completeness_score', sa.Float(), nullable=True),
        sa.Column('data_freshness_score', sa.Float(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_platform_accounts_id'), 'platform_accounts', ['id'], unique=False)
    op.create_index(op.f('ix_platform_accounts_user_id'), 'platform_accounts', ['user_id'], unique=False)
    op.create_index(op.f('ix_platform_accounts_platform'), 'platform_accounts', ['platform'], unique=False)
    
    # Create platform_scraping_logs table
    op.create_table('platform_scraping_logs',
        sa.Column('id', postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column('platform_account_id', postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column('scraping_started_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('scraping_completed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('status', sa.Enum('PENDING', 'IN_PROGRESS', 'COMPLETED', 'FAILED', 'RATE_LIMITED', 'UNAUTHORIZED', 'NOT_FOUND', name='scrapingstatus'), nullable=False),
        sa.Column('data_points_collected', sa.Integer(), nullable=False),
        sa.Column('api_requests_made', sa.Integer(), nullable=False),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('error_details', sa.JSON(), nullable=True),
        sa.Column('processing_time_seconds', sa.Float(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['platform_account_id'], ['platform_accounts.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_platform_scraping_logs_id'), 'platform_scraping_logs', ['id'], unique=False)
    op.create_index(op.f('ix_platform_scraping_logs_platform_account_id'), 'platform_scraping_logs', ['platform_account_id'], unique=False)
    
    # Create analysis_results table
    op.create_table('analysis_results',
        sa.Column('id', postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column('user_id', postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column('analysis_type', sa.Enum('SKILL_ASSESSMENT', 'CAREER_TRAJECTORY', 'LEARNING_PATH', 'PROJECT_RECOMMENDATION', 'JOB_MATCHING', 'COMPREHENSIVE', name='analysistype'), nullable=False),
        sa.Column('analysis_version', sa.String(length=50), nullable=False),
        sa.Column('status', sa.Enum('PENDING', 'IN_PROGRESS', 'COMPLETED', 'FAILED', 'EXPIRED', name='analysisstatus'), nullable=False),
        sa.Column('resume_data_id', postgresql.UUID(as_uuid=False), nullable=True),
        sa.Column('platform_data_snapshot', sa.JSON(), nullable=True),
        sa.Column('skill_assessment', sa.JSON(), nullable=True),
        sa.Column('career_recommendations', sa.JSON(), nullable=True),
        sa.Column('learning_paths', sa.JSON(), nullable=True),
        sa.Column('project_suggestions', sa.JSON(), nullable=True),
        sa.Column('skill_gaps', sa.JSON(), nullable=True),
        sa.Column('market_insights', sa.JSON(), nullable=True),
        sa.Column('overall_score', sa.Float(), nullable=True),
        sa.Column('skill_diversity_score', sa.Float(), nullable=True),
        sa.Column('experience_relevance_score', sa.Float(), nullable=True),
        sa.Column('market_readiness_score', sa.Float(), nullable=True),
        sa.Column('gemini_request_id', sa.String(length=255), nullable=True),
        sa.Column('processing_time_seconds', sa.Float(), nullable=True),
        sa.Column('confidence_score', sa.Float(), nullable=True),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('retry_count', sa.Integer(), nullable=False),
        sa.Column('expires_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('is_current', sa.Boolean(), nullable=False),
        sa.Column('analysis_started_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('analysis_completed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['resume_data_id'], ['resume_data.id'], ),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_analysis_results_id'), 'analysis_results', ['id'], unique=False)
    op.create_index(op.f('ix_analysis_results_user_id'), 'analysis_results', ['user_id'], unique=False)
    op.create_index(op.f('ix_analysis_results_analysis_type'), 'analysis_results', ['analysis_type'], unique=False)
    
    # Create job_recommendations table
    op.create_table('job_recommendations',
        sa.Column('id', postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column('user_id', postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column('analysis_result_id', postgresql.UUID(as_uuid=False), nullable=True),
        sa.Column('job_title', sa.String(length=255), nullable=False),
        sa.Column('company_name', sa.String(length=255), nullable=False),
        sa.Column('job_description', sa.Text(), nullable=True),
        sa.Column('location', sa.String(length=255), nullable=True),
        sa.Column('salary_range', sa.JSON(), nullable=True),
        sa.Column('required_skills', sa.JSON(), nullable=True),
        sa.Column('experience_level', sa.String(length=100), nullable=True),
        sa.Column('source_platform', sa.String(length=100), nullable=False),
        sa.Column('job_url', sa.String(length=500), nullable=True),
        sa.Column('external_job_id', sa.String(length=255), nullable=True),
        sa.Column('posted_date', sa.DateTime(timezone=True), nullable=True),
        sa.Column('match_score', sa.Float(), nullable=False),
        sa.Column('skill_match_percentage', sa.Float(), nullable=True),
        sa.Column('skill_gaps', sa.JSON(), nullable=True),
        sa.Column('recommendation_reason', sa.Text(), nullable=True),
        sa.Column('is_viewed', sa.Boolean(), nullable=False),
        sa.Column('is_saved', sa.Boolean(), nullable=False),
        sa.Column('is_applied', sa.Boolean(), nullable=False),
        sa.Column('user_rating', sa.Integer(), nullable=True),
        sa.Column('user_feedback', sa.Text(), nullable=True),
        sa.Column('recommended_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('viewed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('applied_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['analysis_result_id'], ['analysis_results.id'], ),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_job_recommendations_id'), 'job_recommendations', ['id'], unique=False)
    op.create_index(op.f('ix_job_recommendations_user_id'), 'job_recommendations', ['user_id'], unique=False)


def downgrade():
    """Remove enhanced profile analysis tables"""
    
    # Drop job_recommendations table
    op.drop_index(op.f('ix_job_recommendations_user_id'), table_name='job_recommendations')
    op.drop_index(op.f('ix_job_recommendations_id'), table_name='job_recommendations')
    op.drop_table('job_recommendations')
    
    # Drop analysis_results table
    op.drop_index(op.f('ix_analysis_results_analysis_type'), table_name='analysis_results')
    op.drop_index(op.f('ix_analysis_results_user_id'), table_name='analysis_results')
    op.drop_index(op.f('ix_analysis_results_id'), table_name='analysis_results')
    op.drop_table('analysis_results')
    
    # Drop platform_scraping_logs table
    op.drop_index(op.f('ix_platform_scraping_logs_platform_account_id'), table_name='platform_scraping_logs')
    op.drop_index(op.f('ix_platform_scraping_logs_id'), table_name='platform_scraping_logs')
    op.drop_table('platform_scraping_logs')
    
    # Drop platform_accounts table
    op.drop_index(op.f('ix_platform_accounts_platform'), table_name='platform_accounts')
    op.drop_index(op.f('ix_platform_accounts_user_id'), table_name='platform_accounts')
    op.drop_index(op.f('ix_platform_accounts_id'), table_name='platform_accounts')
    op.drop_table('platform_accounts')
    
    # Drop resume_data table
    op.drop_index(op.f('ix_resume_data_user_id'), table_name='resume_data')
    op.drop_index(op.f('ix_resume_data_id'), table_name='resume_data')
    op.drop_table('resume_data')
    
    # Drop enum types
    op.execute('DROP TYPE IF EXISTS processingstatus')
    op.execute('DROP TYPE IF EXISTS platformtype')
    op.execute('DROP TYPE IF EXISTS scrapingstatus')
    op.execute('DROP TYPE IF EXISTS analysistype')
    op.execute('DROP TYPE IF EXISTS analysisstatus')