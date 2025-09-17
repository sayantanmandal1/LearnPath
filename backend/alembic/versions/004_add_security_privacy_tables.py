"""Add security and privacy tables

Revision ID: 004_add_security_privacy_tables
Revises: 003_seed_skill_taxonomy
Create Date: 2024-01-01 12:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '004_add_security_privacy_tables'
down_revision = '003'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create audit_logs table
    op.create_table(
        'audit_logs',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('event_type', sa.String(50), nullable=False, index=True),
        sa.Column('severity', sa.String(20), nullable=False, index=True),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=True, index=True),
        sa.Column('session_id', sa.String(100), nullable=True, index=True),
        sa.Column('ip_address', sa.String(45), nullable=True, index=True),
        sa.Column('user_agent', sa.Text, nullable=True),
        sa.Column('endpoint', sa.String(200), nullable=True, index=True),
        sa.Column('method', sa.String(10), nullable=True),
        sa.Column('status_code', sa.Integer, nullable=True),
        sa.Column('message', sa.Text, nullable=False),
        sa.Column('details', postgresql.JSONB, nullable=True),
        sa.Column('timestamp', sa.DateTime, nullable=False, index=True),
        sa.Column('request_id', sa.String(100), nullable=True, index=True),
        sa.Column('correlation_id', sa.String(100), nullable=True, index=True),
        sa.Column('source_system', sa.String(50), nullable=True),
    )
    
    # Create user_consents table
    op.create_table(
        'user_consents',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('users.id'), nullable=False, index=True),
        sa.Column('consent_type', sa.String(50), nullable=False, index=True),
        sa.Column('granted', sa.Boolean, nullable=False, default=False),
        sa.Column('granted_at', sa.DateTime, nullable=True),
        sa.Column('withdrawn_at', sa.DateTime, nullable=True),
        sa.Column('ip_address', sa.String(45), nullable=True),
        sa.Column('user_agent', sa.Text, nullable=True),
        sa.Column('consent_version', sa.String(20), nullable=False, default='1.0'),
        sa.Column('metadata', postgresql.JSONB, nullable=True),
        sa.Column('created_at', sa.DateTime, nullable=False),
        sa.Column('updated_at', sa.DateTime, nullable=False),
    )
    
    # Create privacy_requests table
    op.create_table(
        'privacy_requests',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('users.id'), nullable=False, index=True),
        sa.Column('request_type', sa.String(50), nullable=False, index=True),
        sa.Column('status', sa.String(20), nullable=False, default='pending', index=True),
        sa.Column('description', sa.Text, nullable=True),
        sa.Column('requested_data_categories', postgresql.JSONB, nullable=True),
        sa.Column('response_data', postgresql.JSONB, nullable=True),
        sa.Column('ip_address', sa.String(45), nullable=True),
        sa.Column('verification_token', sa.String(100), nullable=True, index=True),
        sa.Column('verified_at', sa.DateTime, nullable=True),
        sa.Column('completed_at', sa.DateTime, nullable=True),
        sa.Column('expires_at', sa.DateTime, nullable=False),
        sa.Column('created_at', sa.DateTime, nullable=False),
        sa.Column('updated_at', sa.DateTime, nullable=False),
    )
    
    # Create data_retention_policies table
    op.create_table(
        'data_retention_policies',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('data_category', sa.String(50), nullable=False, index=True),
        sa.Column('retention_period_days', sa.Integer, nullable=False),
        sa.Column('description', sa.Text, nullable=True),
        sa.Column('legal_basis', sa.String(100), nullable=True),
        sa.Column('active', sa.Boolean, nullable=False, default=True),
        sa.Column('created_at', sa.DateTime, nullable=False),
        sa.Column('updated_at', sa.DateTime, nullable=False),
    )
    
    # Create indexes for better performance
    op.create_index('idx_audit_logs_user_timestamp', 'audit_logs', ['user_id', 'timestamp'])
    op.create_index('idx_audit_logs_event_severity', 'audit_logs', ['event_type', 'severity'])
    op.create_index('idx_user_consents_user_type', 'user_consents', ['user_id', 'consent_type'])
    op.create_index('idx_privacy_requests_user_status', 'privacy_requests', ['user_id', 'status'])
    
    # Insert default data retention policies
    op.execute("""
        INSERT INTO data_retention_policies (id, data_category, retention_period_days, description, legal_basis, active, created_at, updated_at)
        VALUES 
        (gen_random_uuid(), 'basic_profile', 30, 'Basic user profile data retained for 30 days after account deletion', 'Contract performance', true, NOW(), NOW()),
        (gen_random_uuid(), 'professional_data', 365, 'Professional data retained for 1 year after account deletion', 'Legitimate interest', true, NOW(), NOW()),
        (gen_random_uuid(), 'platform_data', 30, 'External platform data retained for 30 days after account deletion', 'Consent', true, NOW(), NOW()),
        (gen_random_uuid(), 'behavioral_data', 730, 'Usage patterns retained for 2 years', 'Legitimate interest', true, NOW(), NOW()),
        (gen_random_uuid(), 'generated_data', 180, 'AI-generated insights retained for 6 months after account deletion', 'Contract performance', true, NOW(), NOW()),
        (gen_random_uuid(), 'audit_logs', 2555, 'Audit logs retained for 7 years for security and compliance', 'Legal obligation', true, NOW(), NOW())
    """)


def downgrade() -> None:
    # Drop indexes
    op.drop_index('idx_privacy_requests_user_status')
    op.drop_index('idx_user_consents_user_type')
    op.drop_index('idx_audit_logs_event_severity')
    op.drop_index('idx_audit_logs_user_timestamp')
    
    # Drop tables
    op.drop_table('data_retention_policies')
    op.drop_table('privacy_requests')
    op.drop_table('user_consents')
    op.drop_table('audit_logs')