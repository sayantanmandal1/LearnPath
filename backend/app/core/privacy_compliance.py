"""
Privacy compliance features and consent management (GDPR, CCPA)
"""
import json
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from enum import Enum
import uuid
from sqlalchemy.dialects.postgresql import UUID
import structlog
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import Column, String, DateTime, Text, Boolean, ForeignKey, Integer, JSON

from app.core.database import Base
from app.core.audit_logging import audit_logger, AuditEventType

logger = structlog.get_logger()


class ConsentType(str, Enum):
    """Types of user consent"""
    DATA_PROCESSING = "data_processing"
    MARKETING = "marketing"
    ANALYTICS = "analytics"
    THIRD_PARTY_SHARING = "third_party_sharing"
    COOKIES = "cookies"
    PROFILE_ANALYSIS = "profile_analysis"
    RECOMMENDATION_ENGINE = "recommendation_engine"


class DataCategory(str, Enum):
    """Categories of personal data"""
    BASIC_PROFILE = "basic_profile"  # Name, email, basic info
    PROFESSIONAL_DATA = "professional_data"  # Resume, skills, experience
    PLATFORM_DATA = "platform_data"  # GitHub, LinkedIn, LeetCode profiles
    BEHAVIORAL_DATA = "behavioral_data"  # Usage patterns, preferences
    GENERATED_DATA = "generated_data"  # AI-generated insights, recommendations
    SENSITIVE_DATA = "sensitive_data"  # Any sensitive personal information


class PrivacyRequestType(str, Enum):
    """Types of privacy requests"""
    DATA_EXPORT = "data_export"
    DATA_DELETION = "data_deletion"
    DATA_CORRECTION = "data_correction"
    CONSENT_WITHDRAWAL = "consent_withdrawal"
    DATA_PORTABILITY = "data_portability"
    PROCESSING_RESTRICTION = "processing_restriction"


class PrivacyRequestStatus(str, Enum):
    """Status of privacy requests"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    REJECTED = "rejected"
    EXPIRED = "expired"


class UserConsent(Base):
    """User consent tracking"""
    __tablename__ = "user_consents"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True)
    consent_type = Column(String(50), nullable=False, index=True)
    granted = Column(Boolean, nullable=False, default=False)
    granted_at = Column(DateTime, nullable=True)
    withdrawn_at = Column(DateTime, nullable=True)
    ip_address = Column(String(45), nullable=True)
    user_agent = Column(Text, nullable=True)
    consent_version = Column(String(20), nullable=False, default="1.0")
    consent_metadata = Column(JSON, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": str(self.id),
            "user_id": str(self.user_id),
            "consent_type": self.consent_type,
            "granted": self.granted,
            "granted_at": self.granted_at.isoformat() if self.granted_at else None,
            "withdrawn_at": self.withdrawn_at.isoformat() if self.withdrawn_at else None,
            "consent_version": self.consent_version,
            "metadata": self.consent_metadata,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }


class PrivacyRequest(Base):
    """Privacy requests (GDPR Article 15-22, CCPA)"""
    __tablename__ = "privacy_requests"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True)
    request_type = Column(String(50), nullable=False, index=True)
    status = Column(String(20), nullable=False, default=PrivacyRequestStatus.PENDING.value, index=True)
    description = Column(Text, nullable=True)
    requested_data_categories = Column(JSON, nullable=True)
    response_data = Column(JSON, nullable=True)
    ip_address = Column(String(45), nullable=True)
    verification_token = Column(String(100), nullable=True, index=True)
    verified_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    expires_at = Column(DateTime, nullable=False)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": str(self.id),
            "user_id": str(self.user_id),
            "request_type": self.request_type,
            "status": self.status,
            "description": self.description,
            "requested_data_categories": self.requested_data_categories,
            "verified_at": self.verified_at.isoformat() if self.verified_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "expires_at": self.expires_at.isoformat(),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }


class DataRetentionPolicy(Base):
    """Data retention policies"""
    __tablename__ = "data_retention_policies"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    data_category = Column(String(50), nullable=False, index=True)
    retention_period_days = Column(Integer, nullable=False)
    description = Column(Text, nullable=True)
    legal_basis = Column(String(100), nullable=True)
    active = Column(Boolean, nullable=False, default=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)


class PrivacyComplianceService:
    """Service for managing privacy compliance"""
    
    def __init__(self):
        self.logger = structlog.get_logger("privacy")
    
    async def grant_consent(
        self,
        db: AsyncSession,
        user_id: str,
        consent_type: ConsentType,
        ip_address: str,
        user_agent: str,
        consent_version: str = "1.0",
        metadata: Optional[Dict[str, Any]] = None
    ) -> UserConsent:
        """Grant user consent"""
        
        # Check if consent already exists
        from sqlalchemy import and_
        existing_consent = await db.execute(
            db.query(UserConsent).filter(
                and_(
                    UserConsent.user_id == uuid.UUID(user_id),
                    UserConsent.consent_type == consent_type.value
                )
            )
        )
        existing_consent = existing_consent.scalar_one_or_none()
        
        if existing_consent:
            # Update existing consent
            existing_consent.granted = True
            existing_consent.granted_at = datetime.utcnow()
            existing_consent.withdrawn_at = None
            existing_consent.ip_address = ip_address
            existing_consent.user_agent = user_agent
            existing_consent.consent_version = consent_version
            existing_consent.metadata = metadata or {}
            existing_consent.updated_at = datetime.utcnow()
            
            consent = existing_consent
        else:
            # Create new consent
            consent = UserConsent(
                user_id=uuid.UUID(user_id),
                consent_type=consent_type.value,
                granted=True,
                granted_at=datetime.utcnow(),
                ip_address=ip_address,
                user_agent=user_agent,
                consent_version=consent_version,
                metadata=metadata or {}
            )
            db.add(consent)
        
        await db.commit()
        await db.refresh(consent)
        
        # Log consent event
        await audit_logger.log_privacy_event(
            db=db,
            event_type=AuditEventType.CONSENT_GIVEN,
            user_id=user_id,
            action=f"Granted consent for {consent_type.value}",
            details={
                "consent_type": consent_type.value,
                "consent_version": consent_version,
                "ip_address": ip_address,
                "metadata": metadata
            }
        )
        """Withdraw user consent"""
        
        from sqlalchemy import and_
        consent = await db.execute(
            db.query(UserConsent).filter(
                and_(
                    UserConsent.user_id == uuid.UUID(user_id),
                    UserConsent.consent_type == consent_type.value
                )
            )
        )
        consent = consent.scalar_one_or_none()
        
        if consent:
            consent.granted = False
            consent.withdrawn_at = datetime.utcnow()
            consent.ip_address = ip_address
            consent.updated_at = datetime.utcnow()
            
            await db.commit()
            await db.refresh(consent)
            
            # Log consent withdrawal
            await audit_logger.log_privacy_event(
                db=db,
                event_type=AuditEventType.CONSENT_WITHDRAWN,
                user_id=user_id,
                action=f"Withdrew consent for {consent_type.value}",
                details={
                    "consent_type": consent_type.value,
                    "ip_address": ip_address
                }
            )
        
        return consent
    
    async def check_consent(
        self,
        db: AsyncSession,
        user_id: str,
        consent_type: ConsentType
    ) -> bool:
        """Check if user has granted specific consent"""
        
        from sqlalchemy import and_
        consent = await db.execute(
            db.query(UserConsent).filter(
                and_(
                    UserConsent.user_id == uuid.UUID(user_id),
                    UserConsent.consent_type == consent_type.value,
                    UserConsent.granted
                )
            )
        )
        consent = consent.scalar_one_or_none()
        
        return consent is not None
    
    async def get_user_consents(
        self,
        db: AsyncSession,
        user_id: str
    ) -> List[UserConsent]:
        """Get all consents for a user"""
        
        consents = await db.execute(
            db.query(UserConsent).filter(
                UserConsent.user_id == uuid.UUID(user_id)
            ).order_by(UserConsent.created_at.desc())
        )
        
        return consents.scalars().all()
    
    async def create_privacy_request(
        self,
        db: AsyncSession,
        user_id: str,
        request_type: PrivacyRequestType,
        description: Optional[str] = None,
        requested_data_categories: Optional[List[DataCategory]] = None,
        ip_address: Optional[str] = None
    ) -> PrivacyRequest:
        """Create a privacy request"""
        
        # Generate verification token
        import secrets
        verification_token = secrets.token_urlsafe(32)
        
        # Set expiration (30 days for most requests)
        expires_at = datetime.utcnow() + timedelta(days=30)
        
        privacy_request = PrivacyRequest(
            user_id=uuid.UUID(user_id),
            request_type=request_type.value,
            description=description,
            requested_data_categories=[cat.value for cat in requested_data_categories] if requested_data_categories else None,
            ip_address=ip_address,
            verification_token=verification_token,
            expires_at=expires_at
        )
        
        db.add(privacy_request)
        await db.commit()
        await db.refresh(privacy_request)
        
        # Log privacy request
        await audit_logger.log_privacy_event(
            db=db,
            event_type=AuditEventType.DATA_EXPORT_REQUEST if request_type == PrivacyRequestType.DATA_EXPORT else AuditEventType.DATA_DELETION_REQUEST,
            user_id=user_id,
            action=f"Created {request_type.value} request",
            details={
                "request_id": str(privacy_request.id),
                "request_type": request_type.value,
                "description": description,
                "requested_data_categories": [cat.value for cat in requested_data_categories] if requested_data_categories else None
            }
        )
        
        return privacy_request
    
    async def export_user_data(
        self,
        db: AsyncSession,
        user_id: str,
        data_categories: Optional[List[DataCategory]] = None
    ) -> Dict[str, Any]:
        """Export all user data for GDPR compliance"""
        
        exported_data = {
            "user_id": user_id,
            "export_timestamp": datetime.utcnow().isoformat(),
            "data_categories": [cat.value for cat in data_categories] if data_categories else "all",
            "data": {}
        }
        
        try:
            # Export user profile data
            if not data_categories or DataCategory.BASIC_PROFILE in data_categories:
                from app.models.user import User
                user = await db.get(User, uuid.UUID(user_id))
                if user:
                    exported_data["data"]["basic_profile"] = {
                        "id": str(user.id),
                        "email": user.email,
                        "created_at": user.created_at.isoformat(),
                        "is_active": user.is_active
                    }
            
            # Export professional data
            if not data_categories or DataCategory.PROFESSIONAL_DATA in data_categories:
                from app.models.profile import UserProfile
                profiles = await db.execute(
                    db.query(UserProfile).filter(UserProfile.user_id == uuid.UUID(user_id))
                )
                profiles = profiles.scalars().all()
                
                exported_data["data"]["professional_data"] = []
                for profile in profiles:
                    profile_data = {
                        "id": str(profile.id),
                        "skills": profile.skills,
                        "experience_years": profile.experience_years,
                        "dream_job": profile.dream_job,
                        "created_at": profile.created_at.isoformat(),
                        "updated_at": profile.updated_at.isoformat()
                    }
                    
                    # Decrypt sensitive fields
                    if profile.github_username:
                        # Decryption skipped: pii_encryption not defined
                        profile_data["github_username"] = profile.github_username
                    
                    exported_data["data"]["professional_data"].append(profile_data)
            
            # Export consent data
            consents = await self.get_user_consents(db, user_id)
            exported_data["data"]["consents"] = [consent.to_dict() for consent in consents]
            
            # Export privacy requests
            privacy_requests = await db.execute(
                db.query(PrivacyRequest).filter(PrivacyRequest.user_id == uuid.UUID(user_id))
            )
            privacy_requests = privacy_requests.scalars().all()
            exported_data["data"]["privacy_requests"] = [req.to_dict() for req in privacy_requests]
            
            return exported_data
        
        except Exception as e:
            self.logger.error("Failed to export user data", user_id=user_id, error=str(e))
            raise
    
    async def delete_user_data(
        self,
        db: AsyncSession,
        user_id: str,
        data_categories: Optional[List[DataCategory]] = None
    ) -> Dict[str, Any]:
        """Delete user data for GDPR compliance"""
        
        deletion_summary = {
            "user_id": user_id,
            "deletion_timestamp": datetime.utcnow().isoformat(),
            "data_categories": [cat.value for cat in data_categories] if data_categories else "all",
            "deleted_records": {}
        }
        
        try:
            # Import all relevant models
            from app.models.user import User
            from app.models.profile import UserProfile
            from app.models.skill import UserSkill
            from app.models.resume import ResumeData
            from app.models.platform_account import PlatformAccount
            from app.models.job_application import JobApplication, JobApplicationFeedback
            from app.models.analysis_result import AnalysisResult, JobRecommendation
            from app.models.job_application import JobRecommendationFeedback
            from app.models.user import RefreshToken

            # Delete user skills
            user_skills = await db.execute(db.query(UserSkill).filter(UserSkill.user_id == uuid.UUID(user_id)))
            user_skills = user_skills.scalars().all()
            for skill in user_skills:
                await db.delete(skill)
            deletion_summary["deleted_records"]["user_skills"] = len(user_skills)

            # Delete resumes
            resumes = await db.execute(db.query(ResumeData).filter(ResumeData.user_id == uuid.UUID(user_id)))
            resumes = resumes.scalars().all()
            for resume in resumes:
                await db.delete(resume)
            deletion_summary["deleted_records"]["resumes"] = len(resumes)

            # Delete platform accounts
            platform_accounts = await db.execute(db.query(PlatformAccount).filter(PlatformAccount.user_id == uuid.UUID(user_id)))
            platform_accounts = platform_accounts.scalars().all()
            for account in platform_accounts:
                await db.delete(account)
            deletion_summary["deleted_records"]["platform_accounts"] = len(platform_accounts)

            # Delete job applications
            job_apps = await db.execute(db.query(JobApplication).filter(JobApplication.user_id == uuid.UUID(user_id)))
            job_apps = job_apps.scalars().all()
            for app in job_apps:
                await db.delete(app)
            deletion_summary["deleted_records"]["job_applications"] = len(job_apps)

            # Delete job application feedback
            job_app_feedback = await db.execute(db.query(JobApplicationFeedback).join(JobApplication, JobApplicationFeedback.application_id == JobApplication.id).filter(JobApplication.user_id == uuid.UUID(user_id)))
            job_app_feedback = job_app_feedback.scalars().all()
            for feedback in job_app_feedback:
                await db.delete(feedback)
            deletion_summary["deleted_records"]["job_application_feedback"] = len(job_app_feedback)

            # Delete job recommendation feedback
            job_rec_feedback = await db.execute(db.query(JobRecommendationFeedback).filter(JobRecommendationFeedback.user_id == uuid.UUID(user_id)))
            job_rec_feedback = job_rec_feedback.scalars().all()
            for feedback in job_rec_feedback:
                await db.delete(feedback)
            deletion_summary["deleted_records"]["job_recommendation_feedback"] = len(job_rec_feedback)

            # Delete analysis results
            analysis_results = await db.execute(db.query(AnalysisResult).filter(AnalysisResult.user_id == uuid.UUID(user_id)))
            analysis_results = analysis_results.scalars().all()
            for result in analysis_results:
                await db.delete(result)
            deletion_summary["deleted_records"]["analysis_results"] = len(analysis_results)

            # Delete job recommendations
            job_recs = await db.execute(db.query(JobRecommendation).filter(JobRecommendation.user_id == uuid.UUID(user_id)))
            job_recs = job_recs.scalars().all()
            for rec in job_recs:
                await db.delete(rec)
            deletion_summary["deleted_records"]["job_recommendations"] = len(job_recs)

            # Delete user profiles
            profiles = await db.execute(db.query(UserProfile).filter(UserProfile.user_id == uuid.UUID(user_id)))
            profiles = profiles.scalars().all()
            for profile in profiles:
                await db.delete(profile)
            deletion_summary["deleted_records"]["profiles"] = len(profiles)

            # Delete refresh tokens
            refresh_tokens = await db.execute(db.query(RefreshToken).filter(RefreshToken.user_id == str(user_id)))
            refresh_tokens = refresh_tokens.scalars().all()
            for token in refresh_tokens:
                await db.delete(token)
            deletion_summary["deleted_records"]["refresh_tokens"] = len(refresh_tokens)

            # Delete consent records
            consents = await db.execute(db.query(UserConsent).filter(UserConsent.user_id == uuid.UUID(user_id)))
            consents = consents.scalars().all()
            for consent in consents:
                await db.delete(consent)
            deletion_summary["deleted_records"]["consents"] = len(consents)

            # Finally, delete the user account
            user = await db.get(User, uuid.UUID(user_id))
            if user:
                await db.delete(user)
                deletion_summary["deleted_records"]["user"] = 1

            await db.commit()

            # Log deletion
            await audit_logger.log_privacy_event(
                db=db,
                event_type=AuditEventType.DATA_DELETION_REQUEST,
                user_id=user_id,
                action="User data deleted",
                details=deletion_summary
            )

            return deletion_summary
        
        except Exception as e:
            await db.rollback()
            self.logger.error("Failed to delete user data", user_id=user_id, error=str(e))
            raise
    
    async def process_privacy_request(
        self,
        db: AsyncSession,
        request_id: str,
        verification_token: str
    ) -> PrivacyRequest:
        """Process a verified privacy request"""
        
        # Get and verify request
        privacy_request = await db.get(PrivacyRequest, uuid.UUID(request_id))
        if not privacy_request:
            raise ValueError("Privacy request not found")
        
        if privacy_request.verification_token != verification_token:
            raise ValueError("Invalid verification token")
        
        if privacy_request.expires_at < datetime.utcnow():
            privacy_request.status = PrivacyRequestStatus.EXPIRED.value
            await db.commit()
            raise ValueError("Privacy request has expired")
        
        # Mark as verified
        privacy_request.verified_at = datetime.utcnow()
        privacy_request.status = PrivacyRequestStatus.IN_PROGRESS.value
        
        try:
            # Process based on request type
            if privacy_request.request_type == PrivacyRequestType.DATA_EXPORT.value:
                data_categories = [DataCategory(cat) for cat in privacy_request.requested_data_categories] if privacy_request.requested_data_categories else None
                exported_data = await self.export_user_data(
                    db, str(privacy_request.user_id), data_categories
                )
                privacy_request.response_data = exported_data
            
            elif privacy_request.request_type == PrivacyRequestType.DATA_DELETION.value:
                data_categories = [DataCategory(cat) for cat in privacy_request.requested_data_categories] if privacy_request.requested_data_categories else None
                deletion_summary = await self.delete_user_data(
                    db, str(privacy_request.user_id), data_categories
                )
                privacy_request.response_data = deletion_summary
            
            # Mark as completed
            privacy_request.status = PrivacyRequestStatus.COMPLETED.value
            privacy_request.completed_at = datetime.utcnow()
            
            await db.commit()
            await db.refresh(privacy_request)
            
            return privacy_request
        
        except Exception as e:
            privacy_request.status = PrivacyRequestStatus.REJECTED.value
            privacy_request.response_data = {"error": str(e)}
            await db.commit()
            raise


# Global privacy compliance service
privacy_service = PrivacyComplianceService()