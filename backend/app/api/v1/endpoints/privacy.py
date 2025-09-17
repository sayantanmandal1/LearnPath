"""
Privacy and data protection API endpoints
"""
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Request
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel, Field

from app.core.database import get_db
from app.api.dependencies import get_current_user
from app.core.privacy_compliance import (
    privacy_service, ConsentType, DataCategory, PrivacyRequestType,
    UserConsent, PrivacyRequest
)
from app.core.audit_logging import audit_logger, AuditEventType
from app.models.user import User

router = APIRouter()


# Request/Response Models
class ConsentRequest(BaseModel):
    consent_type: ConsentType
    granted: bool
    metadata: Optional[dict] = None


class ConsentResponse(BaseModel):
    id: str
    consent_type: str
    granted: bool
    granted_at: Optional[str]
    withdrawn_at: Optional[str]
    consent_version: str
    created_at: str
    updated_at: str


class PrivacyRequestCreate(BaseModel):
    request_type: PrivacyRequestType
    description: Optional[str] = None
    requested_data_categories: Optional[List[DataCategory]] = None


class PrivacyRequestResponse(BaseModel):
    id: str
    request_type: str
    status: str
    description: Optional[str]
    requested_data_categories: Optional[List[str]]
    verified_at: Optional[str]
    completed_at: Optional[str]
    expires_at: str
    created_at: str


class DataExportResponse(BaseModel):
    user_id: str
    export_timestamp: str
    data_categories: List[str]
    data: dict


@router.post("/consent", response_model=ConsentResponse)
async def manage_consent(
    request: Request,
    consent_request: ConsentRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Grant or withdraw user consent"""
    
    ip_address = request.client.host if request.client else "unknown"
    user_agent = request.headers.get("user-agent", "")
    
    try:
        if consent_request.granted:
            consent = await privacy_service.grant_consent(
                db=db,
                user_id=str(current_user.id),
                consent_type=consent_request.consent_type,
                ip_address=ip_address,
                user_agent=user_agent,
                metadata=consent_request.metadata
            )
        else:
            consent = await privacy_service.withdraw_consent(
                db=db,
                user_id=str(current_user.id),
                consent_type=consent_request.consent_type,
                ip_address=ip_address
            )
        
        if not consent:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Consent record not found"
            )
        
        return ConsentResponse(**consent.to_dict())
    
    except Exception as e:
        await audit_logger.log_privacy_event(
            db=db,
            event_type=AuditEventType.CONSENT_GIVEN if consent_request.granted else AuditEventType.CONSENT_WITHDRAWN,
            user_id=str(current_user.id),
            action=f"Failed to {'grant' if consent_request.granted else 'withdraw'} consent",
            details={"error": str(e), "consent_type": consent_request.consent_type.value}
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process consent request"
        )


@router.get("/consent", response_model=List[ConsentResponse])
async def get_user_consents(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get all user consents"""
    
    try:
        consents = await privacy_service.get_user_consents(
            db=db,
            user_id=str(current_user.id)
        )
        
        return [ConsentResponse(**consent.to_dict()) for consent in consents]
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve consent information"
        )


@router.get("/consent/{consent_type}", response_model=bool)
async def check_consent(
    consent_type: ConsentType,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Check if user has granted specific consent"""
    
    try:
        has_consent = await privacy_service.check_consent(
            db=db,
            user_id=str(current_user.id),
            consent_type=consent_type
        )
        
        return has_consent
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to check consent status"
        )


@router.post("/request", response_model=PrivacyRequestResponse)
async def create_privacy_request(
    request: Request,
    privacy_request: PrivacyRequestCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Create a privacy request (data export, deletion, etc.)"""
    
    ip_address = request.client.host if request.client else "unknown"
    
    try:
        privacy_req = await privacy_service.create_privacy_request(
            db=db,
            user_id=str(current_user.id),
            request_type=privacy_request.request_type,
            description=privacy_request.description,
            requested_data_categories=privacy_request.requested_data_categories,
            ip_address=ip_address
        )
        
        # TODO: Send verification email with token
        # This would typically involve sending an email with a verification link
        
        return PrivacyRequestResponse(**privacy_req.to_dict())
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create privacy request"
        )


@router.get("/requests", response_model=List[PrivacyRequestResponse])
async def get_privacy_requests(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get all privacy requests for the current user"""
    
    try:
        from app.core.privacy_compliance import PrivacyRequest
        import uuid
        
        requests = await db.execute(
            db.query(PrivacyRequest).filter(
                PrivacyRequest.user_id == current_user.id
            ).order_by(PrivacyRequest.created_at.desc())
        )
        requests = requests.scalars().all()
        
        return [PrivacyRequestResponse(**req.to_dict()) for req in requests]
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve privacy requests"
        )


class VerificationRequest(BaseModel):
    verification_token: str = Field(..., description="Verification token from email")

@router.post("/request/{request_id}/verify")
async def verify_privacy_request(
    request_id: str,
    request: VerificationRequest,
    db: AsyncSession = Depends(get_db)
):
    """Verify and process a privacy request"""
    
    try:
        privacy_request = await privacy_service.process_privacy_request(
            db=db,
            request_id=request_id,
            verification_token=request.verification_token
        )
        
        return {
            "message": "Privacy request processed successfully",
            "request_id": str(privacy_request.id),
            "status": privacy_request.status,
            "completed_at": privacy_request.completed_at.isoformat() if privacy_request.completed_at else None
        }
    
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process privacy request"
        )


@router.get("/export", response_model=DataExportResponse)
async def export_user_data(
    data_categories: Optional[List[DataCategory]] = None,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Export user data (GDPR Article 15 - Right of Access)"""
    
    try:
        # Check if user has consent for data processing
        has_consent = await privacy_service.check_consent(
            db=db,
            user_id=str(current_user.id),
            consent_type=ConsentType.DATA_PROCESSING
        )
        
        if not has_consent:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Data processing consent required for export"
            )
        
        exported_data = await privacy_service.export_user_data(
            db=db,
            user_id=str(current_user.id),
            data_categories=data_categories
        )
        
        # Log data export
        await audit_logger.log_privacy_event(
            db=db,
            event_type=AuditEventType.DATA_EXPORT,
            user_id=str(current_user.id),
            action="User data exported",
            details={
                "data_categories": [cat.value for cat in data_categories] if data_categories else "all",
                "export_size": len(str(exported_data))
            }
        )
        
        return DataExportResponse(**exported_data)
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to export user data"
        )


class DeleteAccountRequest(BaseModel):
    confirmation: str = Field(..., description="Type 'DELETE' to confirm")

@router.delete("/delete-account")
async def delete_user_account(
    request: DeleteAccountRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Delete user account and all associated data (GDPR Article 17 - Right to Erasure)"""
    
    if request.confirmation != "DELETE":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Account deletion must be confirmed by typing 'DELETE'"
        )
    
    try:
        deletion_summary = await privacy_service.delete_user_data(
            db=db,
            user_id=str(current_user.id)
        )
        
        return {
            "message": "Account and all associated data have been permanently deleted",
            "deletion_summary": deletion_summary
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete user account"
        )


@router.get("/policy")
async def get_privacy_policy():
    """Get current privacy policy and data processing information"""
    
    return {
        "privacy_policy_version": "1.0",
        "last_updated": "2024-01-01",
        "data_categories": {
            "basic_profile": {
                "description": "Basic user information like name and email",
                "retention_period": "Account lifetime + 30 days",
                "legal_basis": "Contract performance"
            },
            "professional_data": {
                "description": "Resume, skills, experience, and career information",
                "retention_period": "Account lifetime + 1 year",
                "legal_basis": "Contract performance and legitimate interest"
            },
            "platform_data": {
                "description": "Data from connected platforms (GitHub, LinkedIn, etc.)",
                "retention_period": "Account lifetime + 30 days",
                "legal_basis": "Consent"
            },
            "behavioral_data": {
                "description": "Usage patterns and preferences",
                "retention_period": "2 years",
                "legal_basis": "Legitimate interest"
            },
            "generated_data": {
                "description": "AI-generated insights and recommendations",
                "retention_period": "Account lifetime + 6 months",
                "legal_basis": "Contract performance"
            }
        },
        "user_rights": [
            "Right of access (Article 15)",
            "Right to rectification (Article 16)",
            "Right to erasure (Article 17)",
            "Right to restrict processing (Article 18)",
            "Right to data portability (Article 20)",
            "Right to object (Article 21)"
        ],
        "contact_info": {
            "data_protection_officer": "privacy@aicareer.com",
            "support_email": "support@aicareer.com"
        }
    }