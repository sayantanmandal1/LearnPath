"""
Authentication endpoints
"""
from typing import Any

import structlog
from fastapi import APIRouter, Depends, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.api.dependencies import AuthServiceDep, CurrentUserDep
from app.schemas.auth import (
    TokenRefresh,
    TokenResponse,
    TokenWithUserResponse,
    UserLogin,
    UserRegister,
    UserResponse,
    UserWithProfileResponse,
)

logger = structlog.get_logger()
router = APIRouter()
security = HTTPBearer()


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register(
    user_data: UserRegister,
    auth_service: AuthServiceDep,
) -> Any:
    """Register a new user"""
    user = await auth_service.register_user(user_data)
    return user


@router.post("/login", response_model=TokenWithUserResponse)
async def login(
    user_data: UserLogin,
    request: Request,
    auth_service: AuthServiceDep,
) -> Any:
    """Login user and return tokens with user data"""
    # Authenticate user
    user = await auth_service.authenticate_user(user_data.email, user_data.password)
    
    # Get device info from request
    device_info = {
        "user_agent": request.headers.get("user-agent"),
        "ip_address": request.client.host if request.client else None,
    }
    
    # Create tokens with user data
    tokens = await auth_service.create_tokens_with_user(user, str(device_info))
    
    return tokens


@router.post("/refresh", response_model=TokenWithUserResponse)
async def refresh_token(
    token_data: TokenRefresh,
    auth_service: AuthServiceDep,
) -> Any:
    """Refresh access token with user data"""
    tokens = await auth_service.refresh_tokens_with_user(token_data.refresh_token)
    return tokens


@router.post("/logout")
async def logout(
    auth_service: AuthServiceDep,
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> Any:
    """Logout user and revoke refresh token"""
    # Note: In a real implementation, you might want to maintain a blacklist
    # of access tokens or use shorter expiration times
    return {"message": "Successfully logged out"}


@router.get("/me", response_model=UserWithProfileResponse)
async def get_current_user_info(
    current_user: CurrentUserDep,
    auth_service: AuthServiceDep,
) -> Any:
    """Get current user information with profile data"""
    # Get user profile if exists
    from sqlalchemy import select
    from app.models.profile import UserProfile
    
    profile_data = None
    stmt = select(UserProfile).where(UserProfile.user_id == current_user.id)
    result = await auth_service.db.execute(stmt)
    profile = result.scalar_one_or_none()
    
    if profile:
        profile_data = {
            "id": profile.id,
            "dream_job": profile.dream_job,
            "experience_years": profile.experience_years,
            "current_role": profile.current_role,
            "location": profile.location,
            "github_username": profile.github_username,
            "leetcode_id": profile.leetcode_id,
            "linkedin_url": profile.linkedin_url,
            "skills": profile.skills,
            "created_at": profile.created_at.isoformat() if profile.created_at else None,
            "updated_at": profile.updated_at.isoformat() if profile.updated_at else None,
        }
    
    return UserWithProfileResponse(
        id=current_user.id,
        email=current_user.email,
        full_name=current_user.full_name,
        is_active=current_user.is_active,
        is_verified=current_user.is_verified,
        created_at=current_user.created_at,
        last_login=current_user.last_login,
        profile=profile_data
    )


@router.get("/verify")
async def verify_token(
    current_user: CurrentUserDep,
) -> Any:
    """Verify if the current token is valid"""
    return {"valid": True, "user_id": current_user.id}