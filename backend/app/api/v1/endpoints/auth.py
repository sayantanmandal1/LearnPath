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
    UserLogin,
    UserRegister,
    UserResponse,
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


@router.post("/login", response_model=TokenResponse)
async def login(
    user_data: UserLogin,
    request: Request,
    auth_service: AuthServiceDep,
) -> Any:
    """Login user and return tokens"""
    # Authenticate user
    user = await auth_service.authenticate_user(user_data.email, user_data.password)
    
    # Get device info from request
    device_info = {
        "user_agent": request.headers.get("user-agent"),
        "ip_address": request.client.host if request.client else None,
    }
    
    # Create tokens
    tokens = await auth_service.create_tokens(user, str(device_info))
    
    return tokens


@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(
    token_data: TokenRefresh,
    auth_service: AuthServiceDep,
) -> Any:
    """Refresh access token"""
    tokens = await auth_service.refresh_tokens(token_data.refresh_token)
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


@router.get("/me", response_model=UserResponse)
async def get_current_user_info(
    current_user: CurrentUserDep,
) -> Any:
    """Get current user information"""
    return UserResponse.model_validate(current_user)


@router.get("/verify")
async def verify_token(
    current_user: CurrentUserDep,
) -> Any:
    """Verify if the current token is valid"""
    return {"valid": True, "user_id": current_user.id}