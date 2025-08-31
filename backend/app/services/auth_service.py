"""
Authentication service for user management
"""
from datetime import datetime, timedelta
from typing import Optional

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import AuthenticationError, ConflictError, ValidationError
from app.core.security import (
    create_access_token,
    create_refresh_token,
    generate_token_hash,
    get_password_hash,
    verify_password,
    verify_token,
)
from app.models.user import RefreshToken, User
from app.schemas.auth import TokenResponse, UserRegister, UserResponse

logger = structlog.get_logger()


class AuthService:
    """Authentication service"""
    
    def __init__(self, db: AsyncSession):
        self.db = db
    
    async def register_user(self, user_data: UserRegister) -> UserResponse:
        """Register a new user"""
        # Check if user already exists
        stmt = select(User).where(User.email == user_data.email)
        result = await self.db.execute(stmt)
        existing_user = result.scalar_one_or_none()
        
        if existing_user:
            raise ConflictError("User with this email already exists")
        
        # Create new user
        hashed_password = get_password_hash(user_data.password)
        user = User(
            email=user_data.email,
            hashed_password=hashed_password,
            full_name=user_data.full_name,
        )
        
        self.db.add(user)
        await self.db.commit()
        await self.db.refresh(user)
        
        logger.info("User registered successfully", user_id=user.id, email=user.email)
        
        return UserResponse.model_validate(user)
    
    async def authenticate_user(self, email: str, password: str) -> User:
        """Authenticate user with email and password"""
        stmt = select(User).where(User.email == email)
        result = await self.db.execute(stmt)
        user = result.scalar_one_or_none()
        
        if not user or not verify_password(password, user.hashed_password):
            raise AuthenticationError("Invalid email or password")
        
        if not user.is_active:
            raise AuthenticationError("Account is deactivated")
        
        # Update last login
        user.last_login = datetime.utcnow()
        await self.db.commit()
        
        logger.info("User authenticated successfully", user_id=user.id, email=user.email)
        
        return user
    
    async def create_tokens(self, user: User, device_info: Optional[str] = None) -> TokenResponse:
        """Create access and refresh tokens for user"""
        # Create access token
        access_token = create_access_token(subject=user.id)
        
        # Create refresh token
        refresh_token = create_refresh_token(subject=user.id)
        
        # Store refresh token in database
        token_hash = generate_token_hash(refresh_token)
        refresh_token_obj = RefreshToken(
            user_id=user.id,
            token_hash=token_hash,
            expires_at=datetime.utcnow() + timedelta(days=7),
            device_info=device_info,
        )
        
        self.db.add(refresh_token_obj)
        await self.db.commit()
        
        return TokenResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in=30 * 60,  # 30 minutes
        )
    
    async def refresh_tokens(self, refresh_token: str) -> TokenResponse:
        """Refresh access token using refresh token"""
        # Verify refresh token
        user_id = verify_token(refresh_token, token_type="refresh")
        if not user_id:
            raise AuthenticationError("Invalid refresh token")
        
        # Check if refresh token exists and is not revoked
        token_hash = generate_token_hash(refresh_token)
        stmt = select(RefreshToken).where(
            RefreshToken.token_hash == token_hash,
            RefreshToken.user_id == user_id,
            RefreshToken.is_revoked == False,
            RefreshToken.expires_at > datetime.utcnow(),
        )
        result = await self.db.execute(stmt)
        stored_token = result.scalar_one_or_none()
        
        if not stored_token:
            raise AuthenticationError("Invalid or expired refresh token")
        
        # Get user
        stmt = select(User).where(User.id == user_id)
        result = await self.db.execute(stmt)
        user = result.scalar_one_or_none()
        
        if not user or not user.is_active:
            raise AuthenticationError("User not found or inactive")
        
        # Revoke old refresh token
        stored_token.is_revoked = True
        
        # Create new tokens
        tokens = await self.create_tokens(user)
        
        logger.info("Tokens refreshed successfully", user_id=user.id)
        
        return tokens
    
    async def revoke_refresh_token(self, refresh_token: str) -> bool:
        """Revoke a refresh token"""
        token_hash = generate_token_hash(refresh_token)
        stmt = select(RefreshToken).where(RefreshToken.token_hash == token_hash)
        result = await self.db.execute(stmt)
        stored_token = result.scalar_one_or_none()
        
        if stored_token:
            stored_token.is_revoked = True
            await self.db.commit()
            return True
        
        return False
    
    async def get_current_user(self, token: str) -> User:
        """Get current user from access token"""
        user_id = verify_token(token, token_type="access")
        if not user_id:
            raise AuthenticationError("Invalid access token")
        
        stmt = select(User).where(User.id == user_id)
        result = await self.db.execute(stmt)
        user = result.scalar_one_or_none()
        
        if not user or not user.is_active:
            raise AuthenticationError("User not found or inactive")
        
        return user