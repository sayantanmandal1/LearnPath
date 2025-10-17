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
from app.models.profile import UserProfile
from app.schemas.auth import TokenResponse, UserRegister, UserResponse, UserWithProfileResponse, TokenWithUserResponse

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
    
    async def create_tokens_with_user(self, user: User, device_info: Optional[str] = None) -> TokenWithUserResponse:
        """Create access and refresh tokens for user with user data"""
        # Create tokens
        tokens = await self.create_tokens(user, device_info)
        
        # Get user profile if exists
        profile_data = None
        stmt = select(UserProfile).where(UserProfile.user_id == user.id)
        result = await self.db.execute(stmt)
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
        
        # Create user response with profile
        user_with_profile = UserWithProfileResponse(
            id=user.id,
            email=user.email,
            full_name=user.full_name,
            is_active=user.is_active,
            is_verified=user.is_verified,
            created_at=user.created_at,
            last_login=user.last_login,
            profile=profile_data
        )
        
        return TokenWithUserResponse(
            access_token=tokens.access_token,
            refresh_token=tokens.refresh_token,
            token_type=tokens.token_type,
            expires_in=tokens.expires_in,
            user=user_with_profile
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
    
    async def refresh_tokens_with_user(self, refresh_token: str) -> TokenWithUserResponse:
        """Refresh access token using refresh token with user data"""
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
        
        # Create new tokens with user data
        tokens = await self.create_tokens_with_user(user)
        
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
        """Get current user from access token. Auto-create user if not found and JWT is valid (Supabase)."""
        logger.info("Attempting to get current user from token")
        user_id = verify_token(token, token_type="access")
        logger.info(f"Token verification result: user_id = {user_id}")
        
        if not user_id:
            logger.error("Token verification failed - invalid access token")
            raise AuthenticationError("Invalid access token")

        stmt = select(User).where(User.id == user_id)
        result = await self.db.execute(stmt)
        user = result.scalar_one_or_none()
        logger.info(f"Database lookup result: user found = {user is not None}")

        if user and user.is_active:
            logger.info(f"Returning existing active user: {user.email}")
            return user

        # Try to decode JWT and extract user info
        from jose import jwt
        from app.core.config import settings
        payload = None
        try:
            logger.info("Attempting to decode JWT for user creation")
            # Use relaxed verification for Supabase JWTs
            options = {
                "verify_signature": False,
                "verify_aud": False,
                "verify_iss": False,
                "verify_exp": False
            }
            payload = jwt.decode(token, settings.SUPABASE_JWT_SECRET, algorithms=[settings.JWT_ALGORITHM], options=options)
            logger.info(f"JWT payload decoded successfully")
        except Exception as e:
            logger.error(f"JWT decoding failed: {str(e)}")
            
        if payload:
            email = payload.get("email")
            full_name = payload.get("user_metadata", {}).get("name") or payload.get("name") or email
            logger.info(f"Creating new user: email={email}, full_name={full_name}, id={user_id}")
            # Create new user (use placeholder password for Supabase users)
            new_user = User(
                id=user_id,
                email=email,
                hashed_password="supabase_managed_user",  # Skip bcrypt for now
                full_name=full_name,
                is_active=True,
                is_verified=True,
                created_at=datetime.utcnow(),
                last_login=datetime.utcnow(),
            )
            self.db.add(new_user)
            await self.db.commit()
            await self.db.refresh(new_user)
            logger.info("Auto-created user from Supabase JWT", user_id=new_user.id, email=new_user.email)
            return new_user

        logger.error("Could not create user - JWT payload invalid")
        raise AuthenticationError("User not found or inactive")