"""
User repository for user-specific database operations
"""
from typing import Optional
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.user import User, RefreshToken
from app.schemas.auth import UserCreate, UserUpdate
from .base import BaseRepository


class UserRepository(BaseRepository[User, UserCreate, UserUpdate]):
    """Repository for User model operations"""
    
    def __init__(self):
        super().__init__(User)
    
    async def get_by_email(self, db: AsyncSession, email: str) -> Optional[User]:
        """
        Get user by email address
        
        Args:
            db: Database session
            email: User email address
            
        Returns:
            User instance or None
        """
        query = select(User).where(User.email == email)
        result = await db.execute(query)
        return result.scalar_one_or_none()
    
    async def get_active_user(self, db: AsyncSession, user_id: str) -> Optional[User]:
        """
        Get active user by ID
        
        Args:
            db: Database session
            user_id: User ID
            
        Returns:
            Active user instance or None
        """
        query = select(User).where(
            User.id == user_id,
            User.is_active == True
        )
        result = await db.execute(query)
        return result.scalar_one_or_none()
    
    async def update_last_login(self, db: AsyncSession, user_id: str) -> Optional[User]:
        """
        Update user's last login timestamp
        
        Args:
            db: Database session
            user_id: User ID
            
        Returns:
            Updated user instance or None
        """
        from datetime import datetime
        return await self.update(db, user_id, {"last_login": datetime.utcnow()})


class RefreshTokenRepository(BaseRepository[RefreshToken, dict, dict]):
    """Repository for RefreshToken model operations"""
    
    def __init__(self):
        super().__init__(RefreshToken)
    
    async def get_by_token_hash(
        self, 
        db: AsyncSession, 
        token_hash: str
    ) -> Optional[RefreshToken]:
        """
        Get refresh token by token hash
        
        Args:
            db: Database session
            token_hash: Hashed token value
            
        Returns:
            RefreshToken instance or None
        """
        query = select(RefreshToken).where(
            RefreshToken.token_hash == token_hash,
            RefreshToken.is_revoked == False
        )
        result = await db.execute(query)
        return result.scalar_one_or_none()
    
    async def revoke_user_tokens(self, db: AsyncSession, user_id: str) -> int:
        """
        Revoke all refresh tokens for a user
        
        Args:
            db: Database session
            user_id: User ID
            
        Returns:
            Number of tokens revoked
        """
        from sqlalchemy import update
        
        query = (
            update(RefreshToken)
            .where(
                RefreshToken.user_id == user_id,
                RefreshToken.is_revoked == False
            )
            .values(is_revoked=True)
        )
        
        result = await db.execute(query)
        await db.commit()
        return result.rowcount