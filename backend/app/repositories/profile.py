"""
Profile repository for user profile database operations
"""
from typing import Optional
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models.profile import UserProfile
from app.schemas.profile import ProfileCreate, ProfileUpdate
from .base import BaseRepository


class ProfileRepository(BaseRepository[UserProfile, ProfileCreate, ProfileUpdate]):
    """Repository for UserProfile model operations"""
    
    def __init__(self):
        super().__init__(UserProfile)
    
    async def get_by_user_id(
        self, 
        db: AsyncSession, 
        user_id: str,
        include_user: bool = False
    ) -> Optional[UserProfile]:
        """
        Get user profile by user ID
        
        Args:
            db: Database session
            user_id: User ID
            include_user: Whether to include user relationship
            
        Returns:
            UserProfile instance or None
        """
        query = select(UserProfile).where(UserProfile.user_id == user_id)
        
        if include_user:
            query = query.options(selectinload(UserProfile.user))
        
        result = await db.execute(query)
        return result.scalar_one_or_none()
    
    async def get_profiles_by_dream_job(
        self,
        db: AsyncSession,
        dream_job: str,
        skip: int = 0,
        limit: int = 100
    ) -> list[UserProfile]:
        """
        Get profiles by dream job for similarity analysis
        
        Args:
            db: Database session
            dream_job: Target dream job
            skip: Number of records to skip
            limit: Maximum number of records to return
            
        Returns:
            List of UserProfile instances
        """
        query = (
            select(UserProfile)
            .where(UserProfile.dream_job.ilike(f"%{dream_job}%"))
            .offset(skip)
            .limit(limit)
        )
        
        result = await db.execute(query)
        return result.scalars().all()
    
    async def get_profiles_with_platform_data(
        self,
        db: AsyncSession,
        platform: str,
        skip: int = 0,
        limit: int = 100
    ) -> list[UserProfile]:
        """
        Get profiles that have data from a specific platform
        
        Args:
            db: Database session
            platform: Platform name (github, leetcode, linkedin)
            skip: Number of records to skip
            limit: Maximum number of records to return
            
        Returns:
            List of UserProfile instances
        """
        platform_field = f"{platform}_username" if platform != "linkedin" else f"{platform}_url"
        
        query = select(UserProfile).where(
            getattr(UserProfile, platform_field).is_not(None)
        ).offset(skip).limit(limit)
        
        result = await db.execute(query)
        return result.scalars().all()
    
    async def update_platform_data(
        self,
        db: AsyncSession,
        user_id: str,
        platform_data: dict
    ) -> Optional[UserProfile]:
        """
        Update platform data for a user profile
        
        Args:
            db: Database session
            user_id: User ID
            platform_data: Platform data to merge
            
        Returns:
            Updated UserProfile instance or None
        """
        profile = await self.get_by_user_id(db, user_id)
        if not profile:
            return None
        
        # Merge platform data
        current_data = profile.platform_data or {}
        current_data.update(platform_data)
        
        from datetime import datetime
        return await self.update(db, profile.id, {
            "platform_data": current_data,
            "data_last_updated": datetime.utcnow()
        })
    
    async def update_skills(
        self,
        db: AsyncSession,
        user_id: str,
        skills: dict
    ) -> Optional[UserProfile]:
        """
        Update skills for a user profile
        
        Args:
            db: Database session
            user_id: User ID
            skills: Skills dictionary with confidence scores
            
        Returns:
            Updated UserProfile instance or None
        """
        profile = await self.get_by_user_id(db, user_id)
        if not profile:
            return None
        
        # Merge skills data
        current_skills = profile.skills or {}
        current_skills.update(skills)
        
        return await self.update(db, profile.id, {"skills": current_skills})