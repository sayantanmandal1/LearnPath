"""
Skill repository for skill-related database operations
"""
from typing import List, Optional
from sqlalchemy import select, and_, or_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models.skill import Skill, UserSkill, SkillCategory
from app.schemas.skill import SkillCreate, SkillUpdate, UserSkillCreate, UserSkillUpdate
from .base import BaseRepository


class SkillRepository(BaseRepository[Skill, SkillCreate, SkillUpdate]):
    """Repository for Skill model operations"""
    
    def __init__(self):
        super().__init__(Skill)
    
    async def get_by_name(self, db: AsyncSession, name: str) -> Optional[Skill]:
        """
        Get skill by name (case-insensitive)
        
        Args:
            db: Database session
            name: Skill name
            
        Returns:
            Skill instance or None
        """
        query = select(Skill).where(Skill.name.ilike(name))
        result = await db.execute(query)
        return result.scalar_one_or_none()
    
    async def search_skills(
        self,
        db: AsyncSession,
        query_text: str,
        category: Optional[str] = None,
        limit: int = 20
    ) -> List[Skill]:
        """
        Search skills by name or aliases
        
        Args:
            db: Database session
            query_text: Search text
            category: Optional category filter
            limit: Maximum number of results
            
        Returns:
            List of matching skills
        """
        query = select(Skill).where(
            and_(
                Skill.is_active == True,
                or_(
                    Skill.name.ilike(f"%{query_text}%"),
                    Skill.aliases.ilike(f"%{query_text}%")
                )
            )
        )
        
        if category:
            query = query.where(Skill.category == category)
        
        query = query.limit(limit)
        result = await db.execute(query)
        return result.scalars().all()
    
    async def get_by_category(
        self,
        db: AsyncSession,
        category: str,
        skip: int = 0,
        limit: int = 100
    ) -> List[Skill]:
        """
        Get skills by category
        
        Args:
            db: Database session
            category: Skill category
            skip: Number of records to skip
            limit: Maximum number of records to return
            
        Returns:
            List of skills in category
        """
        query = (
            select(Skill)
            .where(
                and_(
                    Skill.category == category,
                    Skill.is_active == True
                )
            )
            .offset(skip)
            .limit(limit)
        )
        
        result = await db.execute(query)
        return result.scalars().all()
    
    async def get_high_demand_skills(
        self,
        db: AsyncSession,
        min_demand: float = 0.7,
        limit: int = 50
    ) -> List[Skill]:
        """
        Get skills with high market demand
        
        Args:
            db: Database session
            min_demand: Minimum demand score
            limit: Maximum number of results
            
        Returns:
            List of high-demand skills
        """
        query = (
            select(Skill)
            .where(
                and_(
                    Skill.market_demand >= min_demand,
                    Skill.is_active == True
                )
            )
            .order_by(Skill.market_demand.desc())
            .limit(limit)
        )
        
        result = await db.execute(query)
        return result.scalars().all()
    
    async def find_or_create_skill(
        self,
        db: AsyncSession,
        name: str,
        category: str,
        subcategory: Optional[str] = None
    ) -> Skill:
        """
        Find existing skill or create new one
        
        Args:
            db: Database session
            name: Skill name
            category: Skill category
            subcategory: Optional subcategory
            
        Returns:
            Existing or newly created skill
        """
        # Try to find existing skill
        existing = await self.get_by_name(db, name)
        if existing:
            return existing
        
        # Create new skill
        skill_data = {
            "name": name,
            "category": category,
            "subcategory": subcategory
        }
        return await self.create(db, skill_data)


class UserSkillRepository(BaseRepository[UserSkill, UserSkillCreate, UserSkillUpdate]):
    """Repository for UserSkill model operations"""
    
    def __init__(self):
        super().__init__(UserSkill)
    
    async def get_user_skills(
        self,
        db: AsyncSession,
        user_id: str,
        include_skill_details: bool = True
    ) -> List[UserSkill]:
        """
        Get all skills for a user
        
        Args:
            db: Database session
            user_id: User ID
            include_skill_details: Whether to include skill relationship
            
        Returns:
            List of user skills
        """
        query = select(UserSkill).where(UserSkill.user_id == user_id)
        
        if include_skill_details:
            query = query.options(selectinload(UserSkill.skill))
        
        result = await db.execute(query)
        return result.scalars().all()
    
    async def get_user_skills_by_category(
        self,
        db: AsyncSession,
        user_id: str,
        category: str
    ) -> List[UserSkill]:
        """
        Get user skills filtered by category
        
        Args:
            db: Database session
            user_id: User ID
            category: Skill category
            
        Returns:
            List of user skills in category
        """
        query = (
            select(UserSkill)
            .join(Skill)
            .where(
                and_(
                    UserSkill.user_id == user_id,
                    Skill.category == category
                )
            )
            .options(selectinload(UserSkill.skill))
        )
        
        result = await db.execute(query)
        return result.scalars().all()
    
    async def get_user_skill(
        self,
        db: AsyncSession,
        user_id: str,
        skill_id: str
    ) -> Optional[UserSkill]:
        """
        Get specific user skill relationship
        
        Args:
            db: Database session
            user_id: User ID
            skill_id: Skill ID
            
        Returns:
            UserSkill instance or None
        """
        query = select(UserSkill).where(
            and_(
                UserSkill.user_id == user_id,
                UserSkill.skill_id == skill_id
            )
        )
        
        result = await db.execute(query)
        return result.scalar_one_or_none()
    
    async def upsert_user_skill(
        self,
        db: AsyncSession,
        user_id: str,
        skill_id: str,
        skill_data: dict
    ) -> UserSkill:
        """
        Create or update user skill relationship
        
        Args:
            db: Database session
            user_id: User ID
            skill_id: Skill ID
            skill_data: Skill relationship data
            
        Returns:
            UserSkill instance
        """
        existing = await self.get_user_skill(db, user_id, skill_id)
        
        if existing:
            # Update existing relationship
            return await self.update(db, existing.id, skill_data)
        else:
            # Create new relationship
            skill_data.update({
                "user_id": user_id,
                "skill_id": skill_id
            })
            return await self.create(db, skill_data)
    
    async def get_top_user_skills(
        self,
        db: AsyncSession,
        user_id: str,
        limit: int = 10
    ) -> List[UserSkill]:
        """
        Get top user skills by confidence score
        
        Args:
            db: Database session
            user_id: User ID
            limit: Maximum number of skills to return
            
        Returns:
            List of top user skills
        """
        query = (
            select(UserSkill)
            .where(UserSkill.user_id == user_id)
            .order_by(UserSkill.confidence_score.desc())
            .limit(limit)
            .options(selectinload(UserSkill.skill))
        )
        
        result = await db.execute(query)
        return result.scalars().all()


class SkillCategoryRepository(BaseRepository[SkillCategory, dict, dict]):
    """Repository for SkillCategory model operations"""
    
    def __init__(self):
        super().__init__(SkillCategory)
    
    async def get_root_categories(self, db: AsyncSession) -> List[SkillCategory]:
        """
        Get all root categories (no parent)
        
        Args:
            db: Database session
            
        Returns:
            List of root categories
        """
        query = (
            select(SkillCategory)
            .where(
                and_(
                    SkillCategory.parent_id.is_(None),
                    SkillCategory.is_active == True
                )
            )
            .order_by(SkillCategory.display_order)
        )
        
        result = await db.execute(query)
        return result.scalars().all()
    
    async def get_category_tree(self, db: AsyncSession) -> List[SkillCategory]:
        """
        Get complete category tree with children
        
        Args:
            db: Database session
            
        Returns:
            List of categories with children loaded
        """
        query = (
            select(SkillCategory)
            .where(SkillCategory.is_active == True)
            .options(selectinload(SkillCategory.children))
            .order_by(SkillCategory.display_order)
        )
        
        result = await db.execute(query)
        return result.scalars().all()