"""
Job repository for job posting database operations
"""
from typing import List, Optional
from datetime import datetime, timedelta
from sqlalchemy import select, and_, or_, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models.job import JobPosting, JobSkill, Company
from app.schemas.job import JobPostingCreate, JobPostingUpdate
from .base import BaseRepository


class JobRepository(BaseRepository[JobPosting, JobPostingCreate, JobPostingUpdate]):
    """Repository for JobPosting model operations"""
    
    def __init__(self):
        super().__init__(JobPosting)
    
    async def get_by_external_id(
        self,
        db: AsyncSession,
        external_id: str,
        source: str
    ) -> Optional[JobPosting]:
        """
        Get job posting by external ID and source
        
        Args:
            db: Database session
            external_id: External platform job ID
            source: Source platform name
            
        Returns:
            JobPosting instance or None
        """
        query = select(JobPosting).where(
            and_(
                JobPosting.external_id == external_id,
                JobPosting.source == source
            )
        )
        result = await db.execute(query)
        return result.scalar_one_or_none()
    
    async def search_jobs(
        self,
        db: AsyncSession,
        title: Optional[str] = None,
        company: Optional[str] = None,
        location: Optional[str] = None,
        remote_type: Optional[str] = None,
        experience_level: Optional[str] = None,
        min_salary: Optional[int] = None,
        max_salary: Optional[int] = None,
        skip: int = 0,
        limit: int = 100
    ) -> List[JobPosting]:
        """
        Search job postings with filters
        
        Args:
            db: Database session
            title: Job title filter
            company: Company name filter
            location: Location filter
            remote_type: Remote work type filter
            experience_level: Experience level filter
            min_salary: Minimum salary filter
            max_salary: Maximum salary filter
            skip: Number of records to skip
            limit: Maximum number of records to return
            
        Returns:
            List of matching job postings
        """
        query = select(JobPosting).where(JobPosting.is_active == True)
        
        if title:
            query = query.where(JobPosting.title.ilike(f"%{title}%"))
        
        if company:
            query = query.where(JobPosting.company.ilike(f"%{company}%"))
        
        if location:
            query = query.where(JobPosting.location.ilike(f"%{location}%"))
        
        if remote_type:
            query = query.where(JobPosting.remote_type == remote_type)
        
        if experience_level:
            query = query.where(JobPosting.experience_level == experience_level)
        
        if min_salary:
            query = query.where(JobPosting.salary_min >= min_salary)
        
        if max_salary:
            query = query.where(JobPosting.salary_max <= max_salary)
        
        query = query.offset(skip).limit(limit).order_by(JobPosting.posted_date.desc())
        
        result = await db.execute(query)
        return result.scalars().all()
    
    async def get_recent_jobs(
        self,
        db: AsyncSession,
        days: int = 7,
        limit: int = 100
    ) -> List[JobPosting]:
        """
        Get recently posted jobs
        
        Args:
            db: Database session
            days: Number of days to look back
            limit: Maximum number of jobs to return
            
        Returns:
            List of recent job postings
        """
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        query = (
            select(JobPosting)
            .where(
                and_(
                    JobPosting.is_active == True,
                    JobPosting.posted_date >= cutoff_date
                )
            )
            .order_by(JobPosting.posted_date.desc())
            .limit(limit)
        )
        
        result = await db.execute(query)
        return result.scalars().all()
    
    async def get_jobs_by_skills(
        self,
        db: AsyncSession,
        skill_ids: List[str],
        min_match_count: int = 1,
        limit: int = 50
    ) -> List[JobPosting]:
        """
        Get jobs that require specific skills
        
        Args:
            db: Database session
            skill_ids: List of skill IDs to match
            min_match_count: Minimum number of skills that must match
            limit: Maximum number of jobs to return
            
        Returns:
            List of matching job postings
        """
        # Subquery to count matching skills per job
        skill_count_subquery = (
            select(
                JobSkill.job_posting_id,
                func.count(JobSkill.skill_id).label('skill_match_count')
            )
            .where(JobSkill.skill_id.in_(skill_ids))
            .group_by(JobSkill.job_posting_id)
            .having(func.count(JobSkill.skill_id) >= min_match_count)
            .subquery()
        )
        
        query = (
            select(JobPosting)
            .join(skill_count_subquery, JobPosting.id == skill_count_subquery.c.job_posting_id)
            .where(JobPosting.is_active == True)
            .order_by(skill_count_subquery.c.skill_match_count.desc())
            .limit(limit)
        )
        
        result = await db.execute(query)
        return result.scalars().all()
    
    async def get_unprocessed_jobs(
        self,
        db: AsyncSession,
        limit: int = 100
    ) -> List[JobPosting]:
        """
        Get job postings that haven't been processed by NLP
        
        Args:
            db: Database session
            limit: Maximum number of jobs to return
            
        Returns:
            List of unprocessed job postings
        """
        query = (
            select(JobPosting)
            .where(
                and_(
                    JobPosting.is_active == True,
                    JobPosting.is_processed == False
                )
            )
            .order_by(JobPosting.created_at.asc())
            .limit(limit)
        )
        
        result = await db.execute(query)
        return result.scalars().all()
    
    async def mark_as_processed(
        self,
        db: AsyncSession,
        job_id: str,
        processed_skills: dict
    ) -> Optional[JobPosting]:
        """
        Mark job as processed and store extracted skills
        
        Args:
            db: Database session
            job_id: Job posting ID
            processed_skills: Extracted skills data
            
        Returns:
            Updated JobPosting instance or None
        """
        return await self.update(db, job_id, {
            "is_processed": True,
            "processed_skills": processed_skills
        })


class JobSkillRepository(BaseRepository[JobSkill, dict, dict]):
    """Repository for JobSkill model operations"""
    
    def __init__(self):
        super().__init__(JobSkill)
    
    async def get_job_skills(
        self,
        db: AsyncSession,
        job_posting_id: str,
        include_skill_details: bool = True
    ) -> List[JobSkill]:
        """
        Get all skills for a job posting
        
        Args:
            db: Database session
            job_posting_id: Job posting ID
            include_skill_details: Whether to include skill relationship
            
        Returns:
            List of job skills
        """
        query = select(JobSkill).where(JobSkill.job_posting_id == job_posting_id)
        
        if include_skill_details:
            query = query.options(selectinload(JobSkill.skill))
        
        result = await db.execute(query)
        return result.scalars().all()
    
    async def get_skill_demand_stats(
        self,
        db: AsyncSession,
        skill_id: str,
        days: int = 30
    ) -> dict:
        """
        Get demand statistics for a skill
        
        Args:
            db: Database session
            skill_id: Skill ID
            days: Number of days to analyze
            
        Returns:
            Dictionary with demand statistics
        """
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        # Count total job postings requiring this skill
        total_query = (
            select(func.count(JobSkill.id))
            .join(JobPosting)
            .where(
                and_(
                    JobSkill.skill_id == skill_id,
                    JobPosting.posted_date >= cutoff_date,
                    JobPosting.is_active == True
                )
            )
        )
        
        # Count by importance level
        importance_query = (
            select(
                JobSkill.importance,
                func.count(JobSkill.id).label('count')
            )
            .join(JobPosting)
            .where(
                and_(
                    JobSkill.skill_id == skill_id,
                    JobPosting.posted_date >= cutoff_date,
                    JobPosting.is_active == True
                )
            )
            .group_by(JobSkill.importance)
        )
        
        total_result = await db.execute(total_query)
        importance_result = await db.execute(importance_query)
        
        total_count = total_result.scalar()
        importance_breakdown = {row.importance: row.count for row in importance_result}
        
        return {
            "total_demand": total_count,
            "importance_breakdown": importance_breakdown,
            "period_days": days
        }


class CompanyRepository(BaseRepository[Company, dict, dict]):
    """Repository for Company model operations"""
    
    def __init__(self):
        super().__init__(Company)
    
    async def get_by_name(self, db: AsyncSession, name: str) -> Optional[Company]:
        """
        Get company by name (case-insensitive)
        
        Args:
            db: Database session
            name: Company name
            
        Returns:
            Company instance or None
        """
        query = select(Company).where(Company.name.ilike(name))
        result = await db.execute(query)
        return result.scalar_one_or_none()
    
    async def search_companies(
        self,
        db: AsyncSession,
        query_text: str,
        industry: Optional[str] = None,
        size: Optional[str] = None,
        limit: int = 20
    ) -> List[Company]:
        """
        Search companies by name
        
        Args:
            db: Database session
            query_text: Search text
            industry: Optional industry filter
            size: Optional company size filter
            limit: Maximum number of results
            
        Returns:
            List of matching companies
        """
        query = select(Company).where(
            and_(
                Company.is_active == True,
                Company.name.ilike(f"%{query_text}%")
            )
        )
        
        if industry:
            query = query.where(Company.industry == industry)
        
        if size:
            query = query.where(Company.size == size)
        
        query = query.limit(limit)
        result = await db.execute(query)
        return result.scalars().all()
    
    async def find_or_create_company(
        self,
        db: AsyncSession,
        name: str,
        domain: Optional[str] = None,
        industry: Optional[str] = None
    ) -> Company:
        """
        Find existing company or create new one
        
        Args:
            db: Database session
            name: Company name
            domain: Company domain
            industry: Company industry
            
        Returns:
            Existing or newly created company
        """
        # Try to find existing company
        existing = await self.get_by_name(db, name)
        if existing:
            return existing
        
        # Create new company
        company_data = {
            "name": name,
            "domain": domain,
            "industry": industry
        }
        return await self.create(db, company_data)