"""
Job scraper manager for coordinating multiple job platforms
"""
import asyncio
import logging
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
from dataclasses import asdict

from sqlalchemy.ext.asyncio import AsyncSession

from .base_job_scraper import JobSearchParams, ScrapedJob
from .indeed_scraper import IndeedScraper
from .linkedin_jobs_scraper import LinkedInJobsScraper
from .glassdoor_scraper import GlassdoorScraper
from app.repositories.job import JobRepository
from app.schemas.job import JobPostingCreate
from app.core.exceptions import ExternalServiceError as ExternalAPIError


logger = logging.getLogger(__name__)


class JobScraperManager:
    """Manager for coordinating job scraping across multiple platforms"""
    
    def __init__(self):
        self.scrapers = {
            'indeed': IndeedScraper(rate_limit_delay=1.5),
            'linkedin': LinkedInJobsScraper(rate_limit_delay=2.0),
            'glassdoor': GlassdoorScraper(rate_limit_delay=2.5)
        }
        self.job_repository = JobRepository()
    
    async def scrape_jobs(
        self,
        search_params: JobSearchParams,
        platforms: Optional[List[str]] = None,
        max_concurrent: int = 2
    ) -> Dict[str, List[ScrapedJob]]:
        """
        Scrape jobs from multiple platforms concurrently
        
        Args:
            search_params: Job search parameters
            platforms: List of platforms to scrape (default: all)
            max_concurrent: Maximum concurrent scrapers
            
        Returns:
            Dictionary mapping platform names to scraped jobs
        """
        if platforms is None:
            platforms = list(self.scrapers.keys())
        
        # Filter to available scrapers
        platforms = [p for p in platforms if p in self.scrapers]
        
        if not platforms:
            raise ValueError("No valid platforms specified")
        
        logger.info(f"Starting job scraping for platforms: {platforms}")
        
        # Create semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def scrape_platform(platform: str) -> tuple[str, List[ScrapedJob]]:
            """Scrape jobs from a single platform"""
            async with semaphore:
                try:
                    scraper = self.scrapers[platform]
                    async with scraper:
                        jobs = await scraper.search_jobs(search_params)
                        logger.info(f"Scraped {len(jobs)} jobs from {platform}")
                        return platform, jobs
                except Exception as e:
                    logger.error(f"Failed to scrape {platform}: {str(e)}")
                    return platform, []
        
        # Run scrapers concurrently
        tasks = [scrape_platform(platform) for platform in platforms]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        scraped_jobs = {}
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Scraping task failed: {str(result)}")
                continue
            
            platform, jobs = result
            scraped_jobs[platform] = jobs
        
        total_jobs = sum(len(jobs) for jobs in scraped_jobs.values())
        logger.info(f"Total jobs scraped: {total_jobs}")
        
        return scraped_jobs
    
    async def scrape_and_store_jobs(
        self,
        db: AsyncSession,
        search_params: JobSearchParams,
        platforms: Optional[List[str]] = None,
        deduplicate: bool = True
    ) -> Dict[str, int]:
        """
        Scrape jobs and store them in the database
        
        Args:
            db: Database session
            search_params: Job search parameters
            platforms: List of platforms to scrape
            deduplicate: Whether to deduplicate jobs
            
        Returns:
            Dictionary with statistics per platform
        """
        # Scrape jobs from all platforms
        scraped_jobs = await self.scrape_jobs(search_params, platforms)
        
        # Flatten all jobs
        all_jobs = []
        for platform, jobs in scraped_jobs.items():
            all_jobs.extend(jobs)
        
        # Deduplicate if requested
        if deduplicate:
            all_jobs = self._deduplicate_jobs(all_jobs)
            logger.info(f"After deduplication: {len(all_jobs)} unique jobs")
        
        # Store jobs in database
        stats = {}
        for platform in scraped_jobs.keys():
            platform_jobs = [job for job in all_jobs if job.source == platform]
            stored_count = await self._store_jobs(db, platform_jobs)
            stats[platform] = {
                'scraped': len(scraped_jobs[platform]),
                'stored': stored_count
            }
        
        return stats
    
    def _deduplicate_jobs(self, jobs: List[ScrapedJob]) -> List[ScrapedJob]:
        """
        Deduplicate jobs based on title, company, and location
        
        Args:
            jobs: List of scraped jobs
            
        Returns:
            Deduplicated list of jobs
        """
        seen = set()
        unique_jobs = []
        
        for job in jobs:
            # Create a key for deduplication
            key = (
                job.title.lower().strip(),
                job.company.lower().strip(),
                (job.location or "").lower().strip()
            )
            
            if key not in seen:
                seen.add(key)
                unique_jobs.append(job)
            else:
                logger.debug(f"Duplicate job found: {job.title} at {job.company}")
        
        return unique_jobs
    
    async def _store_jobs(self, db: AsyncSession, jobs: List[ScrapedJob]) -> int:
        """
        Store jobs in the database
        
        Args:
            db: Database session
            jobs: List of jobs to store
            
        Returns:
            Number of jobs successfully stored
        """
        stored_count = 0
        
        for job in jobs:
            try:
                # Check if job already exists
                existing = await self.job_repository.get_by_external_id(
                    db, job.external_id, job.source
                )
                
                if existing:
                    logger.debug(f"Job already exists: {job.external_id} from {job.source}")
                    continue
                
                # Convert to database schema
                job_data = JobPostingCreate(
                    external_id=job.external_id,
                    title=job.title,
                    company=job.company,
                    location=job.location,
                    remote_type=job.remote_type,
                    employment_type=job.employment_type,
                    experience_level=job.experience_level,
                    description=job.description,
                    requirements=job.requirements,
                    salary_min=job.salary_min,
                    salary_max=job.salary_max,
                    salary_currency=job.salary_currency,
                    salary_period=job.salary_period,
                    source=job.source,
                    source_url=job.source_url,
                    posted_date=job.posted_date,
                    expires_date=job.expires_date
                )
                
                # Store in database
                await self.job_repository.create(db, job_data)
                stored_count += 1
                
            except Exception as e:
                logger.error(f"Failed to store job {job.external_id}: {str(e)}")
                continue
        
        logger.info(f"Stored {stored_count} jobs in database")
        return stored_count
    
    async def get_trending_skills(
        self,
        db: AsyncSession,
        days: int = 30,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Get trending skills from recent job postings
        
        Args:
            db: Database session
            days: Number of days to analyze
            limit: Maximum number of skills to return
            
        Returns:
            List of trending skills with statistics
        """
        # Get recent jobs
        recent_jobs = await self.job_repository.get_recent_jobs(db, days, 1000)
        
        # Extract skills from job descriptions
        skill_counts = {}
        total_jobs = len(recent_jobs)
        
        for job in recent_jobs:
            if job.processed_skills:
                for skill, confidence in job.processed_skills.items():
                    if confidence > 0.7:  # Only high-confidence skills
                        skill_counts[skill] = skill_counts.get(skill, 0) + 1
        
        # Calculate trending scores
        trending_skills = []
        for skill, count in skill_counts.items():
            frequency = count / total_jobs if total_jobs > 0 else 0
            trending_skills.append({
                'skill': skill,
                'job_count': count,
                'frequency': frequency,
                'trend_score': frequency * count  # Simple trending score
            })
        
        # Sort by trend score and return top skills
        trending_skills.sort(key=lambda x: x['trend_score'], reverse=True)
        return trending_skills[:limit]
    
    async def get_salary_trends(
        self,
        db: AsyncSession,
        skill: Optional[str] = None,
        location: Optional[str] = None,
        days: int = 90
    ) -> Dict[str, Any]:
        """
        Get salary trends for specific skills or locations
        
        Args:
            db: Database session
            skill: Specific skill to analyze
            location: Specific location to analyze
            days: Number of days to analyze
            
        Returns:
            Salary trend analysis
        """
        # Build query filters
        filters = {}
        if location:
            filters['location'] = location
        
        # Get recent jobs with salary data
        recent_jobs = await self.job_repository.get_recent_jobs(db, days, 5000)
        
        # Filter jobs
        filtered_jobs = []
        for job in recent_jobs:
            if job.salary_min is None and job.salary_max is None:
                continue
            
            if location and location.lower() not in (job.location or "").lower():
                continue
            
            if skill:
                if not job.processed_skills or skill.lower() not in [
                    s.lower() for s in job.processed_skills.keys()
                ]:
                    continue
            
            filtered_jobs.append(job)
        
        if not filtered_jobs:
            return {
                'skill': skill,
                'location': location,
                'job_count': 0,
                'salary_stats': None
            }
        
        # Calculate salary statistics
        salaries = []
        for job in filtered_jobs:
            if job.salary_min and job.salary_max:
                avg_salary = (job.salary_min + job.salary_max) / 2
            elif job.salary_min:
                avg_salary = job.salary_min
            elif job.salary_max:
                avg_salary = job.salary_max
            else:
                continue
            
            # Normalize to yearly salary
            if job.salary_period == 'hourly':
                avg_salary *= 2080  # 40 hours/week * 52 weeks
            elif job.salary_period == 'monthly':
                avg_salary *= 12
            
            salaries.append(avg_salary)
        
        if not salaries:
            return {
                'skill': skill,
                'location': location,
                'job_count': len(filtered_jobs),
                'salary_stats': None
            }
        
        # Calculate statistics
        salaries.sort()
        n = len(salaries)
        
        return {
            'skill': skill,
            'location': location,
            'job_count': len(filtered_jobs),
            'salary_stats': {
                'min': min(salaries),
                'max': max(salaries),
                'median': salaries[n // 2],
                'mean': sum(salaries) / n,
                'percentile_25': salaries[n // 4],
                'percentile_75': salaries[3 * n // 4],
                'sample_size': n
            }
        }
    
    async def schedule_regular_scraping(
        self,
        db: AsyncSession,
        search_configs: List[Dict[str, Any]],
        interval_hours: int = 24
    ):
        """
        Schedule regular job scraping
        
        Args:
            db: Database session
            search_configs: List of search configurations
            interval_hours: Scraping interval in hours
        """
        logger.info(f"Starting scheduled scraping every {interval_hours} hours")
        
        while True:
            try:
                for config in search_configs:
                    search_params = JobSearchParams(**config)
                    stats = await self.scrape_and_store_jobs(db, search_params)
                    logger.info(f"Scheduled scraping completed: {stats}")
                
                # Wait for next interval
                await asyncio.sleep(interval_hours * 3600)
                
            except Exception as e:
                logger.error(f"Scheduled scraping failed: {str(e)}")
                # Wait 1 hour before retrying
                await asyncio.sleep(3600)