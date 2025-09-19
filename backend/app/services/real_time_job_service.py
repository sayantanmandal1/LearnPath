"""
Real-time job scraping and matching service.
Coordinates LinkedIn and Naukri scrapers with AI-powered matching.
"""

import asyncio
import logging
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
import json

from ..schemas.job import JobPosting, JobMatch
from ..models.profile import UserProfile
from .job_scrapers.linkedin_jobs_scraper import LinkedInJobsScraper
from .job_scrapers.naukri_scraper import NaukriScraper
from .job_matching_service import JobMatchingService
from .ai_analysis_service import AIAnalysisService
from ..core.exceptions import ScrapingError, MatchingError
from ..core.redis import get_redis_client

logger = logging.getLogger(__name__)

class RealTimeJobService:
    """
    Real-time job scraping and matching service for Indian tech market.
    Integrates LinkedIn and Naukri scrapers with AI-powered job matching.
    """
    
    def __init__(self):
        self.linkedin_scraper = LinkedInJobsScraper()
        self.naukri_scraper = NaukriScraper()
        self.ai_service = AIAnalysisService()
        self.matching_service = JobMatchingService(self.ai_service)
        self.redis_client = None
        
        # Indian tech cities prioritization
        self.priority_cities = [
            'bangalore', 'hyderabad', 'pune', 'mumbai', 
            'delhi_ncr', 'chennai', 'kolkata', 'ahmedabad'
        ]
        
        # Cache settings
        self.cache_ttl = 3600  # 1 hour cache for job data
        self.match_cache_ttl = 1800  # 30 minutes cache for matches

    async def initialize(self):
        """Initialize the service and its dependencies."""
        try:
            self.redis_client = await get_redis_client()
            logger.info("Real-time job service initialized successfully")
        except Exception as e:
            logger.warning(f"Redis not available, caching disabled: {str(e)}")

    async def get_indian_tech_jobs(
        self,
        role: str,
        preferred_cities: List[str] = None,
        experience_level: str = "mid",
        limit: int = 100,
        use_cache: bool = True
    ) -> List[JobPosting]:
        """
        Get tech jobs from Indian job portals.
        
        Args:
            role: Job role to search for
            preferred_cities: List of preferred cities
            experience_level: Experience level (entry, mid, senior)
            limit: Maximum number of jobs to return
            use_cache: Whether to use cached results
            
        Returns:
            List of JobPosting objects from Indian tech market
        """
        try:
            # Check cache first
            if use_cache and self.redis_client:
                cached_jobs = await self._get_cached_jobs(role, preferred_cities, experience_level)
                if cached_jobs:
                    logger.info(f"Returning {len(cached_jobs)} cached jobs")
                    return cached_jobs[:limit]
            
            # Set default cities if none provided
            if not preferred_cities:
                preferred_cities = self.priority_cities[:5]  # Top 5 cities
            
            # Scrape jobs from multiple sources concurrently
            linkedin_task = self._scrape_linkedin_jobs(role, preferred_cities, experience_level, limit // 2)
            naukri_task = self._scrape_naukri_jobs(role, preferred_cities, experience_level, limit // 2)
            
            linkedin_jobs, naukri_jobs = await asyncio.gather(
                linkedin_task, naukri_task, return_exceptions=True
            )
            
            # Handle exceptions
            if isinstance(linkedin_jobs, Exception):
                logger.error(f"LinkedIn scraping failed: {str(linkedin_jobs)}")
                linkedin_jobs = []
            
            if isinstance(naukri_jobs, Exception):
                logger.error(f"Naukri scraping failed: {str(naukri_jobs)}")
                naukri_jobs = []
            
            # Combine and deduplicate jobs
            all_jobs = linkedin_jobs + naukri_jobs
            unique_jobs = self._deduplicate_jobs(all_jobs)
            
            # Sort by posting date (newest first)
            unique_jobs.sort(key=lambda x: x.posted_date or datetime.min, reverse=True)
            
            # Cache results
            if self.redis_client and unique_jobs:
                await self._cache_jobs(role, preferred_cities, experience_level, unique_jobs)
            
            logger.info(f"Retrieved {len(unique_jobs)} unique jobs from Indian tech market")
            return unique_jobs[:limit]
            
        except Exception as e:
            logger.error(f"Error getting Indian tech jobs: {str(e)}")
            raise ScrapingError(f"Failed to get tech jobs: {str(e)}")

    async def get_personalized_job_matches(
        self,
        profile: UserProfile,
        preferred_cities: List[str] = None,
        target_role: str = None,
        limit: int = 50,
        use_cache: bool = True
    ) -> List[JobMatch]:
        """
        Get personalized job matches for a user profile.
        
        Args:
            profile: User profile to match against
            preferred_cities: User's preferred cities
            target_role: User's target role
            limit: Maximum number of matches to return
            use_cache: Whether to use cached results
            
        Returns:
            List of JobMatch objects sorted by compatibility score
        """
        try:
            # Check cache first
            if use_cache and self.redis_client:
                cached_matches = await self._get_cached_matches(profile.user_id, target_role)
                if cached_matches:
                    logger.info(f"Returning {len(cached_matches)} cached job matches")
                    return cached_matches[:limit]
            
            # Infer target role if not provided
            if not target_role:
                target_role = self._infer_target_role_from_profile(profile)
            
            # Get relevant jobs
            jobs = await self.get_indian_tech_jobs(
                role=target_role,
                preferred_cities=preferred_cities,
                experience_level=self._map_experience_level(profile),
                limit=limit * 2,  # Get more jobs for better matching
                use_cache=use_cache
            )
            
            if not jobs:
                logger.warning("No jobs found for matching")
                return []
            
            # Match jobs to profile
            matches = await self.matching_service.match_jobs_to_profile(
                jobs=jobs,
                profile=profile,
                preferred_locations=preferred_cities,
                target_role=target_role
            )
            
            # Cache results
            if self.redis_client and matches:
                await self._cache_matches(profile.user_id, target_role, matches)
            
            logger.info(f"Generated {len(matches)} job matches for user {profile.user_id}")
            return matches[:limit]
            
        except Exception as e:
            logger.error(f"Error getting personalized job matches: {str(e)}")
            raise MatchingError(f"Failed to get job matches: {str(e)}")

    async def _scrape_linkedin_jobs(
        self, 
        role: str, 
        cities: List[str], 
        experience_level: str, 
        limit: int
    ) -> List[JobPosting]:
        """Scrape jobs from LinkedIn."""
        try:
            # Convert to LinkedIn scraper format
            scraped_jobs = await self.linkedin_scraper.search_indian_tech_jobs(
                role=role,
                preferred_cities=cities,
                experience_level=experience_level,
                limit=limit
            )
            
            # Convert to JobPosting format
            job_postings = []
            for job in scraped_jobs:
                posting = self._convert_scraped_to_posting(job, "linkedin")
                if posting:
                    job_postings.append(posting)
            
            return job_postings
            
        except Exception as e:
            logger.error(f"LinkedIn scraping error: {str(e)}")
            return []

    async def _scrape_naukri_jobs(
        self, 
        role: str, 
        cities: List[str], 
        experience_level: str, 
        limit: int
    ) -> List[JobPosting]:
        """Scrape jobs from Naukri."""
        try:
            job_postings = await self.naukri_scraper.get_indian_tech_jobs(
                role=role,
                preferred_cities=cities,
                experience_level=self._map_experience_to_naukri(experience_level)
            )
            
            return job_postings[:limit]
            
        except Exception as e:
            logger.error(f"Naukri scraping error: {str(e)}")
            return []

    def _convert_scraped_to_posting(self, scraped_job, source: str) -> Optional[JobPosting]:
        """Convert scraped job to JobPosting format."""
        try:
            from ..schemas.job import SalaryRange
            
            # Create salary range if available
            salary_range = None
            if scraped_job.salary_min and scraped_job.salary_max:
                salary_range = SalaryRange(
                    min_amount=scraped_job.salary_min,
                    max_amount=scraped_job.salary_max,
                    currency=scraped_job.salary_currency or "INR",
                    period=scraped_job.salary_period or "annual"
                )
            
            return JobPosting(
                job_id=f"{source}_{scraped_job.external_id}",
                title=scraped_job.title,
                company=scraped_job.company,
                location=scraped_job.location or "India",
                description=scraped_job.description or "No description available",
                required_skills=self._extract_skills_from_scraped(scraped_job),
                experience_level=scraped_job.experience_level or "Not specified",
                salary_range=salary_range,
                posted_date=scraped_job.posted_date or datetime.now(),
                source=source,
                url=scraped_job.source_url or ""
            )
            
        except Exception as e:
            logger.warning(f"Error converting scraped job: {str(e)}")
            return None

    def _extract_skills_from_scraped(self, scraped_job) -> List[str]:
        """Extract skills from scraped job data."""
        skills = []
        
        # Extract from requirements if available
        if scraped_job.requirements:
            text = scraped_job.requirements.lower()
            
            # Common tech skills
            tech_skills = [
                'python', 'java', 'javascript', 'typescript', 'react', 'angular',
                'node.js', 'django', 'flask', 'spring', 'sql', 'mongodb',
                'aws', 'azure', 'docker', 'kubernetes', 'git', 'jenkins'
            ]
            
            for skill in tech_skills:
                if skill in text:
                    skills.append(skill)
        
        return skills

    def _deduplicate_jobs(self, jobs: List[JobPosting]) -> List[JobPosting]:
        """Remove duplicate jobs based on title and company."""
        seen = set()
        unique_jobs = []
        
        for job in jobs:
            # Create a key for deduplication
            key = f"{job.title.lower().strip()}_{job.company.lower().strip()}"
            
            if key not in seen:
                seen.add(key)
                unique_jobs.append(job)
        
        return unique_jobs

    def _infer_target_role_from_profile(self, profile: UserProfile) -> str:
        """Infer target role from user profile."""
        # Use most recent experience
        if profile.experience and profile.experience:
            return profile.experience[0].title
        
        # Use skills to infer role
        if profile.skills:
            skill_names = [skill.name.lower() for skill in profile.skills]
            
            if any(skill in skill_names for skill in ['python', 'django', 'flask']):
                return "Python Developer"
            elif any(skill in skill_names for skill in ['java', 'spring']):
                return "Java Developer"
            elif any(skill in skill_names for skill in ['javascript', 'react', 'angular']):
                return "Frontend Developer"
            elif any(skill in skill_names for skill in ['machine learning', 'data science']):
                return "Data Scientist"
        
        return "Software Developer"

    def _map_experience_level(self, profile: UserProfile) -> str:
        """Map profile experience to standard levels."""
        if not profile.experience:
            return "entry"
        
        # Estimate total years of experience
        total_years = len(profile.experience) * 2  # Rough estimate
        
        if total_years <= 2:
            return "entry"
        elif total_years <= 5:
            return "mid"
        else:
            return "senior"

    def _map_experience_to_naukri(self, experience_level: str) -> str:
        """Map experience level to Naukri format."""
        mapping = {
            "entry": "0-2",
            "mid": "2-5",
            "senior": "5-10"
        }
        return mapping.get(experience_level, "0-5")

    async def _get_cached_jobs(
        self, 
        role: str, 
        cities: List[str], 
        experience_level: str
    ) -> Optional[List[JobPosting]]:
        """Get cached jobs from Redis."""
        if not self.redis_client:
            return None
        
        try:
            cache_key = f"jobs:{role}:{'-'.join(sorted(cities or []))}:{experience_level}"
            cached_data = await self.redis_client.get(cache_key)
            
            if cached_data:
                jobs_data = json.loads(cached_data)
                jobs = []
                
                for job_data in jobs_data:
                    # Reconstruct JobPosting objects
                    job = JobPosting(**job_data)
                    jobs.append(job)
                
                return jobs
                
        except Exception as e:
            logger.warning(f"Error getting cached jobs: {str(e)}")
        
        return None

    async def _cache_jobs(
        self, 
        role: str, 
        cities: List[str], 
        experience_level: str, 
        jobs: List[JobPosting]
    ):
        """Cache jobs in Redis."""
        if not self.redis_client:
            return
        
        try:
            cache_key = f"jobs:{role}:{'-'.join(sorted(cities or []))}:{experience_level}"
            
            # Convert jobs to serializable format
            jobs_data = []
            for job in jobs:
                job_dict = job.dict()
                # Convert datetime to string
                if job_dict.get('posted_date'):
                    job_dict['posted_date'] = job_dict['posted_date'].isoformat()
                jobs_data.append(job_dict)
            
            await self.redis_client.setex(
                cache_key, 
                self.cache_ttl, 
                json.dumps(jobs_data, default=str)
            )
            
        except Exception as e:
            logger.warning(f"Error caching jobs: {str(e)}")

    async def _get_cached_matches(
        self, 
        user_id: str, 
        target_role: str
    ) -> Optional[List[JobMatch]]:
        """Get cached job matches from Redis."""
        if not self.redis_client:
            return None
        
        try:
            cache_key = f"matches:{user_id}:{target_role}"
            cached_data = await self.redis_client.get(cache_key)
            
            if cached_data:
                matches_data = json.loads(cached_data)
                matches = []
                
                for match_data in matches_data:
                    # Reconstruct JobMatch objects
                    match = JobMatch(**match_data)
                    matches.append(match)
                
                return matches
                
        except Exception as e:
            logger.warning(f"Error getting cached matches: {str(e)}")
        
        return None

    async def _cache_matches(
        self, 
        user_id: str, 
        target_role: str, 
        matches: List[JobMatch]
    ):
        """Cache job matches in Redis."""
        if not self.redis_client:
            return
        
        try:
            cache_key = f"matches:{user_id}:{target_role}"
            
            # Convert matches to serializable format
            matches_data = []
            for match in matches:
                match_dict = match.dict()
                # Handle nested objects and datetime conversion
                if match_dict.get('job_posting', {}).get('posted_date'):
                    match_dict['job_posting']['posted_date'] = match_dict['job_posting']['posted_date'].isoformat()
                matches_data.append(match_dict)
            
            await self.redis_client.setex(
                cache_key, 
                self.match_cache_ttl, 
                json.dumps(matches_data, default=str)
            )
            
        except Exception as e:
            logger.warning(f"Error caching matches: {str(e)}")

    async def refresh_job_cache(self, role: str, cities: List[str] = None):
        """Manually refresh job cache for a specific role and cities."""
        try:
            if not cities:
                cities = self.priority_cities
            
            # Clear existing cache
            if self.redis_client:
                for exp_level in ['entry', 'mid', 'senior']:
                    cache_key = f"jobs:{role}:{'-'.join(sorted(cities))}:{exp_level}"
                    await self.redis_client.delete(cache_key)
            
            # Fetch fresh data
            await self.get_indian_tech_jobs(
                role=role,
                preferred_cities=cities,
                use_cache=False
            )
            
            logger.info(f"Refreshed job cache for role: {role}")
            
        except Exception as e:
            logger.error(f"Error refreshing job cache: {str(e)}")

    async def get_job_market_insights(
        self, 
        role: str, 
        cities: List[str] = None
    ) -> Dict[str, Any]:
        """
        Get job market insights for a specific role and cities.
        
        Args:
            role: Job role to analyze
            cities: Cities to analyze (defaults to priority cities)
            
        Returns:
            Dictionary with market insights
        """
        try:
            if not cities:
                cities = self.priority_cities[:5]
            
            # Get recent jobs
            jobs = await self.get_indian_tech_jobs(
                role=role,
                preferred_cities=cities,
                limit=200,
                use_cache=True
            )
            
            if not jobs:
                return {"error": "No jobs found for analysis"}
            
            # Analyze job market
            insights = {
                "total_jobs": len(jobs),
                "cities_distribution": self._analyze_city_distribution(jobs),
                "companies": self._analyze_top_companies(jobs),
                "skills_demand": self._analyze_skills_demand(jobs),
                "experience_levels": self._analyze_experience_levels(jobs),
                "salary_insights": self._analyze_salary_ranges(jobs),
                "posting_trends": self._analyze_posting_trends(jobs)
            }
            
            return insights
            
        except Exception as e:
            logger.error(f"Error getting job market insights: {str(e)}")
            return {"error": f"Failed to get insights: {str(e)}"}

    def _analyze_city_distribution(self, jobs: List[JobPosting]) -> Dict[str, int]:
        """Analyze job distribution by city."""
        city_counts = {}
        
        for job in jobs:
            location = job.location.lower() if job.location else "unknown"
            
            # Map to standard city names
            for city, variations in self.naukri_scraper.indian_tech_cities.items():
                if any(var in location for var in variations):
                    city_counts[city] = city_counts.get(city, 0) + 1
                    break
            else:
                city_counts["other"] = city_counts.get("other", 0) + 1
        
        return dict(sorted(city_counts.items(), key=lambda x: x[1], reverse=True))

    def _analyze_top_companies(self, jobs: List[JobPosting]) -> List[Dict[str, Any]]:
        """Analyze top hiring companies."""
        company_counts = {}
        
        for job in jobs:
            company = job.company.strip()
            company_counts[company] = company_counts.get(company, 0) + 1
        
        # Return top 10 companies
        top_companies = sorted(company_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return [{"company": company, "job_count": count} for company, count in top_companies]

    def _analyze_skills_demand(self, jobs: List[JobPosting]) -> Dict[str, int]:
        """Analyze most demanded skills."""
        skill_counts = {}
        
        for job in jobs:
            for skill in job.required_skills:
                skill_lower = skill.lower().strip()
                skill_counts[skill_lower] = skill_counts.get(skill_lower, 0) + 1
        
        # Return top 15 skills
        top_skills = sorted(skill_counts.items(), key=lambda x: x[1], reverse=True)[:15]
        
        return dict(top_skills)

    def _analyze_experience_levels(self, jobs: List[JobPosting]) -> Dict[str, int]:
        """Analyze experience level distribution."""
        exp_counts = {"entry": 0, "mid": 0, "senior": 0, "not_specified": 0}
        
        for job in jobs:
            exp_level = job.experience_level.lower() if job.experience_level else ""
            
            if any(term in exp_level for term in ['junior', 'entry', '0-2', 'fresher']):
                exp_counts["entry"] += 1
            elif any(term in exp_level for term in ['mid', 'intermediate', '2-5', '3-6']):
                exp_counts["mid"] += 1
            elif any(term in exp_level for term in ['senior', 'lead', '5+', '7+']):
                exp_counts["senior"] += 1
            else:
                exp_counts["not_specified"] += 1
        
        return exp_counts

    def _analyze_salary_ranges(self, jobs: List[JobPosting]) -> Dict[str, Any]:
        """Analyze salary ranges."""
        salaries = []
        
        for job in jobs:
            if job.salary_range and job.salary_range.min_amount and job.salary_range.max_amount:
                avg_salary = (job.salary_range.min_amount + job.salary_range.max_amount) / 2
                salaries.append(avg_salary)
        
        if not salaries:
            return {"message": "No salary data available"}
        
        salaries.sort()
        n = len(salaries)
        
        return {
            "count": n,
            "min": min(salaries),
            "max": max(salaries),
            "median": salaries[n // 2] if n % 2 == 1 else (salaries[n // 2 - 1] + salaries[n // 2]) / 2,
            "average": sum(salaries) / n
        }

    def _analyze_posting_trends(self, jobs: List[JobPosting]) -> Dict[str, int]:
        """Analyze job posting trends by date."""
        date_counts = {}
        
        for job in jobs:
            if job.posted_date:
                date_str = job.posted_date.strftime("%Y-%m-%d")
                date_counts[date_str] = date_counts.get(date_str, 0) + 1
        
        return dict(sorted(date_counts.items()))