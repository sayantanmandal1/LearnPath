"""
Naukri.com job scraper for Indian tech opportunities.
Implements scraping with location filtering and tech role focus.
"""

import asyncio
import aiohttp
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
import re
from urllib.parse import urlencode, quote
import logging
from bs4 import BeautifulSoup

from .base_job_scraper import BaseJobScraper, JobSearchParams, ScrapedJob
from ...schemas.job import JobPosting, SalaryRange
from ...core.exceptions import ScrapingError

logger = logging.getLogger(__name__)

class NaukriScraper(BaseJobScraper):
    """Scraper for Naukri.com focused on Indian tech jobs."""
    
    def __init__(self):
        super().__init__()
        self.base_url = "https://www.naukri.com"
        self.search_url = f"{self.base_url}/jobs"
        
        # Indian tech cities for location filtering
        self.indian_tech_cities = {
            'bangalore': ['bangalore', 'bengaluru'],
            'hyderabad': ['hyderabad'],
            'pune': ['pune'],
            'chennai': ['chennai'],
            'mumbai': ['mumbai', 'mumbai suburban'],
            'delhi_ncr': ['delhi', 'gurgaon', 'noida', 'faridabad', 'ghaziabad'],
            'kolkata': ['kolkata'],
            'ahmedabad': ['ahmedabad'],
            'kochi': ['kochi', 'cochin'],
            'coimbatore': ['coimbatore']
        }
        
        # Tech-focused keywords for better filtering
        self.tech_keywords = [
            'software', 'developer', 'engineer', 'programmer', 'architect',
            'devops', 'data scientist', 'machine learning', 'ai', 'python',
            'java', 'javascript', 'react', 'node', 'full stack', 'backend',
            'frontend', 'mobile', 'android', 'ios', 'cloud', 'aws', 'azure'
        ]

    async def scrape_jobs(
        self, 
        role: str, 
        location: str = "bangalore", 
        experience_level: str = "0-5",
        limit: int = 50
    ) -> List[JobPosting]:
        """
        Scrape jobs from Naukri.com with Indian tech focus.
        
        Args:
            role: Job role/title to search for
            location: Indian city or region
            experience_level: Experience range (e.g., "0-5", "5-10")
            limit: Maximum number of jobs to return
            
        Returns:
            List of JobPosting objects
        """
        try:
            # Normalize location to match Naukri's format
            normalized_location = self._normalize_location(location)
            
            # Build search parameters
            search_params = {
                'k': role,
                'l': normalized_location,
                'experience': experience_level,
                'sort': 'date'
            }
            
            jobs = []
            page = 1
            
            while len(jobs) < limit and page <= 5:  # Limit to 5 pages
                page_jobs = await self._scrape_page(search_params, page)
                if not page_jobs:
                    break
                    
                # Filter for tech relevance
                tech_jobs = [job for job in page_jobs if self._is_tech_relevant(job)]
                jobs.extend(tech_jobs)
                
                page += 1
                await asyncio.sleep(2)  # Rate limiting
            
            return jobs[:limit]
            
        except Exception as e:
            logger.error(f"Error scraping Naukri jobs: {str(e)}")
            raise ScrapingError(f"Failed to scrape Naukri jobs: {str(e)}")

    async def _scrape_page(self, search_params: Dict, page: int) -> List[JobPosting]:
        """Scrape a single page of job results."""
        try:
            # Add page parameter
            params = search_params.copy()
            if page > 1:
                params['start'] = (page - 1) * 20
            
            url = f"{self.search_url}?{urlencode(params)}"
            
            async with aiohttp.ClientSession() as session:
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                    'Accept-Language': 'en-US,en;q=0.5',
                    'Accept-Encoding': 'gzip, deflate',
                    'Connection': 'keep-alive',
                }
                
                async with session.get(url, headers=headers, timeout=30) as response:
                    if response.status != 200:
                        logger.warning(f"Non-200 status code: {response.status}")
                        return []
                    
                    html = await response.text()
                    return self._parse_job_listings(html)
                    
        except Exception as e:
            logger.error(f"Error scraping page {page}: {str(e)}")
            return []

    def _parse_job_listings(self, html: str) -> List[JobPosting]:
        """Parse job listings from HTML content."""
        jobs = []
        
        try:
            soup = BeautifulSoup(html, 'html.parser')
            
            # Find job cards (Naukri uses various selectors)
            job_cards = soup.find_all(['div'], class_=re.compile(r'jobTuple|srp-tuple'))
            
            for card in job_cards:
                try:
                    job = self._parse_job_card(card)
                    if job:
                        jobs.append(job)
                except Exception as e:
                    logger.debug(f"Error parsing job card: {str(e)}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error parsing HTML: {str(e)}")
            
        return jobs

    def _parse_job_card(self, card) -> Optional[JobPosting]:
        """Parse individual job card to extract job details."""
        try:
            # Extract title
            title_elem = card.find(['a', 'span'], class_=re.compile(r'title|jobTitle'))
            if not title_elem:
                return None
            title = title_elem.get_text(strip=True)
            
            # Extract company
            company_elem = card.find(['a', 'span'], class_=re.compile(r'company|subTitle'))
            company = company_elem.get_text(strip=True) if company_elem else "Unknown"
            
            # Extract location
            location_elem = card.find(['span'], class_=re.compile(r'location|locationsContainer'))
            location = location_elem.get_text(strip=True) if location_elem else "India"
            
            # Extract experience
            exp_elem = card.find(['span'], class_=re.compile(r'experience|expwdth'))
            experience = exp_elem.get_text(strip=True) if exp_elem else "Not specified"
            
            # Extract salary if available
            salary_elem = card.find(['span'], class_=re.compile(r'salary|sal'))
            salary_range = None
            if salary_elem:
                salary_text = salary_elem.get_text(strip=True)
                salary_range = self._parse_salary(salary_text)
            
            # Extract job URL
            link_elem = card.find('a', href=True)
            job_url = ""
            if link_elem:
                href = link_elem['href']
                if href.startswith('/'):
                    job_url = f"{self.base_url}{href}"
                else:
                    job_url = href
            
            # Extract skills/description
            skills_elem = card.find(['span', 'div'], class_=re.compile(r'skill|tag'))
            skills = []
            if skills_elem:
                skills_text = skills_elem.get_text(strip=True)
                skills = [s.strip() for s in skills_text.split(',') if s.strip()]
            
            # Generate job ID
            job_id = f"naukri_{hash(f'{title}_{company}_{location}')}"
            
            return JobPosting(
                job_id=job_id,
                title=title,
                company=company,
                location=location,
                description=f"Experience: {experience}",
                required_skills=skills,
                experience_level=experience,
                salary_range=salary_range,
                posted_date=datetime.now(),
                source="naukri.com",
                url=job_url
            )
            
        except Exception as e:
            logger.debug(f"Error parsing job card: {str(e)}")
            return None

    def _parse_salary(self, salary_text: str) -> Optional[SalaryRange]:
        """Parse salary text to extract range."""
        try:
            # Remove common prefixes/suffixes
            salary_text = salary_text.lower().replace('₹', '').replace('rs', '').replace('lakh', '00000').replace('crore', '0000000')
            
            # Look for range patterns
            range_match = re.search(r'(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)', salary_text)
            if range_match:
                min_sal = float(range_match.group(1))
                max_sal = float(range_match.group(2))
                
                # Convert to annual if needed
                if 'month' in salary_text or 'pm' in salary_text:
                    min_sal *= 12
                    max_sal *= 12
                
                return SalaryRange(
                    min_amount=int(min_sal),
                    max_amount=int(max_sal),
                    currency="INR",
                    period="annual"
                )
                
        except Exception as e:
            logger.debug(f"Error parsing salary: {str(e)}")
            
        return None

    def _normalize_location(self, location: str) -> str:
        """Normalize location to match Naukri's location format."""
        location_lower = location.lower().strip()
        
        # Map common variations to Naukri format
        location_mapping = {
            'bengaluru': 'bangalore',
            'delhi ncr': 'delhi / ncr',
            'gurgaon': 'gurgaon',
            'noida': 'noida',
            'mumbai': 'mumbai',
            'pune': 'pune',
            'hyderabad': 'hyderabad',
            'chennai': 'chennai',
            'kolkata': 'kolkata'
        }
        
        return location_mapping.get(location_lower, location)

    def _is_tech_relevant(self, job: JobPosting) -> bool:
        """Check if job is relevant to tech roles."""
        text_to_check = f"{job.title} {job.description} {' '.join(job.required_skills)}".lower()
        
        # Check for tech keywords
        for keyword in self.tech_keywords:
            if keyword in text_to_check:
                return True
                
        return False

    async def get_indian_tech_jobs(
        self, 
        role: str, 
        preferred_cities: List[str] = None,
        experience_level: str = "0-5"
    ) -> List[JobPosting]:
        """
        Get tech jobs from major Indian cities.
        
        Args:
            role: Job role to search for
            preferred_cities: List of preferred cities (defaults to top tech cities)
            experience_level: Experience range
            
        Returns:
            List of JobPosting objects from Indian tech hubs
        """
        if not preferred_cities:
            preferred_cities = ['bangalore', 'hyderabad', 'pune', 'mumbai', 'delhi_ncr']
        
        all_jobs = []
        
        for city in preferred_cities:
            try:
                city_jobs = await self.scrape_jobs(
                    role=role,
                    location=city,
                    experience_level=experience_level,
                    limit=20
                )
                all_jobs.extend(city_jobs)
                await asyncio.sleep(3)  # Rate limiting between cities
                
            except Exception as e:
                logger.error(f"Error scraping jobs for {city}: {str(e)}")
                continue
        
        # Remove duplicates based on title and company
        seen = set()
        unique_jobs = []
        
        for job in all_jobs:
            job_key = f"{job.title}_{job.company}".lower()
            if job_key not in seen:
                seen.add(job_key)
                unique_jobs.append(job)
        
        return unique_jobs

    async def search_jobs(self, params: JobSearchParams) -> List[ScrapedJob]:
        """
        Implementation of abstract method from BaseJobScraper.
        Search for jobs based on parameters and return ScrapedJob objects.
        """
        try:
            # Convert JobSearchParams to our internal format
            job_postings = await self.scrape_jobs(
                role=params.keywords,
                location=params.location or "bangalore",
                experience_level="0-5",  # Default experience level
                limit=params.limit
            )
            
            # Convert JobPosting objects to ScrapedJob objects
            scraped_jobs = []
            for job in job_postings:
                scraped_job = ScrapedJob(
                    external_id=job.job_id,
                    title=job.title,
                    company=job.company,
                    location=job.location,
                    description=job.description,
                    requirements=", ".join(job.required_skills) if job.required_skills else None,
                    salary_min=job.salary_range.min_amount if job.salary_range else None,
                    salary_max=job.salary_range.max_amount if job.salary_range else None,
                    salary_currency=job.salary_range.currency if job.salary_range else "INR",
                    salary_period=job.salary_range.period if job.salary_range else None,
                    employment_type=None,  # Not extracted in current implementation
                    experience_level=job.experience_level,
                    remote_type=self._extract_remote_type(job.description),
                    posted_date=job.posted_date,
                    expires_date=None,  # Not available from Naukri
                    source_url=job.url,
                    source="naukri.com",
                    raw_data={"original_job": job.__dict__}
                )
                scraped_jobs.append(scraped_job)
            
            return scraped_jobs
            
        except Exception as e:
            logger.error(f"Error in search_jobs: {str(e)}")
            return []

    def _parse_job_listing(self, job_element: Any, base_url: str) -> Optional[ScrapedJob]:
        """
        Implementation of abstract method from BaseJobScraper.
        Parse individual job listing from HTML element.
        """
        try:
            # This method is used by the base class for parsing individual elements
            # We'll delegate to our existing _parse_job_card method
            job_posting = self._parse_job_card(job_element)
            
            if not job_posting:
                return None
            
            # Convert JobPosting to ScrapedJob
            scraped_job = ScrapedJob(
                external_id=job_posting.job_id,
                title=job_posting.title,
                company=job_posting.company,
                location=job_posting.location,
                description=job_posting.description,
                requirements=", ".join(job_posting.required_skills) if job_posting.required_skills else None,
                salary_min=job_posting.salary_range.min_amount if job_posting.salary_range else None,
                salary_max=job_posting.salary_range.max_amount if job_posting.salary_range else None,
                salary_currency=job_posting.salary_range.currency if job_posting.salary_range else "INR",
                salary_period=job_posting.salary_range.period if job_posting.salary_range else None,
                employment_type=None,
                experience_level=job_posting.experience_level,
                remote_type=self._extract_remote_type(job_posting.description),
                posted_date=job_posting.posted_date,
                expires_date=None,
                source_url=job_posting.url,
                source="naukri.com",
                raw_data={"original_job": job_posting.__dict__}
            )
            
            return scraped_job
            
        except Exception as e:
            logger.debug(f"Error parsing job listing: {str(e)}")
            return None