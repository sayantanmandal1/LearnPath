"""
LinkedIn Jobs scraper for job market data collection
"""
import re
from datetime import datetime
from typing import List, Optional, Any
from urllib.parse import urljoin, quote

from bs4 import BeautifulSoup

from .base_job_scraper import BaseJobScraper, JobSearchParams, ScrapedJob


class LinkedInJobsScraper(BaseJobScraper):
    """LinkedIn Jobs scraper implementation with Indian market focus"""
    
    def __init__(self, rate_limit_delay: float = 2.0):
        super().__init__(rate_limit_delay)
        self.base_url = "https://www.linkedin.com"
        self.search_url = f"{self.base_url}/jobs/search"
        
        # Indian tech cities mapping for LinkedIn location IDs
        self.indian_tech_cities = {
            'bangalore': 'Bengaluru, Karnataka, India',
            'bengaluru': 'Bengaluru, Karnataka, India',
            'hyderabad': 'Hyderabad, Telangana, India',
            'pune': 'Pune, Maharashtra, India',
            'chennai': 'Chennai, Tamil Nadu, India',
            'mumbai': 'Mumbai, Maharashtra, India',
            'delhi_ncr': 'Delhi, India',
            'gurgaon': 'Gurgaon, Haryana, India',
            'noida': 'Noida, Uttar Pradesh, India',
            'kolkata': 'Kolkata, West Bengal, India',
            'ahmedabad': 'Ahmedabad, Gujarat, India',
            'kochi': 'Kochi, Kerala, India'
        }
        
        # Tech-focused keywords for Indian market
        self.tech_keywords = [
            'software engineer', 'developer', 'programmer', 'architect',
            'devops', 'data scientist', 'machine learning', 'ai engineer',
            'python developer', 'java developer', 'javascript', 'react',
            'node.js', 'full stack', 'backend', 'frontend', 'mobile developer',
            'android developer', 'ios developer', 'cloud engineer', 'sre'
        ]
        
        # LinkedIn-specific headers
        self.headers.update({
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        })
    
    async def search_jobs(self, params: JobSearchParams) -> List[ScrapedJob]:
        """
        Search LinkedIn Jobs for job postings
        
        Args:
            params: Job search parameters
            
        Returns:
            List of scraped job data
        """
        jobs = []
        
        try:
            # Build search parameters
            search_params = self._build_search_params(params)
            
            # Get search results page
            response = await self._make_request(self.search_url, search_params)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find job listing elements
            job_elements = soup.find_all('div', {'class': re.compile(r'job-search-card|base-card')})
            
            self.logger.info(f"Found {len(job_elements)} job listings on LinkedIn")
            
            for job_element in job_elements:
                try:
                    job_data = await self._parse_job_listing(job_element, self.base_url)
                    if job_data:
                        jobs.append(job_data)
                except Exception as e:
                    self.logger.warning(f"Failed to parse LinkedIn job listing: {str(e)}")
                    continue
            
            # Get additional pages if needed
            if len(jobs) < params.limit:
                jobs.extend(await self._get_additional_pages(params, len(jobs)))
            
        except Exception as e:
            self.logger.error(f"LinkedIn job search failed: {str(e)}")
            raise
        
        return jobs[:params.limit]
    
    def _build_search_params(self, params: JobSearchParams) -> dict:
        """Build LinkedIn search parameters"""
        search_params = {
            'keywords': params.keywords,
            'f_TPR': f'r{params.posted_days * 86400}',  # Time posted in seconds
            'start': 0
        }
        
        if params.location:
            search_params['location'] = params.location
        
        if params.remote:
            search_params['f_WT'] = '2'  # Remote work filter
        
        if params.experience_level:
            exp_mapping = {
                'entry': '1',
                'mid': '2', 
                'senior': '3',
                'executive': '4'
            }
            if params.experience_level in exp_mapping:
                search_params['f_E'] = exp_mapping[params.experience_level]
        
        if params.job_type:
            type_mapping = {
                'full-time': 'F',
                'part-time': 'P',
                'contract': 'C',
                'internship': 'I'
            }
            if params.job_type in type_mapping:
                search_params['f_JT'] = type_mapping[params.job_type]
        
        return search_params
    
    async def _parse_job_listing(self, job_element: Any, base_url: str) -> Optional[ScrapedJob]:
        """Parse individual LinkedIn job listing"""
        try:
            # Extract job title and link
            title_element = job_element.find('h3', {'class': re.compile(r'base-search-card__title')})
            if not title_element:
                return None
            
            title_link = title_element.find('a')
            if not title_link:
                return None
            
            title = self._clean_text(title_link.get_text())
            job_url = urljoin(base_url, title_link.get('href', ''))
            
            # Extract job ID from URL
            job_id_match = re.search(r'/jobs/view/(\d+)', job_url)
            external_id = job_id_match.group(1) if job_id_match else job_url.split('/')[-1]
            
            # Extract company name
            company_element = job_element.find('h4', {'class': re.compile(r'base-search-card__subtitle')})
            company = self._clean_text(company_element.get_text()) if company_element else "Unknown"
            
            # Extract location
            location_element = job_element.find('span', {'class': re.compile(r'job-search-card__location')})
            location = self._clean_text(location_element.get_text()) if location_element else None
            
            # Extract posted date
            posted_element = job_element.find('time')
            posted_date = None
            if posted_element:
                posted_text = posted_element.get('datetime') or posted_element.get_text()
                posted_date = self._parse_date(posted_text)
            
            # Extract salary if available
            salary_element = job_element.find('span', {'class': re.compile(r'job-search-card__salary')})
            salary_info = {"min": None, "max": None, "currency": "USD", "period": None}
            if salary_element:
                salary_info = self._parse_salary(salary_element.get_text())
            
            # Get job description (requires additional request)
            description, requirements = await self._get_job_details(job_url)
            
            # Determine remote type and experience level
            full_text = f"{title} {company} {location or ''} {description}"
            remote_type = self._extract_remote_type(full_text)
            experience_level = self._extract_experience_level(full_text)
            
            return ScrapedJob(
                external_id=external_id,
                title=title,
                company=company,
                location=location,
                description=description,
                requirements=requirements,
                salary_min=salary_info["min"],
                salary_max=salary_info["max"],
                salary_currency=salary_info["currency"],
                salary_period=salary_info["period"],
                employment_type=None,  # Not easily extractable from search results
                experience_level=experience_level,
                remote_type=remote_type,
                posted_date=posted_date,
                expires_date=None,
                source_url=job_url,
                source="linkedin",
                raw_data={
                    "linkedin_job_id": external_id,
                    "search_result_html": str(job_element)
                }
            )
            
        except Exception as e:
            self.logger.warning(f"Failed to parse LinkedIn job: {str(e)}")
            return None
    
    async def _get_job_details(self, job_url: str) -> tuple[str, Optional[str]]:
        """
        Get detailed job description from job page
        
        Args:
            job_url: URL of the job posting
            
        Returns:
            Tuple of (description, requirements)
        """
        try:
            response = await self._make_request(job_url)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find job description
            desc_element = soup.find('div', {'class': re.compile(r'show-more-less-html__markup')})
            if not desc_element:
                desc_element = soup.find('div', {'class': re.compile(r'description__text')})
            
            if desc_element:
                description = self._clean_text(desc_element.get_text())
                
                # Try to separate requirements from description
                desc_lower = description.lower()
                requirements = None
                
                # Look for requirements section
                req_indicators = ['requirements:', 'qualifications:', 'what you need:', 'skills:']
                for indicator in req_indicators:
                    if indicator in desc_lower:
                        req_start = desc_lower.find(indicator)
                        # Find next major section or end of text
                        next_sections = ['responsibilities:', 'what you\'ll do:', 'benefits:', 'about']
                        req_end = len(description)
                        for section in next_sections:
                            section_pos = desc_lower.find(section, req_start)
                            if section_pos != -1:
                                req_end = min(req_end, section_pos)
                        
                        requirements = description[req_start:req_end].strip()
                        break
                
                return description, requirements
            
            return "No description available", None
            
        except Exception as e:
            self.logger.warning(f"Failed to get LinkedIn job details for {job_url}: {str(e)}")
            return "Description unavailable", None
    
    async def _get_additional_pages(self, params: JobSearchParams, current_count: int) -> List[ScrapedJob]:
        """Get jobs from additional pages"""
        jobs = []
        page = 1
        
        while current_count + len(jobs) < params.limit and page < 5:  # Limit to 5 pages
            try:
                search_params = self._build_search_params(params)
                search_params['start'] = page * 25  # LinkedIn shows 25 jobs per page
                
                response = await self._make_request(self.search_url, search_params)
                soup = BeautifulSoup(response.text, 'html.parser')
                
                job_elements = soup.find_all('div', {'class': re.compile(r'job-search-card|base-card')})
                
                if not job_elements:
                    break
                
                for job_element in job_elements:
                    try:
                        job_data = await self._parse_job_listing(job_element, self.base_url)
                        if job_data:
                            jobs.append(job_data)
                    except Exception as e:
                        self.logger.warning(f"Failed to parse job on page {page}: {str(e)}")
                        continue
                
                page += 1
                
            except Exception as e:
                self.logger.error(f"Failed to get LinkedIn page {page}: {str(e)}")
                break
        
        return jobs
    
    async def search_indian_tech_jobs(
        self, 
        role: str, 
        preferred_cities: List[str] = None,
        experience_level: str = "mid",
        limit: int = 50
    ) -> List[ScrapedJob]:
        """
        Search for tech jobs specifically in Indian cities.
        
        Args:
            role: Job role/title to search for
            preferred_cities: List of preferred Indian cities
            experience_level: Experience level (entry, mid, senior)
            limit: Maximum number of jobs to return
            
        Returns:
            List of ScrapedJob objects from Indian tech hubs
        """
        if not preferred_cities:
            preferred_cities = ['bangalore', 'hyderabad', 'pune', 'mumbai', 'delhi_ncr']
        
        all_jobs = []
        
        for city in preferred_cities:
            try:
                # Map city to LinkedIn location format
                linkedin_location = self.indian_tech_cities.get(city.lower(), city)
                
                # Create search parameters for this city
                params = JobSearchParams(
                    keywords=role,
                    location=linkedin_location,
                    experience_level=experience_level,
                    limit=limit // len(preferred_cities) + 5,  # Distribute limit across cities
                    posted_days=7,  # Recent jobs only
                    remote=False  # Focus on local opportunities
                )
                
                city_jobs = await self.search_jobs(params)
                
                # Filter for tech relevance
                tech_jobs = [job for job in city_jobs if self._is_tech_relevant(job)]
                all_jobs.extend(tech_jobs)
                
                # Rate limiting between cities
                await self._rate_limit()
                
            except Exception as e:
                self.logger.error(f"Error scraping LinkedIn jobs for {city}: {str(e)}")
                continue
        
        # Remove duplicates and prioritize by relevance
        unique_jobs = self._deduplicate_jobs(all_jobs)
        return unique_jobs[:limit]
    
    def _is_tech_relevant(self, job: ScrapedJob) -> bool:
        """Check if job is relevant to tech roles."""
        text_to_check = f"{job.title} {job.description or ''} {job.requirements or ''}".lower()
        
        # Check for tech keywords
        for keyword in self.tech_keywords:
            if keyword in text_to_check:
                return True
        
        # Check for common tech terms
        tech_terms = [
            'software', 'developer', 'engineer', 'programming', 'coding',
            'python', 'java', 'javascript', 'react', 'node', 'aws', 'cloud',
            'api', 'database', 'sql', 'nosql', 'microservices', 'devops'
        ]
        
        for term in tech_terms:
            if term in text_to_check:
                return True
                
        return False
    
    def _deduplicate_jobs(self, jobs: List[ScrapedJob]) -> List[ScrapedJob]:
        """Remove duplicate jobs based on title and company."""
        seen = set()
        unique_jobs = []
        
        for job in jobs:
            job_key = f"{job.title}_{job.company}".lower().strip()
            if job_key not in seen:
                seen.add(job_key)
                unique_jobs.append(job)
        
        return unique_jobs