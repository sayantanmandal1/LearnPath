"""
Glassdoor scraper for job market data collection
"""
import re
from datetime import datetime
from typing import List, Optional, Any
from urllib.parse import urljoin, quote

from bs4 import BeautifulSoup

from .base_job_scraper import BaseJobScraper, JobSearchParams, ScrapedJob


class GlassdoorScraper(BaseJobScraper):
    """Glassdoor job scraper implementation"""
    
    def __init__(self, rate_limit_delay: float = 2.0):
        super().__init__(rate_limit_delay)
        self.base_url = "https://www.glassdoor.com"
        self.search_url = f"{self.base_url}/Job/jobs.htm"
        
        # Glassdoor-specific headers
        self.headers.update({
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
        })
    
    async def search_jobs(self, params: JobSearchParams) -> List[ScrapedJob]:
        """
        Search Glassdoor for job postings
        
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
            job_elements = soup.find_all('li', {'class': re.compile(r'react-job-listing|JobsList_jobListItem')})
            if not job_elements:
                # Try alternative selectors
                job_elements = soup.find_all('div', {'class': re.compile(r'job-search-card|jobContainer')})
            
            self.logger.info(f"Found {len(job_elements)} job listings on Glassdoor")
            
            for job_element in job_elements:
                try:
                    job_data = self._parse_job_listing(job_element, self.base_url)
                    if job_data:
                        jobs.append(job_data)
                except Exception as e:
                    self.logger.warning(f"Failed to parse Glassdoor job listing: {str(e)}")
                    continue
            
            # Get additional pages if needed
            if len(jobs) < params.limit:
                jobs.extend(await self._get_additional_pages(params, len(jobs)))
            
        except Exception as e:
            self.logger.error(f"Glassdoor job search failed: {str(e)}")
            raise
        
        return jobs[:params.limit]
    
    def _build_search_params(self, params: JobSearchParams) -> dict:
        """Build Glassdoor search parameters"""
        search_params = {
            'sc.keyword': params.keywords,
            'fromAge': str(params.posted_days),
            'p': 1
        }
        
        if params.location:
            search_params['locT'] = 'C'
            search_params['locId'] = params.location
        
        if params.remote:
            search_params['remoteWorkType'] = '1'
        
        if params.experience_level:
            exp_mapping = {
                'entry': '1',
                'mid': '2,3',
                'senior': '4,5',
                'executive': '6'
            }
            if params.experience_level in exp_mapping:
                search_params['seniorityType'] = exp_mapping[params.experience_level]
        
        if params.job_type:
            type_mapping = {
                'full-time': '1',
                'part-time': '2',
                'contract': '3',
                'internship': '4'
            }
            if params.job_type in type_mapping:
                search_params['jobType'] = type_mapping[params.job_type]
        
        if params.salary_min:
            search_params['minSalary'] = str(params.salary_min)
        
        return search_params
    
    def _parse_job_listing(self, job_element: Any, base_url: str) -> Optional[ScrapedJob]:
        """Parse individual Glassdoor job listing"""
        try:
            # Extract job title and link
            title_element = job_element.find('a', {'class': re.compile(r'jobLink|job-title')})
            if not title_element:
                title_element = job_element.find('a', href=re.compile(r'/job-listing/'))
            
            if not title_element:
                return None
            
            title = self._clean_text(title_element.get_text())
            job_url = urljoin(base_url, title_element.get('href', ''))
            
            # Extract job ID from URL
            job_id_match = re.search(r'jobListingId=(\d+)', job_url)
            if not job_id_match:
                job_id_match = re.search(r'/(\d+)\.htm', job_url)
            external_id = job_id_match.group(1) if job_id_match else job_url.split('/')[-1]
            
            # Extract company name
            company_element = job_element.find('div', {'class': re.compile(r'employerName')})
            if not company_element:
                company_element = job_element.find('span', {'class': re.compile(r'employer')})
            
            company = self._clean_text(company_element.get_text()) if company_element else "Unknown"
            
            # Extract location
            location_element = job_element.find('div', {'class': re.compile(r'jobLocation')})
            if not location_element:
                location_element = job_element.find('span', {'class': re.compile(r'location')})
            
            location = self._clean_text(location_element.get_text()) if location_element else None
            
            # Extract salary if available
            salary_element = job_element.find('div', {'class': re.compile(r'salaryEstimate')})
            if not salary_element:
                salary_element = job_element.find('span', {'class': re.compile(r'salary')})
            
            salary_info = {"min": None, "max": None, "currency": "USD", "period": None}
            if salary_element:
                salary_info = self._parse_salary(salary_element.get_text())
            
            # Extract job description snippet
            desc_element = job_element.find('div', {'class': re.compile(r'jobDescription|job-search-key-9hn0t')})
            description = self._clean_text(desc_element.get_text()) if desc_element else ""
            
            # Extract posted date
            posted_element = job_element.find('div', {'class': re.compile(r'jobAge')})
            if not posted_element:
                posted_element = job_element.find('span', {'class': re.compile(r'date')})
            
            posted_date = None
            if posted_element:
                posted_text = posted_element.get_text()
                posted_date = self._parse_date(posted_text)
            
            # Extract company rating if available
            rating_element = job_element.find('span', {'class': re.compile(r'rating')})
            company_rating = None
            if rating_element:
                rating_text = rating_element.get_text()
                rating_match = re.search(r'(\d+\.?\d*)', rating_text)
                if rating_match:
                    company_rating = float(rating_match.group(1))
            
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
                requirements=None,  # Would need additional request to get full description
                salary_min=salary_info["min"],
                salary_max=salary_info["max"],
                salary_currency=salary_info["currency"],
                salary_period=salary_info["period"],
                employment_type=None,
                experience_level=experience_level,
                remote_type=remote_type,
                posted_date=posted_date,
                expires_date=None,
                source_url=job_url,
                source="glassdoor",
                raw_data={
                    "glassdoor_job_id": external_id,
                    "company_rating": company_rating,
                    "search_result_html": str(job_element)
                }
            )
            
        except Exception as e:
            self.logger.warning(f"Failed to parse Glassdoor job: {str(e)}")
            return None
    
    async def _get_additional_pages(self, params: JobSearchParams, current_count: int) -> List[ScrapedJob]:
        """Get jobs from additional pages"""
        jobs = []
        page = 2  # Start from page 2
        
        while current_count + len(jobs) < params.limit and page <= 5:  # Limit to 5 pages
            try:
                search_params = self._build_search_params(params)
                search_params['p'] = page
                
                response = await self._make_request(self.search_url, search_params)
                soup = BeautifulSoup(response.text, 'html.parser')
                
                job_elements = soup.find_all('li', {'class': re.compile(r'react-job-listing|JobsList_jobListItem')})
                if not job_elements:
                    job_elements = soup.find_all('div', {'class': re.compile(r'job-search-card|jobContainer')})
                
                if not job_elements:
                    break
                
                for job_element in job_elements:
                    try:
                        job_data = self._parse_job_listing(job_element, self.base_url)
                        if job_data:
                            jobs.append(job_data)
                    except Exception as e:
                        self.logger.warning(f"Failed to parse job on page {page}: {str(e)}")
                        continue
                
                page += 1
                
            except Exception as e:
                self.logger.error(f"Failed to get Glassdoor page {page}: {str(e)}")
                break
        
        return jobs