"""
Indeed scraper for job market data collection
"""
import re
from datetime import datetime
from typing import List, Optional, Any
from urllib.parse import urljoin, quote

from bs4 import BeautifulSoup

from .base_job_scraper import BaseJobScraper, JobSearchParams, ScrapedJob


class IndeedScraper(BaseJobScraper):
    """Indeed job scraper implementation"""
    
    def __init__(self, rate_limit_delay: float = 1.5):
        super().__init__(rate_limit_delay)
        self.base_url = "https://www.indeed.com"
        self.search_url = f"{self.base_url}/jobs"
        
        # Indeed-specific headers
        self.headers.update({
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        })
    
    async def search_jobs(self, params: JobSearchParams) -> List[ScrapedJob]:
        """
        Search Indeed for job postings
        
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
            job_elements = soup.find_all('div', {'class': re.compile(r'job_seen_beacon|jobsearch-SerpJobCard')})
            
            self.logger.info(f"Found {len(job_elements)} job listings on Indeed")
            
            for job_element in job_elements:
                try:
                    job_data = self._parse_job_listing(job_element, self.base_url)
                    if job_data:
                        jobs.append(job_data)
                except Exception as e:
                    self.logger.warning(f"Failed to parse Indeed job listing: {str(e)}")
                    continue
            
            # Get additional pages if needed
            if len(jobs) < params.limit:
                jobs.extend(await self._get_additional_pages(params, len(jobs)))
            
        except Exception as e:
            self.logger.error(f"Indeed job search failed: {str(e)}")
            raise
        
        return jobs[:params.limit]
    
    def _build_search_params(self, params: JobSearchParams) -> dict:
        """Build Indeed search parameters"""
        search_params = {
            'q': params.keywords,
            'fromage': str(params.posted_days),
            'start': 0
        }
        
        if params.location:
            search_params['l'] = params.location
        
        if params.remote:
            search_params['remotejob'] = '1'
        
        if params.salary_min:
            search_params['salary'] = f"${params.salary_min}+"
        
        if params.job_type:
            type_mapping = {
                'full-time': 'fulltime',
                'part-time': 'parttime',
                'contract': 'contract',
                'internship': 'internship'
            }
            if params.job_type in type_mapping:
                search_params['jt'] = type_mapping[params.job_type]
        
        return search_params
    
    def _parse_job_listing(self, job_element: Any, base_url: str) -> Optional[ScrapedJob]:
        """Parse individual Indeed job listing"""
        try:
            # Extract job title and link
            title_element = job_element.find('h2', {'class': re.compile(r'jobTitle')})
            if not title_element:
                title_element = job_element.find('a', {'data-jk': True})
            
            if not title_element:
                return None
            
            title_link = title_element.find('a') if title_element.name != 'a' else title_element
            if not title_link:
                return None
            
            title = self._clean_text(title_link.get_text())
            job_url = urljoin(base_url, title_link.get('href', ''))
            
            # Extract job ID
            job_id = title_link.get('data-jk') or job_element.get('data-jk')
            if not job_id:
                job_id_match = re.search(r'jk=([a-f0-9]+)', job_url)
                job_id = job_id_match.group(1) if job_id_match else job_url.split('/')[-1]
            
            # Extract company name
            company_element = job_element.find('span', {'class': re.compile(r'companyName')})
            if not company_element:
                company_element = job_element.find('a', {'data-testid': 'company-name'})
            
            company = self._clean_text(company_element.get_text()) if company_element else "Unknown"
            
            # Extract location
            location_element = job_element.find('div', {'class': re.compile(r'companyLocation')})
            if not location_element:
                location_element = job_element.find('div', {'data-testid': 'job-location'})
            
            location = self._clean_text(location_element.get_text()) if location_element else None
            
            # Extract salary if available
            salary_element = job_element.find('span', {'class': re.compile(r'salaryText')})
            if not salary_element:
                salary_element = job_element.find('div', {'class': re.compile(r'salary-snippet')})
            
            salary_info = {"min": None, "max": None, "currency": "USD", "period": None}
            if salary_element:
                salary_info = self._parse_salary(salary_element.get_text())
            
            # Extract job snippet/description
            snippet_element = job_element.find('div', {'class': re.compile(r'job-snippet')})
            if not snippet_element:
                snippet_element = job_element.find('div', {'class': re.compile(r'summary')})
            
            description = self._clean_text(snippet_element.get_text()) if snippet_element else ""
            
            # Extract posted date
            posted_element = job_element.find('span', {'class': re.compile(r'date')})
            posted_date = None
            if posted_element:
                posted_text = posted_element.get_text()
                posted_date = self._parse_date(posted_text)
            
            # Determine remote type and experience level
            full_text = f"{title} {company} {location or ''} {description}"
            remote_type = self._extract_remote_type(full_text)
            experience_level = self._extract_experience_level(full_text)
            
            return ScrapedJob(
                external_id=job_id,
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
                source="indeed",
                raw_data={
                    "indeed_job_id": job_id,
                    "search_result_html": str(job_element)
                }
            )
            
        except Exception as e:
            self.logger.warning(f"Failed to parse Indeed job: {str(e)}")
            return None
    
    async def _get_additional_pages(self, params: JobSearchParams, current_count: int) -> List[ScrapedJob]:
        """Get jobs from additional pages"""
        jobs = []
        page = 1
        
        while current_count + len(jobs) < params.limit and page < 5:  # Limit to 5 pages
            try:
                search_params = self._build_search_params(params)
                search_params['start'] = page * 10  # Indeed shows 10 jobs per page
                
                response = await self._make_request(self.search_url, search_params)
                soup = BeautifulSoup(response.text, 'html.parser')
                
                job_elements = soup.find_all('div', {'class': re.compile(r'job_seen_beacon|jobsearch-SerpJobCard')})
                
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
                self.logger.error(f"Failed to get Indeed page {page}: {str(e)}")
                break
        
        return jobs