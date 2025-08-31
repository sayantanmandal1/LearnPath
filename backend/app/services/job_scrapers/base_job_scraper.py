"""
Base job scraper class with common functionality
"""
import asyncio
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
from dataclasses import dataclass

import httpx
from bs4 import BeautifulSoup

from app.core.exceptions import ExternalServiceError as ExternalAPIError


@dataclass
class JobSearchParams:
    """Job search parameters"""
    keywords: str
    location: Optional[str] = None
    remote: bool = False
    experience_level: Optional[str] = None
    job_type: Optional[str] = None
    salary_min: Optional[int] = None
    salary_max: Optional[int] = None
    posted_days: int = 7
    limit: int = 50


@dataclass
class ScrapedJob:
    """Scraped job data structure"""
    external_id: str
    title: str
    company: str
    location: Optional[str]
    description: str
    requirements: Optional[str]
    salary_min: Optional[int]
    salary_max: Optional[int]
    salary_currency: str
    salary_period: Optional[str]
    employment_type: Optional[str]
    experience_level: Optional[str]
    remote_type: Optional[str]
    posted_date: Optional[datetime]
    expires_date: Optional[datetime]
    source_url: str
    source: str
    raw_data: Dict[str, Any]


class BaseJobScraper(ABC):
    """Base class for job scrapers"""
    
    def __init__(self, rate_limit_delay: float = 1.0):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.rate_limit_delay = rate_limit_delay
        self.session: Optional[httpx.AsyncClient] = None
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
    
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = httpx.AsyncClient(
            headers=self.headers,
            timeout=30.0,
            follow_redirects=True
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.aclose()
    
    @abstractmethod
    async def search_jobs(self, params: JobSearchParams) -> List[ScrapedJob]:
        """
        Search for jobs based on parameters
        
        Args:
            params: Job search parameters
            
        Returns:
            List of scraped job data
        """
        pass
    
    @abstractmethod
    def _parse_job_listing(self, job_element: Any, base_url: str) -> Optional[ScrapedJob]:
        """
        Parse individual job listing from HTML element
        
        Args:
            job_element: HTML element containing job data
            base_url: Base URL for constructing full URLs
            
        Returns:
            Parsed job data or None if parsing fails
        """
        pass
    
    async def _make_request(self, url: str, params: Optional[Dict] = None) -> httpx.Response:
        """
        Make HTTP request with rate limiting and error handling
        
        Args:
            url: Request URL
            params: Query parameters
            
        Returns:
            HTTP response
            
        Raises:
            ExternalAPIError: If request fails
        """
        if not self.session:
            raise ExternalAPIError("Session not initialized. Use async context manager.")
        
        try:
            # Rate limiting
            await asyncio.sleep(self.rate_limit_delay)
            
            response = await self.session.get(url, params=params)
            response.raise_for_status()
            
            self.logger.debug(f"Successfully fetched {url}")
            return response
            
        except httpx.HTTPStatusError as e:
            self.logger.error(f"HTTP error for {url}: {e.response.status_code}")
            raise ExternalAPIError(f"HTTP {e.response.status_code} error for {url}")
        except httpx.RequestError as e:
            self.logger.error(f"Request error for {url}: {str(e)}")
            raise ExternalAPIError(f"Request failed for {url}: {str(e)}")
    
    def _parse_salary(self, salary_text: str) -> Dict[str, Any]:
        """
        Parse salary information from text
        
        Args:
            salary_text: Raw salary text
            
        Returns:
            Dictionary with parsed salary data
        """
        import re
        
        if not salary_text:
            return {"min": None, "max": None, "currency": "USD", "period": None}
        
        # Remove common prefixes and clean text
        salary_text = re.sub(r'^(salary|pay|compensation):\s*', '', salary_text.lower())
        salary_text = re.sub(r'[,$]', '', salary_text)
        
        # Extract currency
        currency = "USD"
        if "€" in salary_text or "eur" in salary_text:
            currency = "EUR"
        elif "£" in salary_text or "gbp" in salary_text:
            currency = "GBP"
        
        # Extract period
        period = None
        if any(word in salary_text for word in ["year", "annual", "yearly"]):
            period = "yearly"
        elif any(word in salary_text for word in ["month", "monthly"]):
            period = "monthly"
        elif any(word in salary_text for word in ["hour", "hourly"]):
            period = "hourly"
        
        # Extract numbers
        numbers = re.findall(r'\d+(?:k|000)?', salary_text)
        if not numbers:
            return {"min": None, "max": None, "currency": currency, "period": period}
        
        # Convert k notation to full numbers
        parsed_numbers = []
        for num in numbers:
            if num.endswith('k'):
                parsed_numbers.append(int(num[:-1]) * 1000)
            elif num.endswith('000'):
                parsed_numbers.append(int(num))
            else:
                parsed_numbers.append(int(num))
        
        if len(parsed_numbers) == 1:
            return {"min": parsed_numbers[0], "max": parsed_numbers[0], "currency": currency, "period": period}
        elif len(parsed_numbers) >= 2:
            return {"min": min(parsed_numbers), "max": max(parsed_numbers), "currency": currency, "period": period}
        
        return {"min": None, "max": None, "currency": currency, "period": period}
    
    def _parse_date(self, date_text: str) -> Optional[datetime]:
        """
        Parse date from various text formats
        
        Args:
            date_text: Raw date text
            
        Returns:
            Parsed datetime or None
        """
        import re
        from dateutil import parser
        
        if not date_text:
            return None
        
        try:
            # Handle relative dates like "2 days ago"
            if "ago" in date_text.lower():
                days_match = re.search(r'(\d+)\s*days?\s*ago', date_text.lower())
                hours_match = re.search(r'(\d+)\s*hours?\s*ago', date_text.lower())
                
                if days_match:
                    days = int(days_match.group(1))
                    return datetime.utcnow() - timedelta(days=days)
                elif hours_match:
                    hours = int(hours_match.group(1))
                    return datetime.utcnow() - timedelta(hours=hours)
                elif "today" in date_text.lower():
                    return datetime.utcnow()
                elif "yesterday" in date_text.lower():
                    return datetime.utcnow() - timedelta(days=1)
            
            # Try to parse as regular date
            return parser.parse(date_text)
            
        except (ValueError, parser.ParserError):
            self.logger.warning(f"Could not parse date: {date_text}")
            return None
    
    def _clean_text(self, text: str) -> str:
        """
        Clean and normalize text content
        
        Args:
            text: Raw text
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Remove extra whitespace and normalize
        text = ' '.join(text.split())
        
        # Remove HTML entities
        import html
        text = html.unescape(text)
        
        return text.strip()
    
    def _extract_experience_level(self, text: str) -> Optional[str]:
        """
        Extract experience level from job text
        
        Args:
            text: Job description or title text
            
        Returns:
            Experience level or None
        """
        text_lower = text.lower()
        
        if any(word in text_lower for word in ["entry", "junior", "graduate", "intern"]):
            return "entry"
        elif any(word in text_lower for word in ["senior", "sr.", "lead", "principal"]):
            return "senior"
        elif any(word in text_lower for word in ["manager", "director", "head", "chief"]):
            return "executive"
        elif any(word in text_lower for word in ["mid", "intermediate", "experienced"]):
            return "mid"
        
        return None
    
    def _extract_remote_type(self, text: str) -> Optional[str]:
        """
        Extract remote work type from job text
        
        Args:
            text: Job description or location text
            
        Returns:
            Remote type or None
        """
        text_lower = text.lower()
        
        if any(word in text_lower for word in ["remote", "work from home", "wfh"]):
            return "remote"
        elif any(word in text_lower for word in ["hybrid", "flexible"]):
            return "hybrid"
        elif any(word in text_lower for word in ["on-site", "onsite", "office"]):
            return "onsite"
        
        return None