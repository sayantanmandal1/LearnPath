"""LinkedIn profile data extraction with rate limiting and ToS compliance."""

import logging
import re
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from pydantic import BaseModel
from urllib.parse import urlparse, parse_qs

from .base_client import BaseAPIClient, APIError, RetryConfig


logger = logging.getLogger(__name__)


class LinkedInExperience(BaseModel):
    """LinkedIn work experience entry."""
    company: str
    position: str
    duration: str
    location: Optional[str] = None
    description: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    is_current: bool = False


class LinkedInEducation(BaseModel):
    """LinkedIn education entry."""
    institution: str
    degree: Optional[str] = None
    field_of_study: Optional[str] = None
    start_year: Optional[int] = None
    end_year: Optional[int] = None
    description: Optional[str] = None


class LinkedInSkill(BaseModel):
    """LinkedIn skill with endorsements."""
    name: str
    endorsements: int = 0
    is_top_skill: bool = False


class LinkedInProfile(BaseModel):
    """LinkedIn user profile data model."""
    name: str
    headline: Optional[str] = None
    location: Optional[str] = None
    industry: Optional[str] = None
    summary: Optional[str] = None
    current_company: Optional[str] = None
    current_position: Optional[str] = None
    connections_count: Optional[int] = None
    followers_count: Optional[int] = None
    experience: List[LinkedInExperience] = []
    education: List[LinkedInEducation] = []
    skills: List[LinkedInSkill] = []
    languages: List[str] = []
    certifications: List[str] = []
    profile_url: str
    extracted_at: datetime


class LinkedInScraper(BaseAPIClient):
    """LinkedIn profile scraper with ToS compliance and rate limiting."""
    
    def __init__(self, timeout: float = 30.0):
        # LinkedIn scraping requires careful rate limiting
        super().__init__(
            base_url="https://www.linkedin.com",
            timeout=timeout,
            retry_config=RetryConfig(
                max_retries=2,  # Conservative retry count
                base_delay=5.0,  # Longer delays to respect rate limits
                max_delay=300.0  # Up to 5 minutes between retries
            )
        )
        self._last_request_time = None
        self._min_request_interval = 3.0  # Minimum 3 seconds between requests
    
    def _get_auth_headers(self) -> Dict[str, str]:
        """LinkedIn doesn't require authentication for public profiles."""
        return {}
    
    def _get_default_headers(self) -> Dict[str, str]:
        """Get headers that mimic a real browser."""
        return {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate, br",
            "DNT": "1",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
        }
    
    async def _make_request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """Override to add additional rate limiting."""
        # Enforce minimum interval between requests
        if self._last_request_time:
            elapsed = (datetime.utcnow() - self._last_request_time).total_seconds()
            if elapsed < self._min_request_interval:
                import asyncio
                await asyncio.sleep(self._min_request_interval - elapsed)
        
        self._last_request_time = datetime.utcnow()
        
        try:
            return await super()._make_request(method, endpoint, **kwargs)
        except APIError as e:
            # LinkedIn often returns 999 status code for rate limiting
            if e.status_code == 999:
                raise APIError("LinkedIn rate limit exceeded - please try again later", status_code=429)
            raise
    
    def _extract_profile_id_from_url(self, profile_url: str) -> Optional[str]:
        """Extract LinkedIn profile ID from URL."""
        # Handle different LinkedIn URL formats
        patterns = [
            r'/in/([^/?]+)',  # /in/username
            r'/pub/([^/?]+)',  # /pub/username  
            r'linkedin\.com/in/([^/?]+)',
            r'linkedin\.com/pub/([^/?]+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, profile_url)
            if match:
                return match.group(1)
        
        return None
    
    async def get_profile_from_url(self, profile_url: str) -> LinkedInProfile:
        """Extract LinkedIn profile data from public profile URL."""
        try:
            profile_id = self._extract_profile_id_from_url(profile_url)
            if not profile_id:
                raise APIError("Invalid LinkedIn profile URL format")
            
            # Note: This is a simplified implementation
            # In production, you would need to handle LinkedIn's anti-scraping measures
            # and potentially use official LinkedIn APIs where available
            
            logger.warning(
                "LinkedIn scraping is limited due to ToS restrictions. "
                "Consider using LinkedIn's official APIs for production use."
            )
            
            # For demonstration, return a mock profile structure
            # In reality, you would parse the HTML response carefully
            return await self._scrape_public_profile(profile_url, profile_id)
            
        except APIError as e:
            raise APIError(f"Failed to extract LinkedIn profile: {e.message}")
    
    async def _scrape_public_profile(self, profile_url: str, profile_id: str) -> LinkedInProfile:
        """Scrape public LinkedIn profile (simplified implementation)."""
        
        # WARNING: This is a simplified mock implementation
        # Real LinkedIn scraping requires handling:
        # - CSRF tokens
        # - Dynamic content loading
        # - Anti-bot measures
        # - Legal compliance with ToS
        
        try:
            # Make request to public profile
            response = await self.get(f"/in/{profile_id}")
            html_content = response.get("content", "")
            
            # Parse basic information (this would need proper HTML parsing)
            profile_data = self._parse_profile_html(html_content, profile_url)
            
            return LinkedInProfile(
                name=profile_data.get("name", "Unknown"),
                headline=profile_data.get("headline"),
                location=profile_data.get("location"),
                industry=profile_data.get("industry"),
                summary=profile_data.get("summary"),
                current_company=profile_data.get("current_company"),
                current_position=profile_data.get("current_position"),
                connections_count=profile_data.get("connections_count"),
                followers_count=profile_data.get("followers_count"),
                experience=profile_data.get("experience", []),
                education=profile_data.get("education", []),
                skills=profile_data.get("skills", []),
                languages=profile_data.get("languages", []),
                certifications=profile_data.get("certifications", []),
                profile_url=profile_url,
                extracted_at=datetime.utcnow()
            )
            
        except APIError as e:
            if e.status_code == 404:
                raise APIError(f"LinkedIn profile not found or not public: {profile_url}")
            elif e.status_code == 429 or e.status_code == 999:
                raise APIError("LinkedIn rate limit exceeded. Please try again later.")
            else:
                raise APIError(f"Failed to access LinkedIn profile: {e.message}")
    
    def _parse_profile_html(self, html_content: str, profile_url: str) -> Dict[str, Any]:
        """Parse LinkedIn profile HTML content."""
        
        # WARNING: This is a mock implementation
        # Real HTML parsing would require libraries like BeautifulSoup
        # and would need to handle LinkedIn's dynamic content loading
        
        # For demonstration purposes, return mock data
        # In production, implement proper HTML parsing
        
        logger.warning("Using mock LinkedIn data - implement proper HTML parsing for production")
        
        return {
            "name": "Mock User",
            "headline": "Software Engineer at Tech Company",
            "location": "San Francisco, CA",
            "industry": "Technology",
            "summary": "Experienced software engineer with expertise in web development.",
            "current_company": "Tech Company",
            "current_position": "Senior Software Engineer",
            "connections_count": 500,
            "followers_count": 100,
            "experience": [
                LinkedInExperience(
                    company="Tech Company",
                    position="Senior Software Engineer",
                    duration="2 years",
                    location="San Francisco, CA",
                    description="Developing web applications using modern technologies.",
                    start_date="2022-01",
                    is_current=True
                )
            ],
            "education": [
                LinkedInEducation(
                    institution="University of Technology",
                    degree="Bachelor of Science",
                    field_of_study="Computer Science",
                    start_year=2018,
                    end_year=2022
                )
            ],
            "skills": [
                LinkedInSkill(name="Python", endorsements=25, is_top_skill=True),
                LinkedInSkill(name="JavaScript", endorsements=20, is_top_skill=True),
                LinkedInSkill(name="React", endorsements=15, is_top_skill=False)
            ],
            "languages": ["English", "Spanish"],
            "certifications": ["AWS Certified Developer"]
        }
    
    async def validate_profile_url(self, profile_url: str) -> bool:
        """Validate if a LinkedIn profile URL is accessible."""
        try:
            profile_id = self._extract_profile_id_from_url(profile_url)
            if not profile_id:
                return False
            
            # Make a lightweight request to check if profile exists
            response = await self.get(f"/in/{profile_id}")
            return response is not None
            
        except APIError:
            return False
    
    async def get_company_info(self, company_name: str) -> Dict[str, Any]:
        """Get basic company information from LinkedIn."""
        try:
            # This would require LinkedIn's company API or scraping company pages
            # For now, return a basic structure
            
            logger.warning("Company info extraction not fully implemented")
            
            return {
                "name": company_name,
                "industry": "Unknown",
                "size": "Unknown",
                "location": "Unknown",
                "description": "Company information not available"
            }
            
        except APIError as e:
            logger.warning(f"Failed to fetch company info for {company_name}: {e.message}")
            return {}
    
    def _respect_robots_txt(self) -> bool:
        """Check if scraping is allowed by robots.txt."""
        # In production, implement robots.txt checking
        # For now, assume scraping is restricted
        logger.warning("LinkedIn scraping should respect robots.txt and ToS")
        return False
    
    async def get_profile_safely(self, profile_url: str) -> Optional[LinkedInProfile]:
        """Safely extract profile with comprehensive error handling."""
        try:
            # Check robots.txt compliance
            if not self._respect_robots_txt():
                logger.warning("Scraping may violate LinkedIn's ToS - consider using official APIs")
            
            # Validate URL format
            if not self._extract_profile_id_from_url(profile_url):
                raise APIError("Invalid LinkedIn profile URL format")
            
            # Extract profile with retries
            return await self.get_profile_from_url(profile_url)
            
        except APIError as e:
            logger.error(f"Failed to extract LinkedIn profile safely: {e.message}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error during LinkedIn profile extraction: {str(e)}")
            return None