"""Integration service for coordinating external API clients with error handling and graceful degradation."""

import logging
import asyncio
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from pydantic import BaseModel

from .github_client import GitHubClient, GitHubProfile
from .leetcode_scraper import LeetCodeScraper, LeetCodeProfile
from .linkedin_scraper import LinkedInScraper, LinkedInProfile
from .data_validator import DataValidator, ValidationResult, DataQuality
from .base_client import APIError, RateLimitError


logger = logging.getLogger(__name__)


class ProfileExtractionRequest(BaseModel):
    """Request model for profile extraction."""
    github_username: Optional[str] = None
    leetcode_username: Optional[str] = None
    linkedin_url: Optional[str] = None
    timeout_seconds: int = 60
    enable_validation: bool = True
    enable_graceful_degradation: bool = True


class ProfileExtractionResult(BaseModel):
    """Result of profile extraction from multiple sources."""
    success: bool
    github_profile: Optional[Dict[str, Any]] = None
    leetcode_profile: Optional[Dict[str, Any]] = None
    linkedin_profile: Optional[Dict[str, Any]] = None
    validation_results: Dict[str, ValidationResult] = {}
    errors: Dict[str, str] = {}
    warnings: List[str] = []
    extraction_time: float = 0.0
    sources_attempted: List[str] = []
    sources_successful: List[str] = []


class ExternalAPIIntegrationService:
    """Service for coordinating external API integrations with comprehensive error handling."""
    
    def __init__(
        self,
        github_token: Optional[str] = None,
        enable_caching: bool = True,
        cache_ttl_seconds: int = 3600
    ):
        self.github_client = GitHubClient(api_token=github_token)
        self.leetcode_scraper = LeetCodeScraper()
        self.linkedin_scraper = LinkedInScraper()
        self.data_validator = DataValidator()
        self.enable_caching = enable_caching
        self.cache_ttl_seconds = cache_ttl_seconds
        self._cache: Dict[str, Tuple[Any, datetime]] = {}
    
    async def extract_comprehensive_profile(
        self,
        request: ProfileExtractionRequest
    ) -> ProfileExtractionResult:
        """Extract comprehensive profile from multiple sources with error handling."""
        start_time = datetime.utcnow()
        result = ProfileExtractionResult(
            success=False,
            sources_attempted=[],
            sources_successful=[]
        )
        
        # Prepare extraction tasks
        tasks = []
        
        if request.github_username:
            result.sources_attempted.append("github")
            tasks.append(self._extract_github_profile_safe(request.github_username))
        
        if request.leetcode_username:
            result.sources_attempted.append("leetcode")
            tasks.append(self._extract_leetcode_profile_safe(request.leetcode_username))
        
        if request.linkedin_url:
            result.sources_attempted.append("linkedin")
            tasks.append(self._extract_linkedin_profile_safe(request.linkedin_url))
        
        if not tasks:
            result.errors["general"] = "No profile sources provided"
            return result
        
        try:
            # Execute all tasks concurrently with timeout
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=request.timeout_seconds
            )
            
            # Process results
            source_index = 0
            
            if request.github_username:
                github_result = results[source_index]
                source_index += 1
                
                if isinstance(github_result, Exception):
                    result.errors["github"] = str(github_result)
                    logger.error(f"GitHub extraction failed: {github_result}")
                elif github_result:
                    result.github_profile = github_result
                    result.sources_successful.append("github")
                    
                    # Validate if requested
                    if request.enable_validation:
                        try:
                            github_profile_obj = GitHubProfile(**github_result)
                            validation = self.data_validator.validate_github_profile(github_profile_obj)
                            result.validation_results["github"] = validation
                            
                            if validation.quality == DataQuality.INVALID:
                                result.warnings.append("GitHub profile data quality is invalid")
                            elif validation.quality == DataQuality.LOW:
                                result.warnings.append("GitHub profile data quality is low")
                        except Exception as e:
                            result.warnings.append(f"GitHub validation failed: {str(e)}")
            
            if request.leetcode_username:
                leetcode_result = results[source_index]
                source_index += 1
                
                if isinstance(leetcode_result, Exception):
                    result.errors["leetcode"] = str(leetcode_result)
                    logger.error(f"LeetCode extraction failed: {leetcode_result}")
                elif leetcode_result:
                    result.leetcode_profile = leetcode_result
                    result.sources_successful.append("leetcode")
                    
                    # Validate if requested
                    if request.enable_validation:
                        try:
                            leetcode_profile_obj = LeetCodeProfile(**leetcode_result)
                            validation = self.data_validator.validate_leetcode_profile(leetcode_profile_obj)
                            result.validation_results["leetcode"] = validation
                            
                            if validation.quality == DataQuality.INVALID:
                                result.warnings.append("LeetCode profile data quality is invalid")
                            elif validation.quality == DataQuality.LOW:
                                result.warnings.append("LeetCode profile data quality is low")
                        except Exception as e:
                            result.warnings.append(f"LeetCode validation failed: {str(e)}")
            
            if request.linkedin_url:
                linkedin_result = results[source_index]
                source_index += 1
                
                if isinstance(linkedin_result, Exception):
                    result.errors["linkedin"] = str(linkedin_result)
                    logger.error(f"LinkedIn extraction failed: {linkedin_result}")
                elif linkedin_result:
                    result.linkedin_profile = linkedin_result
                    result.sources_successful.append("linkedin")
                    
                    # Validate if requested
                    if request.enable_validation:
                        try:
                            linkedin_profile_obj = LinkedInProfile(**linkedin_result)
                            validation = self.data_validator.validate_linkedin_profile(linkedin_profile_obj)
                            result.validation_results["linkedin"] = validation
                            
                            if validation.quality == DataQuality.INVALID:
                                result.warnings.append("LinkedIn profile data quality is invalid")
                            elif validation.quality == DataQuality.LOW:
                                result.warnings.append("LinkedIn profile data quality is low")
                        except Exception as e:
                            result.warnings.append(f"LinkedIn validation failed: {str(e)}")
            
            # Determine overall success
            result.success = len(result.sources_successful) > 0
            
            if not result.success and request.enable_graceful_degradation:
                result.warnings.append("All profile extractions failed, but service continued gracefully")
                result.success = True  # Allow graceful degradation
            
        except asyncio.TimeoutError:
            result.errors["general"] = f"Profile extraction timed out after {request.timeout_seconds} seconds"
            logger.error(f"Profile extraction timeout: {request.timeout_seconds}s")
            
            if request.enable_graceful_degradation:
                result.success = True
                result.warnings.append("Extraction timed out but continuing with partial data")
        
        except Exception as e:
            result.errors["general"] = f"Unexpected error during profile extraction: {str(e)}"
            logger.error(f"Unexpected error in profile extraction: {str(e)}")
        
        finally:
            # Calculate extraction time
            end_time = datetime.utcnow()
            result.extraction_time = (end_time - start_time).total_seconds()
        
        return result
    
    async def _extract_github_profile_safe(self, username: str) -> Optional[Dict[str, Any]]:
        """Safely extract GitHub profile with error handling."""
        cache_key = f"github:{username}"
        
        # Check cache
        if self.enable_caching and cache_key in self._cache:
            cached_data, cached_time = self._cache[cache_key]
            if (datetime.utcnow() - cached_time).total_seconds() < self.cache_ttl_seconds:
                logger.info(f"Using cached GitHub profile for {username}")
                return cached_data
        
        try:
            async with self.github_client as client:
                profile = await client.get_user_profile(username)
                profile_dict = profile.dict()
                
                # Cache result
                if self.enable_caching:
                    self._cache[cache_key] = (profile_dict, datetime.utcnow())
                
                return profile_dict
                
        except RateLimitError as e:
            logger.warning(f"GitHub rate limit exceeded for {username}: {e.message}")
            raise APIError(f"GitHub rate limit exceeded. Please try again later.")
        
        except APIError as e:
            if e.status_code == 404:
                raise APIError(f"GitHub user '{username}' not found")
            else:
                raise APIError(f"GitHub API error: {e.message}")
        
        except Exception as e:
            logger.error(f"Unexpected error extracting GitHub profile for {username}: {str(e)}")
            raise APIError(f"Failed to extract GitHub profile: {str(e)}")
    
    async def _extract_leetcode_profile_safe(self, username: str) -> Optional[Dict[str, Any]]:
        """Safely extract LeetCode profile with error handling."""
        cache_key = f"leetcode:{username}"
        
        # Check cache
        if self.enable_caching and cache_key in self._cache:
            cached_data, cached_time = self._cache[cache_key]
            if (datetime.utcnow() - cached_time).total_seconds() < self.cache_ttl_seconds:
                logger.info(f"Using cached LeetCode profile for {username}")
                return cached_data
        
        try:
            async with self.leetcode_scraper as scraper:
                profile = await scraper.get_user_profile(username)
                profile_dict = profile.dict()
                
                # Cache result
                if self.enable_caching:
                    self._cache[cache_key] = (profile_dict, datetime.utcnow())
                
                return profile_dict
                
        except APIError as e:
            if e.status_code == 404:
                raise APIError(f"LeetCode user '{username}' not found")
            else:
                raise APIError(f"LeetCode scraping error: {e.message}")
        
        except Exception as e:
            logger.error(f"Unexpected error extracting LeetCode profile for {username}: {str(e)}")
            raise APIError(f"Failed to extract LeetCode profile: {str(e)}")
    
    async def _extract_linkedin_profile_safe(self, profile_url: str) -> Optional[Dict[str, Any]]:
        """Safely extract LinkedIn profile with error handling."""
        cache_key = f"linkedin:{profile_url}"
        
        # Check cache
        if self.enable_caching and cache_key in self._cache:
            cached_data, cached_time = self._cache[cache_key]
            if (datetime.utcnow() - cached_time).total_seconds() < self.cache_ttl_seconds:
                logger.info(f"Using cached LinkedIn profile for {profile_url}")
                return cached_data
        
        try:
            async with self.linkedin_scraper as scraper:
                profile = await scraper.get_profile_safely(profile_url)
                if profile:
                    profile_dict = profile.dict()
                    
                    # Cache result
                    if self.enable_caching:
                        self._cache[cache_key] = (profile_dict, datetime.utcnow())
                    
                    return profile_dict
                else:
                    raise APIError("LinkedIn profile extraction returned no data")
                
        except APIError as e:
            if e.status_code == 404:
                raise APIError(f"LinkedIn profile not found or not accessible: {profile_url}")
            elif e.status_code == 429:
                raise APIError("LinkedIn rate limit exceeded. Please try again later.")
            else:
                raise APIError(f"LinkedIn scraping error: {e.message}")
        
        except Exception as e:
            logger.error(f"Unexpected error extracting LinkedIn profile for {profile_url}: {str(e)}")
            raise APIError(f"Failed to extract LinkedIn profile: {str(e)}")
    
    async def validate_profile_sources(
        self,
        github_username: Optional[str] = None,
        leetcode_username: Optional[str] = None,
        linkedin_url: Optional[str] = None
    ) -> Dict[str, bool]:
        """Validate that profile sources exist and are accessible."""
        validation_results = {}
        
        tasks = []
        sources = []
        
        if github_username:
            sources.append("github")
            tasks.append(self._validate_github_username(github_username))
        
        if leetcode_username:
            sources.append("leetcode")
            tasks.append(self._validate_leetcode_username(leetcode_username))
        
        if linkedin_url:
            sources.append("linkedin")
            tasks.append(self._validate_linkedin_url(linkedin_url))
        
        if tasks:
            try:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                for i, result in enumerate(results):
                    source = sources[i]
                    if isinstance(result, Exception):
                        validation_results[source] = False
                        logger.warning(f"Validation failed for {source}: {str(result)}")
                    else:
                        validation_results[source] = result
            
            except Exception as e:
                logger.error(f"Error during profile source validation: {str(e)}")
                for source in sources:
                    validation_results[source] = False
        
        return validation_results
    
    async def _validate_github_username(self, username: str) -> bool:
        """Validate GitHub username exists."""
        try:
            async with self.github_client as client:
                await client.get(f"/users/{username}")
                return True
        except:
            return False
    
    async def _validate_leetcode_username(self, username: str) -> bool:
        """Validate LeetCode username exists."""
        try:
            async with self.leetcode_scraper as scraper:
                return await scraper.validate_username(username)
        except:
            return False
    
    async def _validate_linkedin_url(self, profile_url: str) -> bool:
        """Validate LinkedIn profile URL is accessible."""
        try:
            async with self.linkedin_scraper as scraper:
                return await scraper.validate_profile_url(profile_url)
        except:
            return False
    
    def clear_cache(self):
        """Clear the profile extraction cache."""
        self._cache.clear()
        logger.info("Profile extraction cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        now = datetime.utcnow()
        valid_entries = 0
        expired_entries = 0
        
        for cache_key, (data, cached_time) in self._cache.items():
            if (now - cached_time).total_seconds() < self.cache_ttl_seconds:
                valid_entries += 1
            else:
                expired_entries += 1
        
        return {
            "total_entries": len(self._cache),
            "valid_entries": valid_entries,
            "expired_entries": expired_entries,
            "cache_ttl_seconds": self.cache_ttl_seconds
        }