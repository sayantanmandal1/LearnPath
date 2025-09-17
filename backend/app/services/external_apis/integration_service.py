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
from .profile_merger import ProfileMerger, MergedProfile
from .circuit_breaker import circuit_breaker_manager, CircuitBreakerError
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
    merged_profile: Optional[Dict[str, Any]] = None
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
        self.profile_merger = ProfileMerger()
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
            
            # Generate merged profile if we have any successful extractions
            if result.success and len(result.sources_successful) > 0:
                try:
                    merged_profile = self.profile_merger.merge_profiles(
                        github_profile=result.github_profile,
                        leetcode_profile=result.leetcode_profile,
                        linkedin_profile=result.linkedin_profile,
                        validation_results=result.validation_results
                    )
                    result.merged_profile = merged_profile.dict()
                    logger.info(f"Successfully merged profile data from {len(result.sources_successful)} sources")
                except Exception as e:
                    logger.warning(f"Failed to merge profile data: {str(e)}")
                    result.warnings.append(f"Profile merging failed: {str(e)}")
            
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
        
        max_retries = 3
        base_delay = 2.0
        
        for attempt in range(max_retries):
            try:
                # Use circuit breaker for GitHub API calls
                github_breaker = circuit_breaker_manager.get_breaker("github")
                
                async def github_call():
                    async with self.github_client as client:
                        profile = await client.get_user_profile(username)
                        return profile.dict()
                
                profile_dict = await github_breaker.call(github_call)
                
                # Cache result
                if self.enable_caching:
                    self._cache[cache_key] = (profile_dict, datetime.utcnow())
                
                return profile_dict
                    
            except (RateLimitError, CircuitBreakerError) as e:
                if attempt == max_retries - 1:
                    logger.warning(f"GitHub service unavailable for {username} after {max_retries} attempts: {str(e)}")
                    raise APIError(f"GitHub service temporarily unavailable. Please try again later.")
                
                # Calculate exponential backoff delay
                delay = base_delay * (2 ** attempt)
                if isinstance(e, RateLimitError):
                    retry_after = getattr(e, 'retry_after', delay)
                    wait_time = max(delay, retry_after) if retry_after else delay
                else:
                    wait_time = delay
                
                logger.info(f"GitHub service issue, waiting {wait_time}s before retry {attempt + 1}/{max_retries}")
                await asyncio.sleep(wait_time)
                continue
            
            except APIError as e:
                if e.status_code == 404:
                    raise APIError(f"GitHub user '{username}' not found", status_code=404)
                elif e.status_code in [500, 502, 503, 504] and attempt < max_retries - 1:
                    # Retry on server errors
                    delay = base_delay * (2 ** attempt)
                    logger.info(f"GitHub server error, retrying in {delay}s (attempt {attempt + 1}/{max_retries})")
                    await asyncio.sleep(delay)
                    continue
                else:
                    raise APIError(f"GitHub API error: {e.message}", status_code=e.status_code)
            
            except Exception as e:
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    logger.warning(f"Unexpected error extracting GitHub profile, retrying in {delay}s: {str(e)}")
                    await asyncio.sleep(delay)
                    continue
                else:
                    logger.error(f"Failed to extract GitHub profile for {username} after {max_retries} attempts: {str(e)}")
                    raise APIError(f"Failed to extract GitHub profile: {str(e)}")
        
        return None
    
    async def _extract_leetcode_profile_safe(self, username: str) -> Optional[Dict[str, Any]]:
        """Safely extract LeetCode profile with error handling."""
        cache_key = f"leetcode:{username}"
        
        # Check cache
        if self.enable_caching and cache_key in self._cache:
            cached_data, cached_time = self._cache[cache_key]
            if (datetime.utcnow() - cached_time).total_seconds() < self.cache_ttl_seconds:
                logger.info(f"Using cached LeetCode profile for {username}")
                return cached_data
        
        max_retries = 3
        base_delay = 3.0  # Longer delay for LeetCode due to anti-bot measures
        
        for attempt in range(max_retries):
            try:
                # Use circuit breaker for LeetCode API calls
                leetcode_breaker = circuit_breaker_manager.get_breaker("leetcode")
                
                async def leetcode_call():
                    async with self.leetcode_scraper as scraper:
                        profile = await scraper.get_user_profile(username)
                        return profile.dict()
                
                profile_dict = await leetcode_breaker.call(leetcode_call)
                
                # Cache result
                if self.enable_caching:
                    self._cache[cache_key] = (profile_dict, datetime.utcnow())
                
                return profile_dict
                    
            except (APIError, CircuitBreakerError) as e:
                if isinstance(e, APIError) and e.status_code == 404:
                    raise APIError(f"LeetCode user '{username}' not found", status_code=404)
                elif (isinstance(e, APIError) and (e.status_code == 429 or "rate limit" in str(e).lower())) or isinstance(e, CircuitBreakerError):
                    if attempt == max_retries - 1:
                        logger.warning(f"LeetCode service unavailable for {username} after {max_retries} attempts: {str(e)}")
                        raise APIError(f"LeetCode service temporarily unavailable. Please try again later.")
                    
                    # Exponential backoff with longer delays for LeetCode
                    delay = base_delay * (2 ** attempt)
                    logger.info(f"LeetCode service issue, waiting {delay}s before retry {attempt + 1}/{max_retries}")
                    await asyncio.sleep(delay)
                    continue
                elif isinstance(e, APIError) and e.status_code in [500, 502, 503, 504] and attempt < max_retries - 1:
                    # Retry on server errors
                    delay = base_delay * (2 ** attempt)
                    logger.info(f"LeetCode server error, retrying in {delay}s (attempt {attempt + 1}/{max_retries})")
                    await asyncio.sleep(delay)
                    continue
                else:
                    raise APIError(f"LeetCode scraping error: {str(e)}", status_code=getattr(e, 'status_code', None))
            
            except Exception as e:
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    logger.warning(f"Unexpected error extracting LeetCode profile, retrying in {delay}s: {str(e)}")
                    await asyncio.sleep(delay)
                    continue
                else:
                    logger.error(f"Failed to extract LeetCode profile for {username} after {max_retries} attempts: {str(e)}")
                    raise APIError(f"Failed to extract LeetCode profile: {str(e)}")
        
        return None
    
    async def _extract_linkedin_profile_safe(self, profile_url: str) -> Optional[Dict[str, Any]]:
        """Safely extract LinkedIn profile with error handling."""
        cache_key = f"linkedin:{profile_url}"
        
        # Check cache
        if self.enable_caching and cache_key in self._cache:
            cached_data, cached_time = self._cache[cache_key]
            if (datetime.utcnow() - cached_time).total_seconds() < self.cache_ttl_seconds:
                logger.info(f"Using cached LinkedIn profile for {profile_url}")
                return cached_data
        
        max_retries = 2  # Conservative retry count for LinkedIn
        base_delay = 5.0  # Longer delay to respect LinkedIn's ToS
        
        for attempt in range(max_retries):
            try:
                # Use circuit breaker for LinkedIn scraping
                linkedin_breaker = circuit_breaker_manager.get_breaker("linkedin")
                
                async def linkedin_call():
                    async with self.linkedin_scraper as scraper:
                        profile = await scraper.get_profile_safely(profile_url)
                        if profile:
                            return profile.dict()
                        else:
                            raise APIError("LinkedIn profile extraction returned no data")
                
                profile_dict = await linkedin_breaker.call(linkedin_call)
                
                # Cache result
                if self.enable_caching:
                    self._cache[cache_key] = (profile_dict, datetime.utcnow())
                
                return profile_dict
                        
            except (APIError, CircuitBreakerError) as e:
                if isinstance(e, APIError) and e.status_code == 404:
                    raise APIError(f"LinkedIn profile not found or not accessible: {profile_url}", status_code=404)
                elif (isinstance(e, APIError) and (e.status_code in [429, 999] or "rate limit" in str(e).lower())) or isinstance(e, CircuitBreakerError):
                    if attempt == max_retries - 1:
                        logger.warning(f"LinkedIn service unavailable for {profile_url} after {max_retries} attempts: {str(e)}")
                        raise APIError("LinkedIn service temporarily unavailable. Please try again later.")
                    
                    # Longer delays for LinkedIn due to strict rate limiting
                    delay = base_delay * (2 ** attempt)
                    logger.info(f"LinkedIn service issue, waiting {delay}s before retry {attempt + 1}/{max_retries}")
                    await asyncio.sleep(delay)
                    continue
                elif isinstance(e, APIError) and e.status_code in [500, 502, 503, 504] and attempt < max_retries - 1:
                    # Retry on server errors
                    delay = base_delay * (2 ** attempt)
                    logger.info(f"LinkedIn server error, retrying in {delay}s (attempt {attempt + 1}/{max_retries})")
                    await asyncio.sleep(delay)
                    continue
                else:
                    raise APIError(f"LinkedIn scraping error: {str(e)}", status_code=getattr(e, 'status_code', None))
            
            except Exception as e:
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    logger.warning(f"Unexpected error extracting LinkedIn profile, retrying in {delay}s: {str(e)}")
                    await asyncio.sleep(delay)
                    continue
                else:
                    logger.error(f"Failed to extract LinkedIn profile for {profile_url} after {max_retries} attempts: {str(e)}")
                    raise APIError(f"Failed to extract LinkedIn profile: {str(e)}")
        
        return None
    
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
    
    def get_circuit_breaker_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics for all services."""
        stats = circuit_breaker_manager.get_all_stats()
        return {
            service: {
                "state": breaker_stats.state.value,
                "failure_count": breaker_stats.failure_count,
                "success_count": breaker_stats.success_count,
                "total_requests": breaker_stats.total_requests,
                "total_failures": breaker_stats.total_failures,
                "total_successes": breaker_stats.total_successes,
                "last_failure_time": breaker_stats.last_failure_time.isoformat() if breaker_stats.last_failure_time else None,
                "last_success_time": breaker_stats.last_success_time.isoformat() if breaker_stats.last_success_time else None
            }
            for service, breaker_stats in stats.items()
        }
    
    async def reset_circuit_breakers(self):
        """Reset all circuit breakers."""
        await circuit_breaker_manager.reset_all()
        logger.info("All circuit breakers have been reset")
    
    async def reset_circuit_breaker(self, service: str):
        """Reset a specific circuit breaker."""
        await circuit_breaker_manager.reset_breaker(service)
        logger.info(f"Circuit breaker for {service} has been reset")