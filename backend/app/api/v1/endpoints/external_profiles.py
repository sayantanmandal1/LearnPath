"""API endpoints for external profile extraction."""

from typing import Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, Field

from app.services.external_apis import (
    ExternalAPIIntegrationService,
    ProfileExtractionRequest,
    ProfileExtractionResult
)
from app.core.external_api_config import external_api_config


router = APIRouter()


class ProfileExtractionRequestAPI(BaseModel):
    """API model for profile extraction request."""
    github_username: Optional[str] = Field(None, description="GitHub username")
    leetcode_username: Optional[str] = Field(None, description="LeetCode username")
    linkedin_url: Optional[str] = Field(None, description="LinkedIn profile URL")
    timeout_seconds: int = Field(60, ge=10, le=300, description="Extraction timeout in seconds")
    enable_validation: bool = Field(True, description="Enable data validation")
    enable_graceful_degradation: bool = Field(True, description="Enable graceful degradation on errors")


class ProfileExtractionResponse(BaseModel):
    """API response for profile extraction."""
    success: bool
    github_profile: Optional[Dict[str, Any]] = None
    leetcode_profile: Optional[Dict[str, Any]] = None
    linkedin_profile: Optional[Dict[str, Any]] = None
    merged_profile: Optional[Dict[str, Any]] = None
    validation_results: Dict[str, Dict[str, Any]] = {}
    errors: Dict[str, str] = {}
    warnings: list[str] = []
    extraction_time: float
    sources_attempted: list[str] = []
    sources_successful: list[str] = []


class ProfileValidationRequest(BaseModel):
    """API model for profile source validation."""
    github_username: Optional[str] = None
    leetcode_username: Optional[str] = None
    linkedin_url: Optional[str] = None


class ProfileValidationResponse(BaseModel):
    """API response for profile validation."""
    validation_results: Dict[str, bool]


def get_integration_service() -> ExternalAPIIntegrationService:
    """Dependency to get integration service instance."""
    return ExternalAPIIntegrationService(
        github_token=external_api_config.github_token,
        enable_caching=external_api_config.enable_caching,
        cache_ttl_seconds=external_api_config.cache_ttl_seconds
    )


@router.post("/extract", response_model=ProfileExtractionResponse)
async def extract_profiles(
    request: ProfileExtractionRequestAPI,
    service: ExternalAPIIntegrationService = Depends(get_integration_service)
) -> ProfileExtractionResponse:
    """
    Extract comprehensive profile data from multiple external sources.
    
    This endpoint extracts profile information from GitHub, LeetCode, and LinkedIn
    based on the provided usernames/URLs. It includes data validation, error handling,
    and graceful degradation.
    
    - **github_username**: GitHub username to extract profile from
    - **leetcode_username**: LeetCode username to extract profile from  
    - **linkedin_url**: LinkedIn profile URL to extract data from
    - **timeout_seconds**: Maximum time to wait for extraction (10-300 seconds)
    - **enable_validation**: Whether to validate extracted data
    - **enable_graceful_degradation**: Whether to continue on partial failures
    """
    
    # Validate that at least one source is provided
    if not any([request.github_username, request.leetcode_username, request.linkedin_url]):
        raise HTTPException(
            status_code=400,
            detail="At least one profile source must be provided"
        )
    
    try:
        # Create internal request
        extraction_request = ProfileExtractionRequest(
            github_username=request.github_username,
            leetcode_username=request.leetcode_username,
            linkedin_url=request.linkedin_url,
            timeout_seconds=request.timeout_seconds,
            enable_validation=request.enable_validation,
            enable_graceful_degradation=request.enable_graceful_degradation
        )
        
        # Extract profiles
        result = await service.extract_comprehensive_profile(extraction_request)
        
        # Convert validation results to serializable format
        validation_results_dict = {}
        for source, validation in result.validation_results.items():
            validation_results_dict[source] = {
                "is_valid": validation.is_valid,
                "quality": validation.quality.value,
                "confidence_score": validation.confidence_score,
                "errors": validation.errors,
                "warnings": validation.warnings
            }
        
        return ProfileExtractionResponse(
            success=result.success,
            github_profile=result.github_profile,
            leetcode_profile=result.leetcode_profile,
            linkedin_profile=result.linkedin_profile,
            merged_profile=result.merged_profile,
            validation_results=validation_results_dict,
            errors=result.errors,
            warnings=result.warnings,
            extraction_time=result.extraction_time,
            sources_attempted=result.sources_attempted,
            sources_successful=result.sources_successful
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Profile extraction failed: {str(e)}"
        )


@router.post("/validate", response_model=ProfileValidationResponse)
async def validate_profile_sources(
    request: ProfileValidationRequest,
    service: ExternalAPIIntegrationService = Depends(get_integration_service)
) -> ProfileValidationResponse:
    """
    Validate that profile sources exist and are accessible.
    
    This endpoint quickly checks if the provided usernames/URLs are valid
    and accessible without extracting full profile data.
    
    - **github_username**: GitHub username to validate
    - **leetcode_username**: LeetCode username to validate
    - **linkedin_url**: LinkedIn profile URL to validate
    """
    
    # Validate that at least one source is provided
    if not any([request.github_username, request.leetcode_username, request.linkedin_url]):
        raise HTTPException(
            status_code=400,
            detail="At least one profile source must be provided"
        )
    
    try:
        validation_results = await service.validate_profile_sources(
            github_username=request.github_username,
            leetcode_username=request.leetcode_username,
            linkedin_url=request.linkedin_url
        )
        
        return ProfileValidationResponse(
            validation_results=validation_results
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Profile validation failed: {str(e)}"
        )


@router.get("/cache/stats")
async def get_cache_stats(
    service: ExternalAPIIntegrationService = Depends(get_integration_service)
) -> Dict[str, Any]:
    """
    Get cache statistics for the external API service.
    
    Returns information about cached profile data including:
    - Total cache entries
    - Valid (non-expired) entries
    - Expired entries
    - Cache TTL configuration
    """
    return service.get_cache_stats()


@router.delete("/cache/clear")
async def clear_cache(
    service: ExternalAPIIntegrationService = Depends(get_integration_service)
) -> Dict[str, str]:
    """
    Clear the profile extraction cache.
    
    This will remove all cached profile data, forcing fresh extraction
    on subsequent requests.
    """
    service.clear_cache()
    return {"message": "Cache cleared successfully"}


@router.get("/config")
async def get_api_config() -> Dict[str, Any]:
    """
    Get current external API configuration.
    
    Returns the current configuration settings for external API integrations,
    excluding sensitive information like API tokens.
    """
    return {
        "github_api_url": external_api_config.github_api_url,
        "github_timeout": external_api_config.github_timeout,
        "github_max_repos": external_api_config.github_max_repos,
        "leetcode_api_url": external_api_config.leetcode_api_url,
        "leetcode_timeout": external_api_config.leetcode_timeout,
        "linkedin_api_url": external_api_config.linkedin_api_url,
        "linkedin_timeout": external_api_config.linkedin_timeout,
        "rate_limit_max_retries": external_api_config.rate_limit_max_retries,
        "rate_limit_base_delay": external_api_config.rate_limit_base_delay,
        "rate_limit_max_delay": external_api_config.rate_limit_max_delay,
        "enable_caching": external_api_config.enable_caching,
        "cache_ttl_seconds": external_api_config.cache_ttl_seconds,
        "enable_data_validation": external_api_config.enable_data_validation,
        "enable_graceful_degradation": external_api_config.enable_graceful_degradation,
        "profile_extraction_timeout": external_api_config.profile_extraction_timeout,
        "has_github_token": external_api_config.github_token is not None
    }


@router.get("/circuit-breakers/stats")
async def get_circuit_breaker_stats(
    service: ExternalAPIIntegrationService = Depends(get_integration_service)
) -> Dict[str, Any]:
    """
    Get circuit breaker statistics for all external services.
    
    Returns the current state and statistics for circuit breakers protecting
    external API calls to GitHub, LeetCode, and LinkedIn.
    """
    return service.get_circuit_breaker_stats()


@router.post("/circuit-breakers/reset")
async def reset_all_circuit_breakers(
    service: ExternalAPIIntegrationService = Depends(get_integration_service)
) -> Dict[str, str]:
    """
    Reset all circuit breakers.
    
    This will reset all circuit breakers to the closed state, allowing
    requests to flow through normally.
    """
    await service.reset_circuit_breakers()
    return {"message": "All circuit breakers have been reset"}


@router.post("/circuit-breakers/{service_name}/reset")
async def reset_circuit_breaker(
    service_name: str,
    service: ExternalAPIIntegrationService = Depends(get_integration_service)
) -> Dict[str, str]:
    """
    Reset a specific circuit breaker.
    
    - **service_name**: Name of the service (github, leetcode, linkedin)
    """
    if service_name not in ["github", "leetcode", "linkedin"]:
        raise HTTPException(
            status_code=400,
            detail="Invalid service name. Must be one of: github, leetcode, linkedin"
        )
    
    await service.reset_circuit_breaker(service_name)
    return {"message": f"Circuit breaker for {service_name} has been reset"}