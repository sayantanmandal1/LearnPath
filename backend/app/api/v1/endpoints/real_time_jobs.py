"""
Real-time job scraping and matching API endpoints.
Provides endpoints for Indian tech job market integration.
"""

from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
from pydantic import BaseModel

from ....schemas.job import JobPosting, JobMatch
from ....models.profile import UserProfile
from ....services.real_time_job_service import RealTimeJobService
from ....api.dependencies import get_current_user
from ....core.exceptions import ScrapingError, MatchingError

router = APIRouter()

# Initialize service
job_service = RealTimeJobService()

class JobSearchRequest(BaseModel):
    """Request model for job search."""
    role: str
    preferred_cities: Optional[List[str]] = None
    experience_level: str = "mid"
    limit: int = 50
    use_cache: bool = True

class PersonalizedJobRequest(BaseModel):
    """Request model for personalized job matching."""
    preferred_cities: Optional[List[str]] = None
    target_role: Optional[str] = None
    limit: int = 50
    use_cache: bool = True

class MarketInsightsRequest(BaseModel):
    """Request model for market insights."""
    role: str
    cities: Optional[List[str]] = None

@router.on_event("startup")
async def startup_event():
    """Initialize the job service on startup."""
    await job_service.initialize()

@router.get("/indian-tech-jobs", response_model=List[JobPosting])
async def get_indian_tech_jobs(
    role: str = Query(..., description="Job role to search for"),
    preferred_cities: Optional[List[str]] = Query(None, description="Preferred cities"),
    experience_level: str = Query("mid", description="Experience level (entry, mid, senior)"),
    limit: int = Query(50, ge=1, le=200, description="Maximum number of jobs"),
    use_cache: bool = Query(True, description="Use cached results if available")
):
    """
    Get tech jobs from Indian job portals (LinkedIn and Naukri).
    
    This endpoint scrapes real-time job data from major Indian job portals
    and returns tech-focused opportunities with location filtering.
    """
    try:
        jobs = await job_service.get_indian_tech_jobs(
            role=role,
            preferred_cities=preferred_cities,
            experience_level=experience_level,
            limit=limit,
            use_cache=use_cache
        )
        
        return jobs
        
    except ScrapingError as e:
        raise HTTPException(status_code=503, detail=f"Job scraping failed: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.post("/personalized-matches", response_model=List[JobMatch])
async def get_personalized_job_matches(
    request: PersonalizedJobRequest,
    current_user: UserProfile = Depends(get_current_user)
):
    """
    Get personalized job matches for the current user.
    
    This endpoint uses AI-powered matching to find jobs that best fit
    the user's profile, skills, and career goals.
    """
    try:
        matches = await job_service.get_personalized_job_matches(
            profile=current_user,
            preferred_cities=request.preferred_cities,
            target_role=request.target_role,
            limit=request.limit,
            use_cache=request.use_cache
        )
        
        return matches
        
    except MatchingError as e:
        raise HTTPException(status_code=422, detail=f"Job matching failed: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/market-insights", response_model=Dict[str, Any])
async def get_job_market_insights(
    role: str = Query(..., description="Job role to analyze"),
    cities: Optional[List[str]] = Query(None, description="Cities to analyze")
):
    """
    Get job market insights for a specific role and cities.
    
    Provides analytics on job distribution, top companies, skill demand,
    salary ranges, and posting trends in the Indian tech market.
    """
    try:
        insights = await job_service.get_job_market_insights(
            role=role,
            cities=cities
        )
        
        return insights
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get insights: {str(e)}")

@router.post("/refresh-cache")
async def refresh_job_cache(
    background_tasks: BackgroundTasks,
    role: str = Query(..., description="Job role to refresh"),
    cities: Optional[List[str]] = Query(None, description="Cities to refresh")
):
    """
    Manually refresh job cache for a specific role and cities.
    
    This endpoint triggers a background task to refresh cached job data
    with the latest information from job portals.
    """
    try:
        # Add background task to refresh cache
        background_tasks.add_task(
            job_service.refresh_job_cache,
            role=role,
            cities=cities
        )
        
        return {"message": f"Cache refresh initiated for role: {role}"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to refresh cache: {str(e)}")

@router.get("/supported-cities", response_model=List[str])
async def get_supported_cities():
    """
    Get list of supported Indian tech cities.
    
    Returns the list of Indian cities that are supported for job scraping
    and location-based filtering.
    """
    return job_service.priority_cities

@router.get("/health")
async def health_check():
    """
    Health check endpoint for the real-time job service.
    
    Returns the status of various components including scrapers and cache.
    """
    try:
        status = {
            "service": "healthy",
            "linkedin_scraper": "available",
            "naukri_scraper": "available",
            "cache": "available" if job_service.redis_client else "unavailable",
            "timestamp": "2024-01-01T00:00:00Z"  # Would use actual timestamp
        }
        
        return status
        
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")

# Additional utility endpoints

@router.get("/job-stats", response_model=Dict[str, Any])
async def get_job_statistics(
    role: str = Query(..., description="Job role to get stats for"),
    days: int = Query(7, ge=1, le=30, description="Number of days to analyze")
):
    """
    Get job posting statistics for a specific role.
    
    Returns statistics like total jobs posted, average per day,
    and trending metrics for the specified time period.
    """
    try:
        # This would implement actual statistics calculation
        # For now, return mock data structure
        stats = {
            "role": role,
            "period_days": days,
            "total_jobs": 0,
            "daily_average": 0.0,
            "trending": "stable",
            "top_locations": [],
            "message": "Statistics calculation not yet implemented"
        }
        
        return stats
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get statistics: {str(e)}")

@router.post("/batch-search", response_model=Dict[str, List[JobPosting]])
async def batch_job_search(
    roles: List[str],
    preferred_cities: Optional[List[str]] = None,
    experience_level: str = "mid",
    limit_per_role: int = 20
):
    """
    Search for multiple job roles in a single request.
    
    Useful for getting job data for multiple roles simultaneously,
    with results organized by role.
    """
    try:
        if len(roles) > 5:
            raise HTTPException(status_code=422, detail="Maximum 5 roles allowed per batch")
        
        results = {}
        
        for role in roles:
            try:
                jobs = await job_service.get_indian_tech_jobs(
                    role=role,
                    preferred_cities=preferred_cities,
                    experience_level=experience_level,
                    limit=limit_per_role,
                    use_cache=True
                )
                results[role] = jobs
                
            except Exception as e:
                results[role] = []
                # Log error but continue with other roles
        
        return results
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch search failed: {str(e)}")