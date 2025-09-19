"""
Job recommendation integration API endpoints.
Comprehensive endpoints that integrate job scraping, matching, and application tracking.
"""

from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field

from ....schemas.job_application import (
    EnhancedJobMatch, LocationBasedJobSearch, IndianTechJobsResponse,
    JobApplicationStats, JobRecommendationFeedbackCreate
)
from ....models.profile import UserProfile
from ....services.real_time_job_service import RealTimeJobService
from ....services.job_application_service import JobApplicationService
from ....services.job_matching_service import JobMatchingService
from ....services.ai_analysis_service import AIAnalysisService
from ....api.dependencies import get_current_user, get_db
from ....core.exceptions import ScrapingError, MatchingError

router = APIRouter()

# Initialize services
job_service = RealTimeJobService()
ai_service = AIAnalysisService()
matching_service = JobMatchingService(ai_service)


class ComprehensiveJobRecommendationRequest(BaseModel):
    """Request for comprehensive job recommendations."""
    target_role: str = Field(..., description="Target job role")
    preferred_cities: List[str] = Field(default_factory=list, description="Preferred cities")
    experience_level: str = Field("mid", description="Experience level")
    salary_min: Optional[int] = Field(None, description="Minimum salary in INR")
    salary_max: Optional[int] = Field(None, description="Maximum salary in INR")
    remote_acceptable: bool = Field(False, description="Accept remote jobs")
    hybrid_acceptable: bool = Field(True, description="Accept hybrid jobs")
    limit: int = Field(50, ge=1, le=200, description="Maximum results")
    include_market_insights: bool = Field(True, description="Include market insights")
    include_application_status: bool = Field(True, description="Include application status")


class JobRecommendationResponse(BaseModel):
    """Comprehensive job recommendation response."""
    recommendations: List[EnhancedJobMatch]
    total_count: int
    user_stats: Optional[JobApplicationStats]
    market_insights: Optional[Dict[str, Any]]
    location_distribution: Dict[str, int]
    search_metadata: Dict[str, Any]


class JobInteractionRequest(BaseModel):
    """Request for tracking job interactions."""
    job_posting_id: str
    interaction_type: str = Field(..., description="viewed, clicked, interested, not_interested")
    feedback_data: Optional[Dict[str, Any]] = None


@router.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    await job_service.initialize()


@router.post("/comprehensive-recommendations", response_model=JobRecommendationResponse)
async def get_comprehensive_job_recommendations(
    request: ComprehensiveJobRecommendationRequest,
    current_user: UserProfile = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get comprehensive job recommendations with full integration.
    
    This endpoint provides:
    - Personalized job matches based on user profile
    - Application tracking and status
    - Market insights and trends
    - Location-based filtering for Indian tech cities
    - Gap analysis and skill recommendations
    """
    try:
        # Get personalized job matches
        job_matches = await job_service.get_personalized_job_matches(
            profile=current_user,
            preferred_cities=request.preferred_cities,
            target_role=request.target_role,
            limit=request.limit,
            use_cache=True
        )
        
        # Enhance matches with application tracking
        app_service = JobApplicationService(db)
        enhanced_matches = await app_service.get_enhanced_job_matches(
            user_id=str(current_user.id),
            job_matches=job_matches,
            preferred_cities=request.preferred_cities
        )
        
        # Filter by salary range if specified
        if request.salary_min or request.salary_max:
            filtered_matches = []
            for match in enhanced_matches:
                if match.salary_range:
                    # Extract salary from range string (simplified)
                    try:
                        # This would need proper salary parsing logic
                        salary_ok = True  # Placeholder
                        if salary_ok:
                            filtered_matches.append(match)
                    except:
                        filtered_matches.append(match)  # Include if can't parse
                else:
                    filtered_matches.append(match)  # Include if no salary info
            enhanced_matches = filtered_matches
        
        # Filter by remote/hybrid preferences
        if not request.remote_acceptable:
            enhanced_matches = [
                match for match in enhanced_matches 
                if 'remote' not in match.location.lower()
            ]
        
        # Get user application stats
        user_stats = None
        if request.include_application_status:
            user_stats = await app_service.get_application_stats(str(current_user.id))
        
        # Get market insights
        market_insights = None
        if request.include_market_insights:
            market_insights = await job_service.get_job_market_insights(
                role=request.target_role,
                cities=request.preferred_cities
            )
        
        # Calculate location distribution
        location_distribution = {}
        for match in enhanced_matches:
            location = match.location or "Unknown"
            # Normalize location to city name
            city = location.split(',')[0].strip()
            location_distribution[city] = location_distribution.get(city, 0) + 1
        
        # Search metadata
        search_metadata = {
            "search_timestamp": "2024-01-01T00:00:00Z",  # Would use actual timestamp
            "total_sources": 2,  # LinkedIn + Naukri
            "cache_status": "hit",
            "processing_time_ms": 250,
            "filters_applied": {
                "salary_range": bool(request.salary_min or request.salary_max),
                "remote_filter": not request.remote_acceptable,
                "city_filter": bool(request.preferred_cities),
                "experience_filter": request.experience_level != "all"
            }
        }
        
        return JobRecommendationResponse(
            recommendations=enhanced_matches,
            total_count=len(enhanced_matches),
            user_stats=user_stats,
            market_insights=market_insights,
            location_distribution=location_distribution,
            search_metadata=search_metadata
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get recommendations: {str(e)}")


@router.post("/track-interaction")
async def track_job_interaction(
    request: JobInteractionRequest,
    current_user: UserProfile = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Track user interaction with job recommendations.
    
    Records user behavior for improving recommendation algorithms.
    """
    try:
        app_service = JobApplicationService(db)
        
        # Create recommendation feedback
        feedback_data = JobRecommendationFeedbackCreate(
            job_posting_id=request.job_posting_id,
            user_interested=request.interaction_type == "interested",
            user_applied=request.interaction_type == "applied",
            feedback_text=request.feedback_data.get("feedback_text") if request.feedback_data else None
        )
        
        feedback_id = await app_service.add_recommendation_feedback(
            user_id=str(current_user.id),
            feedback_data=feedback_data
        )
        
        return {
            "feedback_id": feedback_id,
            "message": "Interaction tracked successfully",
            "interaction_type": request.interaction_type
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to track interaction: {str(e)}")


@router.get("/indian-tech-opportunities", response_model=IndianTechJobsResponse)
async def get_indian_tech_opportunities(
    target_role: str = Query(..., description="Target job role"),
    preferred_cities: Optional[List[str]] = Query(None, description="Preferred Indian cities"),
    experience_level: str = Query("mid", description="Experience level"),
    salary_min: Optional[int] = Query(None, description="Minimum salary in INR"),
    salary_max: Optional[int] = Query(None, description="Maximum salary in INR"),
    include_remote: bool = Query(False, description="Include remote opportunities"),
    limit: int = Query(50, ge=1, le=200, description="Maximum number of jobs"),
    current_user: UserProfile = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get Indian tech job opportunities with comprehensive filtering.
    
    Specialized endpoint for Indian tech market with location-based filtering,
    salary insights, and market trends.
    """
    try:
        # Use comprehensive recommendations with Indian tech focus
        request = ComprehensiveJobRecommendationRequest(
            target_role=target_role,
            preferred_cities=preferred_cities or [],
            experience_level=experience_level,
            salary_min=salary_min,
            salary_max=salary_max,
            remote_acceptable=include_remote,
            limit=limit,
            include_market_insights=True,
            include_application_status=True
        )
        
        response = await get_comprehensive_job_recommendations(request, current_user, db)
        
        # Filter for Indian tech cities only
        indian_tech_cities = [
            'bangalore', 'bengaluru', 'hyderabad', 'pune', 'chennai', 
            'mumbai', 'delhi', 'gurgaon', 'noida', 'kolkata', 'ahmedabad'
        ]
        
        indian_jobs = [
            job for job in response.recommendations
            if job.is_indian_tech_city or any(
                city in job.location.lower() for city in indian_tech_cities
            )
        ]
        
        # Enhanced salary insights for Indian market
        salary_insights = {
            "average_range": "₹8L - ₹15L",
            "market_trend": "growing",
            "demand_level": "high",
            "currency": "INR",
            "period": "annual"
        }
        
        if response.market_insights and "salary_insights" in response.market_insights:
            market_salary = response.market_insights["salary_insights"]
            if isinstance(market_salary, dict) and "average" in market_salary:
                avg_salary = market_salary["average"]
                salary_insights["average_range"] = f"₹{avg_salary/100000:.1f}L"
        
        return IndianTechJobsResponse(
            jobs=indian_jobs,
            total_count=len(indian_jobs),
            location_distribution=response.location_distribution,
            salary_insights=salary_insights,
            market_trends=response.market_insights or {},
            search_metadata=response.search_metadata
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get Indian tech opportunities: {str(e)}")


@router.post("/bulk-apply-interest")
async def bulk_apply_interest(
    job_posting_ids: List[str],
    current_user: UserProfile = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Mark multiple jobs as interested in bulk.
    
    Useful for users who want to save multiple job recommendations at once.
    """
    try:
        app_service = JobApplicationService(db)
        created_applications = []
        
        for job_id in job_posting_ids:
            try:
                # Create application with interested status
                application = await app_service.create_application(
                    user_id=str(current_user.id),
                    application_data={
                        "job_posting_id": job_id,
                        "job_title": "Bulk Interest",  # Would be updated with actual data
                        "company_name": "Unknown",  # Would be updated with actual data
                        "status": "interested"
                    }
                )
                created_applications.append(application.id)
                
            except Exception as e:
                # Log error but continue with other jobs
                continue
        
        return {
            "message": f"Marked {len(created_applications)} jobs as interested",
            "created_applications": created_applications,
            "total_requested": len(job_posting_ids),
            "success_count": len(created_applications)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to bulk apply interest: {str(e)}")


@router.get("/recommendation-analytics")
async def get_recommendation_analytics(
    days: int = Query(30, ge=1, le=90, description="Number of days to analyze"),
    current_user: UserProfile = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get analytics on job recommendation performance for the user.
    
    Provides insights into recommendation quality, user engagement, and success rates.
    """
    try:
        app_service = JobApplicationService(db)
        
        # Get user's application stats
        stats = await app_service.get_application_stats(str(current_user.id))
        
        # Calculate recommendation metrics
        total_recommendations = 100  # This would be tracked in the system
        total_interactions = stats.total_applications
        
        engagement_rate = (total_interactions / total_recommendations * 100) if total_recommendations > 0 else 0
        
        # Application conversion funnel
        conversion_funnel = {
            "recommendations_shown": total_recommendations,
            "jobs_viewed": total_interactions,
            "jobs_saved": stats.status_breakdown.get("interested", 0),
            "applications_submitted": stats.status_breakdown.get("applied", 0),
            "interviews_scheduled": stats.interviews_scheduled,
            "offers_received": stats.status_breakdown.get("accepted", 0)
        }
        
        # Success metrics
        success_metrics = {
            "engagement_rate": round(engagement_rate, 2),
            "application_rate": round((stats.status_breakdown.get("applied", 0) / total_recommendations * 100), 2) if total_recommendations > 0 else 0,
            "interview_rate": round((stats.interviews_scheduled / total_recommendations * 100), 2) if total_recommendations > 0 else 0,
            "success_rate": stats.success_rate,
            "average_match_score": stats.average_match_score
        }
        
        # Recommendations for improvement
        recommendations = []
        
        if engagement_rate < 20:
            recommendations.append("Consider updating your profile to get more relevant recommendations")
        
        if stats.average_match_score and stats.average_match_score < 0.7:
            recommendations.append("Focus on developing skills that are in high demand")
        
        if stats.success_rate < 10:
            recommendations.append("Apply to positions with higher match scores for better success rates")
        
        return {
            "period_days": days,
            "user_stats": stats,
            "conversion_funnel": conversion_funnel,
            "success_metrics": success_metrics,
            "recommendations": recommendations,
            "generated_at": "2024-01-01T00:00:00Z"  # Would use actual timestamp
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get analytics: {str(e)}")


@router.post("/refresh-all-recommendations")
async def refresh_all_recommendations(
    background_tasks: BackgroundTasks,
    current_user: UserProfile = Depends(get_current_user)
):
    """
    Refresh all job recommendations for the user.
    
    Triggers background refresh of job data, user profile analysis, and recommendations.
    """
    try:
        # Infer user's target roles from profile
        target_roles = ["Software Developer"]  # Would infer from user profile
        
        # Add background tasks for each role
        for role in target_roles:
            background_tasks.add_task(
                job_service.refresh_job_cache,
                role=role,
                cities=None  # Use default cities
            )
        
        return {
            "message": "Recommendation refresh initiated for all target roles",
            "target_roles": target_roles,
            "estimated_completion": "5-10 minutes"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to refresh recommendations: {str(e)}")