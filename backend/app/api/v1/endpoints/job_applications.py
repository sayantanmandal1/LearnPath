"""
Job application tracking API endpoints.
"""

from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
from sqlalchemy.orm import Session

from ....schemas.job_application import (
    JobApplicationCreate, JobApplicationUpdate, JobApplicationResponse,
    JobApplicationFeedbackCreate, JobRecommendationFeedbackCreate,
    JobApplicationStats, LocationBasedJobSearch, IndianTechJobsResponse,
    EnhancedJobMatch
)
from ....services.job_application_service import JobApplicationService
from ....services.real_time_job_service import RealTimeJobService
from ....models.profile import UserProfile
from ....api.dependencies import get_current_user, get_db
from ....core.exceptions import NotFoundError, ValidationError

router = APIRouter()

# Initialize services
job_service = RealTimeJobService()


@router.post("/applications", response_model=JobApplicationResponse)
async def create_job_application(
    application_data: JobApplicationCreate,
    current_user: UserProfile = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Create a new job application.
    
    Track a job that the user is interested in or has applied to.
    """
    try:
        app_service = JobApplicationService(db)
        return await app_service.create_application(
            user_id=str(current_user.id),
            application_data=application_data
        )
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create application: {str(e)}")


@router.get("/applications", response_model=List[JobApplicationResponse])
async def get_user_applications(
    status: Optional[str] = Query(None, description="Filter by application status"),
    limit: int = Query(50, ge=1, le=200, description="Maximum number of applications"),
    offset: int = Query(0, ge=0, description="Number of applications to skip"),
    current_user: UserProfile = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get user's job applications.
    
    Retrieve all job applications for the current user with optional filtering.
    """
    try:
        app_service = JobApplicationService(db)
        return await app_service.get_user_applications(
            user_id=str(current_user.id),
            status=status,
            limit=limit,
            offset=offset
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get applications: {str(e)}")


@router.put("/applications/{application_id}", response_model=JobApplicationResponse)
async def update_job_application(
    application_id: str,
    update_data: JobApplicationUpdate,
    current_user: UserProfile = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Update a job application.
    
    Update the status, notes, or other details of a job application.
    """
    try:
        app_service = JobApplicationService(db)
        return await app_service.update_application(
            application_id=application_id,
            user_id=str(current_user.id),
            update_data=update_data
        )
    except NotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update application: {str(e)}")


@router.get("/applications/stats", response_model=JobApplicationStats)
async def get_application_statistics(
    current_user: UserProfile = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get user's job application statistics.
    
    Provides insights into application patterns, success rates, and trends.
    """
    try:
        app_service = JobApplicationService(db)
        return await app_service.get_application_stats(str(current_user.id))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get statistics: {str(e)}")


@router.post("/applications/{application_id}/feedback")
async def add_application_feedback(
    application_id: str,
    feedback_data: JobApplicationFeedbackCreate,
    current_user: UserProfile = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Add feedback for a job application.
    
    Provide feedback on match accuracy, recommendation quality, or application outcome.
    """
    try:
        app_service = JobApplicationService(db)
        feedback_id = await app_service.add_application_feedback(
            application_id=application_id,
            user_id=str(current_user.id),
            feedback_data=feedback_data
        )
        return {"feedback_id": feedback_id, "message": "Feedback added successfully"}
    except NotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to add feedback: {str(e)}")


@router.post("/recommendations/feedback")
async def add_recommendation_feedback(
    feedback_data: JobRecommendationFeedbackCreate,
    current_user: UserProfile = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Add feedback for job recommendations.
    
    Help improve the recommendation algorithm by providing feedback on match quality.
    """
    try:
        app_service = JobApplicationService(db)
        feedback_id = await app_service.add_recommendation_feedback(
            user_id=str(current_user.id),
            feedback_data=feedback_data
        )
        return {"feedback_id": feedback_id, "message": "Recommendation feedback added successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to add feedback: {str(e)}")


@router.post("/applications/mark-applied/{job_posting_id}")
async def mark_job_as_applied(
    job_posting_id: str,
    application_method: str = Query("external", description="How the user applied"),
    current_user: UserProfile = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Mark a job as applied.
    
    Track when a user applies to a job externally (e.g., on company website).
    """
    try:
        app_service = JobApplicationService(db)
        application_id = await app_service.mark_job_as_applied(
            user_id=str(current_user.id),
            job_posting_id=job_posting_id,
            application_method=application_method
        )
        return {"application_id": application_id, "message": "Job marked as applied"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to mark as applied: {str(e)}")


@router.post("/enhanced-recommendations", response_model=List[EnhancedJobMatch])
async def get_enhanced_job_recommendations(
    search_params: LocationBasedJobSearch,
    current_user: UserProfile = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get enhanced job recommendations with application tracking.
    
    Provides job matches with location scoring, gap analysis, and application status.
    """
    try:
        # Initialize job service if needed
        await job_service.initialize()
        
        # Get personalized job matches
        job_matches = await job_service.get_personalized_job_matches(
            profile=current_user,
            preferred_cities=search_params.preferred_cities,
            target_role=search_params.target_role,
            limit=search_params.limit,
            use_cache=True
        )
        
        # Enhance matches with application tracking
        app_service = JobApplicationService(db)
        enhanced_matches = await app_service.get_enhanced_job_matches(
            user_id=str(current_user.id),
            job_matches=job_matches,
            preferred_cities=search_params.preferred_cities
        )
        
        return enhanced_matches
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get recommendations: {str(e)}")


@router.get("/indian-tech-jobs", response_model=IndianTechJobsResponse)
async def get_indian_tech_jobs_with_tracking(
    target_role: str = Query(..., description="Target job role"),
    preferred_cities: Optional[List[str]] = Query(None, description="Preferred Indian cities"),
    salary_min: Optional[int] = Query(None, description="Minimum salary in INR"),
    salary_max: Optional[int] = Query(None, description="Maximum salary in INR"),
    experience_level: str = Query("mid", description="Experience level"),
    limit: int = Query(50, ge=1, le=200, description="Maximum number of jobs"),
    current_user: UserProfile = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get Indian tech jobs with enhanced tracking and insights.
    
    Provides location-based job search with market insights and application tracking.
    """
    try:
        # Initialize job service
        await job_service.initialize()
        
        # Get Indian tech jobs
        jobs = await job_service.get_indian_tech_jobs(
            role=target_role,
            preferred_cities=preferred_cities or [],
            experience_level=experience_level,
            limit=limit,
            use_cache=True
        )
        
        # Get job matches for the user
        job_matches = await job_service.get_personalized_job_matches(
            profile=current_user,
            preferred_cities=preferred_cities,
            target_role=target_role,
            limit=limit,
            use_cache=True
        )
        
        # Enhance with application tracking
        app_service = JobApplicationService(db)
        enhanced_matches = await app_service.get_enhanced_job_matches(
            user_id=str(current_user.id),
            job_matches=job_matches,
            preferred_cities=preferred_cities
        )
        
        # Calculate location distribution
        location_distribution = {}
        for job in jobs:
            location = job.location or "Unknown"
            location_distribution[location] = location_distribution.get(location, 0) + 1
        
        # Get market insights
        market_insights = await job_service.get_job_market_insights(
            role=target_role,
            cities=preferred_cities
        )
        
        # Salary insights
        salary_insights = {
            "average_range": "₹8L - ₹15L",  # This would be calculated from actual data
            "market_trend": "growing",
            "demand_level": "high"
        }
        
        # Search metadata
        search_metadata = {
            "search_timestamp": "2024-01-01T00:00:00Z",  # Would use actual timestamp
            "total_sources": 2,  # LinkedIn + Naukri
            "cache_status": "hit",
            "processing_time_ms": 150
        }
        
        return IndianTechJobsResponse(
            jobs=enhanced_matches,
            total_count=len(enhanced_matches),
            location_distribution=location_distribution,
            salary_insights=salary_insights,
            market_trends=market_insights,
            search_metadata=search_metadata
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get Indian tech jobs: {str(e)}")


@router.post("/refresh-recommendations")
async def refresh_job_recommendations(
    background_tasks: BackgroundTasks,
    target_role: str = Query(..., description="Target role to refresh"),
    preferred_cities: Optional[List[str]] = Query(None, description="Preferred cities"),
    current_user: UserProfile = Depends(get_current_user)
):
    """
    Refresh job recommendations for the user.
    
    Triggers a background refresh of job data and recommendations.
    """
    try:
        # Add background task to refresh recommendations
        background_tasks.add_task(
            job_service.refresh_job_cache,
            role=target_role,
            cities=preferred_cities
        )
        
        return {
            "message": "Job recommendations refresh initiated",
            "target_role": target_role,
            "cities": preferred_cities or []
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to refresh recommendations: {str(e)}")


@router.get("/application-insights")
async def get_application_insights(
    current_user: UserProfile = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get insights and analytics for job applications.
    
    Provides personalized insights based on application history and market trends.
    """
    try:
        app_service = JobApplicationService(db)
        stats = await app_service.get_application_stats(str(current_user.id))
        
        # Generate insights based on stats
        insights = {
            "application_velocity": {
                "current_month": stats.applications_this_month,
                "recommendation": "Consider applying to 2-3 more positions this month" if stats.applications_this_month < 5 else "Good application pace!"
            },
            "success_optimization": {
                "current_success_rate": stats.success_rate,
                "benchmark": 15.0,  # Industry benchmark
                "recommendation": "Focus on higher match score positions" if stats.success_rate < 10 else "Great success rate!"
            },
            "skill_development": {
                "avg_match_score": stats.average_match_score,
                "recommendation": "Consider developing skills in high-demand areas" if (stats.average_match_score or 0) < 0.7 else "Strong skill alignment!"
            },
            "market_positioning": {
                "top_companies": stats.top_companies[:3],
                "recommendation": "Diversify company targets" if len(stats.top_companies) < 3 else "Good company diversity"
            }
        }
        
        return {
            "stats": stats,
            "insights": insights,
            "recommendations": [
                "Apply to positions with 70%+ match score for better success rates",
                "Follow up on applications after 1 week",
                "Update your profile with latest skills and projects"
            ]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get insights: {str(e)}")