"""
Dashboard API endpoints for user dashboard data
"""
import logging
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession

from ....core.database import get_db
from ....api.dependencies import get_current_user
from ....models.user import User
from ....services.dashboard_service import DashboardService
from ....schemas.dashboard import (
    DashboardSummary, UserProgressSummary, PersonalizedContent, DashboardConfiguration
)
from ....core.exceptions import ServiceError

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/summary", response_model=DashboardSummary)
async def get_dashboard_summary(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get comprehensive dashboard summary data for the current user.
    
    Returns dashboard summary including:
    - Overall career score and profile completion
    - Key metrics and progress indicators
    - Active milestones and completion stats
    - Top recommendations and recent activities
    - Quick stats for skills, job matches, and learning paths
    """
    try:
        dashboard_service = DashboardService(db)
        summary = await dashboard_service.get_dashboard_summary(current_user.id)
        
        return summary
        
    except ServiceError as e:
        logger.error(f"Service error getting dashboard summary for user {current_user.id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error getting dashboard summary for user {current_user.id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get dashboard summary"
        )


@router.get("/progress", response_model=UserProgressSummary)
async def get_user_progress_summary(
    tracking_period_days: int = Query(90, ge=7, le=365, description="Progress tracking period in days"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get user progress tracking and milestone data.
    
    Returns progress summary including:
    - Overall progress percentage and career score trends
    - Skill improvements and learning progress
    - Milestone completion rates and tracking
    - Job market progress and interview readiness
    - Time-based progress metrics
    """
    try:
        dashboard_service = DashboardService(db)
        progress = await dashboard_service.get_user_progress_summary(
            current_user.id, tracking_period_days
        )
        
        return progress
        
    except ServiceError as e:
        logger.error(f"Service error getting progress summary for user {current_user.id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error getting progress summary for user {current_user.id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get progress summary"
        )


@router.get("/personalized-content", response_model=PersonalizedContent)
async def get_personalized_dashboard_content(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get personalized dashboard content for the current user.
    
    Returns personalized content including:
    - Featured job recommendations and suggested skills
    - Recommended learning paths and market insights
    - Salary insights and industry updates
    - Networking opportunities and similar profiles
    - Personalization score and content categories
    """
    try:
        dashboard_service = DashboardService(db)
        content = await dashboard_service.get_personalized_content(current_user.id)
        
        return content
        
    except ServiceError as e:
        logger.error(f"Service error getting personalized content for user {current_user.id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error getting personalized content for user {current_user.id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get personalized content"
        )


@router.get("/metrics", response_model=dict)
async def get_dashboard_metrics(
    metric_types: Optional[str] = Query(None, description="Comma-separated list of metric types to include"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get specific dashboard metrics for the current user.
    
    Available metric types:
    - career_score: Overall career development score
    - skills: Skills count and distribution
    - market_position: Market position percentile
    - experience: Experience score and analysis
    - progress: Progress tracking metrics
    - recommendations: Recommendation counts and priorities
    """
    try:
        dashboard_service = DashboardService(db)
        
        # Get full summary to extract metrics
        summary = await dashboard_service.get_dashboard_summary(current_user.id)
        
        # Filter metrics if specific types requested
        if metric_types:
            requested_types = [t.strip().lower() for t in metric_types.split(',')]
            filtered_metrics = [
                metric for metric in summary.key_metrics
                if any(req_type in metric.name.lower() for req_type in requested_types)
            ]
        else:
            filtered_metrics = summary.key_metrics
        
        # Build response
        response = {
            "user_id": current_user.id,
            "metrics": [metric.dict() for metric in filtered_metrics],
            "overall_career_score": summary.overall_career_score,
            "profile_completion": summary.profile_completion,
            "generated_at": summary.generated_at.isoformat()
        }
        
        return response
        
    except ServiceError as e:
        logger.error(f"Service error getting dashboard metrics for user {current_user.id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error getting dashboard metrics for user {current_user.id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get dashboard metrics"
        )


@router.get("/milestones", response_model=dict)
async def get_user_milestones(
    status_filter: Optional[str] = Query(None, description="Filter by milestone status: active, completed, all"),
    category_filter: Optional[str] = Query(None, description="Filter by category: skill, career, learning"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get user milestones with optional filtering.
    
    Returns milestone data including:
    - Active and completed milestones
    - Progress percentages and target dates
    - Milestone categories and priorities
    - Completion statistics
    """
    try:
        dashboard_service = DashboardService(db)
        
        # Get progress summary which includes milestones
        progress = await dashboard_service.get_user_progress_summary(current_user.id)
        
        milestones = progress.milestones
        
        # Apply filters
        if status_filter:
            if status_filter.lower() == "active":
                milestones = [m for m in milestones if not m.completed]
            elif status_filter.lower() == "completed":
                milestones = [m for m in milestones if m.completed]
        
        if category_filter:
            milestones = [m for m in milestones if m.category.lower() == category_filter.lower()]
        
        response = {
            "user_id": current_user.id,
            "milestones": [milestone.dict() for milestone in milestones],
            "milestone_completion_rate": progress.milestone_completion_rate,
            "total_milestones": len(progress.milestones),
            "filtered_count": len(milestones),
            "generated_at": progress.generated_at.isoformat()
        }
        
        return response
        
    except ServiceError as e:
        logger.error(f"Service error getting milestones for user {current_user.id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error getting milestones for user {current_user.id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get milestones"
        )


@router.get("/activities", response_model=dict)
async def get_recent_activities(
    limit: int = Query(10, ge=1, le=50, description="Maximum number of activities to return"),
    activity_type: Optional[str] = Query(None, description="Filter by activity type"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get recent user activities for the dashboard.
    
    Returns recent activities including:
    - Profile updates and analysis completions
    - Skill additions and learning progress
    - Job applications and recommendations
    - System interactions and achievements
    """
    try:
        dashboard_service = DashboardService(db)
        
        # Get dashboard summary which includes recent activities
        summary = await dashboard_service.get_dashboard_summary(current_user.id)
        
        activities = summary.recent_activities
        
        # Apply type filter
        if activity_type:
            activities = [a for a in activities if a.type.lower() == activity_type.lower()]
        
        # Apply limit
        activities = activities[:limit]
        
        response = {
            "user_id": current_user.id,
            "activities": [activity.dict() for activity in activities],
            "total_activities": len(summary.recent_activities),
            "filtered_count": len(activities),
            "generated_at": summary.generated_at.isoformat()
        }
        
        return response
        
    except ServiceError as e:
        logger.error(f"Service error getting activities for user {current_user.id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error getting activities for user {current_user.id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get activities"
        )


@router.get("/quick-stats", response_model=dict)
async def get_dashboard_quick_stats(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get quick statistics for dashboard widgets.
    
    Returns quick stats including:
    - Skills count and job matches
    - Learning paths and recommendations
    - Profile completion and career score
    - Recent activity counts
    """
    try:
        dashboard_service = DashboardService(db)
        summary = await dashboard_service.get_dashboard_summary(current_user.id)
        
        response = {
            "user_id": current_user.id,
            "stats": {
                "overall_career_score": summary.overall_career_score,
                "profile_completion": summary.profile_completion,
                "skills_count": summary.skills_count,
                "job_matches_count": summary.job_matches_count,
                "learning_paths_count": summary.learning_paths_count,
                "active_milestones": len(summary.active_milestones),
                "completed_milestones": summary.completed_milestones_count,
                "total_milestones": summary.total_milestones_count,
                "recent_activities": len(summary.recent_activities),
                "top_recommendations": len(summary.top_recommendations)
            },
            "generated_at": summary.generated_at.isoformat()
        }
        
        return response
        
    except ServiceError as e:
        logger.error(f"Service error getting quick stats for user {current_user.id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error getting quick stats for user {current_user.id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get quick stats"
        )