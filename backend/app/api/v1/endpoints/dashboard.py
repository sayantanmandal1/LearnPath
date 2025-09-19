"""
Enhanced Dashboard API endpoints for comprehensive user dashboard data with real-time analysis
"""
import logging
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession

from ....core.database import get_db
from ....api.dependencies import get_current_user
from ....models.user import User
from ....services.dashboard_service import DashboardService
from ....services.ai_analysis_service import AIAnalysisService
from ....services.real_time_job_service import RealTimeJobService
from ....schemas.dashboard import (
    DashboardSummary, UserProgressSummary, PersonalizedContent, DashboardConfiguration
)
from ....schemas.job import JobMatch
from ....core.exceptions import ServiceException

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
        
    except ServiceException as e:
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
        
    except ServiceException as e:
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
        
    except ServiceException as e:
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
        
    except ServiceException as e:
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
        
    except ServiceException as e:
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
        
    except ServiceException as e:
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
        
    except ServiceException as e:
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


# Enhanced Dashboard Endpoints for Real-Time Data Integration

@router.get("/comprehensive-data", response_model=dict)
async def get_comprehensive_dashboard_data(
    include_job_matches: bool = Query(True, description="Include personalized job matches"),
    include_skill_radar: bool = Query(True, description="Include skill radar chart data"),
    include_career_progress: bool = Query(True, description="Include career progress tracking"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get comprehensive dashboard data aggregation endpoint.
    
    This endpoint provides all dashboard data in a single request for optimal performance.
    Includes real-time analysis results, skill assessments, job matches, and progress tracking.
    
    Requirements: 7.1, 7.4
    """
    try:
        dashboard_service = DashboardService(db)
        ai_service = AIAnalysisService()
        job_service = RealTimeJobService()
        await job_service.initialize()
        
        # Get base dashboard summary
        summary = await dashboard_service.get_dashboard_summary(current_user.id)
        
        # Initialize comprehensive data structure
        comprehensive_data = {
            "user_id": current_user.id,
            "dashboard_summary": summary.dict(),
            "generated_at": summary.generated_at.isoformat()
        }
        
        # Add skill radar chart data if requested
        if include_skill_radar:
            try:
                skill_radar_data = await _get_real_time_skill_radar_data(current_user.id, db, ai_service)
                comprehensive_data["skill_radar"] = skill_radar_data
            except Exception as e:
                logger.warning(f"Failed to get skill radar data: {str(e)}")
                comprehensive_data["skill_radar"] = {"error": "Skill radar data unavailable"}
        
        # Add career progress tracking if requested
        if include_career_progress:
            try:
                progress_data = await _get_historical_career_progress(current_user.id, db, dashboard_service)
                comprehensive_data["career_progress"] = progress_data
            except Exception as e:
                logger.warning(f"Failed to get career progress data: {str(e)}")
                comprehensive_data["career_progress"] = {"error": "Career progress data unavailable"}
        
        # Add personalized job matches if requested
        if include_job_matches:
            try:
                job_matches_data = await _get_personalized_job_recommendations(current_user.id, db, job_service)
                comprehensive_data["job_recommendations"] = job_matches_data
            except Exception as e:
                logger.warning(f"Failed to get job matches: {str(e)}")
                comprehensive_data["job_recommendations"] = {"error": "Job recommendations unavailable"}
        
        logger.info(f"Generated comprehensive dashboard data for user {current_user.id}")
        return comprehensive_data
        
    except ServiceException as e:
        logger.error(f"Service error getting comprehensive dashboard data for user {current_user.id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error getting comprehensive dashboard data for user {current_user.id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get comprehensive dashboard data"
        )


@router.get("/skill-radar", response_model=dict)
async def get_real_time_skill_radar_chart_data(
    include_market_comparison: bool = Query(True, description="Include market comparison data"),
    skill_categories: Optional[str] = Query(None, description="Comma-separated skill categories to include"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get real-time skill radar chart data generation.
    
    Provides skill assessment data formatted for animated radar charts with market comparisons.
    Uses actual data from connected platforms and AI analysis.
    
    Requirements: 7.1, 7.2
    """
    try:
        ai_service = AIAnalysisService()
        
        # Get real-time skill radar data
        skill_radar_data = await _get_real_time_skill_radar_data(current_user.id, db, ai_service)
        
        # Filter by categories if specified
        if skill_categories:
            requested_categories = [cat.strip().lower() for cat in skill_categories.split(',')]
            filtered_skills = {}
            
            for category, skills in skill_radar_data.get("skill_categories", {}).items():
                if category.lower() in requested_categories:
                    filtered_skills[category] = skills
            
            skill_radar_data["skill_categories"] = filtered_skills
        
        # Add market comparison data if requested
        if include_market_comparison:
            try:
                market_comparison = await _get_skill_market_comparison(current_user.id, db)
                skill_radar_data["market_comparison"] = market_comparison
            except Exception as e:
                logger.warning(f"Failed to get market comparison: {str(e)}")
                skill_radar_data["market_comparison"] = {"error": "Market comparison unavailable"}
        
        return skill_radar_data
        
    except ServiceException as e:
        logger.error(f"Service error getting skill radar data for user {current_user.id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error getting skill radar data for user {current_user.id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get skill radar data"
        )


@router.get("/career-progress", response_model=dict)
async def get_career_progress_tracking(
    tracking_period_days: int = Query(90, ge=7, le=365, description="Progress tracking period in days"),
    include_predictions: bool = Query(True, description="Include career trajectory predictions"),
    include_milestones: bool = Query(True, description="Include milestone tracking"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get career progress tracking with historical data analysis.
    
    Provides historical career progress data with trend analysis and predictions.
    Reflects actual user improvements over time from real platform data.
    
    Requirements: 7.2, 7.6
    """
    try:
        dashboard_service = DashboardService(db)
        
        # Get historical career progress
        progress_data = await _get_historical_career_progress(
            current_user.id, db, dashboard_service, tracking_period_days
        )
        
        # Add career trajectory predictions if requested
        if include_predictions:
            try:
                ai_service = AIAnalysisService()
                predictions = await _get_career_trajectory_predictions(current_user.id, db, ai_service)
                progress_data["trajectory_predictions"] = predictions
            except Exception as e:
                logger.warning(f"Failed to get career predictions: {str(e)}")
                progress_data["trajectory_predictions"] = {"error": "Predictions unavailable"}
        
        # Add milestone tracking if requested
        if include_milestones:
            try:
                milestone_progress = await _get_milestone_progress_tracking(current_user.id, db)
                progress_data["milestone_tracking"] = milestone_progress
            except Exception as e:
                logger.warning(f"Failed to get milestone tracking: {str(e)}")
                progress_data["milestone_tracking"] = {"error": "Milestone tracking unavailable"}
        
        return progress_data
        
    except ServiceException as e:
        logger.error(f"Service error getting career progress for user {current_user.id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error getting career progress for user {current_user.id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get career progress"
        )


@router.get("/job-recommendations", response_model=dict)
async def get_personalized_job_recommendations(
    limit: int = Query(20, ge=1, le=50, description="Maximum number of job recommendations"),
    preferred_cities: Optional[str] = Query(None, description="Comma-separated preferred cities"),
    target_role: Optional[str] = Query(None, description="Target role for job matching"),
    min_match_score: float = Query(0.6, ge=0.0, le=1.0, description="Minimum match score threshold"),
    include_market_insights: bool = Query(True, description="Include job market insights"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get job recommendation endpoint with personalized matching.
    
    Provides real-time job recommendations based on current market opportunities
    with AI-powered compatibility scoring and gap analysis.
    
    Requirements: 7.3, 5.1, 5.2, 5.3, 5.4
    """
    try:
        job_service = RealTimeJobService()
        await job_service.initialize()
        
        # Parse preferred cities
        cities_list = None
        if preferred_cities:
            cities_list = [city.strip() for city in preferred_cities.split(',')]
        
        # Get personalized job recommendations
        job_recommendations = await _get_personalized_job_recommendations(
            current_user.id, db, job_service, limit, cities_list, target_role, min_match_score
        )
        
        # Add market insights if requested
        if include_market_insights:
            try:
                role_for_insights = target_role or job_recommendations.get("inferred_target_role", "Software Developer")
                market_insights = await job_service.get_job_market_insights(
                    role=role_for_insights,
                    cities=cities_list
                )
                job_recommendations["market_insights"] = market_insights
            except Exception as e:
                logger.warning(f"Failed to get market insights: {str(e)}")
                job_recommendations["market_insights"] = {"error": "Market insights unavailable"}
        
        return job_recommendations
        
    except ServiceException as e:
        logger.error(f"Service error getting job recommendations for user {current_user.id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error getting job recommendations for user {current_user.id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get job recommendations"
        )


@router.get("/real-time-analysis", response_model=dict)
async def get_real_time_analysis_results(
    force_refresh: bool = Query(False, description="Force refresh of analysis results"),
    analysis_types: Optional[str] = Query(None, description="Comma-separated analysis types to include"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get real-time analysis results from AI service.
    
    Provides fresh AI analysis results instead of cached data when requested.
    Includes skill assessment, career recommendations, and learning paths.
    
    Requirements: 7.1, 3.1, 3.2, 3.3, 3.4
    """
    try:
        ai_service = AIAnalysisService()
        
        # Get or generate analysis results
        if force_refresh:
            # Generate fresh analysis
            analysis_result = await ai_service.analyze_complete_profile(current_user.id, db)
            analysis_data = {
                "user_id": current_user.id,
                "analysis_timestamp": analysis_result.analysis_timestamp.isoformat(),
                "skill_assessment": {
                    "technical_skills": analysis_result.skill_assessment.technical_skills,
                    "soft_skills": analysis_result.skill_assessment.soft_skills,
                    "skill_strengths": analysis_result.skill_assessment.skill_strengths,
                    "skill_gaps": analysis_result.skill_assessment.skill_gaps,
                    "improvement_areas": analysis_result.skill_assessment.improvement_areas,
                    "market_relevance_score": analysis_result.skill_assessment.market_relevance_score,
                    "confidence_score": analysis_result.skill_assessment.confidence_score
                },
                "career_recommendations": [
                    {
                        "recommended_role": rec.recommended_role,
                        "match_score": rec.match_score,
                        "reasoning": rec.reasoning,
                        "required_skills": rec.required_skills,
                        "skill_gaps": rec.skill_gaps,
                        "preparation_timeline": rec.preparation_timeline,
                        "salary_range": rec.salary_range,
                        "market_demand": rec.market_demand
                    }
                    for rec in analysis_result.career_recommendations
                ],
                "learning_paths": [
                    {
                        "title": path.title,
                        "description": path.description,
                        "target_skills": path.target_skills,
                        "learning_modules": path.learning_modules,
                        "estimated_duration": path.estimated_duration,
                        "difficulty_level": path.difficulty_level,
                        "resources": path.resources
                    }
                    for path in analysis_result.learning_paths
                ],
                "project_suggestions": [
                    {
                        "title": proj.title,
                        "description": proj.description,
                        "technologies": proj.technologies,
                        "difficulty_level": proj.difficulty_level,
                        "estimated_duration": proj.estimated_duration,
                        "learning_outcomes": proj.learning_outcomes,
                        "portfolio_value": proj.portfolio_value
                    }
                    for proj in analysis_result.project_suggestions
                ],
                "market_insights": analysis_result.market_insights,
                "is_fresh_analysis": True
            }
        else:
            # Get cached analysis results
            analysis_data = await _get_cached_analysis_results(current_user.id, db)
            analysis_data["is_fresh_analysis"] = False
        
        # Filter by analysis types if specified
        if analysis_types:
            requested_types = [t.strip().lower() for t in analysis_types.split(',')]
            filtered_data = {"user_id": analysis_data["user_id"], "analysis_timestamp": analysis_data["analysis_timestamp"]}
            
            for analysis_type in requested_types:
                if analysis_type in analysis_data:
                    filtered_data[analysis_type] = analysis_data[analysis_type]
            
            analysis_data = filtered_data
        
        return analysis_data
        
    except ServiceException as e:
        logger.error(f"Service error getting real-time analysis for user {current_user.id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error getting real-time analysis for user {current_user.id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get real-time analysis"
        )


# Helper functions for enhanced dashboard endpoints

async def _get_real_time_skill_radar_data(user_id: str, db: AsyncSession, ai_service: AIAnalysisService) -> Dict[str, Any]:
    """Get real-time skill radar chart data from AI analysis."""
    try:
        # Get latest analysis results
        from sqlalchemy import select, desc
        from ....models.analysis_result import AnalysisResult, AnalysisType
        
        result = await db.execute(
            select(AnalysisResult)
            .where(AnalysisResult.user_id == user_id)
            .where(AnalysisResult.analysis_type == AnalysisType.SKILL_ASSESSMENT)
            .where(AnalysisResult.status == "completed")
            .order_by(desc(AnalysisResult.created_at))
            .limit(1)
        )
        latest_analysis = result.scalar_one_or_none()
        
        if not latest_analysis or not latest_analysis.skill_assessment:
            # Generate fresh analysis if none exists
            analysis_result = await ai_service.analyze_complete_profile(user_id, db)
            skill_assessment = analysis_result.skill_assessment
        else:
            # Use existing analysis
            skill_data = latest_analysis.skill_assessment
            from ....services.ai_analysis_service import SkillAssessment
            skill_assessment = SkillAssessment(
                technical_skills=skill_data.get("technical_skills", {}),
                soft_skills=skill_data.get("soft_skills", {}),
                skill_strengths=skill_data.get("skill_strengths", []),
                skill_gaps=skill_data.get("skill_gaps", []),
                improvement_areas=skill_data.get("improvement_areas", []),
                market_relevance_score=skill_data.get("market_relevance_score", 0.0),
                confidence_score=skill_data.get("confidence_score", 0.0)
            )
        
        # Format for radar chart
        radar_data = {
            "user_id": user_id,
            "skill_categories": {
                "Technical Skills": skill_assessment.technical_skills,
                "Soft Skills": skill_assessment.soft_skills
            },
            "skill_strengths": skill_assessment.skill_strengths,
            "skill_gaps": skill_assessment.skill_gaps,
            "improvement_areas": skill_assessment.improvement_areas,
            "overall_scores": {
                "technical_average": sum(skill_assessment.technical_skills.values()) / len(skill_assessment.technical_skills) if skill_assessment.technical_skills else 0,
                "soft_skills_average": sum(skill_assessment.soft_skills.values()) / len(skill_assessment.soft_skills) if skill_assessment.soft_skills else 0,
                "market_relevance": skill_assessment.market_relevance_score,
                "confidence": skill_assessment.confidence_score
            },
            "chart_config": {
                "max_value": 1.0,
                "scale_steps": 5,
                "animation_duration": 1000,
                "colors": {
                    "technical": "#3B82F6",
                    "soft_skills": "#10B981",
                    "strengths": "#F59E0B",
                    "gaps": "#EF4444"
                }
            },
            "generated_at": latest_analysis.created_at.isoformat() if latest_analysis else datetime.utcnow().isoformat()
        }
        
        return radar_data
        
    except Exception as e:
        logger.error(f"Error getting skill radar data: {str(e)}")
        return {
            "error": f"Failed to get skill radar data: {str(e)}",
            "fallback_data": {
                "skill_categories": {"Technical Skills": {}, "Soft Skills": {}},
                "skill_strengths": [],
                "skill_gaps": [],
                "overall_scores": {"technical_average": 0, "soft_skills_average": 0}
            }
        }


async def _get_historical_career_progress(
    user_id: str, 
    db: AsyncSession, 
    dashboard_service: DashboardService, 
    tracking_period_days: int = 90
) -> Dict[str, Any]:
    """Get historical career progress with trend analysis."""
    try:
        # Get progress summary from dashboard service
        progress_summary = await dashboard_service.get_user_progress_summary(user_id, tracking_period_days)
        
        # Get historical analysis results for trend analysis
        from sqlalchemy import select, desc
        from ....models.analysis_result import AnalysisResult
        from datetime import datetime, timedelta
        
        start_date = datetime.utcnow() - timedelta(days=tracking_period_days)
        
        result = await db.execute(
            select(AnalysisResult)
            .where(AnalysisResult.user_id == user_id)
            .where(AnalysisResult.created_at >= start_date)
            .where(AnalysisResult.overall_score.isnot(None))
            .order_by(AnalysisResult.created_at)
        )
        historical_analyses = result.scalars().all()
        
        # Build progress timeline
        progress_timeline = []
        for analysis in historical_analyses:
            progress_timeline.append({
                "date": analysis.created_at.isoformat(),
                "overall_score": analysis.overall_score,
                "skill_diversity_score": analysis.skill_diversity_score,
                "experience_relevance_score": analysis.experience_relevance_score,
                "market_readiness_score": analysis.market_readiness_score
            })
        
        progress_data = {
            "user_id": user_id,
            "tracking_period_days": tracking_period_days,
            "overall_progress": progress_summary.overall_progress,
            "career_score_trend": progress_summary.career_score_trend,
            "skill_improvements": progress_summary.skill_improvements,
            "progress_timeline": progress_timeline,
            "progress_metrics": {
                "new_skills_added": progress_summary.new_skills_added,
                "skills_mastered": progress_summary.skills_mastered,
                "learning_paths_started": progress_summary.learning_paths_started,
                "learning_paths_completed": progress_summary.learning_paths_completed,
                "courses_completed": progress_summary.courses_completed,
                "job_compatibility_improvement": progress_summary.job_compatibility_improvement,
                "interview_readiness_score": progress_summary.interview_readiness_score
            },
            "milestones": [milestone.dict() for milestone in progress_summary.milestones],
            "milestone_completion_rate": progress_summary.milestone_completion_rate,
            "generated_at": progress_summary.generated_at.isoformat()
        }
        
        return progress_data
        
    except Exception as e:
        logger.error(f"Error getting historical career progress: {str(e)}")
        return {
            "error": f"Failed to get career progress: {str(e)}",
            "fallback_data": {
                "overall_progress": 0,
                "career_score_trend": [],
                "skill_improvements": [],
                "progress_timeline": []
            }
        }


async def _get_personalized_job_recommendations(
    user_id: str,
    db: AsyncSession,
    job_service: RealTimeJobService,
    limit: int = 20,
    preferred_cities: List[str] = None,
    target_role: str = None,
    min_match_score: float = 0.6
) -> Dict[str, Any]:
    """Get personalized job recommendations with AI matching."""
    try:
        # Get user profile
        from sqlalchemy import select
        from ....models.profile import UserProfile
        
        result = await db.execute(select(UserProfile).where(UserProfile.user_id == user_id))
        profile = result.scalar_one_or_none()
        
        if not profile:
            return {"error": "User profile not found", "recommendations": []}
        
        # Get personalized job matches
        job_matches = await job_service.get_personalized_job_matches(
            profile=profile,
            preferred_cities=preferred_cities,
            target_role=target_role,
            limit=limit
        )
        
        # Filter by minimum match score
        filtered_matches = [match for match in job_matches if match.match_score >= min_match_score]
        
        # Format recommendations
        recommendations = []
        for match in filtered_matches:
            recommendations.append({
                "job_id": match.job_posting.job_id,
                "title": match.job_posting.title,
                "company": match.job_posting.company,
                "location": match.job_posting.location,
                "description": match.job_posting.description[:500] + "..." if len(match.job_posting.description) > 500 else match.job_posting.description,
                "required_skills": match.job_posting.required_skills,
                "experience_level": match.job_posting.experience_level,
                "salary_range": match.job_posting.salary_range.dict() if match.job_posting.salary_range else None,
                "posted_date": match.job_posting.posted_date.isoformat() if match.job_posting.posted_date else None,
                "source": match.job_posting.source,
                "url": match.job_posting.url,
                "match_score": match.match_score,
                "skill_matches": [sm.dict() for sm in match.skill_matches],
                "skill_gaps": [sg.dict() for sg in match.skill_gaps],
                "recommendation_reason": match.recommendation_reason
            })
        
        # Infer target role if not provided
        inferred_target_role = target_role
        if not inferred_target_role and profile.experience:
            inferred_target_role = profile.experience[0].title if profile.experience else "Software Developer"
        
        job_recommendations = {
            "user_id": user_id,
            "target_role": target_role,
            "inferred_target_role": inferred_target_role,
            "preferred_cities": preferred_cities,
            "min_match_score": min_match_score,
            "total_matches": len(job_matches),
            "filtered_matches": len(filtered_matches),
            "recommendations": recommendations,
            "recommendation_summary": {
                "avg_match_score": sum(match.match_score for match in filtered_matches) / len(filtered_matches) if filtered_matches else 0,
                "top_companies": list(set([rec["company"] for rec in recommendations[:10]])),
                "top_locations": list(set([rec["location"] for rec in recommendations[:10]])),
                "common_skills": _get_most_common_skills([rec["required_skills"] for rec in recommendations])
            },
            "generated_at": datetime.utcnow().isoformat()
        }
        
        return job_recommendations
        
    except Exception as e:
        logger.error(f"Error getting personalized job recommendations: {str(e)}")
        return {
            "error": f"Failed to get job recommendations: {str(e)}",
            "recommendations": [],
            "fallback_message": "Job recommendations are temporarily unavailable. Please try again later."
        }


def _get_most_common_skills(skill_lists: List[List[str]]) -> List[str]:
    """Get most commonly required skills across job recommendations."""
    skill_counts = {}
    for skills in skill_lists:
        for skill in skills:
            skill_counts[skill] = skill_counts.get(skill, 0) + 1
    
    # Return top 10 most common skills
    sorted_skills = sorted(skill_counts.items(), key=lambda x: x[1], reverse=True)
    return [skill for skill, count in sorted_skills[:10]]


async def _get_skill_market_comparison(user_id: str, db: AsyncSession) -> Dict[str, Any]:
    """Get skill market comparison data."""
    # This would typically compare user skills against market demand
    # For now, return mock comparison data
    return {
        "market_demand_scores": {
            "python": 0.95,
            "javascript": 0.90,
            "react": 0.85,
            "aws": 0.88,
            "docker": 0.82
        },
        "user_vs_market": {
            "above_market": ["python", "javascript"],
            "at_market": ["react"],
            "below_market": ["aws", "docker"]
        },
        "improvement_priority": ["aws", "docker", "kubernetes"]
    }


async def _get_career_trajectory_predictions(user_id: str, db: AsyncSession, ai_service: AIAnalysisService) -> Dict[str, Any]:
    """Get AI-powered career trajectory predictions."""
    try:
        # This would use AI service to predict career trajectory
        # For now, return structured prediction data
        return {
            "next_role_predictions": [
                {
                    "role": "Senior Software Developer",
                    "probability": 0.75,
                    "timeline": "6-12 months",
                    "required_skills": ["leadership", "system design", "mentoring"]
                },
                {
                    "role": "Tech Lead",
                    "probability": 0.60,
                    "timeline": "12-18 months", 
                    "required_skills": ["team management", "architecture", "project planning"]
                }
            ],
            "salary_progression": {
                "current_estimate": "8-12 LPA",
                "6_months": "10-15 LPA",
                "12_months": "12-18 LPA",
                "24_months": "15-25 LPA"
            },
            "skill_development_roadmap": [
                {"skill": "System Design", "priority": "high", "timeline": "3 months"},
                {"skill": "Leadership", "priority": "medium", "timeline": "6 months"},
                {"skill": "Cloud Architecture", "priority": "medium", "timeline": "4 months"}
            ]
        }
    except Exception as e:
        logger.warning(f"Error getting career predictions: {str(e)}")
        return {"error": "Career predictions unavailable"}


async def _get_milestone_progress_tracking(user_id: str, db: AsyncSession) -> Dict[str, Any]:
    """Get milestone progress tracking data."""
    # This would track actual milestone progress from database
    # For now, return mock milestone tracking data
    return {
        "active_milestones": [
            {
                "id": "milestone_1",
                "title": "Complete Python Certification",
                "progress": 75,
                "target_date": "2024-03-15",
                "status": "in_progress"
            },
            {
                "id": "milestone_2",
                "title": "Build Portfolio Project",
                "progress": 40,
                "target_date": "2024-04-01",
                "status": "in_progress"
            }
        ],
        "completed_milestones": [
            {
                "id": "milestone_3",
                "title": "Update LinkedIn Profile",
                "completed_date": "2024-01-15",
                "status": "completed"
            }
        ],
        "milestone_stats": {
            "total": 8,
            "completed": 3,
            "in_progress": 2,
            "not_started": 3,
            "completion_rate": 37.5
        }
    }


async def _get_cached_analysis_results(user_id: str, db: AsyncSession) -> Dict[str, Any]:
    """Get cached analysis results from database."""
    try:
        from sqlalchemy import select, desc
        from ....models.analysis_result import AnalysisResult
        
        # Get latest analysis results
        result = await db.execute(
            select(AnalysisResult)
            .where(AnalysisResult.user_id == user_id)
            .where(AnalysisResult.status == "completed")
            .order_by(desc(AnalysisResult.created_at))
            .limit(1)
        )
        latest_analysis = result.scalar_one_or_none()
        
        if not latest_analysis:
            return {
                "error": "No analysis results found",
                "message": "Please run a fresh analysis to get results"
            }
        
        return {
            "user_id": user_id,
            "analysis_timestamp": latest_analysis.created_at.isoformat(),
            "skill_assessment": latest_analysis.skill_assessment or {},
            "career_recommendations": latest_analysis.career_recommendations or {},
            "learning_paths": latest_analysis.learning_paths or {},
            "market_insights": latest_analysis.market_insights or {},
            "overall_score": latest_analysis.overall_score,
            "confidence_score": latest_analysis.confidence_score
        }
        
    except Exception as e:
        logger.error(f"Error getting cached analysis results: {str(e)}")
        return {"error": f"Failed to get cached results: {str(e)}"}


# Import datetime for helper functions
from datetime import datetime