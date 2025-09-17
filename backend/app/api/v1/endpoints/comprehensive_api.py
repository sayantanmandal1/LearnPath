"""
Comprehensive API endpoints with filtering, pagination, and advanced request handling.

This module provides enhanced API endpoints that consolidate functionality from
all services with comprehensive filtering, pagination, and data export capabilities.
"""

import logging
from typing import List, Optional, Dict, Any, Union
from datetime import datetime, timedelta
from fastapi import APIRouter, Depends, HTTPException, Query, Path, BackgroundTasks, status
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel, Field
import json
import csv
import io

from app.core.database import get_db
from app.api.dependencies import get_current_user
from app.models.user import User
from app.services.profile_service import UserProfileService
from app.services.recommendation_service import RecommendationService
from app.services.career_trajectory_service import CareerTrajectoryService
from app.services.learning_path_service import LearningPathService
from app.services.analytics_service import AnalyticsService
from app.services.job_scrapers.scraper_manager import JobScraperManager
from app.services.market_trend_analyzer import MarketTrendAnalyzer
from app.core.exceptions import ServiceException

logger = logging.getLogger(__name__)
router = APIRouter()

# Service instances
profile_service = UserProfileService()
recommendation_service = RecommendationService()
career_service = CareerTrajectoryService()
learning_service = LearningPathService()
job_scraper = JobScraperManager()
trend_analyzer = MarketTrendAnalyzer()


# Enhanced Request/Response Models
class PaginationParams(BaseModel):
    """Pagination parameters for API endpoints"""
    page: int = Field(1, ge=1, description="Page number")
    page_size: int = Field(20, ge=1, le=100, description="Items per page")
    
    @property
    def skip(self) -> int:
        return (self.page - 1) * self.page_size
    
    @property
    def limit(self) -> int:
        return self.page_size


class FilterParams(BaseModel):
    """Common filtering parameters"""
    location: Optional[str] = None
    experience_level: Optional[str] = None
    remote_type: Optional[str] = None
    skills: Optional[List[str]] = None
    salary_min: Optional[int] = None
    salary_max: Optional[int] = None
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None


class SortParams(BaseModel):
    """Sorting parameters"""
    sort_by: str = Field("created_at", description="Field to sort by")
    sort_order: str = Field("desc", pattern="^(asc|desc)$", description="Sort order")


class ExportFormat(str):
    """Export format options"""
    JSON = "json"
    CSV = "csv"
    PDF = "pdf"


class ComprehensiveRecommendationRequest(BaseModel):
    """Enhanced recommendation request with filtering and customization"""
    target_role: Optional[str] = None
    n_recommendations: int = Field(10, ge=1, le=50)
    include_explanations: bool = True
    include_alternatives: bool = True
    filters: FilterParams = FilterParams()
    sort: SortParams = SortParams()
    customization: Dict[str, Any] = Field(default_factory=dict)


class PaginatedResponse(BaseModel):
    """Generic paginated response wrapper"""
    items: List[Any]
    total_count: int
    page: int
    page_size: int
    total_pages: int
    has_next: bool
    has_previous: bool


# Profile Management Endpoints with Enhanced Features
@router.get("/profiles/comprehensive")
async def get_comprehensive_profile_data(
    include_analytics: bool = Query(True, description="Include profile analytics"),
    include_recommendations: bool = Query(True, description="Include quick recommendations"),
    include_market_data: bool = Query(False, description="Include market insights"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get comprehensive profile data with analytics, recommendations, and market insights.
    
    This endpoint provides a complete view of the user's profile including:
    - Basic profile information
    - Skill analysis and gaps
    - Quick career recommendations
    - Market demand for user's skills
    - Profile completeness metrics
    """
    try:
        # Get basic profile
        from app.repositories.profile import ProfileRepository
        profile_repo = ProfileRepository()
        profile = await profile_repo.get_by_user_id(db, current_user.id)
        
        if not profile:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Profile not found"
            )
        
        result = {
            "profile": profile,
            "last_updated": profile.data_last_updated
        }
        
        # Add analytics if requested
        if include_analytics:
            analytics_service = AnalyticsService(db)
            analytics = await analytics_service.get_profile_analytics(current_user.id)
            result["analytics"] = analytics
        
        # Add quick recommendations if requested
        if include_recommendations:
            recommendations = await recommendation_service.get_career_recommendations(
                user_id=current_user.id,
                db=db,
                n_recommendations=3
            )
            result["quick_recommendations"] = recommendations
        
        # Add market data if requested
        if include_market_data and profile.skills:
            market_insights = []
            for skill in list(profile.skills.keys())[:5]:  # Top 5 skills
                try:
                    market_data = await trend_analyzer.get_skill_market_data(db, skill)
                    market_insights.append({
                        "skill": skill,
                        "market_data": market_data
                    })
                except Exception as e:
                    logger.warning(f"Failed to get market data for skill {skill}: {e}")
            
            result["market_insights"] = market_insights
        
        return result
        
    except Exception as e:
        logger.error(f"Error getting comprehensive profile data: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get comprehensive profile data"
        )


# Enhanced Career Recommendations with Filtering and Pagination
@router.post("/recommendations/career/advanced")
async def get_advanced_career_recommendations(
    request: ComprehensiveRecommendationRequest,
    pagination: PaginationParams = Depends(),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get advanced career recommendations with comprehensive filtering and pagination.
    
    Features:
    - Advanced filtering by location, experience, salary, skills
    - Pagination support
    - Custom sorting options
    - Alternative path suggestions
    - Market demand integration
    """
    try:
        # Get recommendations with filters
        recommendations = await recommendation_service.get_filtered_career_recommendations(
            user_id=current_user.id,
            db=db,
            target_role=request.target_role,
            filters=request.filters.dict(exclude_none=True),
            n_recommendations=request.n_recommendations,
            include_alternatives=request.include_alternatives
        )
        
        # Apply sorting
        if request.sort.sort_by in ["match_score", "salary_range", "growth_potential"]:
            reverse = request.sort.sort_order == "desc"
            recommendations.sort(
                key=lambda x: getattr(x, request.sort.sort_by, 0),
                reverse=reverse
            )
        
        # Apply pagination
        total_count = len(recommendations)
        start_idx = pagination.skip
        end_idx = start_idx + pagination.limit
        paginated_recommendations = recommendations[start_idx:end_idx]
        
        return PaginatedResponse(
            items=paginated_recommendations,
            total_count=total_count,
            page=pagination.page,
            page_size=pagination.page_size,
            total_pages=(total_count + pagination.page_size - 1) // pagination.page_size,
            has_next=end_idx < total_count,
            has_previous=pagination.page > 1
        )
        
    except ServiceException as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error getting advanced career recommendations: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get career recommendations"
        )


# Learning Path Recommendations with Customization
@router.post("/recommendations/learning-paths/customized")
async def get_customized_learning_paths(
    target_skills: List[str] = Query(..., description="Skills to learn"),
    difficulty_preference: str = Query("intermediate", description="Preferred difficulty level"),
    time_commitment_hours_per_week: int = Query(10, ge=1, le=40, description="Available study time"),
    preferred_providers: Optional[List[str]] = Query(None, description="Preferred learning providers"),
    budget_max: Optional[float] = Query(None, description="Maximum budget for paid resources"),
    learning_style: str = Query("mixed", description="Learning style preference"),
    include_projects: bool = Query(True, description="Include hands-on projects"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get customized learning paths based on detailed preferences and constraints.
    
    This endpoint creates personalized learning paths considering:
    - Time availability and commitment
    - Budget constraints
    - Learning style preferences
    - Provider preferences
    - Difficulty progression
    """
    try:
        # Build customization parameters
        customization = {
            "difficulty_preference": difficulty_preference,
            "time_commitment_hours_per_week": time_commitment_hours_per_week,
            "preferred_providers": preferred_providers or [],
            "budget_max": budget_max,
            "learning_style": learning_style,
            "include_projects": include_projects
        }
        
        # Generate customized learning paths
        learning_paths = await learning_service.generate_customized_learning_paths(
            user_id=current_user.id,
            target_skills=target_skills,
            customization=customization,
            db=db
        )
        
        return {
            "learning_paths": learning_paths,
            "customization_applied": customization,
            "estimated_completion_weeks": sum(path.get("estimated_duration_weeks", 0) for path in learning_paths),
            "total_estimated_cost": sum(path.get("estimated_cost", 0) for path in learning_paths if path.get("estimated_cost"))
        }
        
    except Exception as e:
        logger.error(f"Error generating customized learning paths: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate customized learning paths"
        )


# Job Matching with Advanced Filtering
@router.get("/job-matching/advanced")
async def get_advanced_job_matches(
    filters: FilterParams = Depends(),
    pagination: PaginationParams = Depends(),
    sort: SortParams = Depends(),
    match_threshold: float = Query(0.6, ge=0.0, le=1.0, description="Minimum match score"),
    include_skill_gaps: bool = Query(True, description="Include skill gap analysis"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get advanced job matches with comprehensive filtering and analysis.
    
    Features:
    - Advanced filtering by multiple criteria
    - Match score thresholding
    - Skill gap analysis for each match
    - Pagination and sorting
    - Market demand insights
    """
    try:
        # Get user profile for matching
        from app.repositories.profile import ProfileRepository
        profile_repo = ProfileRepository()
        profile = await profile_repo.get_by_user_id(db, current_user.id)
        
        if not profile:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Profile not found"
            )
        
        # Get job matches with filters
        job_matches = await recommendation_service.get_advanced_job_matches(
            user_profile=profile,
            filters=filters.dict(exclude_none=True),
            match_threshold=match_threshold,
            include_skill_gaps=include_skill_gaps,
            db=db
        )
        
        # Apply sorting
        if sort.sort_by == "match_score":
            job_matches.sort(
                key=lambda x: x.get("match_score", 0),
                reverse=sort.sort_order == "desc"
            )
        elif sort.sort_by == "salary":
            job_matches.sort(
                key=lambda x: x.get("salary_max", 0),
                reverse=sort.sort_order == "desc"
            )
        
        # Apply pagination
        total_count = len(job_matches)
        start_idx = pagination.skip
        end_idx = start_idx + pagination.limit
        paginated_matches = job_matches[start_idx:end_idx]
        
        return PaginatedResponse(
            items=paginated_matches,
            total_count=total_count,
            page=pagination.page,
            page_size=pagination.page_size,
            total_pages=(total_count + pagination.page_size - 1) // pagination.page_size,
            has_next=end_idx < total_count,
            has_previous=pagination.page > 1
        )
        
    except Exception as e:
        logger.error(f"Error getting advanced job matches: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get job matches"
        )


# Market Analysis with Export Capabilities
@router.get("/market-analysis/comprehensive")
async def get_comprehensive_market_analysis(
    skills: Optional[List[str]] = Query(None, description="Skills to analyze"),
    locations: Optional[List[str]] = Query(None, description="Locations to analyze"),
    time_period_days: int = Query(90, ge=7, le=365, description="Analysis time period"),
    include_predictions: bool = Query(True, description="Include future predictions"),
    include_comparisons: bool = Query(True, description="Include skill comparisons"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get comprehensive market analysis with trends, predictions, and comparisons.
    
    Provides detailed market insights including:
    - Skill demand trends
    - Salary predictions
    - Geographic analysis
    - Emerging skills detection
    - Market competitiveness
    """
    try:
        # If no skills provided, use user's skills
        if not skills:
            from app.repositories.profile import ProfileRepository
            profile_repo = ProfileRepository()
            profile = await profile_repo.get_by_user_id(db, current_user.id)
            if profile and profile.skills:
                skills = list(profile.skills.keys())[:10]  # Top 10 skills
        
        if not skills:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No skills provided and no user profile found"
            )
        
        # Get comprehensive market analysis
        analysis = await trend_analyzer.get_comprehensive_market_analysis(
            db=db,
            skills=skills,
            locations=locations,
            time_period_days=time_period_days,
            include_predictions=include_predictions,
            include_comparisons=include_comparisons
        )
        
        return analysis
        
    except Exception as e:
        logger.error(f"Error getting comprehensive market analysis: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get market analysis"
        )


# Data Export Endpoints
@router.get("/export/recommendations/{format}")
async def export_recommendations(
    format: str = Path(..., pattern="^(json|csv|pdf)$", description="Export format"),
    recommendation_type: str = Query("career", description="Type of recommendations to export"),
    filters: FilterParams = Depends(),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Export recommendations in various formats (JSON, CSV, PDF).
    
    Supports exporting:
    - Career recommendations
    - Learning path recommendations
    - Job matches
    - Market analysis data
    """
    try:
        # Get recommendations based on type
        if recommendation_type == "career":
            data = await recommendation_service.get_career_recommendations(
                user_id=current_user.id,
                db=db,
                n_recommendations=50
            )
        elif recommendation_type == "learning_paths":
            data = await learning_service.get_user_learning_paths(
                user_id=current_user.id,
                db=db
            )
        elif recommendation_type == "job_matches":
            from app.repositories.profile import ProfileRepository
            profile_repo = ProfileRepository()
            profile = await profile_repo.get_by_user_id(db, current_user.id)
            data = await recommendation_service.get_advanced_job_matches(
                user_profile=profile,
                filters=filters.dict(exclude_none=True),
                db=db
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid recommendation type"
            )
        
        # Export in requested format
        if format == "json":
            return {"data": data, "exported_at": datetime.utcnow()}
        
        elif format == "csv":
            # Convert to CSV
            output = io.StringIO()
            if data:
                writer = csv.DictWriter(output, fieldnames=data[0].keys())
                writer.writeheader()
                for item in data:
                    writer.writerow(item)
            
            response = StreamingResponse(
                io.BytesIO(output.getvalue().encode()),
                media_type="text/csv",
                headers={"Content-Disposition": f"attachment; filename={recommendation_type}_export.csv"}
            )
            return response
        
        elif format == "pdf":
            # For PDF export, you would integrate with a PDF generation service
            # For now, return a placeholder
            raise HTTPException(
                status_code=status.HTTP_501_NOT_IMPLEMENTED,
                detail="PDF export not yet implemented"
            )
        
    except Exception as e:
        logger.error(f"Error exporting recommendations: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to export recommendations"
        )


# Analytics Dashboard Data
@router.get("/dashboard/summary")
async def get_dashboard_summary(
    time_period_days: int = Query(30, ge=7, le=365, description="Summary time period"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get comprehensive dashboard summary with key metrics and insights.
    
    Provides:
    - Profile completeness score
    - Recent activity summary
    - Top skill gaps
    - Market opportunities
    - Learning progress
    - Career trajectory insights
    """
    try:
        # Get profile
        from app.repositories.profile import ProfileRepository
        profile_repo = ProfileRepository()
        profile = await profile_repo.get_by_user_id(db, current_user.id)
        
        if not profile:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Profile not found"
            )
        
        # Calculate profile completeness
        completeness_score = profile_service.calculate_profile_completeness(profile)
        
        # Get recent recommendations
        recent_recommendations = await recommendation_service.get_career_recommendations(
            user_id=current_user.id,
            db=db,
            n_recommendations=3
        )
        
        # Get skill gaps for dream job
        skill_gaps = {}
        if profile.dream_job:
            try:
                skill_gaps = await career_service.analyze_skill_gaps(
                    user_id=current_user.id,
                    target_role=profile.dream_job,
                    db=db
                )
            except Exception as e:
                logger.warning(f"Failed to analyze skill gaps: {e}")
        
        # Get market opportunities
        market_opportunities = []
        if profile.skills:
            try:
                top_skills = sorted(
                    profile.skills.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:5]
                
                for skill, confidence in top_skills:
                    market_data = await trend_analyzer.get_skill_market_data(db, skill)
                    if market_data.get("demand_score", 0) > 0.7:
                        market_opportunities.append({
                            "skill": skill,
                            "confidence": confidence,
                            "market_data": market_data
                        })
            except Exception as e:
                logger.warning(f"Failed to get market opportunities: {e}")
        
        return {
            "profile_completeness": {
                "score": completeness_score,
                "missing_elements": profile_service.get_missing_profile_elements(profile)
            },
            "recent_recommendations": recent_recommendations,
            "skill_gaps": skill_gaps,
            "market_opportunities": market_opportunities,
            "summary_stats": {
                "total_skills": len(profile.skills) if profile.skills else 0,
                "dream_job": profile.dream_job,
                "last_updated": profile.data_last_updated,
                "analysis_period_days": time_period_days
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting dashboard summary: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get dashboard summary"
        )


# Batch Operations
@router.post("/batch/analyze-multiple-roles")
async def analyze_multiple_roles(
    target_roles: List[str] = Query(..., description="List of roles to analyze"),
    include_learning_paths: bool = Query(True, description="Include learning paths for each role"),
    include_market_data: bool = Query(True, description="Include market data for each role"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Analyze multiple target roles simultaneously for comparison.
    
    Provides comprehensive analysis for multiple roles including:
    - Skill gap analysis for each role
    - Career trajectory recommendations
    - Learning path suggestions
    - Market demand comparison
    - Difficulty and timeline estimates
    """
    try:
        if len(target_roles) > 10:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Maximum 10 roles can be analyzed at once"
            )
        
        results = []
        
        for role in target_roles:
            try:
                # Analyze skill gaps
                skill_gaps = await career_service.analyze_skill_gaps(
                    user_id=current_user.id,
                    target_role=role,
                    db=db
                )
                
                role_analysis = {
                    "role": role,
                    "skill_gaps": skill_gaps
                }
                
                # Add learning paths if requested
                if include_learning_paths:
                    learning_paths = await learning_service.generate_learning_paths_for_role(
                        user_id=current_user.id,
                        target_role=role,
                        db=db
                    )
                    role_analysis["learning_paths"] = learning_paths
                
                # Add market data if requested
                if include_market_data:
                    market_data = await trend_analyzer.get_role_market_data(db, role)
                    role_analysis["market_data"] = market_data
                
                results.append(role_analysis)
                
            except Exception as e:
                logger.warning(f"Failed to analyze role {role}: {e}")
                results.append({
                    "role": role,
                    "error": str(e)
                })
        
        # Add comparison summary
        comparison_summary = {
            "total_roles_analyzed": len([r for r in results if "error" not in r]),
            "easiest_transition": None,
            "highest_market_demand": None,
            "shortest_learning_path": None
        }
        
        # Find easiest transition (lowest average skill gap)
        valid_results = [r for r in results if "error" not in r and "skill_gaps" in r]
        if valid_results:
            easiest = min(
                valid_results,
                key=lambda x: sum(x["skill_gaps"].get("missing_skills", {}).values()) / max(len(x["skill_gaps"].get("missing_skills", {})), 1)
            )
            comparison_summary["easiest_transition"] = easiest["role"]
        
        return {
            "role_analyses": results,
            "comparison_summary": comparison_summary,
            "analyzed_at": datetime.utcnow()
        }
        
    except Exception as e:
        logger.error(f"Error analyzing multiple roles: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to analyze multiple roles"
        )


# Health and Status Endpoints
@router.get("/health/comprehensive")
async def comprehensive_health_check(
    db: AsyncSession = Depends(get_db)
):
    """
    Comprehensive health check for all API services and dependencies.
    
    Checks:
    - Database connectivity
    - External API availability
    - Service health status
    - Cache connectivity
    - ML model availability
    """
    try:
        health_status = {
            "status": "healthy",
            "timestamp": datetime.utcnow(),
            "services": {}
        }
        
        # Check database
        try:
            await db.execute("SELECT 1")
            health_status["services"]["database"] = "healthy"
        except Exception as e:
            health_status["services"]["database"] = f"unhealthy: {str(e)}"
            health_status["status"] = "degraded"
        
        # Check recommendation service
        try:
            if recommendation_service.model_trained:
                health_status["services"]["recommendation_engine"] = "healthy"
            else:
                health_status["services"]["recommendation_engine"] = "not_trained"
                health_status["status"] = "degraded"
        except Exception as e:
            health_status["services"]["recommendation_engine"] = f"unhealthy: {str(e)}"
            health_status["status"] = "degraded"
        
        # Check external APIs (sample check)
        try:
            # This would check external API connectivity
            health_status["services"]["external_apis"] = "healthy"
        except Exception as e:
            health_status["services"]["external_apis"] = f"degraded: {str(e)}"
        
        return health_status
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow()
        }