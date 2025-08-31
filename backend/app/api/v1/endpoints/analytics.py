"""
Analytics API endpoints for career analysis and reporting
"""
import logging
from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException, status, Query, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession

from ....core.database import get_db
from ....api.dependencies import get_current_user
from ....models.user import User
from ....services.analytics_service import AnalyticsService
from ....services.visualization_service import VisualizationService
# from ....services.pdf_report_service import PDFReportService
from ....schemas.analytics import (
    AnalyticsRequest, CareerAnalysisReport, SkillRadarChart, CareerRoadmapVisualization,
    SkillGapReport, JobCompatibilityReport, HistoricalProgressReport,
    ChartConfiguration, ChartType, VisualizationResponse,
    PDFReportRequest, PDFReportResponse
)
from ....core.exceptions import AnalyticsError, VisualizationError, PDFGenerationError

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/generate-report", response_model=CareerAnalysisReport)
async def generate_comprehensive_analytics_report(
    request: AnalyticsRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Generate comprehensive career analysis report with all requested analytics.
    
    This endpoint generates a complete career analysis including:
    - Skill radar charts
    - Career roadmap visualization
    - Skill gap analysis
    - Job compatibility scoring
    - Historical progress tracking
    """
    try:
        # Ensure user can only request their own analytics
        if request.user_id != current_user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Can only request analytics for your own profile"
            )
        
        analytics_service = AnalyticsService(db)
        report = await analytics_service.generate_comprehensive_report(request)
        
        return report
        
    except AnalyticsError as e:
        logger.error(f"Analytics error for user {current_user.id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error generating analytics report for user {current_user.id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate analytics report"
        )


@router.get("/skill-radar", response_model=SkillRadarChart)
async def get_skill_radar_chart(
    target_role: Optional[str] = Query(None, description="Target role for comparison"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Generate skill radar chart with professional visualization data.
    
    Returns radar chart data showing user's skills compared to market averages
    and optionally target role requirements.
    """
    try:
        analytics_service = AnalyticsService(db)
        radar_chart = await analytics_service.generate_skill_radar_chart(
            current_user.id, target_role
        )
        
        return radar_chart
        
    except AnalyticsError as e:
        logger.error(f"Error generating skill radar chart for user {current_user.id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error generating skill radar chart for user {current_user.id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate skill radar chart"
        )


@router.get("/career-roadmap", response_model=CareerRoadmapVisualization)
async def get_career_roadmap(
    target_role: str = Query(..., description="Target career role"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Generate interactive career roadmap visualization.
    
    Returns roadmap with nodes representing career milestones and edges
    showing progression paths with difficulty and timeline estimates.
    """
    try:
        analytics_service = AnalyticsService(db)
        roadmap = await analytics_service.generate_career_roadmap(
            current_user.id, target_role
        )
        
        return roadmap
        
    except AnalyticsError as e:
        logger.error(f"Error generating career roadmap for user {current_user.id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error generating career roadmap for user {current_user.id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate career roadmap"
        )


@router.get("/skill-gaps", response_model=SkillGapReport)
async def analyze_skill_gaps(
    target_role: str = Query(..., description="Target career role"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Analyze skill gaps with progress tracking capabilities.
    
    Returns detailed analysis of skill gaps between current profile
    and target role requirements, with learning recommendations.
    """
    try:
        analytics_service = AnalyticsService(db)
        gap_report = await analytics_service.analyze_skill_gaps(
            current_user.id, target_role
        )
        
        return gap_report
        
    except AnalyticsError as e:
        logger.error(f"Error analyzing skill gaps for user {current_user.id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error analyzing skill gaps for user {current_user.id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to analyze skill gaps"
        )


@router.get("/job-compatibility", response_model=JobCompatibilityReport)
async def get_job_compatibility_scores(
    location: Optional[str] = Query(None, description="Job location filter"),
    experience_level: Optional[str] = Query(None, description="Experience level filter"),
    remote_type: Optional[str] = Query(None, description="Remote work type filter"),
    limit: int = Query(20, ge=1, le=100, description="Maximum number of jobs to analyze"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Generate job compatibility scoring with requirement overlay.
    
    Returns compatibility scores for available job postings with
    detailed skill matching and recommendations.
    """
    try:
        # Build filters
        filters = {}
        if location:
            filters["location"] = location
        if experience_level:
            filters["experience_level"] = experience_level
        if remote_type:
            filters["remote_type"] = remote_type
        
        analytics_service = AnalyticsService(db)
        compatibility_report = await analytics_service.generate_job_compatibility_scores(
            current_user.id, filters, limit
        )
        
        return compatibility_report
        
    except AnalyticsError as e:
        logger.error(f"Error generating job compatibility scores for user {current_user.id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error generating job compatibility scores for user {current_user.id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate job compatibility scores"
        )


@router.get("/progress-tracking", response_model=HistoricalProgressReport)
async def get_progress_tracking(
    tracking_period_days: int = Query(90, ge=7, le=365, description="Tracking period in days"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Track historical progress and improvement trends.
    
    Returns analysis of skill improvements and career progress
    over the specified tracking period.
    """
    try:
        analytics_service = AnalyticsService(db)
        progress_report = await analytics_service.track_historical_progress(
            current_user.id, tracking_period_days
        )
        
        return progress_report
        
    except AnalyticsError as e:
        logger.error(f"Error tracking progress for user {current_user.id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error tracking progress for user {current_user.id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to track progress"
        )


@router.post("/visualizations/skill-radar", response_model=VisualizationResponse)
async def create_skill_radar_visualization(
    config: ChartConfiguration,
    target_role: Optional[str] = Query(None, description="Target role for comparison"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Create skill radar chart visualization with custom configuration.
    
    Returns visualization data optimized for frontend chart libraries.
    """
    try:
        # Get radar chart data
        analytics_service = AnalyticsService(db)
        radar_data = await analytics_service.generate_skill_radar_chart(
            current_user.id, target_role
        )
        
        # Generate visualization
        viz_service = VisualizationService()
        chart_data = viz_service.generate_skill_radar_chart_data(radar_data, config)
        
        response = viz_service.create_visualization_response(
            ChartType.RADAR, chart_data, config
        )
        
        return response
        
    except (AnalyticsError, VisualizationError) as e:
        logger.error(f"Error creating radar visualization for user {current_user.id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error creating radar visualization for user {current_user.id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create radar visualization"
        )


@router.post("/visualizations/career-roadmap", response_model=VisualizationResponse)
async def create_career_roadmap_visualization(
    config: ChartConfiguration,
    target_role: str = Query(..., description="Target career role"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Create career roadmap visualization with interactive elements.
    
    Returns network graph data for interactive roadmap visualization.
    """
    try:
        # Get roadmap data
        analytics_service = AnalyticsService(db)
        roadmap_data = await analytics_service.generate_career_roadmap(
            current_user.id, target_role
        )
        
        # Generate visualization
        viz_service = VisualizationService()
        chart_data = viz_service.generate_career_roadmap_data(roadmap_data, config)
        
        response = viz_service.create_visualization_response(
            ChartType.SCATTER, chart_data, config  # Using scatter for network graph
        )
        
        return response
        
    except (AnalyticsError, VisualizationError) as e:
        logger.error(f"Error creating roadmap visualization for user {current_user.id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error creating roadmap visualization for user {current_user.id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create roadmap visualization"
        )


@router.post("/visualizations/skill-gaps", response_model=VisualizationResponse)
async def create_skill_gap_visualization(
    config: ChartConfiguration,
    target_role: str = Query(..., description="Target career role"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Create skill gap analysis visualization.
    
    Returns bar chart data showing current vs target skill levels.
    """
    try:
        # Get skill gap data
        analytics_service = AnalyticsService(db)
        gap_data = await analytics_service.analyze_skill_gaps(
            current_user.id, target_role
        )
        
        # Generate visualization
        viz_service = VisualizationService()
        chart_data = viz_service.generate_skill_gap_chart_data(gap_data, config)
        
        response = viz_service.create_visualization_response(
            ChartType.BAR, chart_data, config
        )
        
        return response
        
    except (AnalyticsError, VisualizationError) as e:
        logger.error(f"Error creating skill gap visualization for user {current_user.id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error creating skill gap visualization for user {current_user.id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create skill gap visualization"
        )


# PDF endpoints temporarily disabled due to import issues
# @router.post("/reports/pdf", response_model=PDFReportResponse)
# async def generate_pdf_report(...):
#     pass