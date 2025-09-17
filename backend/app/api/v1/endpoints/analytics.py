"""
Analytics API endpoints for career analysis and reporting
"""
import logging
from datetime import datetime
from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException, status, Query, BackgroundTasks
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession

from ....core.database import get_db
from ....api.dependencies import get_current_user
from ....models.user import User
from ....services.analytics_service import AnalyticsService
from ....services.visualization_service import VisualizationService
from ....services.pdf_report_service import PDFReportService
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


@router.get("/comprehensive-analytics", response_model=dict)
async def get_comprehensive_user_analytics(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get comprehensive user analytics with aggregation across all dimensions.
    
    Returns detailed analytics including skill distribution, experience analysis,
    market position, and career progression metrics.
    """
    try:
        analytics_service = AnalyticsService(db)
        analytics = await analytics_service.calculate_comprehensive_user_analytics(current_user.id)
        
        return analytics
        
    except AnalyticsError as e:
        logger.error(f"Error getting comprehensive analytics for user {current_user.id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error getting comprehensive analytics for user {current_user.id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get comprehensive analytics"
        )


@router.get("/strengths-improvements", response_model=dict)
async def get_strengths_and_improvements_analysis(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Analyze user strengths and improvement areas with detailed recommendations.
    
    Returns comprehensive analysis of user strengths, areas for improvement,
    and actionable recommendations for career development.
    """
    try:
        analytics_service = AnalyticsService(db)
        analysis = await analytics_service.analyze_strengths_and_improvements(current_user.id)
        
        return analysis
        
    except AnalyticsError as e:
        logger.error(f"Error analyzing strengths and improvements for user {current_user.id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error analyzing strengths and improvements for user {current_user.id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to analyze strengths and improvements"
        )


@router.get("/career-score", response_model=dict)
async def get_overall_career_score_and_recommendations(
    target_role: Optional[str] = Query(None, description="Target role for role-specific analysis"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Generate overall career score with comprehensive recommendations.
    
    Returns overall career score, role-specific analysis (if target role provided),
    comprehensive recommendations, priority actions, and career trajectory predictions.
    """
    try:
        analytics_service = AnalyticsService(db)
        score_analysis = await analytics_service.generate_overall_career_score_and_recommendations(
            current_user.id, target_role
        )
        
        return score_analysis
        
    except AnalyticsError as e:
        logger.error(f"Error generating career score for user {current_user.id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error generating career score for user {current_user.id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate career score and recommendations"
        )


@router.get("/analytics-summary", response_model=dict)
async def get_analytics_summary(
    target_role: Optional[str] = Query(None, description="Target role for enhanced analysis"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get a comprehensive analytics summary combining all analytics dimensions.
    
    Returns a consolidated view of user analytics including comprehensive metrics,
    strengths analysis, and career scoring with recommendations.
    """
    try:
        analytics_service = AnalyticsService(db)
        
        # Get all analytics components
        comprehensive_analytics = await analytics_service.calculate_comprehensive_user_analytics(current_user.id)
        strengths_analysis = await analytics_service.analyze_strengths_and_improvements(current_user.id)
        career_score = await analytics_service.generate_overall_career_score_and_recommendations(
            current_user.id, target_role
        )
        
        # Combine into summary
        summary = {
            "user_id": current_user.id,
            "overall_career_score": comprehensive_analytics["overall_career_score"],
            "analytics_summary": {
                "skill_analytics": comprehensive_analytics["skill_analytics"],
                "experience_analytics": comprehensive_analytics["experience_analytics"],
                "market_analytics": comprehensive_analytics["market_analytics"],
                "progression_analytics": comprehensive_analytics["progression_analytics"]
            },
            "strengths_summary": {
                "top_strengths": strengths_analysis["strengths"]["skills"][:5],
                "improvement_areas": strengths_analysis["improvement_areas"]["areas"][:5],
                "strength_score": strengths_analysis["strengths"]["overall_strength_score"],
                "improvement_urgency": strengths_analysis["improvement_areas"]["urgency_score"]
            },
            "career_score_summary": {
                "overall_score": career_score["overall_career_score"],
                "role_specific_score": career_score.get("role_specific_score"),
                "target_role": career_score.get("target_role"),
                "priority_actions": career_score["priority_actions"][:3],
                "trajectory_predictions": career_score["trajectory_predictions"]
            },
            "key_recommendations": career_score["comprehensive_recommendations"][:5],
            "generated_at": datetime.utcnow().isoformat()
        }
        
        return summary
        
    except AnalyticsError as e:
        logger.error(f"Error generating analytics summary for user {current_user.id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error generating analytics summary for user {current_user.id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate analytics summary"
        )


@router.post("/reports/pdf", response_model=PDFReportResponse)
async def generate_pdf_report(
    request: PDFReportRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Generate comprehensive PDF career analysis report"""
    try:
        # Verify user can access this report
        if request.user_id != current_user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Cannot generate report for another user"
            )
        
        # Generate comprehensive analytics report first
        analytics_request = AnalyticsRequest(
            user_id=request.user_id,
            analysis_types=["full_report"],
            include_job_matches=True,
            include_progress_tracking=True
        )
        
        analytics_service = AnalyticsService(db)
        report_data = await analytics_service.generate_comprehensive_report(analytics_request)
        
        # Generate PDF report
        pdf_service = PDFReportService()
        pdf_response = await pdf_service.generate_comprehensive_report(report_data, request)
        
        return pdf_response
        
    except Exception as e:
        logger.error(f"Error generating PDF report: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate PDF report: {str(e)}"
        )


@router.get("/reports/{report_id}/download")
async def download_pdf_report(
    report_id: str,
    current_user: User = Depends(get_current_user)
):
    """Download generated PDF report"""
    try:
        pdf_service = PDFReportService()
        file_path = await pdf_service.get_report_file(report_id)
        
        if not file_path or not file_path.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Report not found or expired"
            )
        
        # Verify user owns this report (basic check via filename)
        if current_user.id not in str(file_path.name):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to this report"
            )
        
        return FileResponse(
            path=str(file_path),
            filename=f"career_analysis_report_{current_user.id}.pdf",
            media_type="application/pdf"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading PDF report: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to download report"
        )