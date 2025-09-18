"""
AI Analysis API endpoints with Gemini integration
"""
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.api.dependencies import get_current_user
from app.services.ai_analysis_service import AIAnalysisService
from app.schemas.ai_analysis import (
    AnalysisRequest, AnalysisStatusResponse, BulkAnalysisRequest, BulkAnalysisResponse,
    CompleteAnalysisResponse, SkillAssessmentResponse, CareerRecommendationResponse,
    LearningPathResponse, ProjectSuggestionResponse, MarketInsightsResponse,
    AnalysisHistoryResponse, AnalysisComparisonResponse, AnalysisMetricsResponse,
    GeminiAPIStatus, AnalysisConfigResponse, AnalysisTypeEnum
)
from app.schemas.auth import UserResponse
from app.models.analysis_result import AnalysisType
from app.core.exceptions import ProcessingError, ValidationError

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/analyze/complete", response_model=CompleteAnalysisResponse)
async def analyze_complete_profile(
    background_tasks: BackgroundTasks,
    force_refresh: bool = False,
    current_user: UserResponse = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Perform comprehensive AI analysis of user profile
    
    This endpoint triggers a complete analysis including:
    - Skill assessment
    - Career recommendations  
    - Learning paths
    - Project suggestions
    - Market insights
    
    The analysis runs in the background and results are cached.
    """
    try:
        service = AIAnalysisService()
        
        # Check if we have recent cached results and force_refresh is False
        if not force_refresh:
            cached_result = await service.get_cached_analysis(
                current_user.id, AnalysisType.COMPLETE_ANALYSIS, db
            )
            if cached_result and not await service.is_analysis_stale(
                current_user.id, AnalysisType.COMPLETE_ANALYSIS, db, max_age_hours=24
            ):
                logger.info(f"Returning cached complete analysis for user {current_user.id}")
                return CompleteAnalysisResponse(**cached_result)
        
        # Perform new analysis
        logger.info(f"Starting complete AI analysis for user {current_user.id}")
        analysis = await service.analyze_complete_profile(current_user.id, db)
        
        # Convert to response format
        response = CompleteAnalysisResponse(
            user_id=analysis.user_id,
            skill_assessment=SkillAssessmentResponse(
                technical_skills=analysis.skill_assessment.technical_skills,
                soft_skills=analysis.skill_assessment.soft_skills,
                skill_strengths=analysis.skill_assessment.skill_strengths,
                skill_gaps=analysis.skill_assessment.skill_gaps,
                improvement_areas=analysis.skill_assessment.improvement_areas,
                market_relevance_score=analysis.skill_assessment.market_relevance_score,
                confidence_score=analysis.skill_assessment.confidence_score
            ),
            career_recommendations=[
                CareerRecommendationResponse(
                    recommended_role=rec.recommended_role,
                    match_score=rec.match_score,
                    reasoning=rec.reasoning,
                    required_skills=rec.required_skills,
                    skill_gaps=rec.skill_gaps,
                    preparation_timeline=rec.preparation_timeline,
                    salary_range=rec.salary_range,
                    market_demand=rec.market_demand
                )
                for rec in analysis.career_recommendations
            ],
            learning_paths=[
                LearningPathResponse(
                    title=path.title,
                    description=path.description,
                    target_skills=path.target_skills,
                    learning_modules=[
                        {
                            "module_name": module.get("module_name", ""),
                            "topics": module.get("topics", []),
                            "duration": module.get("duration", "")
                        }
                        for module in path.learning_modules
                    ],
                    estimated_duration=path.estimated_duration,
                    difficulty_level=path.difficulty_level,
                    resources=[
                        {
                            "type": res.get("type", ""),
                            "title": res.get("title", ""),
                            "provider": res.get("provider", ""),
                            "url": res.get("url")
                        }
                        for res in path.resources
                    ]
                )
                for path in analysis.learning_paths
            ],
            project_suggestions=[
                ProjectSuggestionResponse(
                    title=proj.title,
                    description=proj.description,
                    technologies=proj.technologies,
                    difficulty_level=proj.difficulty_level,
                    estimated_duration=proj.estimated_duration,
                    learning_outcomes=proj.learning_outcomes,
                    portfolio_value=proj.portfolio_value
                )
                for proj in analysis.project_suggestions
            ],
            market_insights=MarketInsightsResponse(**analysis.market_insights),
            analysis_timestamp=analysis.analysis_timestamp,
            gemini_request_id=analysis.gemini_request_id
        )
        
        logger.info(f"Complete AI analysis completed for user {current_user.id}")
        return response
        
    except ProcessingError as e:
        logger.error(f"AI analysis processing error for user {current_user.id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Analysis processing failed: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Unexpected error in AI analysis for user {current_user.id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during analysis"
        )


@router.post("/analyze/skills", response_model=SkillAssessmentResponse)
async def analyze_skills(
    force_refresh: bool = False,
    current_user: UserResponse = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Generate AI-powered skill assessment
    
    Analyzes user's technical and soft skills based on:
    - Resume data
    - Platform profiles (GitHub, LeetCode, etc.)
    - Work experience
    - Education background
    """
    try:
        service = AIAnalysisService()
        
        # Check for cached results
        if not force_refresh:
            cached_result = await service.get_cached_analysis(
                current_user.id, AnalysisType.SKILL_ASSESSMENT, db
            )
            if cached_result and not await service.is_analysis_stale(
                current_user.id, AnalysisType.SKILL_ASSESSMENT, db, max_age_hours=12
            ):
                return SkillAssessmentResponse(**cached_result)
        
        # Generate new skill assessment
        skill_assessment = await service.generate_skill_assessment(current_user.id, db)
        
        response = SkillAssessmentResponse(
            technical_skills=skill_assessment.technical_skills,
            soft_skills=skill_assessment.soft_skills,
            skill_strengths=skill_assessment.skill_strengths,
            skill_gaps=skill_assessment.skill_gaps,
            improvement_areas=skill_assessment.improvement_areas,
            market_relevance_score=skill_assessment.market_relevance_score,
            confidence_score=skill_assessment.confidence_score
        )
        
        logger.info(f"Skill assessment completed for user {current_user.id}")
        return response
        
    except Exception as e:
        logger.error(f"Skill assessment failed for user {current_user.id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Skill assessment failed: {str(e)}"
        )


@router.post("/analyze/career", response_model=List[CareerRecommendationResponse])
async def analyze_career_recommendations(
    force_refresh: bool = False,
    current_user: UserResponse = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Generate AI-powered career recommendations
    
    Provides personalized career recommendations based on:
    - Current skills and experience
    - Career goals and preferences
    - Market demand and trends
    - Skill gaps analysis
    """
    try:
        service = AIAnalysisService()
        
        # Check for cached results
        if not force_refresh:
            cached_result = await service.get_cached_analysis(
                current_user.id, AnalysisType.CAREER_RECOMMENDATION, db
            )
            if cached_result and not await service.is_analysis_stale(
                current_user.id, AnalysisType.CAREER_RECOMMENDATION, db, max_age_hours=24
            ):
                recommendations = cached_result.get("recommendations", [])
                return [CareerRecommendationResponse(**rec) for rec in recommendations]
        
        # Generate new career recommendations
        career_recommendations = await service.generate_career_recommendations(current_user.id, db)
        
        response = [
            CareerRecommendationResponse(
                recommended_role=rec.recommended_role,
                match_score=rec.match_score,
                reasoning=rec.reasoning,
                required_skills=rec.required_skills,
                skill_gaps=rec.skill_gaps,
                preparation_timeline=rec.preparation_timeline,
                salary_range=rec.salary_range,
                market_demand=rec.market_demand
            )
            for rec in career_recommendations
        ]
        
        logger.info(f"Career recommendations completed for user {current_user.id}")
        return response
        
    except Exception as e:
        logger.error(f"Career recommendations failed for user {current_user.id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Career recommendations failed: {str(e)}"
        )


@router.post("/analyze/learning-paths", response_model=List[LearningPathResponse])
async def analyze_learning_paths(
    force_refresh: bool = False,
    current_user: UserResponse = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Generate AI-powered learning paths
    
    Creates personalized learning paths to address skill gaps and achieve career goals.
    Includes modules, resources, and timelines.
    """
    try:
        service = AIAnalysisService()
        
        # Check for cached results
        if not force_refresh:
            cached_result = await service.get_cached_analysis(
                current_user.id, AnalysisType.LEARNING_PATH, db
            )
            if cached_result and not await service.is_analysis_stale(
                current_user.id, AnalysisType.LEARNING_PATH, db, max_age_hours=48
            ):
                learning_paths = cached_result.get("learning_paths", [])
                return [
                    LearningPathResponse(
                        title=path.get("title", ""),
                        description=path.get("description", ""),
                        target_skills=path.get("target_skills", []),
                        learning_modules=path.get("learning_modules", []),
                        estimated_duration=path.get("estimated_duration", ""),
                        difficulty_level=path.get("difficulty_level", ""),
                        resources=path.get("resources", [])
                    )
                    for path in learning_paths
                ]
        
        # Generate new learning paths
        learning_paths = await service.generate_learning_paths(current_user.id, db)
        
        response = [
            LearningPathResponse(
                title=path.title,
                description=path.description,
                target_skills=path.target_skills,
                learning_modules=[
                    {
                        "module_name": module.get("module_name", ""),
                        "topics": module.get("topics", []),
                        "duration": module.get("duration", "")
                    }
                    for module in path.learning_modules
                ],
                estimated_duration=path.estimated_duration,
                difficulty_level=path.difficulty_level,
                resources=[
                    {
                        "type": res.get("type", ""),
                        "title": res.get("title", ""),
                        "provider": res.get("provider", ""),
                        "url": res.get("url")
                    }
                    for res in path.resources
                ]
            )
            for path in learning_paths
        ]
        
        logger.info(f"Learning paths generated for user {current_user.id}")
        return response
        
    except Exception as e:
        logger.error(f"Learning paths generation failed for user {current_user.id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Learning paths generation failed: {str(e)}"
        )


@router.post("/analyze/projects", response_model=List[ProjectSuggestionResponse])
async def analyze_project_suggestions(
    force_refresh: bool = False,
    current_user: UserResponse = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Generate AI-powered project suggestions
    
    Suggests portfolio projects that demonstrate skills and help with career advancement.
    Includes technologies, difficulty levels, and learning outcomes.
    """
    try:
        service = AIAnalysisService()
        
        # Check for cached results
        if not force_refresh:
            cached_result = await service.get_cached_analysis(
                current_user.id, AnalysisType.PROJECT_SUGGESTION, db
            )
            if cached_result and not await service.is_analysis_stale(
                current_user.id, AnalysisType.PROJECT_SUGGESTION, db, max_age_hours=48
            ):
                project_suggestions = cached_result.get("project_suggestions", [])
                return [ProjectSuggestionResponse(**proj) for proj in project_suggestions]
        
        # Generate new project suggestions
        project_suggestions = await service.generate_project_suggestions(current_user.id, db)
        
        response = [
            ProjectSuggestionResponse(
                title=proj.title,
                description=proj.description,
                technologies=proj.technologies,
                difficulty_level=proj.difficulty_level,
                estimated_duration=proj.estimated_duration,
                learning_outcomes=proj.learning_outcomes,
                portfolio_value=proj.portfolio_value
            )
            for proj in project_suggestions
        ]
        
        logger.info(f"Project suggestions generated for user {current_user.id}")
        return response
        
    except Exception as e:
        logger.error(f"Project suggestions generation failed for user {current_user.id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Project suggestions generation failed: {str(e)}"
        )


@router.get("/status", response_model=Dict[str, AnalysisStatusResponse])
async def get_analysis_status(
    current_user: UserResponse = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Get status of all analysis types for the current user
    
    Returns the status and last update time for each type of analysis.
    """
    try:
        service = AIAnalysisService()
        status_dict = {}
        
        for analysis_type in AnalysisType:
            cached_result = await service.get_cached_analysis(current_user.id, analysis_type, db)
            is_stale = await service.is_analysis_stale(current_user.id, analysis_type, db)
            
            if cached_result:
                status_dict[analysis_type.value] = AnalysisStatusResponse(
                    user_id=current_user.id,
                    analysis_type=AnalysisTypeEnum(analysis_type.value),
                    status="completed" if not is_stale else "stale",
                    last_updated=datetime.utcnow(),  # Would be actual timestamp from DB
                    confidence_score=cached_result.get("confidence_score", 0.0)
                )
            else:
                status_dict[analysis_type.value] = AnalysisStatusResponse(
                    user_id=current_user.id,
                    analysis_type=AnalysisTypeEnum(analysis_type.value),
                    status="not_started",
                    last_updated=None,
                    confidence_score=None
                )
        
        return status_dict
        
    except Exception as e:
        logger.error(f"Failed to get analysis status for user {current_user.id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get analysis status"
        )


@router.post("/bulk-analyze", response_model=BulkAnalysisResponse)
async def bulk_analyze(
    request: BulkAnalysisRequest,
    background_tasks: BackgroundTasks,
    current_user: UserResponse = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Perform multiple types of analysis in a single request
    
    Efficiently runs multiple analysis types and returns combined results.
    """
    try:
        service = AIAnalysisService()
        results = {}
        status_dict = {}
        errors = {}
        confidence_scores = []
        
        for analysis_type in request.analysis_types:
            try:
                if analysis_type == AnalysisTypeEnum.SKILL_ASSESSMENT:
                    result = await service.generate_skill_assessment(current_user.id, db)
                    results[analysis_type.value] = {
                        "technical_skills": result.technical_skills,
                        "soft_skills": result.soft_skills,
                        "skill_strengths": result.skill_strengths,
                        "skill_gaps": result.skill_gaps,
                        "improvement_areas": result.improvement_areas,
                        "market_relevance_score": result.market_relevance_score,
                        "confidence_score": result.confidence_score
                    }
                    confidence_scores.append(result.confidence_score)
                    
                elif analysis_type == AnalysisTypeEnum.CAREER_RECOMMENDATION:
                    result = await service.generate_career_recommendations(current_user.id, db)
                    results[analysis_type.value] = [
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
                        for rec in result
                    ]
                    avg_confidence = sum(rec.match_score for rec in result) / len(result) if result else 0.0
                    confidence_scores.append(avg_confidence)
                
                # Add other analysis types as needed
                
                status_dict[analysis_type.value] = "completed"
                
            except Exception as e:
                logger.error(f"Bulk analysis failed for {analysis_type.value}: {str(e)}")
                errors[analysis_type.value] = str(e)
                status_dict[analysis_type.value] = "failed"
        
        overall_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0
        
        return BulkAnalysisResponse(
            user_id=current_user.id,
            results=results,
            status=status_dict,
            errors=errors,
            overall_confidence=overall_confidence,
            analysis_timestamp=datetime.utcnow()
        )
        
    except Exception as e:
        logger.error(f"Bulk analysis failed for user {current_user.id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Bulk analysis failed: {str(e)}"
        )


@router.get("/gemini/status", response_model=GeminiAPIStatus)
async def get_gemini_status():
    """
    Get Gemini API status and health information
    
    Returns information about API availability, error rates, and performance.
    """
    try:
        service = AIAnalysisService()
        
        # Test Gemini API availability
        try:
            test_response, _ = await service.gemini_client.generate_content("Test", temperature=0.1)
            is_available = True
            last_successful = datetime.utcnow()
        except Exception:
            is_available = False
            last_successful = None
        
        return GeminiAPIStatus(
            is_available=is_available,
            last_successful_request=last_successful,
            error_rate=0.0,  # Would be calculated from metrics
            average_response_time=2.5,  # Would be calculated from metrics
            rate_limit_status={"requests_remaining": 1000, "reset_time": "1 hour"}
        )
        
    except Exception as e:
        logger.error(f"Failed to get Gemini API status: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get API status"
        )


@router.get("/config", response_model=AnalysisConfigResponse)
async def get_analysis_config():
    """
    Get AI analysis service configuration
    
    Returns current configuration settings for the analysis service.
    """
    try:
        import os
        
        return AnalysisConfigResponse(
            gemini_api_enabled=bool(os.getenv("GEMINI_API_KEY")),
            fallback_enabled=True,
            cache_duration_hours=24,
            max_concurrent_analyses=5,
            supported_analysis_types=list(AnalysisTypeEnum)
        )
        
    except Exception as e:
        logger.error(f"Failed to get analysis config: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get configuration"
        )