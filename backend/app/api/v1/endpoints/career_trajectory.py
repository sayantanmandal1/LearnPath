"""
Career trajectory API endpoints for personalized career path recommendations.
"""

import logging
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.api.dependencies import get_current_user
from app.models.user import User
from app.services.career_trajectory_service import CareerTrajectoryService
from app.core.exceptions import ServiceException
from app.schemas.career_trajectory import (
    CareerTrajectoryResponse,
    CareerTrajectoryListResponse,
    SkillGapAnalysisResponse,
    JobMatchScoreResponse
)

logger = logging.getLogger(__name__)

router = APIRouter()
career_trajectory_service = CareerTrajectoryService()


@router.get(
    "/trajectories",
    response_model=CareerTrajectoryListResponse,
    summary="Get career trajectory recommendations",
    description="Get personalized career trajectory recommendations based on user profile and goals"
)
async def get_career_trajectories(
    n_recommendations: int = Query(5, ge=1, le=10, description="Number of trajectory recommendations"),
    include_alternatives: bool = Query(True, description="Include alternative paths for each trajectory"),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Get personalized career trajectory recommendations for the current user.
    
    This endpoint analyzes the user's profile, skills, and career goals to generate
    comprehensive career trajectory recommendations including:
    - Dream job optimization paths
    - Natural career progression routes
    - Lateral movement opportunities
    - Market-driven career paths
    - Alternative routes and approaches
    """
    try:
        logger.info(f"Getting career trajectories for user {current_user.id}")
        
        trajectories = await career_trajectory_service.get_career_trajectory_recommendations(
            user_id=current_user.id,
            db=db,
            n_recommendations=n_recommendations,
            include_alternatives=include_alternatives
        )
        
        return CareerTrajectoryListResponse(
            trajectories=[
                CareerTrajectoryResponse.from_domain_model(trajectory)
                for trajectory in trajectories
            ],
            total_count=len(trajectories),
            generated_at=trajectories[0].recommendation_date if trajectories else None
        )
        
    except ServiceException as e:
        logger.error(f"Service error getting career trajectories: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error getting career trajectories: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get(
    "/trajectories/{trajectory_id}",
    response_model=CareerTrajectoryResponse,
    summary="Get specific career trajectory",
    description="Get detailed information about a specific career trajectory recommendation"
)
async def get_career_trajectory(
    trajectory_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Get detailed information about a specific career trajectory.
    
    This endpoint provides comprehensive details about a career trajectory including
    progression steps, skill requirements, market analysis, and alternative paths.
    """
    try:
        # For now, regenerate trajectories and find the matching one
        # In a production system, you'd store and retrieve specific trajectories
        trajectories = await career_trajectory_service.get_career_trajectory_recommendations(
            user_id=current_user.id,
            db=db,
            n_recommendations=10,
            include_alternatives=True
        )
        
        matching_trajectory = None
        for trajectory in trajectories:
            if trajectory.trajectory_id == trajectory_id:
                matching_trajectory = trajectory
                break
        
        if not matching_trajectory:
            raise HTTPException(status_code=404, detail="Career trajectory not found")
        
        return CareerTrajectoryResponse.from_domain_model(matching_trajectory)
        
    except HTTPException:
        raise
    except ServiceException as e:
        logger.error(f"Service error getting career trajectory: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error getting career trajectory: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get(
    "/skill-gap-analysis",
    response_model=SkillGapAnalysisResponse,
    summary="Analyze skill gaps for target role",
    description="Analyze skill gaps between user profile and a target role"
)
async def analyze_skill_gaps(
    target_role: str = Query(..., description="Target job role to analyze gaps for"),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Analyze skill gaps between the user's current profile and a target role.
    
    This endpoint provides detailed analysis including:
    - Missing skills that need to be developed
    - Weak skills that need improvement
    - Strong skills that can be leveraged
    - Overall readiness assessment
    - Learning time estimates
    - Priority skill recommendations
    """
    try:
        logger.info(f"Analyzing skill gaps for user {current_user.id} targeting {target_role}")
        
        gap_analysis = await career_trajectory_service.analyze_skill_gaps(
            user_id=current_user.id,
            target_role=target_role,
            db=db
        )
        
        return SkillGapAnalysisResponse(**gap_analysis)
        
    except ServiceException as e:
        logger.error(f"Service error analyzing skill gaps: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error analyzing skill gaps: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get(
    "/job-match/{job_id}",
    response_model=JobMatchScoreResponse,
    summary="Calculate job match score",
    description="Calculate match score between user profile and a specific job posting"
)
async def calculate_job_match_score(
    job_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Calculate match score between the user's profile and a specific job posting.
    
    This endpoint provides:
    - Overall match score and percentage
    - Skill gap analysis specific to the job
    - Readiness assessment
    - Detailed breakdown of matching and missing skills
    """
    try:
        logger.info(f"Calculating job match score for user {current_user.id} and job {job_id}")
        
        match_analysis = await career_trajectory_service.get_job_match_score(
            user_id=current_user.id,
            job_id=job_id,
            db=db
        )
        
        return JobMatchScoreResponse(**match_analysis)
        
    except ServiceException as e:
        logger.error(f"Service error calculating job match score: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error calculating job match score: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post(
    "/trajectories/refresh",
    response_model=CareerTrajectoryListResponse,
    summary="Refresh career trajectory recommendations",
    description="Force refresh of career trajectory recommendations with latest data"
)
async def refresh_career_trajectories(
    n_recommendations: int = Query(5, ge=1, le=10, description="Number of trajectory recommendations"),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Force refresh of career trajectory recommendations.
    
    This endpoint regenerates career trajectory recommendations using the latest
    market data, job postings, and user profile information. Use this when you
    want to ensure recommendations reflect the most current information.
    """
    try:
        logger.info(f"Refreshing career trajectories for user {current_user.id}")
        
        # Clear any cached data (if implemented)
        career_trajectory_service.market_demand_cache.clear()
        
        trajectories = await career_trajectory_service.get_career_trajectory_recommendations(
            user_id=current_user.id,
            db=db,
            n_recommendations=n_recommendations,
            include_alternatives=True
        )
        
        return CareerTrajectoryListResponse(
            trajectories=[
                CareerTrajectoryResponse.from_domain_model(trajectory)
                for trajectory in trajectories
            ],
            total_count=len(trajectories),
            generated_at=trajectories[0].recommendation_date if trajectories else None
        )
        
    except ServiceException as e:
        logger.error(f"Service error refreshing career trajectories: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error refreshing career trajectories: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get(
    "/market-insights/{role}",
    summary="Get market insights for role",
    description="Get market demand and trend insights for a specific role"
)
async def get_market_insights(
    role: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Get market insights and demand data for a specific role.
    
    This endpoint provides:
    - Current market demand level
    - Growth potential assessment
    - Salary trends and ranges
    - Job posting volume
    - Market competitiveness
    """
    try:
        logger.info(f"Getting market insights for role: {role}")
        
        market_data = await career_trajectory_service._get_market_demand_data(role, db)
        
        return {
            "role": role,
            "market_data": market_data,
            "analysis_date": career_trajectory_service.market_demand_cache.get(
                f"market_demand_{role.lower().replace(' ', '_')}", 
                (None, None)
            )[1]
        }
        
    except Exception as e:
        logger.error(f"Error getting market insights: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")