"""
Career Guidance API Endpoints
"""
from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
import logging

from app.core.database import get_db
from app.api.dependencies import get_current_user
from app.schemas.career_guidance import CareerGuidanceRequest, CareerGuidanceResponse
from app.services.career_guidance_service import CareerGuidanceService
from app.schemas.auth import UserResponse

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/generate", response_model=CareerGuidanceResponse)
async def generate_career_guidance(
    request: CareerGuidanceRequest,
    current_user: UserResponse = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Generate comprehensive career guidance including:
    - Detailed focus area recommendations based on target roles
    - Project specification displays with learning outcomes
    - Preparation roadmap with timelines and milestones
    - Curated resource recommendations with quality ratings
    """
    try:
        logger.info(f"Generating career guidance for user {current_user.id}")
        
        # Ensure the request is for the current user
        request.user_id = current_user.id
        
        service = CareerGuidanceService()
        guidance = await service.generate_career_guidance(request, db)
        
        logger.info(f"Successfully generated career guidance for user {current_user.id}")
        return guidance
        
    except Exception as e:
        logger.error(f"Error generating career guidance: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate career guidance: {str(e)}"
        )


@router.get("/history", response_model=List[CareerGuidanceResponse])
async def get_career_guidance_history(
    current_user: UserResponse = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Get user's career guidance history
    """
    try:
        logger.info(f"Fetching career guidance history for user {current_user.id}")
        
        service = CareerGuidanceService()
        history = await service.get_career_guidance_history(current_user.id, db)
        
        return history
        
    except Exception as e:
        logger.error(f"Error fetching career guidance history: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch career guidance history: {str(e)}"
        )


@router.post("/feedback/{guidance_id}")
async def submit_guidance_feedback(
    guidance_id: str,
    feedback: dict,
    current_user: UserResponse = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Submit feedback for career guidance to improve future recommendations
    """
    try:
        logger.info(f"Submitting feedback for guidance {guidance_id} from user {current_user.id}")
        
        service = CareerGuidanceService()
        success = await service.update_guidance_feedback(guidance_id, feedback, db)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Career guidance not found"
            )
        
        return {"message": "Feedback submitted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error submitting guidance feedback: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to submit feedback: {str(e)}"
        )


@router.get("/focus-areas/{target_role}")
async def get_focus_areas_for_role(
    target_role: str,
    current_user: UserResponse = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Get focus areas for a specific target role
    """
    try:
        logger.info(f"Fetching focus areas for role {target_role}")
        
        # Create a minimal request to get focus areas
        request = CareerGuidanceRequest(
            user_id=current_user.id,
            target_role=target_role,
            current_experience_years=2,  # Default values
            time_commitment_hours_per_week=10,
            career_timeline_months=12
        )
        
        service = CareerGuidanceService()
        user_profile = await service._get_user_profile(current_user.id, db)
        focus_areas = await service._generate_focus_areas(request, user_profile)
        
        return {"focus_areas": focus_areas}
        
    except Exception as e:
        logger.error(f"Error fetching focus areas: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch focus areas: {str(e)}"
        )


@router.get("/resources/{skill}")
async def get_curated_resources_for_skill(
    skill: str,
    difficulty_level: str = "intermediate",
    current_user: UserResponse = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Get curated learning resources for a specific skill
    """
    try:
        logger.info(f"Fetching resources for skill {skill}")
        
        # Create a minimal request to get resources
        request = CareerGuidanceRequest(
            user_id=current_user.id,
            target_role="Software Engineer",
            current_experience_years=2,
            time_commitment_hours_per_week=10,
            career_timeline_months=12,
            specific_interests=[skill]
        )
        
        service = CareerGuidanceService()
        
        # Create a mock focus area for the skill
        from app.schemas.career_guidance import FocusArea, DifficultyLevel
        focus_area = FocusArea(
            id="temp_focus",
            name=f"{skill} Mastery",
            description=f"Master {skill} for career advancement",
            importance_score=8.0,
            current_level=DifficultyLevel.BEGINNER,
            target_level=DifficultyLevel(difficulty_level),
            skills_required=[skill],
            estimated_time_weeks=8,
            priority_rank=1
        )
        
        resources = await service._curate_learning_resources(request, [focus_area])
        
        return {"resources": resources}
        
    except Exception as e:
        logger.error(f"Error fetching resources: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch resources: {str(e)}"
        )
