"""
Learning Path API endpoints for the AI Career Recommender System.

This module provides REST API endpoints for:
- Generating personalized learning paths
- Getting project recommendations
- Tracking learning progress
- Managing learning resources
"""

from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, Path
from fastapi.security import HTTPBearer
import logging

from app.schemas.learning_path import (
    LearningPathRequest, LearningPathResponse, LearningPath,
    ProjectRecommendation, LearningProgress, DifficultyLevel,
    ResourceProvider, LearningPathUpdate
)
from app.models.user import User
from app.services.learning_path_service import LearningPathService
from app.api.dependencies import get_current_user
from app.core.exceptions import ServiceException


logger = logging.getLogger(__name__)
router = APIRouter()
security = HTTPBearer()


@router.post("/generate", response_model=LearningPathResponse)
async def generate_learning_paths(
    request: LearningPathRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Generate personalized learning paths based on user requirements.
    
    This endpoint implements the core learning path generation functionality,
    addressing requirements 4.1, 4.2, 4.4, 4.6, 4.7.
    """
    try:
        logger.info(f"Generating learning paths for user {current_user.id}")
        
        # Set user_id from authenticated user
        request.user_id = current_user.id
        
        # Initialize service and generate paths
        service = LearningPathService()
        response = await service.generate_learning_paths(request)
        
        logger.info(f"Generated {response.total_paths} learning paths for user {current_user.id}")
        return response
        
    except ServiceException as e:
        logger.error(f"Service error generating learning paths: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error generating learning paths: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/projects", response_model=List[ProjectRecommendation])
async def get_project_recommendations(
    skills: List[str] = Query(..., description="Skills to practice"),
    difficulty: DifficultyLevel = Query(DifficultyLevel.INTERMEDIATE, description="Project difficulty level"),
    limit: int = Query(10, ge=1, le=50, description="Maximum number of projects to return"),
    current_user: User = Depends(get_current_user)
):
    """
    Get project recommendations from GitHub trending repositories.
    
    Implements requirement 4.3 for project-based learning.
    """
    try:
        logger.info(f"Getting project recommendations for skills: {skills}")
        
        service = LearningPathService()
        projects = await service.get_project_recommendations(skills, difficulty)
        
        # Limit results
        limited_projects = projects[:limit]
        
        logger.info(f"Returning {len(limited_projects)} project recommendations")
        return limited_projects
        
    except ServiceException as e:
        logger.error(f"Service error getting project recommendations: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error getting project recommendations: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/{path_id}", response_model=LearningPath)
async def get_learning_path(
    path_id: str = Path(..., description="Learning path ID"),
    current_user: User = Depends(get_current_user)
):
    """Get a specific learning path by ID."""
    try:
        # This would typically fetch from database
        # For now, return a mock response
        raise HTTPException(status_code=404, detail="Learning path not found")
        
    except Exception as e:
        logger.error(f"Error getting learning path {path_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.put("/{path_id}", response_model=LearningPath)
async def update_learning_path(
    path_id: str = Path(..., description="Learning path ID"),
    update_data: LearningPathUpdate = ...,
    current_user: User = Depends(get_current_user)
):
    """Update a learning path."""
    try:
        # This would typically update in database
        # For now, return a mock response
        raise HTTPException(status_code=404, detail="Learning path not found")
        
    except Exception as e:
        logger.error(f"Error updating learning path {path_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.delete("/{path_id}")
async def delete_learning_path(
    path_id: str = Path(..., description="Learning path ID"),
    current_user: User = Depends(get_current_user)
):
    """Delete a learning path."""
    try:
        # This would typically delete from database
        # For now, return success
        return {"message": "Learning path deleted successfully"}
        
    except Exception as e:
        logger.error(f"Error deleting learning path {path_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/{path_id}/progress", response_model=LearningProgress)
async def update_learning_progress(
    path_id: str = Path(..., description="Learning path ID"),
    progress: LearningProgress = ...,
    current_user: User = Depends(get_current_user)
):
    """
    Update learning progress for a specific milestone or resource.
    
    Implements requirement 4.7 for milestone tracking.
    """
    try:
        logger.info(f"Updating learning progress for path {path_id}")
        
        # Set user_id from authenticated user
        progress.user_id = current_user.id
        progress.learning_path_id = path_id
        
        # This would typically save to database
        # For now, return the progress object
        logger.info(f"Progress updated: {progress.progress_percentage}% complete")
        return progress
        
    except Exception as e:
        logger.error(f"Error updating learning progress: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/{path_id}/progress", response_model=List[LearningProgress])
async def get_learning_progress(
    path_id: str = Path(..., description="Learning path ID"),
    current_user: User = Depends(get_current_user)
):
    """Get learning progress for a specific path."""
    try:
        logger.info(f"Getting learning progress for path {path_id}")
        
        # This would typically fetch from database
        # For now, return empty list
        return []
        
    except Exception as e:
        logger.error(f"Error getting learning progress: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/user/paths", response_model=List[LearningPath])
async def get_user_learning_paths(
    current_user: User = Depends(get_current_user),
    limit: int = Query(10, ge=1, le=50, description="Maximum number of paths to return"),
    offset: int = Query(0, ge=0, description="Number of paths to skip")
):
    """Get all learning paths for the current user."""
    try:
        logger.info(f"Getting learning paths for user {current_user.id}")
        
        # This would typically fetch from database
        # For now, return empty list
        return []
        
    except Exception as e:
        logger.error(f"Error getting user learning paths: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/resources/rate")
async def rate_learning_resource(
    resource_id: str,
    rating: int = Query(..., ge=1, le=5, description="Rating from 1 to 5"),
    review: Optional[str] = Query(None, description="Optional review text"),
    current_user: User = Depends(get_current_user)
):
    """Rate a learning resource to improve recommendations."""
    try:
        logger.info(f"User {current_user.id} rating resource {resource_id}: {rating}/5")
        
        # This would typically save to database for ML model improvement
        # For now, return success
        return {
            "message": "Resource rating saved successfully",
            "resource_id": resource_id,
            "rating": rating,
            "review": review
        }
        
    except Exception as e:
        logger.error(f"Error saving resource rating: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/providers", response_model=List[str])
async def get_supported_providers():
    """Get list of supported learning resource providers."""
    try:
        providers = [provider.value for provider in ResourceProvider]
        return providers
        
    except Exception as e:
        logger.error(f"Error getting supported providers: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/difficulty-levels", response_model=List[str])
async def get_difficulty_levels():
    """Get list of supported difficulty levels."""
    try:
        levels = [level.value for level in DifficultyLevel]
        return levels
        
    except Exception as e:
        logger.error(f"Error getting difficulty levels: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/feedback")
async def submit_learning_path_feedback(
    path_id: str,
    rating: int = Query(..., ge=1, le=5, description="Overall path rating"),
    feedback: Optional[str] = Query(None, description="Feedback text"),
    completed: bool = Query(False, description="Whether the path was completed"),
    current_user: User = Depends(get_current_user)
):
    """
    Submit feedback on a learning path to improve future recommendations.
    
    This feedback is used to continuously improve the ML models.
    """
    try:
        logger.info(f"User {current_user.id} submitting feedback for path {path_id}")
        
        # This would typically save to database for ML model improvement
        # For now, return success
        return {
            "message": "Feedback submitted successfully",
            "path_id": path_id,
            "rating": rating,
            "feedback": feedback,
            "completed": completed
        }
        
    except Exception as e:
        logger.error(f"Error submitting feedback: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")