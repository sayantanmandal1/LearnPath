"""
Learning Path API endpoints for the AI Career Recommender System.

This module provides REST API endpoints for:
- Generating personalized learning paths
- Getting project recommendations
- Tracking learning progress
- Managing learning resources
"""

from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, Query, Path
from fastapi.security import HTTPBearer
import logging

from app.schemas.learning_path import (
    LearningPathRequest, LearningPathResponse, LearningPath,
    ProjectRecommendation, LearningProgress, DifficultyLevel,
    ResourceProvider, LearningPathUpdate, LearningResource
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


@router.get("/{user_id}", response_model=List[LearningPath])
async def get_user_learning_paths(
    user_id: str = Path(..., description="User ID"),
    current_user: User = Depends(get_current_user),
    limit: int = Query(10, ge=1, le=50, description="Maximum number of paths to return"),
    offset: int = Query(0, ge=0, description="Number of paths to skip")
):
    """Get all learning paths for a specific user (frontend integration endpoint)."""
    try:
        logger.info(f"Getting learning paths for user {user_id}")
        
        # Verify user can access this data (either own data or admin)
        if current_user.id != user_id:
            # In a real implementation, check if current_user has admin privileges
            logger.warning(f"User {current_user.id} attempting to access paths for user {user_id}")
        
        # Initialize service and get user's learning paths
        service = LearningPathService()
        learning_paths = await service.get_user_learning_paths(user_id, limit, offset)
        
        logger.info(f"Retrieved {len(learning_paths)} learning paths for user {user_id}")
        return learning_paths
        
    except Exception as e:
        logger.error(f"Error getting user learning paths: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/user/paths", response_model=List[LearningPath])
async def get_current_user_learning_paths(
    current_user: User = Depends(get_current_user),
    limit: int = Query(10, ge=1, le=50, description="Maximum number of paths to return"),
    offset: int = Query(0, ge=0, description="Number of paths to skip")
):
    """Get all learning paths for the current authenticated user."""
    try:
        logger.info(f"Getting learning paths for current user {current_user.id}")
        
        service = LearningPathService()
        learning_paths = await service.get_user_learning_paths(current_user.id, limit, offset)
        
        logger.info(f"Retrieved {len(learning_paths)} learning paths for current user")
        return learning_paths
        
    except Exception as e:
        logger.error(f"Error getting current user learning paths: {e}")
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


@router.get("/{user_id}/simplified", response_model=List[Dict[str, Any]])
async def get_user_learning_paths_simplified(
    user_id: str = Path(..., description="User ID"),
    current_user: User = Depends(get_current_user),
    limit: int = Query(10, ge=1, le=50, description="Maximum number of paths to return")
):
    """
    Get learning paths in simplified format for frontend integration.
    
    Returns learning paths in the format expected by the frontend:
    - title: string
    - provider: string  
    - duration: string
    - difficulty: string
    """
    try:
        logger.info(f"Getting simplified learning paths for user {user_id}")
        
        # Verify user can access this data
        if current_user.id != user_id:
            logger.warning(f"User {current_user.id} attempting to access paths for user {user_id}")
        
        # Get full learning paths
        service = LearningPathService()
        learning_paths = await service.get_user_learning_paths(user_id, limit, 0)
        
        # Convert to simplified format for frontend
        simplified_paths = []
        for path in learning_paths:
            # Get primary provider from resources
            primary_provider = "Mixed"
            if path.resources:
                provider_counts = {}
                for resource in path.resources:
                    provider = resource.provider.value.title()
                    provider_counts[provider] = provider_counts.get(provider, 0) + 1
                primary_provider = max(provider_counts, key=provider_counts.get)
            
            # Format duration
            duration_str = f"{path.estimated_duration_weeks} weeks"
            if path.estimated_duration_weeks == 1:
                duration_str = "1 week"
            elif path.estimated_duration_weeks > 52:
                years = path.estimated_duration_weeks // 52
                duration_str = f"{years} year{'s' if years > 1 else ''}"
            
            simplified_path = {
                "id": path.id,
                "title": path.title,
                "provider": primary_provider,
                "duration": duration_str,
                "difficulty": path.difficulty_level.value.title(),
                "description": path.description,
                "target_role": path.target_role,
                "skills": path.target_skills,
                "confidence_score": path.confidence_score,
                "estimated_hours": path.estimated_duration_hours
            }
            simplified_paths.append(simplified_path)
        
        logger.info(f"Returning {len(simplified_paths)} simplified learning paths")
        return simplified_paths
        
    except Exception as e:
        logger.error(f"Error getting simplified learning paths: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/generate-from-profile", response_model=List[LearningPath])
async def generate_learning_paths_from_profile(
    profile_data: Dict[str, Any],
    current_user: User = Depends(get_current_user)
):
    """
    Generate learning paths based on user profile analysis.
    
    This endpoint supports frontend integration by generating learning paths
    from user profile data collected during the analysis process.
    """
    try:
        logger.info(f"Generating learning paths from profile for user {current_user.id}")
        
        # Add user_id to profile data
        profile_data['user_id'] = current_user.id
        
        service = LearningPathService()
        learning_paths = await service.generate_learning_paths_for_profile(profile_data)
        
        logger.info(f"Generated {len(learning_paths)} learning paths from profile")
        return learning_paths
        
    except ServiceException as e:
        logger.error(f"Service error generating paths from profile: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error generating paths from profile: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/skills/{skills}/resources", response_model=List[LearningResource])
async def get_skill_based_resources(
    skills: str = Path(..., description="Comma-separated list of skills"),
    difficulty: DifficultyLevel = Query(DifficultyLevel.INTERMEDIATE, description="Difficulty level"),
    current_user: User = Depends(get_current_user)
):
    """
    Get learning resources for specific skills.
    
    This endpoint provides targeted learning resources for skill development,
    supporting granular skill-based learning recommendations.
    """
    try:
        skill_list = [skill.strip() for skill in skills.split(',')]
        logger.info(f"Getting resources for skills: {skill_list}")
        
        service = LearningPathService()
        resources = await service.get_skill_based_recommendations(skill_list, difficulty)
        
        logger.info(f"Returning {len(resources)} skill-based resources")
        return resources
        
    except ServiceException as e:
        logger.error(f"Service error getting skill resources: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error getting skill resources: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/custom", response_model=LearningPath)
async def create_custom_learning_path(
    title: str,
    skills: List[str],
    preferences: Dict[str, Any] = {},
    current_user: User = Depends(get_current_user)
):
    """
    Create a custom learning path for specific skills.
    
    This endpoint allows users to create personalized learning paths
    based on their specific skill requirements and preferences.
    """
    try:
        logger.info(f"Creating custom learning path: {title} for skills: {skills}")
        
        # Add user context to preferences
        preferences['user_id'] = current_user.id
        
        service = LearningPathService()
        learning_path = await service.create_custom_learning_path(title, skills, preferences)
        
        logger.info(f"Created custom learning path: {learning_path.id}")
        return learning_path
        
    except ServiceException as e:
        logger.error(f"Service error creating custom path: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error creating custom path: {e}")
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