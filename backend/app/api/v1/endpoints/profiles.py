"""
Profile API endpoints for user profile management and data aggregation.
"""

import logging
from typing import Optional
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.api.dependencies import get_current_user
from app.models.user import User
from app.schemas.profile import (
    ProfileCreate, ProfileUpdate, ProfileResponse, 
    ResumeUpload, PlatformDataUpdate
)
from app.services.profile_service import UserProfileService
from app.core.exceptions import ValidationError, NotFoundError, ConflictError

logger = logging.getLogger(__name__)

router = APIRouter()
@router.get("/me/stats")
async def get_my_profile_stats(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Return placeholder stats for current user."""
    # TODO: Replace with real stats logic
    return {
        "courses_completed": 0,
        "courses_target": 0,
        "skills_mastered": 0,
        "skills_target": 0,
        "certificates_earned": 0,
        "certificates_target": 0,
        "study_hours": 0,
        "study_hours_target": 0
    }

@router.get("/me/achievements")
async def get_my_profile_achievements(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Return placeholder achievements for current user."""
    # TODO: Replace with real achievements logic
    return []

@router.get("/me/activities")
async def get_my_profile_activities(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Return placeholder activities for current user."""
    # TODO: Replace with real activities logic
    return []

# Initialize profile service (in production, this would be dependency injected)
profile_service = UserProfileService()


@router.post("/", response_model=ProfileResponse, status_code=status.HTTP_201_CREATED)
async def create_profile(
    profile_data: ProfileCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Create user profile with multi-source data integration.
    
    This endpoint creates a comprehensive user profile by:
    - Storing basic profile information
    - Extracting data from external platforms (GitHub, LeetCode, LinkedIn)
    - Merging skills from multiple sources with confidence scoring
    - Generating unified skill profile
    """
    try:
        profile = await profile_service.create_profile_with_integration(
            db=db,
            user_id=current_user.id,
            profile_data=profile_data
        )
        
        logger.info(f"Created profile for user {current_user.id}")
        return profile
        
    except ConflictError as e:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(e)
        )
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error creating profile for user {current_user.id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create profile"
        )


@router.post("/with-resume", response_model=ProfileResponse, status_code=status.HTTP_201_CREATED)
async def create_profile_with_resume(
    profile_data: ProfileCreate,
    resume_file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Create user profile with resume upload and multi-source data integration.
    
    This endpoint creates a comprehensive user profile by:
    - Processing uploaded resume using NLP for skill extraction
    - Storing basic profile information
    - Extracting data from external platforms
    - Merging skills from all sources including resume
    - Generating unified skill profile
    """
    try:
        # Validate file type
        allowed_types = ["application/pdf", "application/msword", 
                        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"]
        
        if resume_file.content_type not in allowed_types:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Only PDF and Word documents are supported"
            )
        
        # Validate file size (10MB limit)
        if resume_file.size > 10 * 1024 * 1024:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="File size must be less than 10MB"
            )
        
        profile = await profile_service.create_profile_with_integration(
            db=db,
            user_id=current_user.id,
            profile_data=profile_data,
            resume_file=resume_file
        )
        
        logger.info(f"Created profile with resume for user {current_user.id}")
        return profile
        
    except ConflictError as e:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(e)
        )
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error creating profile with resume for user {current_user.id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create profile with resume"
        )


@router.get("/me", response_model=ProfileResponse)
async def get_my_profile(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get current user's profile."""
    try:
        # For now, just return default profile data
        # TODO: Implement proper profile lookup from database
        return ProfileResponse(
            id=str(current_user.id),
            user_id=str(current_user.id),
            dream_job="",
            experience_years=None,
            current_role="",
            location="",
            github_username="",
            leetcode_id="",
            linkedin_url="",
            codeforces_id="",
            industry="",
            desired_role="",
            career_goals="",
            timeframe="",
            salary_expectation="",
            education="",
            certifications="",
            languages="",
            work_type="",
            company_size="",
            work_culture="",
            benefits=[],
            skills={},
            platform_data={},
            resume_data={},
            career_interests={},
            skill_gaps={},
            profile_score=0.0,
            completeness_score=0.0,
            data_last_updated=None,
            created_at=current_user.created_at,
            updated_at=current_user.updated_at
        )
        
    except Exception as e:
        logger.error(f"Error getting profile for user {current_user.id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get profile: {str(e)}"
        )


@router.put("/me", response_model=ProfileResponse)
async def update_my_profile(
    profile_update: ProfileUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Update current user's profile with data consistency validation.
    
    This endpoint updates the user profile with:
    - Data consistency validation
    - Conflict resolution using configured strategies
    - Change tracking for audit purposes
    - Automatic skill merging when skills are updated
    """
    try:
        profile = await profile_service.update_profile_with_validation(
            db=db,
            user_id=current_user.id,
            update_data=profile_update
        )
        
        logger.info(f"Updated profile for user {current_user.id}")
        return profile
        
    except NotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error updating profile for user {current_user.id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update profile"
        )


@router.post("/me/refresh", response_model=ProfileResponse)
async def refresh_profile_data(
    force_refresh: bool = False,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Refresh external platform data for current user's profile.
    
    This endpoint:
    - Fetches fresh data from connected external platforms
    - Re-extracts and merges skills from all sources
    - Updates profile with latest information
    - Respects rate limits and caching (unless force_refresh=True)
    """
    try:
        profile = await profile_service.refresh_external_data(
            db=db,
            user_id=current_user.id,
            force_refresh=force_refresh
        )
        
        logger.info(f"Refreshed profile data for user {current_user.id}")
        return profile
        
    except NotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error refreshing profile data for user {current_user.id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to refresh profile data"
        )


@router.post("/me/upload-resume", response_model=ProfileResponse)
async def upload_resume(
    resume_file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Upload and process resume for existing profile.
    
    This endpoint:
    - Processes uploaded resume using NLP for skill extraction
    - Merges extracted skills with existing profile skills
    - Updates profile with resume data and merged skills
    """
    try:
        # Validate file type
        allowed_types = ["application/pdf", "application/msword", 
                        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"]
        
        if resume_file.content_type not in allowed_types:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Only PDF and Word documents are supported"
            )
        
        # Validate file size (10MB limit)
        if resume_file.size > 10 * 1024 * 1024:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="File size must be less than 10MB"
            )
        
        # Get existing profile
        from app.repositories.profile import ProfileRepository
        profile_repo = ProfileRepository()
        
        existing_profile = await profile_repo.get_by_user_id(db, current_user.id)
        if not existing_profile:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Profile not found. Create a profile first."
            )
        
        # Process resume and update profile
        resume_data = await profile_service._process_resume_file(resume_file)
        resume_skills = profile_service._extract_skills_from_resume(resume_data)
        
        # Merge with existing skills
        existing_skills = existing_profile.skills or {}
        skill_sources = {
            'existing': existing_skills,
            'resume': resume_skills
        }
        
        skill_merge_result = profile_service._merge_skill_profiles(skill_sources)
        
        # Update profile
        updated_profile = await profile_repo.update(db, existing_profile.id, {
            'resume_data': resume_data.dict() if hasattr(resume_data, 'dict') else resume_data,
            'skills': skill_merge_result.merged_skills,
            'data_last_updated': datetime.utcnow()
        })
        
        logger.info(f"Uploaded resume for user {current_user.id}")
        return ProfileResponse.from_orm(updated_profile)
        
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error uploading resume for user {current_user.id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to upload resume"
        )


@router.get("/me/analytics")
async def get_profile_analytics(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get comprehensive profile analytics and insights.
    
    Returns:
    - Profile completeness score
    - Skill distribution analysis
    - Data freshness metrics
    - Platform coverage analysis
    - Skill gaps summary
    - Personalized recommendations
    """
    try:
        analytics = await profile_service.get_profile_analytics(
            db=db,
            user_id=current_user.id
        )
        
        return analytics
        
    except NotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error getting profile analytics for user {current_user.id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get profile analytics"
        )


@router.post("/me/validate-sources")
async def validate_profile_sources(
    github_username: Optional[str] = None,
    leetcode_username: Optional[str] = None,
    linkedin_url: Optional[str] = None,
    current_user: User = Depends(get_current_user)
):
    """
    Validate that external profile sources exist and are accessible.
    
    This endpoint checks:
    - GitHub username exists and is accessible
    - LeetCode username exists and has data
    - LinkedIn profile URL is accessible
    
    Returns validation results for each provided source.
    """
    try:
        validation_results = await profile_service.external_api_service.validate_profile_sources(
            github_username=github_username,
            leetcode_username=leetcode_username,
            linkedin_url=linkedin_url
        )
        
        return {
            'validation_results': validation_results,
            'all_valid': all(validation_results.values()),
            'valid_sources': [source for source, valid in validation_results.items() if valid],
            'invalid_sources': [source for source, valid in validation_results.items() if not valid]
        }
        
    except Exception as e:
        logger.error(f"Error validating profile sources for user {current_user.id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to validate profile sources"
        )


@router.get("/me/skill-gaps")
async def get_skill_gaps(
    target_role: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get detailed skill gap analysis for current user.
    
    Args:
        target_role: Optional target role to analyze gaps for (overrides dream_job)
    
    Returns detailed skill gap analysis including:
    - Critical, moderate, and minor skill gaps
    - Learning recommendations for each gap
    - Estimated time to close gaps
    """
    try:
        from app.repositories.profile import ProfileRepository
        profile_repo = ProfileRepository()
        
        profile = await profile_repo.get_by_user_id(db, current_user.id)
        if not profile:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Profile not found"
            )
        
        # Use provided target role or profile's dream job
        role_to_analyze = target_role or profile.dream_job
        if not role_to_analyze:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="No target role specified. Provide target_role parameter or set dream_job in profile."
            )
        
        # Calculate skill gaps
        current_skills = profile.skills or {}
        skill_gaps = await profile_service._calculate_skill_gaps(current_skills, role_to_analyze)
        
        return {
            'target_role': role_to_analyze,
            'current_skills_count': len(current_skills),
            'skill_gaps': skill_gaps,
            'recommendations': [
                {
                    'skill': skill,
                    'gap_size': gap,
                    'priority': 'critical' if gap >= 0.7 else 'moderate' if gap >= 0.4 else 'minor',
                    'estimated_learning_time_weeks': int(gap * 12),  # Rough estimate
                    'suggested_resources': f"Learn {skill} through online courses and practice projects"
                }
                for skill, gap in sorted(skill_gaps.items(), key=lambda x: x[1], reverse=True)
            ]
        }
        
    except Exception as e:
        logger.error(f"Error getting skill gaps for user {current_user.id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get skill gaps"
        )


@router.post("/me/calculate-score")
async def calculate_profile_score(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Calculate and update profile score based on current data.
    
    Returns:
    - Updated profile score
    - Completeness score
    - Breakdown of scoring factors
    - Recommendations for improvement
    """
    try:
        analytics = await profile_service.get_profile_analytics(
            db=db,
            user_id=current_user.id
        )
        
        return {
            'profile_score': analytics['profile_score'],
            'completeness_score': analytics['completeness_score'],
            'scoring_breakdown': {
                'completeness_contribution': analytics['completeness_score'] * 0.4,
                'skills_contribution': analytics['skill_analysis']['total_skills'] * 5 * 0.3,  # Simplified
                'platform_contribution': analytics['platform_coverage']['coverage_percentage'] * 0.2,
                'freshness_contribution': 80 * 0.1  # Simplified
            },
            'recommendations': analytics['recommendations'],
            'summary': analytics['summary']
        }
        
    except Exception as e:
        logger.error(f"Error calculating profile score for user {current_user.id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to calculate profile score"
        )