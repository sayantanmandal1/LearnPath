"""
Career Analysis API endpoints for processing frontend analysis form data
and generating comprehensive AI recommendations.
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel, Field

from app.core.database import get_db
from app.api.dependencies import get_current_user
from app.models.user import User
from app.repositories.profile import ProfileRepository
from app.core.exceptions import ServiceException, ValidationError

logger = logging.getLogger(__name__)

router = APIRouter()


class CareerAnalysisRequest(BaseModel):
    """Request model for career analysis form data from frontend"""
    # Personal Information
    current_role: str = Field(..., description="Current job role")
    experience: str = Field(..., description="Years of experience (e.g., '2-3', '4-6')")
    industry: str = Field(..., description="Current industry")
    location: str = Field(..., description="Current location")
    
    # Career Goals
    desired_role: str = Field(..., description="Desired job role")
    career_goals: str = Field(..., description="Career aspirations and goals")
    timeframe: str = Field(..., description="Timeframe for career transition")
    salary_expectation: str = Field(..., description="Salary expectation range")
    
    # Skills & Education
    skills: str = Field(..., description="Technical skills (comma-separated)")
    education: str = Field(..., description="Educational background")
    certifications: str = Field(default="", description="Professional certifications")
    languages: str = Field(default="", description="Programming/spoken languages")
    
    # Work Preferences
    work_type: str = Field(..., description="Preferred work type (remote, hybrid, onsite)")
    company_size: str = Field(..., description="Preferred company size")
    work_culture: str = Field(default="", description="Work culture preferences")
    benefits: List[str] = Field(default=[], description="Important benefits")


class JobRecommendation(BaseModel):
    """Job recommendation model matching frontend expectations"""
    type: str = Field(default="job", description="Recommendation type")
    title: str = Field(..., description="Job title")
    company: str = Field(..., description="Company name")
    match: int = Field(..., description="Match percentage (0-100)")
    salary: str = Field(..., description="Salary range")
    location: str = Field(..., description="Job location")


class LearningPathRecommendation(BaseModel):
    """Learning path recommendation model"""
    title: str = Field(..., description="Learning path title")
    provider: str = Field(..., description="Learning provider")
    duration: str = Field(..., description="Estimated duration")
    difficulty: str = Field(..., description="Difficulty level")


class MarketInsights(BaseModel):
    """Market insights model"""
    demand_trend: str = Field(..., description="Market demand trend")
    salary_growth: str = Field(..., description="Salary growth trend")
    top_skills: List[str] = Field(..., description="Top skills in demand")
    competition_level: str = Field(..., description="Competition level")


class CareerAnalysisResponse(BaseModel):
    """Response model matching frontend expectations"""
    overall_score: int = Field(..., description="Overall career score (0-100)")
    strengths: List[str] = Field(..., description="User's career strengths")
    improvements: List[str] = Field(..., description="Areas for improvement")
    recommendations: List[JobRecommendation] = Field(..., description="Job recommendations")
    learning_paths: List[LearningPathRecommendation] = Field(..., description="Learning path recommendations")
    market_insights: MarketInsights = Field(..., description="Market insights")


@router.post("/analyze", response_model=CareerAnalysisResponse)
async def analyze_career(
    analysis_request: CareerAnalysisRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Process career analysis form data and generate AI recommendations.
    
    This endpoint:
    1. Receives analysis form data from frontend
    2. Creates/updates user profile with the provided data
    3. Generates career recommendations using AI services
    4. Returns analysis results in format expected by frontend
    """
    try:
        logger.info(f"Processing career analysis for user {current_user.id}")
        
        # Initialize repository
        profile_repo = ProfileRepository(db)
        
        # Process and normalize the form data
        processed_data = await _process_analysis_form_data(analysis_request, current_user.id)
        
        # Create or update user profile with analysis data
        await _create_or_update_profile(processed_data, current_user.id, profile_repo, db)
        
        # Generate mock analysis results for now
        # In production, this would use the ML services
        response = await _generate_mock_analysis_response(analysis_request, processed_data)
        
        logger.info(f"Career analysis completed for user {current_user.id}")
        return response
        
    except ValidationError as e:
        logger.error(f"Validation error in career analysis: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid analysis data: {str(e)}"
        )
    except ServiceException as e:
        logger.error(f"Service error in career analysis: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analysis service error: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Unexpected error in career analysis: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred during analysis"
        )


async def _process_analysis_form_data(
    request: CareerAnalysisRequest, 
    user_id: str
) -> Dict[str, Any]:
    """Process and normalize analysis form data"""
    
    # Parse experience years from string format
    experience_years = _parse_experience_years(request.experience)
    
    # Parse skills from comma-separated string
    skills_list = [skill.strip() for skill in request.skills.split(',') if skill.strip()]
    skills_dict = {skill: 0.8 for skill in skills_list}  # Default confidence
    
    # Parse certifications
    certifications_list = [cert.strip() for cert in request.certifications.split(',') if cert.strip()]
    
    # Parse languages
    languages_list = [lang.strip() for lang in request.languages.split(',') if lang.strip()]
    
    return {
        'user_id': user_id,
        'current_role': request.current_role,
        'dream_job': request.desired_role,
        'experience_years': experience_years,
        'industry': request.industry,
        'location': request.location,
        'skills': skills_dict,
        'education': request.education,
        'certifications': certifications_list,
        'languages': languages_list,
        'career_goals': request.career_goals,
        'timeframe': request.timeframe,
        'salary_expectation': request.salary_expectation,
        'work_preferences': {
            'work_type': request.work_type,
            'company_size': request.company_size,
            'work_culture': request.work_culture,
            'benefits': request.benefits
        }
    }


async def _create_or_update_profile(
    processed_data: Dict[str, Any],
    user_id: str,
    profile_repo: ProfileRepository,
    db: AsyncSession
):
    """Create or update user profile with analysis data"""
    try:
        existing_profile = await profile_repo.get_by_user_id(user_id)
        
        if existing_profile:
            # Update existing profile
            update_data = {
                'current_role': processed_data['current_role'],
                'dream_job': processed_data['dream_job'],
                'experience_years': processed_data['experience_years'],
                'industry': processed_data['industry'],
                'location': processed_data['location'],
                'skills': processed_data['skills'],
                'education': processed_data['education'],
                'certifications': processed_data['certifications'],
                'languages': processed_data['languages'],
                'career_goals': processed_data['career_goals'],
                'work_preferences': processed_data['work_preferences'],
                'updated_at': datetime.utcnow()
            }
            await profile_repo.update(existing_profile.id, update_data)
        else:
            # Create new profile
            profile_data = {
                'user_id': user_id,
                'current_role': processed_data['current_role'],
                'dream_job': processed_data['dream_job'],
                'experience_years': processed_data['experience_years'],
                'industry': processed_data['industry'],
                'location': processed_data['location'],
                'skills': processed_data['skills'],
                'education': processed_data['education'],
                'certifications': processed_data['certifications'],
                'languages': processed_data['languages'],
                'career_goals': processed_data['career_goals'],
                'work_preferences': processed_data['work_preferences']
            }
            await profile_repo.create(profile_data)
            
    except Exception as e:
        logger.error(f"Error creating/updating profile: {str(e)}")
        raise ServiceException(f"Failed to save profile data: {str(e)}")


async def _generate_market_insights(
    desired_role: str,
    skills_str: str,
    db: AsyncSession
) -> MarketInsights:
    """Generate market insights for the desired role and skills using real market data"""
    
    try:
        # Import the market insights service
        from app.services.market_insights_service import MarketInsightsService
        
        market_service = MarketInsightsService()
        
        # Parse skills
        skills_list = [skill.strip() for skill in skills_str.split(',') if skill.strip()]
        
        # Get comprehensive market insights
        insights_data = await market_service.get_comprehensive_market_insights(
            db=db,
            role=desired_role,
            skills=skills_list,
            location=None,
            experience_level=None,
            days=90
        )
        
        # Convert to the expected format
        return MarketInsights(
            demand_trend=insights_data['demand_trend'],
            salary_growth=insights_data['salary_growth'],
            top_skills=insights_data['top_skills'][:5],  # Limit to top 5
            competition_level=insights_data['competition_level']
        )
        
    except Exception as e:
        logger.warning(f"Failed to get real market insights, using fallback: {str(e)}")
        
        # Fallback to simulated data if real analysis fails
        skills_list = [skill.strip().lower() for skill in skills_str.split(',') if skill.strip()]
        
        role_insights = {
            'software engineer': {
                'demand_trend': 'High',
                'salary_growth': '+15% YoY',
                'top_skills': ['Python', 'JavaScript', 'React', 'AWS', 'Docker'],
                'competition_level': 'Medium'
            },
            'data scientist': {
                'demand_trend': 'Very High',
                'salary_growth': '+18% YoY',
                'top_skills': ['Python', 'Machine Learning', 'SQL', 'TensorFlow', 'Statistics'],
                'competition_level': 'High'
            },
            'frontend developer': {
                'demand_trend': 'High',
                'salary_growth': '+12% YoY',
                'top_skills': ['React', 'JavaScript', 'TypeScript', 'CSS', 'Vue.js'],
                'competition_level': 'Medium'
            },
            'backend developer': {
                'demand_trend': 'High',
                'salary_growth': '+14% YoY',
                'top_skills': ['Python', 'Java', 'Node.js', 'SQL', 'Microservices'],
                'competition_level': 'Medium'
            }
        }
        
        # Find matching role insights
        role_key = desired_role.lower()
        insights = None
        
        for key, data in role_insights.items():
            if key in role_key or any(word in role_key for word in key.split()):
                insights = data
                break
        
        # Default insights if role not found
        if not insights:
            insights = {
                'demand_trend': 'Medium',
                'salary_growth': '+10% YoY',
                'top_skills': skills_list[:5] if skills_list else ['Communication', 'Problem Solving'],
                'competition_level': 'Medium'
            }
        
        return MarketInsights(**insights)


async def _generate_mock_analysis_response(
    analysis_request: CareerAnalysisRequest,
    processed_data: Dict[str, Any]
) -> CareerAnalysisResponse:
    """Generate mock analysis response for testing"""
    
    # Calculate overall score based on experience and skills
    experience_years = processed_data.get('experience_years', 0)
    skills_count = len(processed_data.get('skills', {}))
    
    base_score = 60
    experience_bonus = min(experience_years * 5, 25)
    skills_bonus = min(skills_count * 2, 15)
    overall_score = base_score + experience_bonus + skills_bonus
    
    # Extract strengths from user data
    strengths = []
    user_skills = list(processed_data.get('skills', {}).keys())
    if user_skills:
        strengths.append(f"Strong technical skills in {', '.join(user_skills[:3])}")
    
    if experience_years >= 5:
        strengths.append("Extensive professional experience")
    elif experience_years >= 2:
        strengths.append("Solid professional foundation")
    
    if processed_data.get('education'):
        strengths.append("Strong educational background")
    
    if processed_data.get('certifications'):
        strengths.append("Professional certifications")
    
    # Default strengths if none identified
    if not strengths:
        strengths = [
            "Good problem-solving abilities",
            "Strong communication skills",
            "Adaptability and learning mindset"
        ]
    
    # Generate improvement suggestions
    improvements = [
        f"Consider learning cloud technologies (AWS, Azure) for {analysis_request.desired_role}",
        "Develop project management skills",
        "Build a stronger professional network",
        "Gain experience in emerging technologies"
    ]
    
    # Generate job recommendations
    job_recommendations = [
        JobRecommendation(
            title=f"Senior {analysis_request.current_role}",
            company="TechCorp Inc.",
            match=92,
            salary="$95,000 - $120,000",
            location=analysis_request.location or "Remote"
        ),
        JobRecommendation(
            title=analysis_request.desired_role,
            company="StartupXYZ",
            match=88,
            salary="$85,000 - $110,000",
            location="Austin, TX"
        ),
        JobRecommendation(
            title=f"Lead {analysis_request.current_role}",
            company="Digital Agency",
            match=85,
            salary="$100,000 - $130,000",
            location="Remote"
        )
    ]
    
    # Generate learning path recommendations
    learning_paths = [
        LearningPathRecommendation(
            title="Cloud Computing Fundamentals",
            provider="AWS",
            duration="6 weeks",
            difficulty="Intermediate"
        ),
        LearningPathRecommendation(
            title=f"Advanced {analysis_request.desired_role} Skills",
            provider="Frontend Masters",
            duration="4 weeks",
            difficulty="Advanced"
        ),
        LearningPathRecommendation(
            title="Leadership and Management",
            provider="Coursera",
            duration="8 weeks",
            difficulty="Beginner"
        )
    ]
    
    # Generate market insights
    market_insights = await _generate_market_insights(
        analysis_request.desired_role, analysis_request.skills, None
    )
    
    return CareerAnalysisResponse(
        overall_score=overall_score,
        strengths=strengths,
        improvements=improvements,
        recommendations=job_recommendations,
        learning_paths=learning_paths,
        market_insights=market_insights
    )


def _parse_experience_years(experience_str: str) -> int:
    """Parse experience string to years"""
    experience_mapping = {
        '0-1': 1,
        '2-3': 3,
        '4-6': 5,
        '7-10': 8,
        '10+': 12
    }
    return experience_mapping.get(experience_str, 2)


def _format_salary_range(salary_range: Dict[str, Any]) -> str:
    """Format salary range for display"""
    if not salary_range:
        return "$70,000 - $90,000"
    
    min_salary = salary_range.get('min', 70000)
    max_salary = salary_range.get('max', 90000)
    currency = salary_range.get('currency', 'USD')
    
    if currency == 'USD':
        return f"${min_salary:,} - ${max_salary:,}"
    else:
        return f"{min_salary:,} - {max_salary:,} {currency}"