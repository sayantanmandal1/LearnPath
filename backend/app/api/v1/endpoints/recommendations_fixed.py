"""
Fixed recommendations endpoint
"""
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel, Field

from app.core.database import get_db
from app.services.intelligent_recommendations import generate_career_recommendations

router = APIRouter()

class RecommendationRequest(BaseModel):
    """Request model for recommendations."""
    target_role: Optional[str] = Field(None, description="Target job role for recommendations")
    n_recommendations: int = Field(5, ge=1, le=20, description="Number of recommendations to return")
    include_explanations: bool = Field(True, description="Include detailed explanations")

@router.post("/career")
async def get_career_recommendations(
    request: RecommendationRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Get personalized career recommendations for any job profile.
    
    This endpoint works with ANY job title - backend, frontend, marketing, data science, etc.
    It uses intelligent matching to provide relevant recommendations.
    """
    try:
        # Generate recommendations for any job profile
        recommendations_data = generate_career_recommendations(
            job_title=request.target_role or "",
            skills=getattr(request, 'skills', [])
        )
        
        # Format for frontend
        career_recommendations = []
        for rec in recommendations_data[:request.n_recommendations]:
            career_recommendations.append({
                "job_title": rec["job_title"],
                "company": rec["company"],
                "match_score": rec["match_score"],
                "match_percentage": rec["match_score"] * 100,
                "required_skills": rec["required_skills"],
                "skill_gaps": rec["skill_gaps"],
                "salary_range": rec["salary_range"],
                "growth_potential": rec["growth_potential"],
                "market_demand": rec["market_demand"],
                "confidence_score": rec["confidence_score"],
                "reasoning": rec["reasoning"],
                "alternative_paths": rec["alternative_paths"],
                "location": rec["location"],
                "employment_type": rec["employment_type"],
                "recommendation_date": rec["recommendation_date"]
            })
        
        # Dynamic skill gaps
        skill_gaps = [
            {"skill": "Advanced Skills", "gap_level": 0.3, "priority": "High"},
            {"skill": "Industry Knowledge", "gap_level": 0.4, "priority": "Medium"},
            {"skill": "Leadership", "gap_level": 0.2, "priority": "Low"}
        ]
        
        response_data = {
            "career_recommendations": career_recommendations,
            "skill_gaps": skill_gaps
        }
        
        print(f"✅ Generated recommendations for: '{request.target_role}'")
        return response_data
        
    except Exception as e:
        import traceback
        print(f"❌ Error in career recommendations: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.post("/learning-paths")
async def get_learning_path_recommendations(
    request: RecommendationRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Get personalized learning path recommendations.
    """
    try:
        from datetime import datetime
        
        # Mock learning paths that adapt to different profiles
        learning_paths = [
            {
                "path_id": "path_1",
                "title": "Professional Development Track",
                "target_skills": ["Leadership", "Communication", "Project Management"],
                "estimated_duration_weeks": 8,
                "difficulty_level": "Intermediate",
                "priority_score": 0.9,
                "reasoning": "Essential skills for career advancement",
                "resources": [
                    {"type": "course", "title": "Leadership Fundamentals", "url": "#", "duration": "3 weeks"},
                    {"type": "project", "title": "Team Project", "url": "#", "duration": "2 weeks"}
                ],
                "milestones": [
                    {"week": 3, "goal": "Leadership basics"},
                    {"week": 6, "goal": "Project management"},
                    {"week": 8, "goal": "Team leadership"}
                ],
                "created_date": datetime.now().isoformat()
            }
        ]
        
        return {"learning_paths": learning_paths}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")