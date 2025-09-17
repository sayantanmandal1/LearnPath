"""
Tests for career analysis endpoint
"""

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.main import app
from app.models.user import User
from app.models.profile import UserProfile


@pytest.mark.asyncio
async def test_career_analysis_endpoint_success(
    async_client: AsyncClient,
    test_user: User,
    auth_headers: dict
):
    """Test successful career analysis"""
    
    analysis_data = {
        "current_role": "Software Developer",
        "experience": "2-3",
        "industry": "Technology",
        "location": "San Francisco, CA",
        "desired_role": "Senior Software Engineer",
        "career_goals": "I want to advance to a senior role and lead technical projects",
        "timeframe": "medium",
        "salary_expectation": "$90,000 - $120,000",
        "skills": "Python, JavaScript, React, SQL",
        "education": "Bachelor's in Computer Science",
        "certifications": "AWS Certified Developer",
        "languages": "English, Spanish",
        "work_type": "hybrid",
        "company_size": "medium",
        "work_culture": "Collaborative and innovative environment",
        "benefits": ["Health Insurance", "401(k) Matching", "Remote Work"]
    }
    
    response = await async_client.post(
        "/api/v1/career-analysis/analyze",
        json=analysis_data,
        headers=auth_headers
    )
    
    assert response.status_code == 200
    data = response.json()
    
    # Verify response structure matches frontend expectations
    assert "overall_score" in data
    assert "strengths" in data
    assert "improvements" in data
    assert "recommendations" in data
    assert "learning_paths" in data
    assert "market_insights" in data
    
    # Verify data types
    assert isinstance(data["overall_score"], int)
    assert isinstance(data["strengths"], list)
    assert isinstance(data["improvements"], list)
    assert isinstance(data["recommendations"], list)
    assert isinstance(data["learning_paths"], list)
    assert isinstance(data["market_insights"], dict)
    
    # Verify job recommendations structure
    if data["recommendations"]:
        job_rec = data["recommendations"][0]
        assert "type" in job_rec
        assert "title" in job_rec
        assert "company" in job_rec
        assert "match" in job_rec
        assert "salary" in job_rec
        assert "location" in job_rec
    
    # Verify learning paths structure
    if data["learning_paths"]:
        learning_path = data["learning_paths"][0]
        assert "title" in learning_path
        assert "provider" in learning_path
        assert "duration" in learning_path
        assert "difficulty" in learning_path
    
    # Verify market insights structure
    market_insights = data["market_insights"]
    assert "demand_trend" in market_insights
    assert "salary_growth" in market_insights
    assert "top_skills" in market_insights
    assert "competition_level" in market_insights


@pytest.mark.asyncio
async def test_career_analysis_endpoint_unauthorized(async_client: AsyncClient):
    """Test career analysis without authentication"""
    
    analysis_data = {
        "current_role": "Software Developer",
        "experience": "2-3",
        "industry": "Technology",
        "location": "San Francisco, CA",
        "desired_role": "Senior Software Engineer",
        "career_goals": "I want to advance to a senior role",
        "timeframe": "medium",
        "salary_expectation": "$90,000 - $120,000",
        "skills": "Python, JavaScript",
        "education": "Bachelor's in Computer Science",
        "work_type": "hybrid",
        "company_size": "medium"
    }
    
    response = await async_client.post(
        "/api/v1/career-analysis/analyze",
        json=analysis_data
    )
    
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_career_analysis_endpoint_validation_error(
    async_client: AsyncClient,
    auth_headers: dict
):
    """Test career analysis with invalid data"""
    
    # Missing required fields
    analysis_data = {
        "current_role": "Software Developer",
        # Missing other required fields
    }
    
    response = await async_client.post(
        "/api/v1/career-analysis/analyze",
        json=analysis_data,
        headers=auth_headers
    )
    
    assert response.status_code == 422  # Validation error


@pytest.mark.asyncio
async def test_career_analysis_creates_profile(
    async_client: AsyncClient,
    test_user: User,
    auth_headers: dict,
    db_session: AsyncSession
):
    """Test that career analysis creates user profile"""
    
    analysis_data = {
        "current_role": "Software Developer",
        "experience": "2-3",
        "industry": "Technology",
        "location": "San Francisco, CA",
        "desired_role": "Senior Software Engineer",
        "career_goals": "I want to advance to a senior role",
        "timeframe": "medium",
        "salary_expectation": "$90,000 - $120,000",
        "skills": "Python, JavaScript, React",
        "education": "Bachelor's in Computer Science",
        "work_type": "hybrid",
        "company_size": "medium"
    }
    
    response = await async_client.post(
        "/api/v1/career-analysis/analyze",
        json=analysis_data,
        headers=auth_headers
    )
    
    assert response.status_code == 200
    
    # Verify profile was created/updated
    from sqlalchemy import select
    result = await db_session.execute(
        select(UserProfile).where(UserProfile.user_id == test_user.id)
    )
    profile = result.scalar_one_or_none()
    
    assert profile is not None
    assert profile.current_role == "Software Developer"
    assert profile.dream_job == "Senior Software Engineer"
    assert profile.experience_years == 3  # Parsed from "2-3"
    assert profile.industry == "Technology"
    assert profile.location == "San Francisco, CA"