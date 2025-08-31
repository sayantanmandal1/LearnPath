"""
Tests for database models
"""
import os
import pytest
from datetime import datetime
from uuid import uuid4

# Set test environment variables before importing models
os.environ["SECRET_KEY"] = "test_secret_key_for_testing_only_32_chars"
os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///./test.db"
os.environ["REDIS_URL"] = "redis://localhost:6379/1"
os.environ["JWT_SECRET_KEY"] = "test_jwt_secret_key_for_testing_only_32_chars"

from app.models.user import User, RefreshToken
from app.models.profile import UserProfile
from app.models.skill import Skill, UserSkill, SkillCategory
from app.models.job import JobPosting, JobSkill, Company


def test_user_model():
    """Test User model creation"""
    user = User(
        id=str(uuid4()),
        email="test@example.com",
        hashed_password="hashed_password",
        full_name="Test User",
        is_active=True,
        is_verified=False
    )
    
    assert user.email == "test@example.com"
    assert user.full_name == "Test User"
    assert user.is_active is True
    assert user.is_verified is False


def test_skill_model():
    """Test Skill model creation"""
    skill = Skill(
        id=str(uuid4()),
        name="Python",
        category="programming",
        subcategory="general_purpose",
        description="High-level programming language",
        aliases="python3,py",
        market_demand=0.95,
        average_salary_impact=15.0,
        is_active=True
    )
    
    assert skill.name == "Python"
    assert skill.category == "programming"
    assert skill.market_demand == 0.95
    assert skill.is_active is True


def test_user_skill_model():
    """Test UserSkill model creation"""
    user_skill = UserSkill(
        id=str(uuid4()),
        user_id=str(uuid4()),
        skill_id=str(uuid4()),
        confidence_score=0.85,
        proficiency_level="advanced",
        source="resume",
        evidence="5 years of Python development",
        years_experience=5.0,
        is_verified=True
    )
    
    assert user_skill.confidence_score == 0.85
    assert user_skill.proficiency_level == "advanced"
    assert user_skill.source == "resume"
    assert user_skill.years_experience == 5.0


def test_job_posting_model():
    """Test JobPosting model creation"""
    job = JobPosting(
        id=str(uuid4()),
        external_id="job123",
        title="Senior Python Developer",
        company="Tech Corp",
        location="San Francisco, CA",
        remote_type="hybrid",
        employment_type="full-time",
        experience_level="senior",
        description="We are looking for a senior Python developer...",
        salary_min=120000,
        salary_max=180000,
        salary_currency="USD",
        salary_period="yearly",
        source="linkedin",
        is_active=True,
        is_processed=False
    )
    
    assert job.title == "Senior Python Developer"
    assert job.company == "Tech Corp"
    assert job.remote_type == "hybrid"
    assert job.salary_min == 120000
    assert job.source == "linkedin"


def test_company_model():
    """Test Company model creation"""
    company = Company(
        id=str(uuid4()),
        name="Tech Corp",
        domain="techcorp.com",
        industry="technology",
        size="large",
        description="Leading technology company",
        headquarters="San Francisco, CA",
        founded_year=2010,
        glassdoor_rating=4.2,
        employee_count=5000,
        is_active=True
    )
    
    assert company.name == "Tech Corp"
    assert company.industry == "technology"
    assert company.size == "large"
    assert company.glassdoor_rating == 4.2
    assert company.employee_count == 5000


def test_skill_category_model():
    """Test SkillCategory model creation"""
    category = SkillCategory(
        id=str(uuid4()),
        name="Programming Languages",
        parent_id=None,
        description="Programming languages and scripting languages",
        display_order=1,
        is_active=True
    )
    
    assert category.name == "Programming Languages"
    assert category.parent_id is None
    assert category.display_order == 1
    assert category.is_active is True


def test_user_profile_model():
    """Test UserProfile model creation"""
    profile = UserProfile(
        id=str(uuid4()),
        user_id=str(uuid4()),
        dream_job="Senior Software Engineer",
        experience_years=5,
        current_role="Software Developer",
        location="New York, NY",
        github_username="testuser",
        leetcode_id="testuser123",
        linkedin_url="https://linkedin.com/in/testuser",
        skills={"Python": 0.9, "JavaScript": 0.8},
        career_interests={"backend": True, "ai": True}
    )
    
    assert profile.dream_job == "Senior Software Engineer"
    assert profile.experience_years == 5
    assert profile.github_username == "testuser"
    assert profile.skills["Python"] == 0.9
    assert profile.career_interests["backend"] is True