"""
Integration tests for database models and repositories
"""
import os
import pytest
import asyncio
from uuid import uuid4
from datetime import datetime

# Set test environment variables before importing models
os.environ["SECRET_KEY"] = "test_secret_key_for_testing_only_32_chars"
os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///./test_integration.db"
os.environ["REDIS_URL"] = "redis://localhost:6379/1"
os.environ["JWT_SECRET_KEY"] = "test_jwt_secret_key_for_testing_only_32_chars"

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.database import Base
from app.models.user import User
from app.models.profile import UserProfile
from app.models.skill import Skill, UserSkill, SkillCategory
from app.models.job import JobPosting, JobSkill, Company
from app.repositories.user import UserRepository
from app.repositories.skill import SkillRepository, UserSkillRepository
from app.repositories.job import JobRepository
from app.repositories.profile import ProfileRepository


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
async def engine():
    """Create test database engine"""
    # Use SQLite with string UUIDs for testing
    engine = create_async_engine(
        "sqlite+aiosqlite:///./test_integration.db",
        echo=False,
        poolclass=StaticPool,
        connect_args={"check_same_thread": False}
    )
    
    # Create all tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    yield engine
    
    # Cleanup
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


@pytest.fixture
async def db_session(engine):
    """Create database session for testing"""
    async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    
    async with async_session() as session:
        yield session


@pytest.mark.asyncio
async def test_user_crud_operations(db_session):
    """Test basic CRUD operations for User model"""
    user_repo = UserRepository()
    
    # Create user
    user_data = {
        "email": "test@example.com",
        "hashed_password": "hashed_password",
        "full_name": "Test User",
        "is_active": True,
        "is_verified": False
    }
    
    created_user = await user_repo.create(db_session, user_data)
    assert created_user.email == "test@example.com"
    assert created_user.full_name == "Test User"
    
    # Get user by ID
    retrieved_user = await user_repo.get(db_session, created_user.id)
    assert retrieved_user is not None
    assert retrieved_user.email == "test@example.com"
    
    # Get user by email
    user_by_email = await user_repo.get_by_email(db_session, "test@example.com")
    assert user_by_email is not None
    assert user_by_email.id == created_user.id
    
    # Update user
    updated_user = await user_repo.update(db_session, created_user.id, {"full_name": "Updated User"})
    assert updated_user.full_name == "Updated User"
    
    # Delete user
    deleted = await user_repo.delete(db_session, created_user.id)
    assert deleted is True
    
    # Verify deletion
    deleted_user = await user_repo.get(db_session, created_user.id)
    assert deleted_user is None


@pytest.mark.asyncio
async def test_skill_crud_operations(db_session):
    """Test basic CRUD operations for Skill model"""
    skill_repo = SkillRepository()
    
    # Create skill
    skill_data = {
        "name": "Python",
        "category": "programming",
        "subcategory": "general_purpose",
        "description": "High-level programming language",
        "aliases": "python3,py",
        "market_demand": 0.95,
        "average_salary_impact": 15.0,
        "is_active": True
    }
    
    created_skill = await skill_repo.create(db_session, skill_data)
    assert created_skill.name == "Python"
    assert created_skill.category == "programming"
    assert created_skill.market_demand == 0.95
    
    # Get skill by name
    skill_by_name = await skill_repo.get_by_name(db_session, "Python")
    assert skill_by_name is not None
    assert skill_by_name.id == created_skill.id
    
    # Search skills
    search_results = await skill_repo.search_skills(db_session, "Py", limit=10)
    assert len(search_results) > 0
    assert any(skill.name == "Python" for skill in search_results)
    
    # Get skills by category
    category_skills = await skill_repo.get_by_category(db_session, "programming")
    assert len(category_skills) > 0
    assert any(skill.name == "Python" for skill in category_skills)


@pytest.mark.asyncio
async def test_user_skill_relationship(db_session):
    """Test UserSkill relationship operations"""
    user_repo = UserRepository()
    skill_repo = SkillRepository()
    user_skill_repo = UserSkillRepository()
    
    # Create user
    user_data = {
        "email": "skilltest@example.com",
        "hashed_password": "hashed_password",
        "is_active": True,
        "is_verified": False
    }
    user = await user_repo.create(db_session, user_data)
    
    # Create skill
    skill_data = {
        "name": "JavaScript",
        "category": "programming",
        "is_active": True
    }
    skill = await skill_repo.create(db_session, skill_data)
    
    # Create user-skill relationship
    user_skill_data = {
        "user_id": user.id,
        "skill_id": skill.id,
        "confidence_score": 0.85,
        "proficiency_level": "advanced",
        "source": "resume",
        "evidence": "5 years of JavaScript development",
        "years_experience": 5.0,
        "is_verified": True
    }
    
    user_skill = await user_skill_repo.create(db_session, user_skill_data)
    assert user_skill.confidence_score == 0.85
    assert user_skill.proficiency_level == "advanced"
    
    # Get user skills
    user_skills = await user_skill_repo.get_user_skills(db_session, user.id)
    assert len(user_skills) == 1
    assert user_skills[0].skill_id == skill.id
    
    # Test upsert functionality
    updated_data = {
        "confidence_score": 0.90,
        "proficiency_level": "expert"
    }
    upserted_skill = await user_skill_repo.upsert_user_skill(
        db_session, user.id, skill.id, updated_data
    )
    assert upserted_skill.confidence_score == 0.90
    assert upserted_skill.proficiency_level == "expert"


@pytest.mark.asyncio
async def test_job_posting_operations(db_session):
    """Test JobPosting operations"""
    job_repo = JobRepository()
    
    # Create job posting
    job_data = {
        "external_id": "job123",
        "title": "Senior Python Developer",
        "company": "Tech Corp",
        "location": "San Francisco, CA",
        "remote_type": "hybrid",
        "employment_type": "full-time",
        "experience_level": "senior",
        "description": "We are looking for a senior Python developer...",
        "salary_min": 120000,
        "salary_max": 180000,
        "salary_currency": "USD",
        "salary_period": "yearly",
        "source": "linkedin",
        "is_active": True,
        "is_processed": False
    }
    
    created_job = await job_repo.create(db_session, job_data)
    assert created_job.title == "Senior Python Developer"
    assert created_job.company == "Tech Corp"
    assert created_job.salary_min == 120000
    
    # Get job by external ID
    job_by_external_id = await job_repo.get_by_external_id(
        db_session, "job123", "linkedin"
    )
    assert job_by_external_id is not None
    assert job_by_external_id.id == created_job.id
    
    # Search jobs
    search_results = await job_repo.search_jobs(
        db_session, title="Python", company="Tech"
    )
    assert len(search_results) > 0
    assert any(job.title == "Senior Python Developer" for job in search_results)
    
    # Mark as processed
    processed_skills = {"Python": 0.9, "Django": 0.7}
    processed_job = await job_repo.mark_as_processed(
        db_session, created_job.id, processed_skills
    )
    assert processed_job.is_processed is True
    assert processed_job.processed_skills == processed_skills


@pytest.mark.asyncio
async def test_profile_operations(db_session):
    """Test UserProfile operations"""
    user_repo = UserRepository()
    profile_repo = ProfileRepository()
    
    # Create user
    user_data = {
        "email": "profile@example.com",
        "hashed_password": "hashed_password",
        "is_active": True,
        "is_verified": False
    }
    user = await user_repo.create(db_session, user_data)
    
    # Create profile
    profile_data = {
        "user_id": user.id,
        "dream_job": "Senior Software Engineer",
        "experience_years": 5,
        "current_role": "Software Developer",
        "location": "New York, NY",
        "github_username": "testuser",
        "leetcode_id": "testuser123",
        "linkedin_url": "https://linkedin.com/in/testuser",
        "skills": {"Python": 0.9, "JavaScript": 0.8},
        "career_interests": {"backend": True, "ai": True}
    }
    
    created_profile = await profile_repo.create(db_session, profile_data)
    assert created_profile.dream_job == "Senior Software Engineer"
    assert created_profile.experience_years == 5
    assert created_profile.skills["Python"] == 0.9
    
    # Get profile by user ID
    profile_by_user = await profile_repo.get_by_user_id(db_session, user.id)
    assert profile_by_user is not None
    assert profile_by_user.id == created_profile.id
    
    # Update platform data
    platform_data = {
        "github": {
            "repositories": 25,
            "followers": 100,
            "languages": ["Python", "JavaScript", "Go"]
        }
    }
    updated_profile = await profile_repo.update_platform_data(
        db_session, user.id, platform_data
    )
    assert updated_profile.platform_data["github"]["repositories"] == 25
    
    # Update skills
    new_skills = {"Go": 0.7, "Docker": 0.8}
    skills_updated_profile = await profile_repo.update_skills(
        db_session, user.id, new_skills
    )
    assert skills_updated_profile.skills["Go"] == 0.7
    assert skills_updated_profile.skills["Python"] == 0.9  # Should preserve existing


if __name__ == "__main__":
    pytest.main([__file__, "-v"])