"""
Comprehensive test configuration and fixtures for the AI Career Recommender.
"""
import asyncio
import os
import tempfile
from typing import AsyncGenerator, Generator
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from httpx import AsyncClient
from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.database import Base, get_db
from app.core.redis import get_redis
from app.main import app
from app.models.user import User
from app.models.profile import UserProfile
from app.models.skill import Skill, UserSkill
from app.models.job import JobPosting
from app.schemas.profile import UserProfileCreate
from app.schemas.auth import UserCreate


# Test database configuration
TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"
TEST_SYNC_DATABASE_URL = "sqlite:///:memory:"


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
async def async_engine():
    """Create async database engine for testing."""
    engine = create_async_engine(
        TEST_DATABASE_URL,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    yield engine
    
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    
    await engine.dispose()


@pytest.fixture
async def async_session(async_engine) -> AsyncGenerator[AsyncSession, None]:
    """Create async database session for testing."""
    async_session_maker = sessionmaker(
        async_engine, class_=AsyncSession, expire_on_commit=False
    )
    
    async with async_session_maker() as session:
        yield session


@pytest.fixture
def sync_engine():
    """Create sync database engine for testing."""
    engine = create_engine(
        TEST_SYNC_DATABASE_URL,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    
    Base.metadata.create_all(bind=engine)
    yield engine
    Base.metadata.drop_all(bind=engine)


@pytest.fixture
def sync_session(sync_engine):
    """Create sync database session for testing."""
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=sync_engine)
    session = SessionLocal()
    yield session
    session.close()


@pytest.fixture
def mock_redis():
    """Mock Redis client for testing."""
    redis_mock = AsyncMock()
    redis_mock.get.return_value = None
    redis_mock.set.return_value = True
    redis_mock.delete.return_value = 1
    redis_mock.exists.return_value = False
    return redis_mock


@pytest.fixture
def override_dependencies(async_session, mock_redis):
    """Override FastAPI dependencies for testing."""
    def get_test_db():
        return async_session
    
    def get_test_redis():
        return mock_redis
    
    app.dependency_overrides[get_db] = get_test_db
    app.dependency_overrides[get_redis] = get_test_redis
    
    yield
    
    app.dependency_overrides.clear()


@pytest.fixture
def client(override_dependencies) -> Generator[TestClient, None, None]:
    """Create test client for FastAPI app."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
async def async_client(override_dependencies) -> AsyncGenerator[AsyncClient, None]:
    """Create async test client for FastAPI app."""
    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac


# Test data fixtures
@pytest.fixture
def sample_user_data():
    """Sample user data for testing."""
    return {
        "email": "test@example.com",
        "password": "testpassword123",
        "full_name": "Test User"
    }


@pytest.fixture
def sample_profile_data():
    """Sample profile data for testing."""
    return {
        "skills": ["Python", "Machine Learning", "FastAPI"],
        "dream_job": "Senior ML Engineer",
        "experience_years": 3,
        "github_username": "testuser",
        "leetcode_id": "testuser",
        "linkedin_url": "https://linkedin.com/in/testuser"
    }


@pytest.fixture
def sample_resume_file():
    """Create a temporary resume file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("""
        John Doe
        Software Engineer
        
        Skills: Python, JavaScript, React, Node.js, Machine Learning
        Experience: 3 years in web development and data science
        Education: BS Computer Science
        """)
        f.flush()
        yield f.name
    
    os.unlink(f.name)


@pytest.fixture
async def test_user(async_session):
    """Create a test user in the database."""
    user = User(
        email="test@example.com",
        hashed_password="$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW",  # secret
        full_name="Test User",
        is_active=True
    )
    async_session.add(user)
    await async_session.commit()
    await async_session.refresh(user)
    return user


@pytest.fixture
async def test_profile(async_session, test_user):
    """Create a test user profile in the database."""
    profile = UserProfile(
        user_id=test_user.id,
        dream_job="Senior ML Engineer",
        experience_years=3,
        github_username="testuser",
        leetcode_id="testuser",
        linkedin_url="https://linkedin.com/in/testuser"
    )
    async_session.add(profile)
    await async_session.commit()
    await async_session.refresh(profile)
    return profile


@pytest.fixture
async def test_skills(async_session):
    """Create test skills in the database."""
    skills = [
        Skill(name="Python", category="programming", description="Python programming language"),
        Skill(name="Machine Learning", category="technical", description="ML algorithms and techniques"),
        Skill(name="FastAPI", category="framework", description="FastAPI web framework"),
        Skill(name="React", category="frontend", description="React JavaScript library"),
    ]
    
    for skill in skills:
        async_session.add(skill)
    
    await async_session.commit()
    
    for skill in skills:
        await async_session.refresh(skill)
    
    return skills


@pytest.fixture
async def test_job_postings(async_session):
    """Create test job postings in the database."""
    jobs = [
        JobPosting(
            title="Senior Python Developer",
            company="Tech Corp",
            location="San Francisco, CA",
            description="Looking for experienced Python developer",
            required_skills=["Python", "FastAPI", "PostgreSQL"],
            experience_level="senior",
            salary_min=120000,
            salary_max=180000,
            source="linkedin"
        ),
        JobPosting(
            title="ML Engineer",
            company="AI Startup",
            location="Remote",
            description="Machine learning engineer position",
            required_skills=["Python", "Machine Learning", "TensorFlow"],
            experience_level="mid",
            salary_min=100000,
            salary_max=150000,
            source="indeed"
        )
    ]
    
    for job in jobs:
        async_session.add(job)
    
    await async_session.commit()
    
    for job in jobs:
        await async_session.refresh(job)
    
    return jobs


# Mock external services
@pytest.fixture
def mock_github_client():
    """Mock GitHub API client."""
    mock = AsyncMock()
    mock.get_user_profile.return_value = {
        "username": "testuser",
        "repositories": [
            {"name": "ml-project", "language": "Python", "stars": 10},
            {"name": "web-app", "language": "JavaScript", "stars": 5}
        ],
        "languages": {"Python": 70, "JavaScript": 30},
        "total_commits": 150
    }
    return mock


@pytest.fixture
def mock_leetcode_scraper():
    """Mock LeetCode scraper."""
    mock = AsyncMock()
    mock.get_user_stats.return_value = {
        "username": "testuser",
        "problems_solved": 250,
        "easy_solved": 100,
        "medium_solved": 120,
        "hard_solved": 30,
        "contest_rating": 1650,
        "skills": ["Dynamic Programming", "Graph Theory", "Binary Search"]
    }
    return mock


@pytest.fixture
def mock_linkedin_scraper():
    """Mock LinkedIn scraper."""
    mock = AsyncMock()
    mock.get_profile_data.return_value = {
        "name": "Test User",
        "headline": "Software Engineer",
        "experience": [
            {
                "title": "Software Engineer",
                "company": "Tech Company",
                "duration": "2 years",
                "skills": ["Python", "React", "AWS"]
            }
        ],
        "skills": ["Python", "JavaScript", "Machine Learning"],
        "connections": 500
    }
    return mock


@pytest.fixture
def mock_nlp_engine():
    """Mock NLP engine for testing."""
    mock = MagicMock()
    mock.extract_skills_from_text.return_value = [
        {"skill": "Python", "confidence": 0.95},
        {"skill": "Machine Learning", "confidence": 0.88},
        {"skill": "FastAPI", "confidence": 0.82}
    ]
    mock.generate_embeddings.return_value = [0.1] * 384  # Mock embedding vector
    return mock


@pytest.fixture
def mock_recommendation_engine():
    """Mock recommendation engine for testing."""
    mock = MagicMock()
    mock.recommend_careers.return_value = [
        {
            "job_title": "Senior ML Engineer",
            "match_score": 0.92,
            "required_skills": ["Python", "Machine Learning", "TensorFlow"],
            "skill_gaps": {"TensorFlow": 0.3},
            "salary_range": (120000, 180000),
            "reasoning": "Strong match based on current skills"
        }
    ]
    mock.recommend_learning_paths.return_value = [
        {
            "title": "Advanced Machine Learning",
            "duration_weeks": 12,
            "skills": ["Deep Learning", "Neural Networks"],
            "resources": ["Coursera ML Course", "PyTorch Tutorial"]
        }
    ]
    return mock


# Performance testing fixtures
@pytest.fixture
def performance_test_data():
    """Generate data for performance testing."""
    return {
        "users": [
            {"email": f"user{i}@test.com", "password": "password123"}
            for i in range(100)
        ],
        "profiles": [
            {
                "skills": ["Python", "JavaScript", "React"],
                "dream_job": f"Engineer {i}",
                "experience_years": i % 10
            }
            for i in range(100)
        ]
    }


# ML testing fixtures
@pytest.fixture
def synthetic_resume_data():
    """Generate synthetic resume data for ML testing."""
    return [
        {
            "text": "Experienced Python developer with 5 years in web development. Skilled in Django, Flask, and FastAPI.",
            "expected_skills": ["Python", "Django", "Flask", "FastAPI", "Web Development"]
        },
        {
            "text": "Data scientist with expertise in machine learning, deep learning, and statistical analysis using Python and R.",
            "expected_skills": ["Machine Learning", "Deep Learning", "Python", "R", "Statistics"]
        },
        {
            "text": "Frontend developer specializing in React, Vue.js, and modern JavaScript frameworks.",
            "expected_skills": ["React", "Vue.js", "JavaScript", "Frontend Development"]
        }
    ]


@pytest.fixture
def ml_model_test_data():
    """Test data for ML model validation."""
    return {
        "skill_embeddings": {
            "Python": [0.1, 0.2, 0.3] * 128,  # 384-dim vector
            "JavaScript": [0.2, 0.1, 0.4] * 128,
            "Machine Learning": [0.3, 0.4, 0.1] * 128
        },
        "job_profiles": [
            {
                "title": "Python Developer",
                "skills": ["Python", "Django", "PostgreSQL"],
                "embedding": [0.1, 0.2, 0.3] * 128
            },
            {
                "title": "Frontend Developer", 
                "skills": ["JavaScript", "React", "CSS"],
                "embedding": [0.2, 0.1, 0.4] * 128
            }
        ]
    }