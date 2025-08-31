"""
Simple test configuration without full app loading.
"""
import asyncio
import pytest
from typing import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock

from tests.utils.test_data_generator import TestDataGenerator


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def test_data_generator():
    """Test data generator fixture."""
    return TestDataGenerator()


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
    return mock