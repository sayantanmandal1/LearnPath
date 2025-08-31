"""
Tests for repository classes
"""
import os
import pytest
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

# Set test environment variables before importing models
os.environ["SECRET_KEY"] = "test_secret_key_for_testing_only_32_chars"
os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///./test.db"
os.environ["REDIS_URL"] = "redis://localhost:6379/1"
os.environ["JWT_SECRET_KEY"] = "test_jwt_secret_key_for_testing_only_32_chars"

from app.repositories.user import UserRepository
from app.repositories.skill import SkillRepository, UserSkillRepository
from app.repositories.job import JobRepository
from app.models.user import User
from app.models.skill import Skill, UserSkill
from app.models.job import JobPosting


@pytest.fixture
def mock_db_session():
    """Mock database session"""
    session = AsyncMock()
    return session


@pytest.fixture
def user_repo():
    """User repository fixture"""
    return UserRepository()


@pytest.fixture
def skill_repo():
    """Skill repository fixture"""
    return SkillRepository()


@pytest.fixture
def user_skill_repo():
    """User skill repository fixture"""
    return UserSkillRepository()


@pytest.fixture
def job_repo():
    """Job repository fixture"""
    return JobRepository()


class TestUserRepository:
    """Test UserRepository methods"""
    
    def test_init(self, user_repo):
        """Test repository initialization"""
        assert user_repo.model == User
    
    @pytest.mark.asyncio
    async def test_get_by_email(self, user_repo, mock_db_session):
        """Test get user by email"""
        # Mock the database result
        mock_user = User(
            id=str(uuid4()),
            email="test@example.com",
            hashed_password="hashed",
            is_active=True,
            is_verified=False
        )
        
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_user
        mock_db_session.execute.return_value = mock_result
        
        result = await user_repo.get_by_email(mock_db_session, "test@example.com")
        
        assert result == mock_user
        mock_db_session.execute.assert_called_once()


class TestSkillRepository:
    """Test SkillRepository methods"""
    
    def test_init(self, skill_repo):
        """Test repository initialization"""
        assert skill_repo.model == Skill
    
    @pytest.mark.asyncio
    async def test_get_by_name(self, skill_repo, mock_db_session):
        """Test get skill by name"""
        mock_skill = Skill(
            id=str(uuid4()),
            name="Python",
            category="programming",
            is_active=True
        )
        
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_skill
        mock_db_session.execute.return_value = mock_result
        
        result = await skill_repo.get_by_name(mock_db_session, "Python")
        
        assert result == mock_skill
        mock_db_session.execute.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_find_or_create_skill_existing(self, skill_repo, mock_db_session):
        """Test find_or_create_skill with existing skill"""
        mock_skill = Skill(
            id=str(uuid4()),
            name="Python",
            category="programming",
            is_active=True
        )
        
        # Mock get_by_name to return existing skill
        skill_repo.get_by_name = AsyncMock(return_value=mock_skill)
        
        result = await skill_repo.find_or_create_skill(
            mock_db_session, "Python", "programming"
        )
        
        assert result == mock_skill
        skill_repo.get_by_name.assert_called_once_with(mock_db_session, "Python")


class TestUserSkillRepository:
    """Test UserSkillRepository methods"""
    
    def test_init(self, user_skill_repo):
        """Test repository initialization"""
        assert user_skill_repo.model == UserSkill
    
    @pytest.mark.asyncio
    async def test_get_user_skill(self, user_skill_repo, mock_db_session):
        """Test get specific user skill"""
        user_id = str(uuid4())
        skill_id = str(uuid4())
        
        mock_user_skill = UserSkill(
            id=str(uuid4()),
            user_id=user_id,
            skill_id=skill_id,
            confidence_score=0.8,
            proficiency_level="intermediate",
            source="resume",
            is_verified=False
        )
        
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_user_skill
        mock_db_session.execute.return_value = mock_result
        
        result = await user_skill_repo.get_user_skill(
            mock_db_session, user_id, skill_id
        )
        
        assert result == mock_user_skill
        mock_db_session.execute.assert_called_once()


class TestJobRepository:
    """Test JobRepository methods"""
    
    def test_init(self, job_repo):
        """Test repository initialization"""
        assert job_repo.model == JobPosting
    
    @pytest.mark.asyncio
    async def test_get_by_external_id(self, job_repo, mock_db_session):
        """Test get job by external ID"""
        mock_job = JobPosting(
            id=str(uuid4()),
            external_id="job123",
            title="Software Engineer",
            company="Tech Corp",
            description="Job description",
            source="linkedin",
            is_active=True,
            is_processed=False
        )
        
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_job
        mock_db_session.execute.return_value = mock_result
        
        result = await job_repo.get_by_external_id(
            mock_db_session, "job123", "linkedin"
        )
        
        assert result == mock_job
        mock_db_session.execute.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_mark_as_processed(self, job_repo, mock_db_session):
        """Test marking job as processed"""
        job_id = str(uuid4())
        processed_skills = {"Python": 0.9, "JavaScript": 0.7}
        
        # Mock the update method
        job_repo.update = AsyncMock(return_value=MagicMock())
        
        result = await job_repo.mark_as_processed(
            mock_db_session, job_id, processed_skills
        )
        
        job_repo.update.assert_called_once_with(
            mock_db_session, job_id, {
                "is_processed": True,
                "processed_skills": processed_skills
            }
        )