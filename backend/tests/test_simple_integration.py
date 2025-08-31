"""
Simple integration tests for database models and repositories
"""
import os
import pytest
import asyncio
from uuid import uuid4

# Set test environment variables before importing models
os.environ["SECRET_KEY"] = "test_secret_key_for_testing_only_32_chars"
os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///./test_simple.db"
os.environ["REDIS_URL"] = "redis://localhost:6379/1"
os.environ["JWT_SECRET_KEY"] = "test_jwt_secret_key_for_testing_only_32_chars"

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.database import Base
from app.models.user import User
from app.models.skill import Skill, UserSkill
from app.repositories.user import UserRepository
from app.repositories.skill import SkillRepository, UserSkillRepository


@pytest.mark.asyncio
async def test_database_models_and_repositories():
    """Test that all models and repositories work correctly"""
    
    # Create test database engine
    engine = create_async_engine(
        "sqlite+aiosqlite:///./test_simple.db",
        echo=False,
        poolclass=StaticPool,
        connect_args={"check_same_thread": False}
    )
    
    # Create all tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    # Create session
    async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    
    async with async_session() as session:
        # Test User operations
        user_repo = UserRepository()
        
        user_data = {
            "email": "test@example.com",
            "hashed_password": "hashed_password",
            "full_name": "Test User",
            "is_active": True,
            "is_verified": False
        }
        
        # Create user
        created_user = await user_repo.create(session, user_data)
        assert created_user.email == "test@example.com"
        assert created_user.full_name == "Test User"
        
        # Get user by ID
        retrieved_user = await user_repo.get(session, created_user.id)
        assert retrieved_user is not None
        assert retrieved_user.email == "test@example.com"
        
        # Get user by email
        user_by_email = await user_repo.get_by_email(session, "test@example.com")
        assert user_by_email is not None
        assert user_by_email.id == created_user.id
        
        # Test Skill operations
        skill_repo = SkillRepository()
        
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
        
        # Create skill
        created_skill = await skill_repo.create(session, skill_data)
        assert created_skill.name == "Python"
        assert created_skill.category == "programming"
        assert created_skill.market_demand == 0.95
        
        # Get skill by name
        skill_by_name = await skill_repo.get_by_name(session, "Python")
        assert skill_by_name is not None
        assert skill_by_name.id == created_skill.id
        
        # Test UserSkill relationship
        user_skill_repo = UserSkillRepository()
        
        user_skill_data = {
            "user_id": created_user.id,
            "skill_id": created_skill.id,
            "confidence_score": 0.85,
            "proficiency_level": "advanced",
            "source": "resume",
            "evidence": "5 years of Python development",
            "years_experience": 5.0,
            "is_verified": True
        }
        
        # Create user-skill relationship
        user_skill = await user_skill_repo.create(session, user_skill_data)
        assert user_skill.confidence_score == 0.85
        assert user_skill.proficiency_level == "advanced"
        
        # Get user skills
        user_skills = await user_skill_repo.get_user_skills(session, created_user.id)
        assert len(user_skills) == 1
        assert user_skills[0].skill_id == created_skill.id
        
        # Test search functionality
        search_results = await skill_repo.search_skills(session, "Py", limit=10)
        assert len(search_results) > 0
        assert any(skill.name == "Python" for skill in search_results)
        
        # Test category filtering
        category_skills = await skill_repo.get_by_category(session, "programming")
        assert len(category_skills) > 0
        assert any(skill.name == "Python" for skill in category_skills)
        
        print("✅ All database models and repositories are working correctly!")
    
    # Cleanup
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(test_database_models_and_repositories())