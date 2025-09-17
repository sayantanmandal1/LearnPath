"""
Simple tests for learning path service without full app dependencies.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch
import sys
import os

# Add the backend directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app.schemas.learning_path import DifficultyLevel


class TestLearningPathServiceSimple:
    """Test learning path service methods directly."""
    
    @pytest.mark.asyncio
    async def test_get_user_learning_paths_service(self):
        """Test the service method for getting user learning paths."""
        from app.services.learning_path_service import LearningPathService
        
        service = LearningPathService()
        paths = await service.get_user_learning_paths("test-user", limit=5)
        
        assert isinstance(paths, list)
        assert len(paths) <= 5
        
        # Verify each path has required attributes
        for path in paths:
            assert hasattr(path, 'title')
            assert hasattr(path, 'difficulty_level')
            assert hasattr(path, 'estimated_duration_weeks')
            assert hasattr(path, 'target_skills')
            assert hasattr(path, 'milestones')
            assert hasattr(path, 'resources')
            
            # Verify data types
            assert isinstance(path.title, str)
            assert isinstance(path.target_skills, list)
            assert isinstance(path.milestones, list)
            assert isinstance(path.resources, list)
    
    @pytest.mark.asyncio
    async def test_skill_based_recommendations_service(self):
        """Test the service method for skill-based recommendations."""
        from app.services.learning_path_service import LearningPathService
        
        service = LearningPathService()
        resources = await service.get_skill_based_recommendations(
            ["python", "javascript"], 
            DifficultyLevel.INTERMEDIATE
        )
        
        assert isinstance(resources, list)
        
        # Verify resources are relevant to requested skills
        for resource in resources:
            assert hasattr(resource, 'skills_taught')
            skills_taught = resource.skills_taught
            assert any(skill in ["python", "javascript"] for skill in skills_taught)
            
            # Verify resource structure
            assert hasattr(resource, 'title')
            assert hasattr(resource, 'provider')
            assert hasattr(resource, 'difficulty_level')
            assert hasattr(resource, 'url')
    
    @pytest.mark.asyncio
    async def test_custom_learning_path_creation(self):
        """Test creating custom learning paths."""
        from app.services.learning_path_service import LearningPathService
        
        service = LearningPathService()
        preferences = {
            "difficulty": "intermediate",
            "hours_per_week": 15
        }
        
        path = await service.create_custom_learning_path(
            "Test Custom Path",
            ["python", "django"],
            preferences
        )
        
        assert path.title == "Test Custom Path"
        assert "python" in path.target_skills
        assert "django" in path.target_skills
        assert path.difficulty_level == DifficultyLevel.INTERMEDIATE
        assert len(path.milestones) == 2  # One per skill
        
        # Verify milestones
        for milestone in path.milestones:
            assert hasattr(milestone, 'title')
            assert hasattr(milestone, 'skills_to_acquire')
            assert hasattr(milestone, 'estimated_duration_hours')
    
    @pytest.mark.asyncio
    async def test_generate_learning_paths_for_profile(self):
        """Test generating learning paths from user profile."""
        from app.services.learning_path_service import LearningPathService
        
        service = LearningPathService()
        profile_data = {
            "user_id": "test-user",
            "skills": ["python", "sql"],
            "target_role": "Data Scientist",
            "experience_level": "intermediate",
            "time_commitment_hours_per_week": 12,
            "target_skills": ["machine_learning", "data_visualization"]
        }
        
        paths = await service.generate_learning_paths_for_profile(profile_data)
        
        assert isinstance(paths, list)
        assert len(paths) > 0
        
        # Verify paths are relevant to profile
        for path in paths:
            assert hasattr(path, 'target_skills')
            target_skills = path.target_skills
            # Should include some of the target skills or related skills
            relevant_skills = ["machine_learning", "data_visualization", "python", "data_science"]
            assert any(skill in target_skills for skill in relevant_skills)


if __name__ == "__main__":
    pytest.main([__file__])