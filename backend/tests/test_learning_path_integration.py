"""
Integration tests for learning path generation backend.

This module tests the learning path generation functionality to ensure
it works correctly for frontend integration.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
import json

from app.main import app
from app.models.user import User
from app.schemas.learning_path import DifficultyLevel


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def mock_user():
    """Create mock user for testing."""
    user = Mock(spec=User)
    user.id = "test-user-123"
    user.email = "test@example.com"
    user.first_name = "Test"
    user.last_name = "User"
    return user


@pytest.fixture
def auth_headers():
    """Create mock authentication headers."""
    return {"Authorization": "Bearer test-token"}


class TestLearningPathGeneration:
    """Test learning path generation functionality."""
    
    @patch('app.api.dependencies.get_current_user')
    def test_generate_learning_paths(self, mock_get_user, client, mock_user, auth_headers):
        """Test basic learning path generation."""
        mock_get_user.return_value = mock_user
        
        request_data = {
            "user_id": "test-user-123",
            "target_role": "Full Stack Developer",
            "target_skills": ["javascript", "react", "node.js"],
            "current_skills": {"html": 0.8, "css": 0.7},
            "time_commitment_hours_per_week": 15,
            "difficulty_preference": "intermediate"
        }
        
        response = client.post(
            "/api/v1/learning-paths/generate",
            json=request_data,
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify response structure
        assert "learning_paths" in data
        assert "total_paths" in data
        assert "skill_gaps_identified" in data
        
        # Verify learning paths have required fields
        if data["learning_paths"]:
            path = data["learning_paths"][0]
            assert "title" in path
            assert "difficulty_level" in path
            assert "estimated_duration_weeks" in path
            assert "milestones" in path
            assert "resources" in path
    
    @patch('app.api.dependencies.get_current_user')
    def test_get_user_learning_paths(self, mock_get_user, client, mock_user, auth_headers):
        """Test getting user learning paths."""
        mock_get_user.return_value = mock_user
        
        response = client.get(
            f"/api/v1/learning-paths/{mock_user.id}",
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Should return a list of learning paths
        assert isinstance(data, list)
        
        # Verify each path has required fields
        for path in data:
            assert "id" in path
            assert "title" in path
            assert "difficulty_level" in path
            assert "estimated_duration_weeks" in path
            assert "target_skills" in path
    
    @patch('app.api.dependencies.get_current_user')
    def test_get_simplified_learning_paths(self, mock_get_user, client, mock_user, auth_headers):
        """Test getting learning paths in simplified format for frontend."""
        mock_get_user.return_value = mock_user
        
        response = client.get(
            f"/api/v1/learning-paths/{mock_user.id}/simplified",
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Should return a list of simplified paths
        assert isinstance(data, list)
        
        # Verify simplified format matches frontend expectations
        for path in data:
            assert "title" in path
            assert "provider" in path
            assert "duration" in path
            assert "difficulty" in path
            
            # Verify data types
            assert isinstance(path["title"], str)
            assert isinstance(path["provider"], str)
            assert isinstance(path["duration"], str)
            assert isinstance(path["difficulty"], str)
    
    @patch('app.api.dependencies.get_current_user')
    def test_generate_from_profile(self, mock_get_user, client, mock_user, auth_headers):
        """Test generating learning paths from user profile."""
        mock_get_user.return_value = mock_user
        
        profile_data = {
            "skills": ["python", "sql"],
            "target_role": "Data Scientist",
            "experience_level": "intermediate",
            "time_commitment_hours_per_week": 12,
            "learning_style": "hands_on",
            "target_skills": ["machine_learning", "data_visualization"]
        }
        
        response = client.post(
            "/api/v1/learning-paths/generate-from-profile",
            json=profile_data,
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Should return a list of learning paths
        assert isinstance(data, list)
        
        # Verify paths are relevant to profile
        for path in data:
            assert "target_skills" in path
            # Should include some of the target skills
            target_skills = path["target_skills"]
            assert any(skill in ["machine_learning", "data_visualization", "python"] for skill in target_skills)
    
    @patch('app.api.dependencies.get_current_user')
    def test_get_skill_based_resources(self, mock_get_user, client, mock_user, auth_headers):
        """Test getting resources for specific skills."""
        mock_get_user.return_value = mock_user
        
        skills = "python,machine_learning"
        response = client.get(
            f"/api/v1/learning-paths/skills/{skills}/resources",
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Should return a list of learning resources
        assert isinstance(data, list)
        
        # Verify resource structure
        for resource in data:
            assert "title" in resource
            assert "type" in resource
            assert "provider" in resource
            assert "url" in resource
            assert "difficulty_level" in resource
            assert "skills_taught" in resource
            
            # Verify skills taught include requested skills
            skills_taught = resource["skills_taught"]
            assert any(skill in ["python", "machine_learning"] for skill in skills_taught)
    
    @patch('app.api.dependencies.get_current_user')
    def test_create_custom_learning_path(self, mock_get_user, client, mock_user, auth_headers):
        """Test creating a custom learning path."""
        mock_get_user.return_value = mock_user
        
        request_data = {
            "title": "Custom Web Development Path",
            "skills": ["javascript", "react", "css"],
            "preferences": {
                "difficulty": "intermediate",
                "hours_per_week": 10
            }
        }
        
        response = client.post(
            "/api/v1/learning-paths/custom",
            json=request_data,
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify custom path structure
        assert data["title"] == "Custom Web Development Path"
        assert "javascript" in data["target_skills"]
        assert "react" in data["target_skills"]
        assert "css" in data["target_skills"]
        assert data["difficulty_level"] == "intermediate"
        assert len(data["milestones"]) == 3  # One per skill
    
    @patch('app.api.dependencies.get_current_user')
    def test_get_project_recommendations(self, mock_get_user, client, mock_user, auth_headers):
        """Test getting project recommendations."""
        mock_get_user.return_value = mock_user
        
        response = client.get(
            "/api/v1/learning-paths/projects",
            params={
                "skills": ["python", "machine_learning"],
                "difficulty": "intermediate",
                "limit": 5
            },
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Should return a list of project recommendations
        assert isinstance(data, list)
        assert len(data) <= 5  # Respects limit
        
        # Verify project structure
        for project in data:
            assert "title" in project
            assert "description" in project
            assert "difficulty_level" in project
            assert "skills_practiced" in project
            assert "technologies" in project
            assert "estimated_duration_hours" in project
    
    @patch('app.api.dependencies.get_current_user')
    def test_submit_feedback(self, mock_get_user, client, mock_user, auth_headers):
        """Test submitting learning path feedback."""
        mock_get_user.return_value = mock_user
        
        response = client.post(
            "/api/v1/learning-paths/feedback",
            params={
                "path_id": "test-path-123",
                "rating": 4,
                "feedback": "Great learning path!",
                "completed": True
            },
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["message"] == "Feedback submitted successfully"
        assert data["path_id"] == "test-path-123"
        assert data["rating"] == 4
        assert data["completed"] is True


class TestLearningPathService:
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


if __name__ == "__main__":
    pytest.main([__file__])