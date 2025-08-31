"""
Integration tests for Profile API endpoints.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timedelta
from typing import Dict, Any
import json
import io

from fastapi.testclient import TestClient
from fastapi import UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from app.main import app
from app.models.user import User
from app.models.profile import UserProfile
from app.schemas.profile import ProfileCreate, ProfileUpdate


class TestProfileEndpoints:
    """Integration tests for profile API endpoints."""
    
    @pytest.fixture
    def client(self):
        """Test client."""
        return TestClient(app)
    
    @pytest.fixture
    def mock_user(self):
        """Mock authenticated user."""
        return User(
            id="user-123",
            email="test@example.com",
            password_hash="hashed_password",
            is_active=True,
            created_at=datetime.utcnow()
        )
    
    @pytest.fixture
    def mock_profile(self):
        """Mock user profile."""
        return UserProfile(
            id="profile-123",
            user_id="user-123",
            dream_job="Software Engineer",
            experience_years=3,
            current_role="Junior Developer",
            location="New York",
            github_username="testuser",
            leetcode_id="testuser",
            linkedin_url="https://linkedin.com/in/testuser",
            skills={"Python": 0.8, "JavaScript": 0.6},
            platform_data={
                "github": {"languages": {"Python": 2000, "JavaScript": 1000}},
                "leetcode": {"skill_tags": {"algorithms": 10}}
            },
            resume_data={"skills": [{"skill_name": "Python", "confidence_score": 0.8}]},
            career_interests={"web_development": 0.9},
            skill_gaps={"React": 0.4, "Docker": 0.6},
            data_last_updated=datetime.utcnow(),
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
    
    @pytest.fixture
    def auth_headers(self):
        """Mock authentication headers."""
        return {"Authorization": "Bearer mock-jwt-token"}
    
    @pytest.fixture
    def sample_profile_data(self):
        """Sample profile creation data."""
        return {
            "dream_job": "Software Engineer",
            "experience_years": 3,
            "current_role": "Junior Developer",
            "location": "New York",
            "github_username": "testuser",
            "leetcode_id": "testuser",
            "linkedin_url": "https://linkedin.com/in/testuser",
            "skills": {"Python": 0.8, "JavaScript": 0.6},
            "career_interests": {"web_development": 0.9}
        }
    
    @patch('app.api.dependencies.get_current_user')
    @patch('app.services.profile_service.UserProfileService.create_profile_with_integration')
    def test_create_profile_success(self, mock_create_profile, mock_get_user, client, mock_user, mock_profile, auth_headers, sample_profile_data):
        """Test successful profile creation."""
        # Setup mocks
        mock_get_user.return_value = mock_user
        mock_create_profile.return_value = mock_profile
        
        # Make request
        response = client.post(
            "/api/v1/profiles/",
            json=sample_profile_data,
            headers=auth_headers
        )
        
        # Verify response
        assert response.status_code == 201
        data = response.json()
        assert data["id"] == "profile-123"
        assert data["user_id"] == "user-123"
        assert data["dream_job"] == "Software Engineer"
        assert data["skills"]["Python"] == 0.8
        
        # Verify service was called
        mock_create_profile.assert_called_once()
    
    @patch('app.api.dependencies.get_current_user')
    @patch('app.services.profile_service.UserProfileService.create_profile_with_integration')
    def test_create_profile_conflict(self, mock_create_profile, mock_get_user, client, mock_user, auth_headers, sample_profile_data):
        """Test profile creation when profile already exists."""
        from app.core.exceptions import ConflictError
        
        # Setup mocks
        mock_get_user.return_value = mock_user
        mock_create_profile.side_effect = ConflictError("Profile already exists for user user-123")
        
        # Make request
        response = client.post(
            "/api/v1/profiles/",
            json=sample_profile_data,
            headers=auth_headers
        )
        
        # Verify response
        assert response.status_code == 409
        assert "already exists" in response.json()["detail"]
    
    @patch('app.api.dependencies.get_current_user')
    @patch('app.services.profile_service.UserProfileService.create_profile_with_integration')
    def test_create_profile_validation_error(self, mock_create_profile, mock_get_user, client, mock_user, auth_headers):
        """Test profile creation with validation error."""
        from app.core.exceptions import ValidationError
        
        # Setup mocks
        mock_get_user.return_value = mock_user
        mock_create_profile.side_effect = ValidationError("Invalid profile data")
        
        # Invalid profile data
        invalid_data = {
            "experience_years": -1,  # Invalid
            "github_username": "invalid@username"  # Invalid format
        }
        
        # Make request
        response = client.post(
            "/api/v1/profiles/",
            json=invalid_data,
            headers=auth_headers
        )
        
        # Verify response
        assert response.status_code == 422
        assert "Invalid" in response.json()["detail"]
    
    @patch('app.api.dependencies.get_current_user')
    @patch('app.services.profile_service.UserProfileService.create_profile_with_integration')
    def test_create_profile_with_resume_success(self, mock_create_profile, mock_get_user, client, mock_user, mock_profile, auth_headers, sample_profile_data):
        """Test successful profile creation with resume upload."""
        # Setup mocks
        mock_get_user.return_value = mock_user
        mock_create_profile.return_value = mock_profile
        
        # Create mock PDF file
        pdf_content = b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\n"
        
        # Make request with multipart form data
        response = client.post(
            "/api/v1/profiles/with-resume",
            data=sample_profile_data,
            files={"resume_file": ("resume.pdf", io.BytesIO(pdf_content), "application/pdf")},
            headers=auth_headers
        )
        
        # Verify response
        assert response.status_code == 201
        data = response.json()
        assert data["id"] == "profile-123"
        
        # Verify service was called with resume file
        mock_create_profile.assert_called_once()
        call_args = mock_create_profile.call_args
        assert call_args.kwargs["resume_file"] is not None
    
    @patch('app.api.dependencies.get_current_user')
    def test_create_profile_with_resume_invalid_file_type(self, mock_get_user, client, mock_user, auth_headers, sample_profile_data):
        """Test profile creation with invalid resume file type."""
        # Setup mocks
        mock_get_user.return_value = mock_user
        
        # Create mock text file (invalid)
        text_content = b"This is a text file, not a resume"
        
        # Make request
        response = client.post(
            "/api/v1/profiles/with-resume",
            data=sample_profile_data,
            files={"resume_file": ("resume.txt", io.BytesIO(text_content), "text/plain")},
            headers=auth_headers
        )
        
        # Verify response
        assert response.status_code == 422
        assert "Only PDF and Word documents are supported" in response.json()["detail"]
    
    @patch('app.api.dependencies.get_current_user')
    def test_create_profile_with_resume_file_too_large(self, mock_get_user, client, mock_user, auth_headers, sample_profile_data):
        """Test profile creation with resume file too large."""
        # Setup mocks
        mock_get_user.return_value = mock_user
        
        # Create mock large PDF file (>10MB)
        large_content = b"x" * (11 * 1024 * 1024)  # 11MB
        
        # Mock file size
        with patch('fastapi.UploadFile.size', 11 * 1024 * 1024):
            response = client.post(
                "/api/v1/profiles/with-resume",
                data=sample_profile_data,
                files={"resume_file": ("resume.pdf", io.BytesIO(large_content), "application/pdf")},
                headers=auth_headers
            )
        
        # Verify response
        assert response.status_code == 422
        assert "File size must be less than 10MB" in response.json()["detail"]
    
    @patch('app.api.dependencies.get_current_user')
    @patch('app.repositories.profile.ProfileRepository.get_by_user_id')
    def test_get_my_profile_success(self, mock_get_profile, mock_get_user, client, mock_user, mock_profile, auth_headers):
        """Test successful profile retrieval."""
        # Setup mocks
        mock_get_user.return_value = mock_user
        mock_get_profile.return_value = mock_profile
        
        # Make request
        response = client.get("/api/v1/profiles/me", headers=auth_headers)
        
        # Verify response
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "profile-123"
        assert data["user_id"] == "user-123"
        assert data["dream_job"] == "Software Engineer"
    
    @patch('app.api.dependencies.get_current_user')
    @patch('app.repositories.profile.ProfileRepository.get_by_user_id')
    def test_get_my_profile_not_found(self, mock_get_profile, mock_get_user, client, mock_user, auth_headers):
        """Test profile retrieval when profile doesn't exist."""
        # Setup mocks
        mock_get_user.return_value = mock_user
        mock_get_profile.return_value = None
        
        # Make request
        response = client.get("/api/v1/profiles/me", headers=auth_headers)
        
        # Verify response
        assert response.status_code == 404
        assert "Profile not found" in response.json()["detail"]
    
    @patch('app.api.dependencies.get_current_user')
    @patch('app.services.profile_service.UserProfileService.update_profile_with_validation')
    def test_update_my_profile_success(self, mock_update_profile, mock_get_user, client, mock_user, mock_profile, auth_headers):
        """Test successful profile update."""
        # Setup mocks
        mock_get_user.return_value = mock_user
        mock_update_profile.return_value = mock_profile
        
        update_data = {
            "experience_years": 5,
            "current_role": "Senior Developer",
            "skills": {"Python": 0.9, "React": 0.7}
        }
        
        # Make request
        response = client.put(
            "/api/v1/profiles/me",
            json=update_data,
            headers=auth_headers
        )
        
        # Verify response
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "profile-123"
        
        # Verify service was called
        mock_update_profile.assert_called_once()
    
    @patch('app.api.dependencies.get_current_user')
    @patch('app.services.profile_service.UserProfileService.update_profile_with_validation')
    def test_update_my_profile_not_found(self, mock_update_profile, mock_get_user, client, mock_user, auth_headers):
        """Test profile update when profile doesn't exist."""
        from app.core.exceptions import NotFoundError
        
        # Setup mocks
        mock_get_user.return_value = mock_user
        mock_update_profile.side_effect = NotFoundError("Profile not found for user user-123")
        
        update_data = {"experience_years": 5}
        
        # Make request
        response = client.put(
            "/api/v1/profiles/me",
            json=update_data,
            headers=auth_headers
        )
        
        # Verify response
        assert response.status_code == 404
        assert "not found" in response.json()["detail"]
    
    @patch('app.api.dependencies.get_current_user')
    @patch('app.services.profile_service.UserProfileService.refresh_external_data')
    def test_refresh_profile_data_success(self, mock_refresh_data, mock_get_user, client, mock_user, mock_profile, auth_headers):
        """Test successful profile data refresh."""
        # Setup mocks
        mock_get_user.return_value = mock_user
        mock_refresh_data.return_value = mock_profile
        
        # Make request
        response = client.post("/api/v1/profiles/me/refresh", headers=auth_headers)
        
        # Verify response
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "profile-123"
        
        # Verify service was called with default force_refresh=False
        mock_refresh_data.assert_called_once()
        call_args = mock_refresh_data.call_args
        assert call_args.kwargs["force_refresh"] is False
    
    @patch('app.api.dependencies.get_current_user')
    @patch('app.services.profile_service.UserProfileService.refresh_external_data')
    def test_refresh_profile_data_force_refresh(self, mock_refresh_data, mock_get_user, client, mock_user, mock_profile, auth_headers):
        """Test profile data refresh with force_refresh=True."""
        # Setup mocks
        mock_get_user.return_value = mock_user
        mock_refresh_data.return_value = mock_profile
        
        # Make request with force_refresh parameter
        response = client.post(
            "/api/v1/profiles/me/refresh?force_refresh=true",
            headers=auth_headers
        )
        
        # Verify response
        assert response.status_code == 200
        
        # Verify service was called with force_refresh=True
        mock_refresh_data.assert_called_once()
        call_args = mock_refresh_data.call_args
        assert call_args.kwargs["force_refresh"] is True
    
    @patch('app.api.dependencies.get_current_user')
    @patch('app.repositories.profile.ProfileRepository.get_by_user_id')
    @patch('app.repositories.profile.ProfileRepository.update')
    @patch('app.services.profile_service.UserProfileService._process_resume_file')
    def test_upload_resume_success(self, mock_process_resume, mock_update_profile, mock_get_profile, mock_get_user, client, mock_user, mock_profile, auth_headers):
        """Test successful resume upload."""
        # Setup mocks
        mock_get_user.return_value = mock_user
        mock_get_profile.return_value = mock_profile
        mock_update_profile.return_value = mock_profile
        
        # Mock resume processing
        mock_resume_data = Mock()
        mock_resume_data.dict.return_value = {"skills": [{"skill_name": "Python", "confidence_score": 0.9}]}
        mock_process_resume.return_value = mock_resume_data
        
        # Create mock PDF file
        pdf_content = b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\n"
        
        # Make request
        response = client.post(
            "/api/v1/profiles/me/upload-resume",
            files={"resume_file": ("resume.pdf", io.BytesIO(pdf_content), "application/pdf")},
            headers=auth_headers
        )
        
        # Verify response
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "profile-123"
        
        # Verify resume was processed
        mock_process_resume.assert_called_once()
        mock_update_profile.assert_called_once()
    
    @patch('app.api.dependencies.get_current_user')
    @patch('app.repositories.profile.ProfileRepository.get_by_user_id')
    def test_upload_resume_profile_not_found(self, mock_get_profile, mock_get_user, client, mock_user, auth_headers):
        """Test resume upload when profile doesn't exist."""
        # Setup mocks
        mock_get_user.return_value = mock_user
        mock_get_profile.return_value = None
        
        # Create mock PDF file
        pdf_content = b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\n"
        
        # Make request
        response = client.post(
            "/api/v1/profiles/me/upload-resume",
            files={"resume_file": ("resume.pdf", io.BytesIO(pdf_content), "application/pdf")},
            headers=auth_headers
        )
        
        # Verify response
        assert response.status_code == 404
        assert "Profile not found" in response.json()["detail"]
    
    @patch('app.api.dependencies.get_current_user')
    @patch('app.services.profile_service.UserProfileService.get_profile_analytics')
    def test_get_profile_analytics_success(self, mock_get_analytics, mock_get_user, client, mock_user, auth_headers):
        """Test successful profile analytics retrieval."""
        # Setup mocks
        mock_get_user.return_value = mock_user
        mock_analytics = {
            "profile_completeness": {"score": 0.8, "completed_fields": 8, "total_fields": 10},
            "skill_distribution": {"total_skills": 5, "categories": {}, "top_skills": []},
            "data_freshness": {"freshness_score": 0.9, "needs_refresh": False},
            "platform_coverage": {"coverage_score": 0.75, "connected_count": 3},
            "skill_gaps_summary": {"total_gaps": 2, "critical_gaps": [], "moderate_gaps": []},
            "recommendations": [{"type": "completeness", "priority": "medium", "title": "Add more skills"}]
        }
        mock_get_analytics.return_value = mock_analytics
        
        # Make request
        response = client.get("/api/v1/profiles/me/analytics", headers=auth_headers)
        
        # Verify response
        assert response.status_code == 200
        data = response.json()
        assert data["profile_completeness"]["score"] == 0.8
        assert data["skill_distribution"]["total_skills"] == 5
        assert data["data_freshness"]["freshness_score"] == 0.9
        assert len(data["recommendations"]) == 1
    
    @patch('app.api.dependencies.get_current_user')
    @patch('app.services.profile_service.UserProfileService.external_api_service.validate_profile_sources')
    def test_validate_profile_sources_success(self, mock_validate_sources, mock_get_user, client, mock_user, auth_headers):
        """Test successful profile sources validation."""
        # Setup mocks
        mock_get_user.return_value = mock_user
        mock_validate_sources.return_value = {
            "github": True,
            "leetcode": True,
            "linkedin": False
        }
        
        # Make request
        response = client.post(
            "/api/v1/profiles/me/validate-sources",
            params={
                "github_username": "testuser",
                "leetcode_username": "testuser",
                "linkedin_url": "https://linkedin.com/in/testuser"
            },
            headers=auth_headers
        )
        
        # Verify response
        assert response.status_code == 200
        data = response.json()
        assert data["validation_results"]["github"] is True
        assert data["validation_results"]["leetcode"] is True
        assert data["validation_results"]["linkedin"] is False
        assert data["all_valid"] is False
        assert "github" in data["valid_sources"]
        assert "linkedin" in data["invalid_sources"]
    
    @patch('app.api.dependencies.get_current_user')
    @patch('app.repositories.profile.ProfileRepository.get_by_user_id')
    @patch('app.services.profile_service.UserProfileService._calculate_skill_gaps')
    def test_get_skill_gaps_success(self, mock_calculate_gaps, mock_get_profile, mock_get_user, client, mock_user, mock_profile, auth_headers):
        """Test successful skill gaps retrieval."""
        # Setup mocks
        mock_get_user.return_value = mock_user
        mock_get_profile.return_value = mock_profile
        mock_calculate_gaps.return_value = {
            "React": 0.6,
            "Docker": 0.4,
            "AWS": 0.8
        }
        
        # Make request
        response = client.get("/api/v1/profiles/me/skill-gaps", headers=auth_headers)
        
        # Verify response
        assert response.status_code == 200
        data = response.json()
        assert data["target_role"] == "Software Engineer"  # From profile's dream_job
        assert "React" in data["skill_gaps"]
        assert "Docker" in data["skill_gaps"]
        assert "AWS" in data["skill_gaps"]
        assert len(data["recommendations"]) == 3
        
        # Check recommendations are sorted by gap size (descending)
        recommendations = data["recommendations"]
        assert recommendations[0]["skill"] == "AWS"  # Highest gap (0.8)
        assert recommendations[0]["priority"] == "critical"
        assert recommendations[1]["skill"] == "React"  # Medium gap (0.6)
        assert recommendations[2]["skill"] == "Docker"  # Lowest gap (0.4)
    
    @patch('app.api.dependencies.get_current_user')
    @patch('app.repositories.profile.ProfileRepository.get_by_user_id')
    @patch('app.services.profile_service.UserProfileService._calculate_skill_gaps')
    def test_get_skill_gaps_with_target_role(self, mock_calculate_gaps, mock_get_profile, mock_get_user, client, mock_user, mock_profile, auth_headers):
        """Test skill gaps retrieval with custom target role."""
        # Setup mocks
        mock_get_user.return_value = mock_user
        mock_get_profile.return_value = mock_profile
        mock_calculate_gaps.return_value = {
            "Machine Learning": 0.9,
            "TensorFlow": 0.7,
            "Pandas": 0.5
        }
        
        # Make request with target role parameter
        response = client.get(
            "/api/v1/profiles/me/skill-gaps?target_role=Data Scientist",
            headers=auth_headers
        )
        
        # Verify response
        assert response.status_code == 200
        data = response.json()
        assert data["target_role"] == "Data Scientist"  # Override dream_job
        assert "Machine Learning" in data["skill_gaps"]
        assert "TensorFlow" in data["skill_gaps"]
        assert "Pandas" in data["skill_gaps"]
        
        # Verify service was called with custom target role
        mock_calculate_gaps.assert_called_once_with(mock_profile.skills, "Data Scientist")
    
    @patch('app.api.dependencies.get_current_user')
    @patch('app.repositories.profile.ProfileRepository.get_by_user_id')
    def test_get_skill_gaps_no_target_role(self, mock_get_profile, mock_get_user, client, mock_user, auth_headers):
        """Test skill gaps retrieval when no target role is available."""
        # Setup mocks
        mock_get_user.return_value = mock_user
        
        # Profile without dream_job
        profile_without_dream_job = UserProfile(
            id="profile-123",
            user_id="user-123",
            dream_job=None,  # No dream job
            skills={"Python": 0.8}
        )
        mock_get_profile.return_value = profile_without_dream_job
        
        # Make request without target_role parameter
        response = client.get("/api/v1/profiles/me/skill-gaps", headers=auth_headers)
        
        # Verify response
        assert response.status_code == 422
        assert "No target role specified" in response.json()["detail"]
    
    def test_unauthorized_access(self, client):
        """Test that endpoints require authentication."""
        # Test various endpoints without auth headers
        endpoints = [
            ("POST", "/api/v1/profiles/"),
            ("GET", "/api/v1/profiles/me"),
            ("PUT", "/api/v1/profiles/me"),
            ("POST", "/api/v1/profiles/me/refresh"),
            ("GET", "/api/v1/profiles/me/analytics"),
            ("GET", "/api/v1/profiles/me/skill-gaps")
        ]
        
        for method, endpoint in endpoints:
            if method == "POST":
                response = client.post(endpoint, json={})
            elif method == "PUT":
                response = client.put(endpoint, json={})
            else:
                response = client.get(endpoint)
            
            assert response.status_code == 401 or response.status_code == 403


if __name__ == "__main__":
    pytest.main([__file__, "-v"])