"""
Tests for comprehensive API endpoints with filtering, pagination, and advanced features.
"""

import pytest
from httpx import AsyncClient
from unittest.mock import Mock, patch
from datetime import datetime

from app.main import app
from app.models.user import User
from app.models.profile import UserProfile


@pytest.fixture
def mock_user():
    """Mock user for testing."""
    user = Mock(spec=User)
    user.id = "test-user-id"
    user.email = "test@example.com"
    user.is_active = True
    return user


@pytest.fixture
def mock_profile():
    """Mock user profile for testing."""
    profile = Mock(spec=UserProfile)
    profile.user_id = "test-user-id"
    profile.current_role = "Software Engineer"
    profile.dream_job = "Senior Software Engineer"
    profile.experience_years = 3
    profile.location = "San Francisco, CA"
    profile.skills = {
        "python": 0.8,
        "javascript": 0.7,
        "sql": 0.6,
        "react": 0.7,
        "docker": 0.5
    }
    profile.resume_data = {"extracted_text": "Sample resume content"}
    profile.platform_data = {
        "github": {"username": "testuser", "repos": 10},
        "leetcode": {"username": "testuser", "problems_solved": 150}
    }
    profile.data_last_updated = datetime.utcnow()
    return profile


class TestComprehensiveProfileEndpoints:
    """Test comprehensive profile endpoints."""
    
    @pytest.mark.asyncio
    async def test_get_comprehensive_profile_data(self, mock_user, mock_profile):
        """Test getting comprehensive profile data."""
        with patch('app.api.dependencies.get_current_user', return_value=mock_user), \
             patch('app.repositories.profile.ProfileRepository.get_by_user_id', return_value=mock_profile), \
             patch('app.services.analytics_service.AnalyticsService.get_profile_analytics') as mock_analytics, \
             patch('app.services.recommendation_service.RecommendationService.get_career_recommendations') as mock_recommendations:
            
            mock_analytics.return_value = {
                "profile_completeness": {"score": 0.85, "percentage": 85.0},
                "skill_analysis": {"total_skills": 5}
            }
            mock_recommendations.return_value = [
                {"job_title": "Senior Developer", "match_score": 0.9}
            ]
            
            async with AsyncClient(app=app, base_url="http://test") as client:
                response = await client.get(
                    "/api/v1/comprehensive/profiles/comprehensive",
                    params={
                        "include_analytics": True,
                        "include_recommendations": True,
                        "include_market_data": False
                    }
                )
            
            assert response.status_code == 200
            data = response.json()
            assert "profile" in data
            assert "analytics" in data
            assert "quick_recommendations" in data
            assert data["profile"]["current_role"] == "Software Engineer"
    
    @pytest.mark.asyncio
    async def test_get_comprehensive_profile_data_not_found(self, mock_user):
        """Test getting comprehensive profile data when profile not found."""
        with patch('app.api.dependencies.get_current_user', return_value=mock_user), \
             patch('app.repositories.profile.ProfileRepository.get_by_user_id', return_value=None):
            
            async with AsyncClient(app=app, base_url="http://test") as client:
                response = await client.get("/api/v1/comprehensive/profiles/comprehensive")
            
            assert response.status_code == 404
            assert "Profile not found" in response.json()["detail"]


class TestAdvancedRecommendations:
    """Test advanced recommendation endpoints."""
    
    @pytest.mark.asyncio
    async def test_get_advanced_career_recommendations(self, mock_user):
        """Test getting advanced career recommendations with filtering."""
        mock_recommendations = [
            {
                "job_title": "Senior Python Developer",
                "match_score": 0.9,
                "location": "San Francisco, CA",
                "salary_range": {"min": 120000, "max": 150000}
            },
            {
                "job_title": "Full Stack Developer",
                "match_score": 0.8,
                "location": "Remote",
                "salary_range": {"min": 100000, "max": 130000}
            }
        ]
        
        with patch('app.api.dependencies.get_current_user', return_value=mock_user), \
             patch('app.services.recommendation_service.RecommendationService.get_filtered_career_recommendations', return_value=mock_recommendations):
            
            async with AsyncClient(app=app, base_url="http://test") as client:
                response = await client.post(
                    "/api/v1/comprehensive/recommendations/career/advanced",
                    json={
                        "target_role": "Senior Developer",
                        "n_recommendations": 10,
                        "include_explanations": True,
                        "include_alternatives": True,
                        "filters": {
                            "location": "San Francisco",
                            "salary_min": 100000
                        },
                        "sort": {
                            "sort_by": "match_score",
                            "sort_order": "desc"
                        }
                    },
                    params={"page": 1, "page_size": 20}
                )
            
            assert response.status_code == 200
            data = response.json()
            assert "items" in data
            assert "total_count" in data
            assert "page" in data
            assert len(data["items"]) == 2
            assert data["items"][0]["job_title"] == "Senior Python Developer"
    
    @pytest.mark.asyncio
    async def test_get_customized_learning_paths(self, mock_user):
        """Test getting customized learning paths."""
        mock_learning_paths = [
            {
                "skill": "python",
                "title": "Master Python",
                "estimated_duration_weeks": 8,
                "estimated_cost": 99.99,
                "resources": [
                    {"title": "Python Course", "type": "course", "provider": "Coursera"}
                ]
            }
        ]
        
        with patch('app.api.dependencies.get_current_user', return_value=mock_user), \
             patch('app.services.learning_path_service.LearningPathService.generate_customized_learning_paths', return_value=mock_learning_paths):
            
            async with AsyncClient(app=app, base_url="http://test") as client:
                response = await client.post(
                    "/api/v1/comprehensive/recommendations/learning-paths/customized",
                    params={
                        "target_skills": ["python", "machine learning"],
                        "difficulty_preference": "intermediate",
                        "time_commitment_hours_per_week": 15,
                        "budget_max": 200.0,
                        "learning_style": "mixed",
                        "include_projects": True
                    }
                )
            
            assert response.status_code == 200
            data = response.json()
            assert "learning_paths" in data
            assert "customization_applied" in data
            assert "estimated_completion_weeks" in data
            assert len(data["learning_paths"]) == 1


class TestJobMatching:
    """Test advanced job matching endpoints."""
    
    @pytest.mark.asyncio
    async def test_get_advanced_job_matches(self, mock_user, mock_profile):
        """Test getting advanced job matches with filtering."""
        mock_job_matches = [
            {
                "job_id": "job-1",
                "job_title": "Python Developer",
                "company": "Tech Corp",
                "match_score": 0.85,
                "skill_gaps": {"machine_learning": 0.7},
                "overall_readiness": 0.8
            }
        ]
        
        with patch('app.api.dependencies.get_current_user', return_value=mock_user), \
             patch('app.repositories.profile.ProfileRepository.get_by_user_id', return_value=mock_profile), \
             patch('app.services.recommendation_service.RecommendationService.get_advanced_job_matches', return_value=mock_job_matches):
            
            async with AsyncClient(app=app, base_url="http://test") as client:
                response = await client.get(
                    "/api/v1/comprehensive/job-matching/advanced",
                    params={
                        "location": "San Francisco",
                        "match_threshold": 0.7,
                        "include_skill_gaps": True,
                        "page": 1,
                        "page_size": 20,
                        "sort_by": "match_score",
                        "sort_order": "desc"
                    }
                )
            
            assert response.status_code == 200
            data = response.json()
            assert "items" in data
            assert "total_count" in data
            assert len(data["items"]) == 1
            assert data["items"][0]["job_title"] == "Python Developer"


class TestMarketAnalysis:
    """Test market analysis endpoints."""
    
    @pytest.mark.asyncio
    async def test_get_comprehensive_market_analysis(self, mock_user, mock_profile):
        """Test getting comprehensive market analysis."""
        mock_analysis = {
            "analysis_date": datetime.utcnow().isoformat(),
            "market_overview": {"total_job_postings": 1000},
            "skill_trends": [
                {"skill": "python", "trend": "growing", "job_count": 150}
            ],
            "emerging_skills": [
                {"skill_name": "rust", "growth_rate": 0.3}
            ]
        }
        
        with patch('app.api.dependencies.get_current_user', return_value=mock_user), \
             patch('app.repositories.profile.ProfileRepository.get_by_user_id', return_value=mock_profile), \
             patch('app.services.market_trend_analyzer.MarketTrendAnalyzer.get_comprehensive_market_analysis', return_value=mock_analysis):
            
            async with AsyncClient(app=app, base_url="http://test") as client:
                response = await client.get(
                    "/api/v1/comprehensive/market-analysis/comprehensive",
                    params={
                        "skills": ["python", "javascript"],
                        "time_period_days": 90,
                        "include_predictions": True,
                        "include_comparisons": True
                    }
                )
            
            assert response.status_code == 200
            data = response.json()
            assert "market_overview" in data
            assert "skill_trends" in data
            assert "emerging_skills" in data


class TestDataExport:
    """Test data export endpoints."""
    
    @pytest.mark.asyncio
    async def test_export_recommendations_json(self, mock_user):
        """Test exporting recommendations in JSON format."""
        mock_recommendations = [
            {"job_title": "Developer", "match_score": 0.9}
        ]
        
        with patch('app.api.dependencies.get_current_user', return_value=mock_user), \
             patch('app.services.recommendation_service.RecommendationService.get_career_recommendations', return_value=mock_recommendations):
            
            async with AsyncClient(app=app, base_url="http://test") as client:
                response = await client.get(
                    "/api/v1/comprehensive/export/recommendations/json",
                    params={"recommendation_type": "career"}
                )
            
            assert response.status_code == 200
            data = response.json()
            assert "data" in data
            assert "exported_at" in data
            assert len(data["data"]) == 1
    
    @pytest.mark.asyncio
    async def test_export_recommendations_pdf_not_implemented(self, mock_user):
        """Test that PDF export returns not implemented."""
        with patch('app.api.dependencies.get_current_user', return_value=mock_user):
            
            async with AsyncClient(app=app, base_url="http://test") as client:
                response = await client.get(
                    "/api/v1/comprehensive/export/recommendations/pdf",
                    params={"recommendation_type": "career"}
                )
            
            assert response.status_code == 501
            assert "not yet implemented" in response.json()["detail"]


class TestDashboard:
    """Test dashboard endpoints."""
    
    @pytest.mark.asyncio
    async def test_get_dashboard_summary(self, mock_user, mock_profile):
        """Test getting dashboard summary."""
        with patch('app.api.dependencies.get_current_user', return_value=mock_user), \
             patch('app.repositories.profile.ProfileRepository.get_by_user_id', return_value=mock_profile), \
             patch('app.services.profile_service.UserProfileService.calculate_profile_completeness', return_value=0.85), \
             patch('app.services.recommendation_service.RecommendationService.get_career_recommendations', return_value=[]), \
             patch('app.services.career_trajectory_service.CareerTrajectoryService.analyze_skill_gaps', return_value={}):
            
            async with AsyncClient(app=app, base_url="http://test") as client:
                response = await client.get(
                    "/api/v1/comprehensive/dashboard/summary",
                    params={"time_period_days": 30}
                )
            
            assert response.status_code == 200
            data = response.json()
            assert "profile_completeness" in data
            assert "recent_recommendations" in data
            assert "summary_stats" in data
            assert data["summary_stats"]["total_skills"] == 5


class TestBatchOperations:
    """Test batch operation endpoints."""
    
    @pytest.mark.asyncio
    async def test_analyze_multiple_roles(self, mock_user):
        """Test analyzing multiple roles simultaneously."""
        mock_analysis = {
            "role": "Software Engineer",
            "skill_gaps": {"machine_learning": 0.6},
            "learning_paths": [{"skill": "python", "duration_weeks": 4}],
            "market_data": {"demand_level": "high"}
        }
        
        with patch('app.api.dependencies.get_current_user', return_value=mock_user), \
             patch('app.services.career_trajectory_service.CareerTrajectoryService.analyze_skill_gaps', return_value=mock_analysis["skill_gaps"]), \
             patch('app.services.learning_path_service.LearningPathService.generate_learning_paths_for_role', return_value=mock_analysis["learning_paths"]), \
             patch('app.services.market_trend_analyzer.MarketTrendAnalyzer.get_role_market_data', return_value=mock_analysis["market_data"]):
            
            async with AsyncClient(app=app, base_url="http://test") as client:
                response = await client.post(
                    "/api/v1/comprehensive/batch/analyze-multiple-roles",
                    params={
                        "target_roles": ["Software Engineer", "Data Scientist"],
                        "include_learning_paths": True,
                        "include_market_data": True
                    }
                )
            
            assert response.status_code == 200
            data = response.json()
            assert "role_analyses" in data
            assert "comparison_summary" in data
            assert len(data["role_analyses"]) == 2


class TestHealthCheck:
    """Test health check endpoints."""
    
    @pytest.mark.asyncio
    async def test_comprehensive_health_check(self):
        """Test comprehensive health check."""
        with patch('app.services.recommendation_service.RecommendationService.model_trained', True):
            
            async with AsyncClient(app=app, base_url="http://test") as client:
                response = await client.get("/api/v1/comprehensive/health/comprehensive")
            
            assert response.status_code == 200
            data = response.json()
            assert "status" in data
            assert "services" in data
            assert "timestamp" in data
            assert data["status"] in ["healthy", "degraded", "unhealthy"]


class TestErrorHandling:
    """Test error handling in comprehensive API."""
    
    @pytest.mark.asyncio
    async def test_invalid_pagination_parameters(self, mock_user):
        """Test handling of invalid pagination parameters."""
        with patch('app.api.dependencies.get_current_user', return_value=mock_user):
            
            async with AsyncClient(app=app, base_url="http://test") as client:
                response = await client.post(
                    "/api/v1/comprehensive/recommendations/career/advanced",
                    json={
                        "target_role": "Developer",
                        "n_recommendations": 5
                    },
                    params={"page": 0, "page_size": 0}  # Invalid pagination
                )
            
            assert response.status_code == 422  # Validation error
    
    @pytest.mark.asyncio
    async def test_service_exception_handling(self, mock_user):
        """Test handling of service exceptions."""
        with patch('app.api.dependencies.get_current_user', return_value=mock_user), \
             patch('app.services.recommendation_service.RecommendationService.get_filtered_career_recommendations', side_effect=Exception("Service error")):
            
            async with AsyncClient(app=app, base_url="http://test") as client:
                response = await client.post(
                    "/api/v1/comprehensive/recommendations/career/advanced",
                    json={
                        "target_role": "Developer",
                        "n_recommendations": 5
                    }
                )
            
            assert response.status_code == 500
            assert "Failed to get career recommendations" in response.json()["detail"]


class TestPaginationAndFiltering:
    """Test pagination and filtering functionality."""
    
    @pytest.mark.asyncio
    async def test_pagination_response_structure(self, mock_user):
        """Test that pagination response has correct structure."""
        mock_recommendations = [
            {"job_title": f"Developer {i}", "match_score": 0.9 - i * 0.1}
            for i in range(25)  # 25 items to test pagination
        ]
        
        with patch('app.api.dependencies.get_current_user', return_value=mock_user), \
             patch('app.services.recommendation_service.RecommendationService.get_filtered_career_recommendations', return_value=mock_recommendations):
            
            async with AsyncClient(app=app, base_url="http://test") as client:
                response = await client.post(
                    "/api/v1/comprehensive/recommendations/career/advanced",
                    json={"target_role": "Developer", "n_recommendations": 25},
                    params={"page": 1, "page_size": 10}
                )
            
            assert response.status_code == 200
            data = response.json()
            
            # Check pagination structure
            assert "items" in data
            assert "total_count" in data
            assert "page" in data
            assert "page_size" in data
            assert "total_pages" in data
            assert "has_next" in data
            assert "has_previous" in data
            
            # Check pagination values
            assert data["total_count"] == 25
            assert data["page"] == 1
            assert data["page_size"] == 10
            assert data["total_pages"] == 3
            assert data["has_next"] is True
            assert data["has_previous"] is False
            assert len(data["items"]) == 10