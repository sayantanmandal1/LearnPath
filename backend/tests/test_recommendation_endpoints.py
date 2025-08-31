"""
Tests for recommendation API endpoints.
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from fastapi.testclient import TestClient
from datetime import datetime

from app.main import app
from app.models.user import User
from app.core.exceptions import ServiceException


client = TestClient(app)


class TestRecommendationEndpoints:
    """Test recommendation API endpoints."""
    
    @pytest.fixture
    def mock_current_user(self):
        """Mock current user."""
        user = User()
        user.id = "user123"
        user.email = "test@example.com"
        return user
    
    @pytest.fixture
    def mock_recommendation_service(self):
        """Mock recommendation service."""
        return AsyncMock()
    
    def test_get_career_recommendations_success(self, mock_current_user):
        """Test successful career recommendations."""
        mock_recommendations = [
            {
                'job_title': 'Senior Python Developer',
                'company': 'Tech Corp',
                'match_score': 0.85,
                'match_percentage': 85.0,
                'required_skills': ['python', 'django'],
                'skill_gaps': {'kubernetes': 0.3},
                'salary_range': {'min': 80000, 'max': 120000, 'currency': 'USD'},
                'growth_potential': 0.8,
                'market_demand': 'high',
                'confidence_score': 0.85,
                'reasoning': 'Strong match based on Python skills',
                'alternative_paths': ['DevOps Engineer'],
                'location': 'San Francisco',
                'employment_type': 'Full-time',
                'recommendation_date': datetime.utcnow().isoformat()
            }
        ]
        
        with patch('app.api.dependencies.get_current_user', return_value=mock_current_user):
            with patch('app.api.v1.endpoints.recommendations.recommendation_service') as mock_service:
                mock_service.get_career_recommendations.return_value = mock_recommendations
                
                response = client.post(
                    "/api/v1/recommendations/career",
                    json={"n_recommendations": 5, "include_explanations": True}
                )
                
                assert response.status_code == 200
                data = response.json()
                assert len(data) == 1
                assert data[0]['job_title'] == 'Senior Python Developer'
                assert data[0]['match_percentage'] == 85.0
    
    def test_get_career_recommendations_service_error(self, mock_current_user):
        """Test career recommendations with service error."""
        with patch('app.api.dependencies.get_current_user', return_value=mock_current_user):
            with patch('app.api.v1.endpoints.recommendations.recommendation_service') as mock_service:
                mock_service.get_career_recommendations.side_effect = ServiceException("Model not trained")
                
                response = client.post(
                    "/api/v1/recommendations/career",
                    json={"n_recommendations": 5}
                )
                
                assert response.status_code == 400
                assert "Model not trained" in response.json()['detail']
    
    def test_get_learning_path_recommendations_success(self, mock_current_user):
        """Test successful learning path recommendations."""
        mock_learning_paths = [
            {
                'path_id': 'path1',
                'title': 'Master Django Development',
                'target_skills': ['django'],
                'estimated_duration_weeks': 8,
                'difficulty_level': 'intermediate',
                'priority_score': 0.9,
                'reasoning': 'Django is critical for your target role',
                'resources': [
                    {
                        'title': 'Django Complete Course',
                        'type': 'course',
                        'provider': 'Coursera',
                        'rating': 4.5,
                        'duration_hours': 40
                    }
                ],
                'milestones': [
                    {
                        'title': 'Complete Django Basics',
                        'estimated_weeks': 2
                    }
                ],
                'created_date': datetime.utcnow().isoformat()
            }
        ]
        
        with patch('app.api.dependencies.get_current_user', return_value=mock_current_user):
            with patch('app.api.v1.endpoints.recommendations.recommendation_service') as mock_service:
                mock_service.get_learning_path_recommendations.return_value = mock_learning_paths
                
                response = client.post(
                    "/api/v1/recommendations/learning-paths",
                    json={"target_role": "Senior Python Developer", "n_recommendations": 3}
                )
                
                assert response.status_code == 200
                data = response.json()
                assert len(data) == 1
                assert data[0]['title'] == 'Master Django Development'
                assert data[0]['target_skills'] == ['django']
    
    def test_get_learning_path_recommendations_missing_target_role(self, mock_current_user):
        """Test learning path recommendations without target role."""
        with patch('app.api.dependencies.get_current_user', return_value=mock_current_user):
            response = client.post(
                "/api/v1/recommendations/learning-paths",
                json={"n_recommendations": 3}
            )
            
            assert response.status_code == 400
            assert "target_role is required" in response.json()['detail']
    
    def test_analyze_skill_gaps_success(self, mock_current_user):
        """Test successful skill gap analysis."""
        mock_analysis = {
            'target_role': 'Senior Python Developer',
            'missing_skills': {'django': 0.8, 'postgresql': 0.7},
            'weak_skills': {'javascript': 0.4},
            'strong_skills': ['python', 'git'],
            'overall_readiness': 0.6,
            'learning_time_estimate_weeks': 12,
            'priority_skills': ['django', 'postgresql'],
            'readiness_percentage': 60.0,
            'analysis_date': datetime.utcnow().isoformat()
        }
        
        with patch('app.api.dependencies.get_current_user', return_value=mock_current_user):
            with patch('app.api.v1.endpoints.recommendations.recommendation_service') as mock_service:
                mock_service.analyze_skill_gaps.return_value = mock_analysis
                
                response = client.get("/api/v1/recommendations/skill-gaps/Senior%20Python%20Developer")
                
                assert response.status_code == 200
                data = response.json()
                assert data['target_role'] == 'Senior Python Developer'
                assert data['readiness_percentage'] == 60.0
                assert 'django' in data['missing_skills']
                assert 'python' in data['strong_skills']
    
    def test_get_job_match_score_success(self, mock_current_user):
        """Test successful job match score calculation."""
        mock_match_analysis = {
            'job_id': 'job123',
            'job_title': 'Senior Python Developer',
            'company': 'Tech Corp',
            'match_score': 0.75,
            'match_percentage': 75.0,
            'skill_gaps': {'django': 0.3},
            'weak_skills': {'javascript': 0.2},
            'strong_skills': ['python', 'sql'],
            'overall_readiness': 0.7,
            'readiness_percentage': 70.0,
            'analysis_date': datetime.utcnow().isoformat()
        }
        
        with patch('app.api.dependencies.get_current_user', return_value=mock_current_user):
            with patch('app.api.v1.endpoints.recommendations.recommendation_service') as mock_service:
                mock_service.get_job_match_score.return_value = mock_match_analysis
                
                response = client.get("/api/v1/recommendations/job-match/job123")
                
                assert response.status_code == 200
                data = response.json()
                assert data['job_id'] == 'job123'
                assert data['match_percentage'] == 75.0
                assert data['company'] == 'Tech Corp'
    
    def test_initialize_recommendation_models_success(self, mock_current_user):
        """Test successful model initialization."""
        with patch('app.api.dependencies.get_current_user', return_value=mock_current_user):
            with patch('app.api.v1.endpoints.recommendations.recommendation_service') as mock_service:
                mock_service.initialize_and_train_models.return_value = None
                
                response = client.post("/api/v1/recommendations/initialize-models")
                
                assert response.status_code == 200
                data = response.json()
                assert "successfully" in data['message']
    
    def test_get_model_status_success(self, mock_current_user):
        """Test getting model status."""
        with patch('app.api.dependencies.get_current_user', return_value=mock_current_user):
            with patch('app.api.v1.endpoints.recommendations.recommendation_service') as mock_service:
                mock_service.model_trained = True
                mock_service.last_training_time = datetime.utcnow()
                mock_service.training_interval.total_seconds.return_value = 86400  # 24 hours
                mock_service.recommendation_engine.get_model_performance_metrics.return_value = {
                    'precision_at_5': 0.82,
                    'recall_at_5': 0.75
                }
                
                response = client.get("/api/v1/recommendations/model-status")
                
                assert response.status_code == 200
                data = response.json()
                assert data['model_trained'] is True
                assert 'last_training_time' in data
                assert 'performance_metrics' in data
    
    def test_get_similar_jobs_success(self, mock_current_user):
        """Test getting similar jobs."""
        with patch('app.api.dependencies.get_current_user', return_value=mock_current_user):
            response = client.get("/api/v1/recommendations/similar-jobs/job123?limit=3")
            
            assert response.status_code == 200
            data = response.json()
            assert 'similar_jobs' in data
            assert len(data['similar_jobs']) <= 3
            
            if data['similar_jobs']:
                job = data['similar_jobs'][0]
                assert 'job_id' in job
                assert 'similarity_score' in job
                assert 'matching_skills' in job
    
    def test_get_trending_skills_success(self, mock_current_user):
        """Test getting trending skills."""
        with patch('app.api.dependencies.get_current_user', return_value=mock_current_user):
            response = client.get("/api/v1/recommendations/trending-skills?limit=5")
            
            assert response.status_code == 200
            data = response.json()
            assert 'trending_skills' in data
            assert len(data['trending_skills']) <= 5
            
            if data['trending_skills']:
                skill = data['trending_skills'][0]
                assert 'skill_name' in skill
                assert 'trend_score' in skill
                assert 'category' in skill
    
    def test_get_trending_skills_with_category_filter(self, mock_current_user):
        """Test getting trending skills with category filter."""
        with patch('app.api.dependencies.get_current_user', return_value=mock_current_user):
            response = client.get("/api/v1/recommendations/trending-skills?category=programming_languages&limit=3")
            
            assert response.status_code == 200
            data = response.json()
            assert 'trending_skills' in data
            
            # All returned skills should match the category filter
            for skill in data['trending_skills']:
                assert skill['category'] == 'programming_languages'
    
    def test_get_recommendation_history_success(self, mock_current_user):
        """Test getting recommendation history."""
        with patch('app.api.dependencies.get_current_user', return_value=mock_current_user):
            response = client.get("/api/v1/recommendations/recommendation-history?limit=10")
            
            assert response.status_code == 200
            data = response.json()
            assert 'recommendation_history' in data
            assert len(data['recommendation_history']) <= 10
            
            if data['recommendation_history']:
                rec = data['recommendation_history'][0]
                assert 'id' in rec
                assert 'type' in rec
                assert 'generated_date' in rec
    
    def test_get_recommendation_history_with_type_filter(self, mock_current_user):
        """Test getting recommendation history with type filter."""
        with patch('app.api.dependencies.get_current_user', return_value=mock_current_user):
            response = client.get("/api/v1/recommendations/recommendation-history?recommendation_type=career_recommendation")
            
            assert response.status_code == 200
            data = response.json()
            assert 'recommendation_history' in data
            
            # All returned recommendations should match the type filter
            for rec in data['recommendation_history']:
                assert rec['type'] == 'career_recommendation'
    
    def test_unauthorized_access(self):
        """Test unauthorized access to endpoints."""
        # Test without authentication
        response = client.post("/api/v1/recommendations/career", json={"n_recommendations": 5})
        assert response.status_code == 401
        
        response = client.get("/api/v1/recommendations/skill-gaps/Developer")
        assert response.status_code == 401
        
        response = client.get("/api/v1/recommendations/model-status")
        assert response.status_code == 401
    
    def test_invalid_request_parameters(self, mock_current_user):
        """Test invalid request parameters."""
        with patch('app.api.dependencies.get_current_user', return_value=mock_current_user):
            # Test invalid n_recommendations (too high)
            response = client.post(
                "/api/v1/recommendations/career",
                json={"n_recommendations": 25}  # Max is 20
            )
            assert response.status_code == 422
            
            # Test invalid n_recommendations (too low)
            response = client.post(
                "/api/v1/recommendations/career",
                json={"n_recommendations": 0}  # Min is 1
            )
            assert response.status_code == 422
    
    def test_service_exception_handling(self, mock_current_user):
        """Test service exception handling in endpoints."""
        with patch('app.api.dependencies.get_current_user', return_value=mock_current_user):
            with patch('app.api.v1.endpoints.recommendations.recommendation_service') as mock_service:
                mock_service.get_career_recommendations.side_effect = ServiceException("Database error")
                
                response = client.post(
                    "/api/v1/recommendations/career",
                    json={"n_recommendations": 5}
                )
                
                assert response.status_code == 400
                assert "Database error" in response.json()['detail']
    
    def test_internal_server_error_handling(self, mock_current_user):
        """Test internal server error handling."""
        with patch('app.api.dependencies.get_current_user', return_value=mock_current_user):
            with patch('app.api.v1.endpoints.recommendations.recommendation_service') as mock_service:
                mock_service.get_career_recommendations.side_effect = Exception("Unexpected error")
                
                response = client.post(
                    "/api/v1/recommendations/career",
                    json={"n_recommendations": 5}
                )
                
                assert response.status_code == 500
                assert "Internal server error" in response.json()['detail']
    
    def test_request_validation(self, mock_current_user):
        """Test request validation."""
        with patch('app.api.dependencies.get_current_user', return_value=mock_current_user):
            # Test missing required fields
            response = client.post("/api/v1/recommendations/career", json={})
            assert response.status_code == 200  # n_recommendations has default value
            
            # Test invalid JSON
            response = client.post(
                "/api/v1/recommendations/career",
                data="invalid json",
                headers={"Content-Type": "application/json"}
            )
            assert response.status_code == 422
    
    def test_query_parameter_validation(self, mock_current_user):
        """Test query parameter validation."""
        with patch('app.api.dependencies.get_current_user', return_value=mock_current_user):
            # Test invalid limit parameter
            response = client.get("/api/v1/recommendations/trending-skills?limit=100")  # Max is 50
            assert response.status_code == 422
            
            response = client.get("/api/v1/recommendations/trending-skills?limit=0")  # Min is 1
            assert response.status_code == 422
    
    def test_response_model_validation(self, mock_current_user):
        """Test response model validation."""
        # Mock service to return invalid data
        invalid_recommendation = {
            'job_title': 'Test Job',
            # Missing required fields
        }
        
        with patch('app.api.dependencies.get_current_user', return_value=mock_current_user):
            with patch('app.api.v1.endpoints.recommendations.recommendation_service') as mock_service:
                mock_service.get_career_recommendations.return_value = [invalid_recommendation]
                
                response = client.post(
                    "/api/v1/recommendations/career",
                    json={"n_recommendations": 1}
                )
                
                # Should handle validation error gracefully
                assert response.status_code in [422, 500]


@pytest.mark.asyncio
async def test_endpoint_integration():
    """Integration test for recommendation endpoints."""
    from app.services.recommendation_service import RecommendationService
    
    # This would be a more comprehensive integration test
    # that tests the full flow from endpoint to service to ML engine
    service = RecommendationService()
    
    # Mock the ML components
    with patch.object(service, 'initialize_and_train_models'):
        service.model_trained = True
        
        # Test that the service can be called (basic integration)
        assert service.recommendation_engine is not None
        assert service.skill_gap_analyzer is not None
        assert service.explainer is not None


class TestRecommendationEndpointSecurity:
    """Test security aspects of recommendation endpoints."""
    
    def test_authentication_required(self):
        """Test that all endpoints require authentication."""
        endpoints = [
            ("POST", "/api/v1/recommendations/career"),
            ("POST", "/api/v1/recommendations/learning-paths"),
            ("GET", "/api/v1/recommendations/skill-gaps/Developer"),
            ("GET", "/api/v1/recommendations/job-match/job123"),
            ("POST", "/api/v1/recommendations/initialize-models"),
            ("GET", "/api/v1/recommendations/model-status"),
            ("GET", "/api/v1/recommendations/similar-jobs/job123"),
            ("GET", "/api/v1/recommendations/trending-skills"),
            ("GET", "/api/v1/recommendations/recommendation-history")
        ]
        
        for method, endpoint in endpoints:
            if method == "POST":
                response = client.post(endpoint, json={})
            else:
                response = client.get(endpoint)
            
            assert response.status_code == 401, f"Endpoint {method} {endpoint} should require authentication"
    
    def test_user_data_isolation(self, mock_current_user):
        """Test that users can only access their own data."""
        # This test would verify that user A cannot access user B's recommendations
        # In practice, this is handled by the get_current_user dependency
        # and the service layer using the authenticated user's ID
        
        with patch('app.api.dependencies.get_current_user', return_value=mock_current_user):
            with patch('app.api.v1.endpoints.recommendations.recommendation_service') as mock_service:
                mock_service.get_career_recommendations.return_value = []
                
                response = client.post(
                    "/api/v1/recommendations/career",
                    json={"n_recommendations": 5}
                )
                
                # Verify that the service was called with the authenticated user's ID
                mock_service.get_career_recommendations.assert_called_once()
                call_args = mock_service.get_career_recommendations.call_args
                assert call_args[1]['user_id'] == mock_current_user.id