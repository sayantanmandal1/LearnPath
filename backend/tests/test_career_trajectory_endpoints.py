"""
Tests for career trajectory API endpoints.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime
from fastapi.testclient import TestClient
from fastapi import status

from app.main import app
from app.services.career_trajectory_service import CareerTrajectoryRecommendation
from app.core.exceptions import ServiceException


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def mock_current_user():
    """Create mock current user."""
    user = Mock()
    user.id = "test-user-123"
    user.email = "test@example.com"
    return user


@pytest.fixture
def mock_trajectory():
    """Create mock career trajectory."""
    return CareerTrajectoryRecommendation(
        trajectory_id="traj-123",
        title="Path to Senior Software Engineer",
        target_role="Senior Software Engineer",
        match_score=0.85,
        confidence_score=0.78,
        progression_steps=[
            {
                'role': 'Software Engineer',
                'duration_months': 0,
                'description': 'Current position',
                'key_activities': ['Strengthen skills'],
                'skills_to_develop': [],
                'milestones': ['Complete projects']
            },
            {
                'role': 'Senior Software Engineer',
                'duration_months': 24,
                'description': 'Target role',
                'key_activities': ['Lead projects'],
                'skills_to_develop': ['system design'],
                'milestones': ['Promotion']
            }
        ],
        estimated_timeline_months=24,
        difficulty_level="moderate",
        required_skills=["python", "javascript", "system design"],
        skill_gaps={"system design": 0.8, "docker": 0.6},
        transferable_skills=["python", "javascript"],
        market_demand="high",
        salary_progression={
            "Software Engineer": (80000, 120000),
            "Senior Software Engineer": (120000, 160000)
        },
        growth_potential=0.8,
        alternative_routes=[],
        lateral_opportunities=["Tech Lead", "Product Manager"],
        reasoning="This trajectory aligns with your experience and market demand.",
        success_factors=["Master system design", "Build leadership skills"],
        potential_challenges=["Competition", "Skill development time"],
        recommendation_date=datetime.utcnow(),
        data_sources=["job_postings", "market_analysis"]
    )


@pytest.fixture
def mock_skill_gap_analysis():
    """Create mock skill gap analysis."""
    return {
        'target_role': 'Senior Software Engineer',
        'missing_skills': {'system design': 0.8, 'docker': 0.6},
        'weak_skills': {'kubernetes': 0.4},
        'strong_skills': ['python', 'javascript', 'sql'],
        'overall_readiness': 0.7,
        'learning_time_estimate_weeks': 16,
        'priority_skills': ['system design', 'docker'],
        'readiness_percentage': 70.0,
        'analysis_date': datetime.utcnow().isoformat()
    }


@pytest.fixture
def mock_job_match_analysis():
    """Create mock job match analysis."""
    return {
        'job_id': 'job-123',
        'job_title': 'Senior Software Engineer',
        'company': 'Tech Corp',
        'match_score': 0.82,
        'match_percentage': 82.0,
        'skill_gaps': {'system design': 0.7},
        'weak_skills': {'docker': 0.5},
        'strong_skills': ['python', 'javascript'],
        'overall_readiness': 0.75,
        'readiness_percentage': 75.0,
        'analysis_date': datetime.utcnow().isoformat()
    }


class TestCareerTrajectoryEndpoints:
    """Test career trajectory API endpoints."""
    
    def test_get_career_trajectories_success(
        self, client, mock_current_user, mock_trajectory
    ):
        """Test successful career trajectories retrieval."""
        with patch('app.api.dependencies.get_current_user', return_value=mock_current_user), \
             patch('app.api.v1.endpoints.career_trajectory.career_trajectory_service') as mock_service:
            
            mock_service.get_career_trajectory_recommendations = AsyncMock(
                return_value=[mock_trajectory]
            )
            
            response = client.get(
                "/api/v1/career-trajectory/trajectories",
                params={"n_recommendations": 3, "include_alternatives": True}
            )
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            
            assert "trajectories" in data
            assert "total_count" in data
            assert data["total_count"] == 1
            assert len(data["trajectories"]) == 1
            
            trajectory = data["trajectories"][0]
            assert trajectory["trajectory_id"] == "traj-123"
            assert trajectory["title"] == "Path to Senior Software Engineer"
            assert trajectory["target_role"] == "Senior Software Engineer"
            assert trajectory["match_score"] == 0.85
            assert trajectory["confidence_score"] == 0.78
    
    def test_get_career_trajectories_with_parameters(
        self, client, mock_current_user, mock_trajectory
    ):
        """Test career trajectories with custom parameters."""
        with patch('app.api.dependencies.get_current_user', return_value=mock_current_user), \
             patch('app.api.v1.endpoints.career_trajectory.career_trajectory_service') as mock_service:
            
            mock_service.get_career_trajectory_recommendations = AsyncMock(
                return_value=[mock_trajectory]
            )
            
            response = client.get(
                "/api/v1/career-trajectory/trajectories",
                params={"n_recommendations": 5, "include_alternatives": False}
            )
            
            assert response.status_code == status.HTTP_200_OK
            
            # Verify service was called with correct parameters
            mock_service.get_career_trajectory_recommendations.assert_called_once()
            call_args = mock_service.get_career_trajectory_recommendations.call_args
            assert call_args.kwargs["n_recommendations"] == 5
            assert call_args.kwargs["include_alternatives"] == False
    
    def test_get_career_trajectories_service_error(
        self, client, mock_current_user
    ):
        """Test career trajectories with service error."""
        with patch('app.api.dependencies.get_current_user', return_value=mock_current_user), \
             patch('app.api.v1.endpoints.career_trajectory.career_trajectory_service') as mock_service:
            
            mock_service.get_career_trajectory_recommendations = AsyncMock(
                side_effect=ServiceException("Profile not found")
            )
            
            response = client.get("/api/v1/career-trajectory/trajectories")
            
            assert response.status_code == status.HTTP_400_BAD_REQUEST
            assert "Profile not found" in response.json()["detail"]
    
    def test_get_career_trajectories_unexpected_error(
        self, client, mock_current_user
    ):
        """Test career trajectories with unexpected error."""
        with patch('app.api.dependencies.get_current_user', return_value=mock_current_user), \
             patch('app.api.v1.endpoints.career_trajectory.career_trajectory_service') as mock_service:
            
            mock_service.get_career_trajectory_recommendations = AsyncMock(
                side_effect=Exception("Unexpected error")
            )
            
            response = client.get("/api/v1/career-trajectory/trajectories")
            
            assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
            assert "Internal server error" in response.json()["detail"]
    
    def test_get_specific_career_trajectory_success(
        self, client, mock_current_user, mock_trajectory
    ):
        """Test successful specific career trajectory retrieval."""
        with patch('app.api.dependencies.get_current_user', return_value=mock_current_user), \
             patch('app.api.v1.endpoints.career_trajectory.career_trajectory_service') as mock_service:
            
            mock_service.get_career_trajectory_recommendations = AsyncMock(
                return_value=[mock_trajectory]
            )
            
            response = client.get("/api/v1/career-trajectory/trajectories/traj-123")
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            
            assert data["trajectory_id"] == "traj-123"
            assert data["title"] == "Path to Senior Software Engineer"
    
    def test_get_specific_career_trajectory_not_found(
        self, client, mock_current_user, mock_trajectory
    ):
        """Test specific career trajectory not found."""
        with patch('app.api.dependencies.get_current_user', return_value=mock_current_user), \
             patch('app.api.v1.endpoints.career_trajectory.career_trajectory_service') as mock_service:
            
            mock_service.get_career_trajectory_recommendations = AsyncMock(
                return_value=[mock_trajectory]  # Different ID
            )
            
            response = client.get("/api/v1/career-trajectory/trajectories/nonexistent-id")
            
            assert response.status_code == status.HTTP_404_NOT_FOUND
            assert "Career trajectory not found" in response.json()["detail"]
    
    def test_analyze_skill_gaps_success(
        self, client, mock_current_user, mock_skill_gap_analysis
    ):
        """Test successful skill gap analysis."""
        with patch('app.api.dependencies.get_current_user', return_value=mock_current_user), \
             patch('app.api.v1.endpoints.career_trajectory.career_trajectory_service') as mock_service:
            
            mock_service.analyze_skill_gaps = AsyncMock(
                return_value=mock_skill_gap_analysis
            )
            
            response = client.get(
                "/api/v1/career-trajectory/skill-gap-analysis",
                params={"target_role": "Senior Software Engineer"}
            )
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            
            assert data["target_role"] == "Senior Software Engineer"
            assert "missing_skills" in data
            assert "weak_skills" in data
            assert "strong_skills" in data
            assert data["overall_readiness"] == 0.7
            assert data["readiness_percentage"] == 70.0
    
    def test_analyze_skill_gaps_missing_parameter(
        self, client, mock_current_user
    ):
        """Test skill gap analysis with missing target_role parameter."""
        with patch('app.api.dependencies.get_current_user', return_value=mock_current_user):
            
            response = client.get("/api/v1/career-trajectory/skill-gap-analysis")
            
            assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_calculate_job_match_score_success(
        self, client, mock_current_user, mock_job_match_analysis
    ):
        """Test successful job match score calculation."""
        with patch('app.api.dependencies.get_current_user', return_value=mock_current_user), \
             patch('app.api.v1.endpoints.career_trajectory.career_trajectory_service') as mock_service:
            
            mock_service.get_job_match_score = AsyncMock(
                return_value=mock_job_match_analysis
            )
            
            response = client.get("/api/v1/career-trajectory/job-match/job-123")
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            
            assert data["job_id"] == "job-123"
            assert data["job_title"] == "Senior Software Engineer"
            assert data["company"] == "Tech Corp"
            assert data["match_score"] == 0.82
            assert data["match_percentage"] == 82.0
    
    def test_refresh_career_trajectories_success(
        self, client, mock_current_user, mock_trajectory
    ):
        """Test successful career trajectories refresh."""
        with patch('app.api.dependencies.get_current_user', return_value=mock_current_user), \
             patch('app.api.v1.endpoints.career_trajectory.career_trajectory_service') as mock_service:
            
            mock_service.get_career_trajectory_recommendations = AsyncMock(
                return_value=[mock_trajectory]
            )
            mock_service.market_demand_cache = {}
            
            response = client.post(
                "/api/v1/career-trajectory/trajectories/refresh",
                params={"n_recommendations": 3}
            )
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            
            assert "trajectories" in data
            assert data["total_count"] == 1
            
            # Verify cache was cleared
            assert len(mock_service.market_demand_cache) == 0
    
    def test_get_market_insights_success(
        self, client, mock_current_user
    ):
        """Test successful market insights retrieval."""
        mock_market_data = {
            'demand_level': 'high',
            'growth_potential': 0.8,
            'salary_trend': 'growing',
            'job_count': 45,
            'average_salary': 125000
        }
        
        with patch('app.api.dependencies.get_current_user', return_value=mock_current_user), \
             patch('app.api.v1.endpoints.career_trajectory.career_trajectory_service') as mock_service:
            
            mock_service._get_market_demand_data = AsyncMock(
                return_value=mock_market_data
            )
            mock_service.market_demand_cache = {}
            
            response = client.get("/api/v1/career-trajectory/market-insights/Software Engineer")
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            
            assert data["role"] == "Software Engineer"
            assert data["market_data"]["demand_level"] == "high"
            assert data["market_data"]["growth_potential"] == 0.8
    
    def test_parameter_validation(self, client, mock_current_user):
        """Test API parameter validation."""
        with patch('app.api.dependencies.get_current_user', return_value=mock_current_user):
            
            # Test invalid n_recommendations (too high)
            response = client.get(
                "/api/v1/career-trajectory/trajectories",
                params={"n_recommendations": 15}
            )
            assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
            
            # Test invalid n_recommendations (too low)
            response = client.get(
                "/api/v1/career-trajectory/trajectories",
                params={"n_recommendations": 0}
            )
            assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_authentication_required(self, client):
        """Test that authentication is required for all endpoints."""
        # Test without authentication
        response = client.get("/api/v1/career-trajectory/trajectories")
        assert response.status_code in [status.HTTP_401_UNAUTHORIZED, status.HTTP_403_FORBIDDEN]
        
        response = client.get("/api/v1/career-trajectory/skill-gap-analysis?target_role=Engineer")
        assert response.status_code in [status.HTTP_401_UNAUTHORIZED, status.HTTP_403_FORBIDDEN]
        
        response = client.get("/api/v1/career-trajectory/job-match/job-123")
        assert response.status_code in [status.HTTP_401_UNAUTHORIZED, status.HTTP_403_FORBIDDEN]


class TestCareerTrajectoryResponseSchemas:
    """Test career trajectory response schema validation."""
    
    def test_career_trajectory_response_from_domain_model(self, mock_trajectory):
        """Test conversion from domain model to response schema."""
        from app.schemas.career_trajectory import CareerTrajectoryResponse
        
        response = CareerTrajectoryResponse.from_domain_model(mock_trajectory)
        
        assert response.trajectory_id == mock_trajectory.trajectory_id
        assert response.title == mock_trajectory.title
        assert response.target_role == mock_trajectory.target_role
        assert response.match_score == mock_trajectory.match_score
        assert response.confidence_score == mock_trajectory.confidence_score
        
        # Test progression steps conversion
        assert len(response.progression_steps) == len(mock_trajectory.progression_steps)
        for i, step in enumerate(response.progression_steps):
            original_step = mock_trajectory.progression_steps[i]
            assert step.role == original_step['role']
            assert step.duration_months == original_step['duration_months']
            assert step.description == original_step['description']
        
        # Test salary progression conversion
        assert len(response.salary_progression) == len(mock_trajectory.salary_progression)
        for role, salary_range in response.salary_progression.items():
            original_range = mock_trajectory.salary_progression[role]
            assert salary_range.min_salary == original_range[0]
            assert salary_range.max_salary == original_range[1]
            assert salary_range.currency == "USD"
    
    def test_skill_gap_analysis_response_validation(self, mock_skill_gap_analysis):
        """Test skill gap analysis response validation."""
        from app.schemas.career_trajectory import SkillGapAnalysisResponse
        
        response = SkillGapAnalysisResponse(**mock_skill_gap_analysis)
        
        assert response.target_role == mock_skill_gap_analysis['target_role']
        assert response.overall_readiness == mock_skill_gap_analysis['overall_readiness']
        assert response.readiness_percentage == mock_skill_gap_analysis['readiness_percentage']
        assert isinstance(response.missing_skills, dict)
        assert isinstance(response.strong_skills, list)
    
    def test_job_match_score_response_validation(self, mock_job_match_analysis):
        """Test job match score response validation."""
        from app.schemas.career_trajectory import JobMatchScoreResponse
        
        response = JobMatchScoreResponse(**mock_job_match_analysis)
        
        assert response.job_id == mock_job_match_analysis['job_id']
        assert response.job_title == mock_job_match_analysis['job_title']
        assert response.company == mock_job_match_analysis['company']
        assert response.match_score == mock_job_match_analysis['match_score']
        assert response.match_percentage == mock_job_match_analysis['match_percentage']


@pytest.mark.integration
class TestCareerTrajectoryEndpointsIntegration:
    """Integration tests for career trajectory endpoints."""
    
    def test_full_trajectory_workflow(
        self, client, mock_current_user, mock_trajectory, mock_skill_gap_analysis
    ):
        """Test complete trajectory workflow from API perspective."""
        with patch('app.api.dependencies.get_current_user', return_value=mock_current_user), \
             patch('app.api.v1.endpoints.career_trajectory.career_trajectory_service') as mock_service:
            
            # Setup service mocks
            mock_service.get_career_trajectory_recommendations = AsyncMock(
                return_value=[mock_trajectory]
            )
            mock_service.analyze_skill_gaps = AsyncMock(
                return_value=mock_skill_gap_analysis
            )
            mock_service.market_demand_cache = {}
            
            # 1. Get career trajectories
            response = client.get("/api/v1/career-trajectory/trajectories")
            assert response.status_code == status.HTTP_200_OK
            trajectories_data = response.json()
            
            # 2. Get specific trajectory
            trajectory_id = trajectories_data["trajectories"][0]["trajectory_id"]
            response = client.get(f"/api/v1/career-trajectory/trajectories/{trajectory_id}")
            assert response.status_code == status.HTTP_200_OK
            
            # 3. Analyze skill gaps for target role
            target_role = trajectories_data["trajectories"][0]["target_role"]
            response = client.get(
                "/api/v1/career-trajectory/skill-gap-analysis",
                params={"target_role": target_role}
            )
            assert response.status_code == status.HTTP_200_OK
            
            # 4. Refresh trajectories
            response = client.post("/api/v1/career-trajectory/trajectories/refresh")
            assert response.status_code == status.HTTP_200_OK
            
            # Verify all service methods were called
            assert mock_service.get_career_trajectory_recommendations.call_count >= 2
            mock_service.analyze_skill_gaps.assert_called_once()
    
    def test_error_propagation_and_handling(
        self, client, mock_current_user
    ):
        """Test error propagation from service to API layer."""
        with patch('app.api.dependencies.get_current_user', return_value=mock_current_user), \
             patch('app.api.v1.endpoints.career_trajectory.career_trajectory_service') as mock_service:
            
            # Test ServiceException propagation
            mock_service.get_career_trajectory_recommendations = AsyncMock(
                side_effect=ServiceException("User profile incomplete")
            )
            
            response = client.get("/api/v1/career-trajectory/trajectories")
            assert response.status_code == status.HTTP_400_BAD_REQUEST
            assert "User profile incomplete" in response.json()["detail"]
            
            # Test generic exception handling
            mock_service.analyze_skill_gaps = AsyncMock(
                side_effect=Exception("Database connection failed")
            )
            
            response = client.get(
                "/api/v1/career-trajectory/skill-gap-analysis",
                params={"target_role": "Engineer"}
            )
            assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
            assert "Internal server error" in response.json()["detail"]