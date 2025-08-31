"""
Comprehensive unit tests for all service methods.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timedelta

from app.services.profile_service import ProfileService
from app.services.recommendation_service import RecommendationService
from app.services.career_trajectory_service import CareerTrajectoryService
from app.services.learning_path_service import LearningPathService
from app.services.analytics_service import AnalyticsService
from app.services.auth_service import AuthService
from app.core.exceptions import ValidationError, NotFoundError, ExternalAPIError


@pytest.mark.unit
class TestProfileService:
    """Unit tests for ProfileService methods."""
    
    @pytest.fixture
    def profile_service(self, async_session, mock_nlp_engine):
        return ProfileService(db=async_session, nlp_engine=mock_nlp_engine)
    
    async def test_create_profile_success(self, profile_service, sample_profile_data, test_user):
        """Test successful profile creation."""
        with patch.object(profile_service, '_extract_resume_skills') as mock_extract:
            mock_extract.return_value = ["Python", "Machine Learning"]
            
            profile = await profile_service.create_profile(
                user_id=test_user.id,
                profile_data=sample_profile_data
            )
            
            assert profile.user_id == test_user.id
            assert profile.dream_job == sample_profile_data["dream_job"]
            assert profile.experience_years == sample_profile_data["experience_years"]
    
    async def test_create_profile_invalid_user(self, profile_service, sample_profile_data):
        """Test profile creation with invalid user ID."""
        with pytest.raises(NotFoundError):
            await profile_service.create_profile(
                user_id="invalid-id",
                profile_data=sample_profile_data
            )
    
    async def test_extract_resume_skills(self, profile_service, sample_resume_file):
        """Test resume skill extraction."""
        with patch('builtins.open', create=True) as mock_open:
            mock_open.return_value.__enter__.return_value.read.return_value = "Python developer with ML experience"
            
            skills = await profile_service._extract_resume_skills(sample_resume_file)
            
            assert isinstance(skills, list)
            assert len(skills) > 0
    
    async def test_merge_skill_profiles(self, profile_service):
        """Test skill profile merging logic."""
        skill_profiles = [
            {"Python": 0.9, "JavaScript": 0.7},
            {"Python": 0.8, "React": 0.6},
            {"Machine Learning": 0.85}
        ]
        
        merged = await profile_service._merge_skill_profiles(skill_profiles)
        
        assert "Python" in merged
        assert merged["Python"] > 0.8  # Should take highest confidence
        assert "JavaScript" in merged
        assert "React" in merged
        assert "Machine Learning" in merged
    
    async def test_update_profile_external_data(self, profile_service, test_profile):
        """Test updating profile with external platform data."""
        with patch.object(profile_service, '_fetch_github_data') as mock_github:
            mock_github.return_value = {"languages": {"Python": 80, "JavaScript": 20}}
            
            updated_profile = await profile_service.update_external_data(test_profile.id)
            
            assert updated_profile is not None
            mock_github.assert_called_once()


@pytest.mark.unit
class TestRecommendationService:
    """Unit tests for RecommendationService methods."""
    
    @pytest.fixture
    def recommendation_service(self, async_session, mock_recommendation_engine):
        return RecommendationService(db=async_session, engine=mock_recommendation_engine)
    
    async def test_get_career_recommendations(self, recommendation_service, test_profile):
        """Test career recommendation generation."""
        recommendations = await recommendation_service.get_career_recommendations(
            user_id=test_profile.user_id,
            limit=5
        )
        
        assert len(recommendations) <= 5
        assert all(rec.match_score >= 0 for rec in recommendations)
        assert all(rec.match_score <= 1 for rec in recommendations)
    
    async def test_get_recommendations_invalid_user(self, recommendation_service):
        """Test recommendations for invalid user."""
        with pytest.raises(NotFoundError):
            await recommendation_service.get_career_recommendations(
                user_id="invalid-id",
                limit=5
            )
    
    async def test_calculate_skill_gaps(self, recommendation_service, test_profile):
        """Test skill gap calculation."""
        target_skills = {"Python": 0.9, "TensorFlow": 0.8, "Docker": 0.7}
        
        gaps = await recommendation_service.calculate_skill_gaps(
            user_id=test_profile.user_id,
            target_skills=target_skills
        )
        
        assert isinstance(gaps, dict)
        assert all(0 <= gap <= 1 for gap in gaps.values())
    
    async def test_filter_recommendations_by_criteria(self, recommendation_service):
        """Test recommendation filtering."""
        mock_recommendations = [
            MagicMock(match_score=0.9, salary_min=100000),
            MagicMock(match_score=0.7, salary_min=80000),
            MagicMock(match_score=0.8, salary_min=120000)
        ]
        
        filtered = await recommendation_service._filter_by_criteria(
            recommendations=mock_recommendations,
            min_match_score=0.75,
            min_salary=90000
        )
        
        assert len(filtered) == 2  # Should filter out the 0.7 score and 80k salary


@pytest.mark.unit
class TestCareerTrajectoryService:
    """Unit tests for CareerTrajectoryService methods."""
    
    @pytest.fixture
    def trajectory_service(self, async_session):
        return CareerTrajectoryService(db=async_session)
    
    async def test_generate_career_paths(self, trajectory_service, test_profile):
        """Test career path generation."""
        with patch.object(trajectory_service, '_analyze_market_trends') as mock_trends:
            mock_trends.return_value = {"demand": "high", "growth": 0.15}
            
            paths = await trajectory_service.generate_career_paths(
                user_id=test_profile.user_id,
                target_role="Senior ML Engineer"
            )
            
            assert isinstance(paths, list)
            assert len(paths) > 0
    
    async def test_calculate_trajectory_feasibility(self, trajectory_service):
        """Test trajectory feasibility calculation."""
        current_skills = {"Python": 0.9, "Machine Learning": 0.7}
        target_skills = {"Python": 0.95, "Deep Learning": 0.8, "MLOps": 0.7}
        
        feasibility = await trajectory_service._calculate_feasibility(
            current_skills=current_skills,
            target_skills=target_skills,
            timeline_months=12
        )
        
        assert 0 <= feasibility <= 1
    
    async def test_identify_skill_progression_path(self, trajectory_service):
        """Test skill progression path identification."""
        current_level = "junior"
        target_level = "senior"
        domain = "machine_learning"
        
        progression = await trajectory_service._identify_progression_path(
            current_level=current_level,
            target_level=target_level,
            domain=domain
        )
        
        assert isinstance(progression, list)
        assert len(progression) > 0


@pytest.mark.unit
class TestLearningPathService:
    """Unit tests for LearningPathService methods."""
    
    @pytest.fixture
    def learning_service(self, async_session):
        return LearningPathService(db=async_session)
    
    async def test_generate_learning_path(self, learning_service, test_profile):
        """Test learning path generation."""
        skill_gaps = {"TensorFlow": 0.8, "Docker": 0.6, "Kubernetes": 0.7}
        
        with patch.object(learning_service, '_fetch_learning_resources') as mock_resources:
            mock_resources.return_value = [
                {"title": "TensorFlow Course", "duration": 40, "rating": 4.5},
                {"title": "Docker Basics", "duration": 20, "rating": 4.2}
            ]
            
            path = await learning_service.generate_learning_path(
                user_id=test_profile.user_id,
                skill_gaps=skill_gaps,
                timeline_weeks=16
            )
            
            assert path is not None
            assert path.estimated_duration_weeks <= 16
    
    async def test_prioritize_skills_by_impact(self, learning_service):
        """Test skill prioritization logic."""
        skill_gaps = {"TensorFlow": 0.8, "Docker": 0.3, "Kubernetes": 0.6}
        market_demand = {"TensorFlow": 0.9, "Docker": 0.7, "Kubernetes": 0.8}
        
        prioritized = await learning_service._prioritize_skills(
            skill_gaps=skill_gaps,
            market_demand=market_demand,
            user_goals=["machine_learning"]
        )
        
        assert prioritized[0]["skill"] == "TensorFlow"  # Highest impact
        assert len(prioritized) == 3
    
    async def test_estimate_learning_duration(self, learning_service):
        """Test learning duration estimation."""
        resources = [
            {"duration_hours": 40, "difficulty": "intermediate"},
            {"duration_hours": 20, "difficulty": "beginner"},
            {"duration_hours": 60, "difficulty": "advanced"}
        ]
        
        duration = await learning_service._estimate_duration(
            resources=resources,
            user_experience_level="intermediate",
            weekly_hours=10
        )
        
        assert duration > 0
        assert isinstance(duration, int)


@pytest.mark.unit
class TestAnalyticsService:
    """Unit tests for AnalyticsService methods."""
    
    @pytest.fixture
    def analytics_service(self, async_session):
        return AnalyticsService(db=async_session)
    
    async def test_generate_skill_radar_data(self, analytics_service, test_profile):
        """Test skill radar chart data generation."""
        with patch.object(analytics_service, '_get_user_skills') as mock_skills:
            mock_skills.return_value = {
                "Python": 0.9,
                "Machine Learning": 0.7,
                "JavaScript": 0.6,
                "React": 0.5
            }
            
            radar_data = await analytics_service.generate_skill_radar(
                user_id=test_profile.user_id
            )
            
            assert "categories" in radar_data
            assert "values" in radar_data
            assert len(radar_data["categories"]) == len(radar_data["values"])
    
    async def test_calculate_career_compatibility(self, analytics_service, test_profile):
        """Test career compatibility scoring."""
        target_job = {
            "required_skills": {"Python": 0.8, "Machine Learning": 0.9, "SQL": 0.6},
            "experience_level": "mid"
        }
        
        compatibility = await analytics_service.calculate_compatibility(
            user_id=test_profile.user_id,
            target_job=target_job
        )
        
        assert 0 <= compatibility.overall_score <= 1
        assert "skill_match" in compatibility.breakdown
        assert "experience_match" in compatibility.breakdown
    
    async def test_track_progress_over_time(self, analytics_service, test_profile):
        """Test progress tracking functionality."""
        # Mock historical data
        with patch.object(analytics_service, '_get_historical_skills') as mock_history:
            mock_history.return_value = [
                {"date": datetime.now() - timedelta(days=30), "skills": {"Python": 0.7}},
                {"date": datetime.now() - timedelta(days=15), "skills": {"Python": 0.8}},
                {"date": datetime.now(), "skills": {"Python": 0.9}}
            ]
            
            progress = await analytics_service.track_progress(
                user_id=test_profile.user_id,
                skill="Python",
                days=30
            )
            
            assert progress.improvement > 0
            assert len(progress.timeline) == 3


@pytest.mark.unit
class TestAuthService:
    """Unit tests for AuthService methods."""
    
    @pytest.fixture
    def auth_service(self, async_session):
        return AuthService(db=async_session)
    
    async def test_create_user_success(self, auth_service, sample_user_data):
        """Test successful user creation."""
        user = await auth_service.create_user(sample_user_data)
        
        assert user.email == sample_user_data["email"]
        assert user.full_name == sample_user_data["full_name"]
        assert user.is_active is True
        assert user.hashed_password != sample_user_data["password"]  # Should be hashed
    
    async def test_create_user_duplicate_email(self, auth_service, sample_user_data, test_user):
        """Test user creation with duplicate email."""
        sample_user_data["email"] = test_user.email
        
        with pytest.raises(ValidationError):
            await auth_service.create_user(sample_user_data)
    
    async def test_authenticate_user_success(self, auth_service, test_user):
        """Test successful user authentication."""
        user = await auth_service.authenticate_user(
            email=test_user.email,
            password="secret"  # From fixture
        )
        
        assert user is not None
        assert user.id == test_user.id
    
    async def test_authenticate_user_wrong_password(self, auth_service, test_user):
        """Test authentication with wrong password."""
        user = await auth_service.authenticate_user(
            email=test_user.email,
            password="wrongpassword"
        )
        
        assert user is None
    
    async def test_generate_access_token(self, auth_service, test_user):
        """Test access token generation."""
        token = await auth_service.create_access_token(
            data={"sub": str(test_user.id)}
        )
        
        assert isinstance(token, str)
        assert len(token) > 0
    
    async def test_verify_token_valid(self, auth_service, test_user):
        """Test token verification with valid token."""
        token = await auth_service.create_access_token(
            data={"sub": str(test_user.id)}
        )
        
        payload = await auth_service.verify_token(token)
        
        assert payload is not None
        assert payload["sub"] == str(test_user.id)
    
    async def test_verify_token_invalid(self, auth_service):
        """Test token verification with invalid token."""
        with pytest.raises(ValidationError):
            await auth_service.verify_token("invalid.token.here")


@pytest.mark.unit
class TestServiceErrorHandling:
    """Test error handling across all services."""
    
    async def test_external_api_timeout_handling(self, async_session):
        """Test handling of external API timeouts."""
        service = ProfileService(db=async_session)
        
        with patch('httpx.AsyncClient.get') as mock_get:
            mock_get.side_effect = asyncio.TimeoutError()
            
            with pytest.raises(ExternalAPIError):
                await service._fetch_github_data("testuser")
    
    async def test_database_connection_error_handling(self, async_session):
        """Test handling of database connection errors."""
        service = ProfileService(db=async_session)
        
        with patch.object(async_session, 'execute') as mock_execute:
            mock_execute.side_effect = Exception("Database connection failed")
            
            with pytest.raises(Exception):
                await service.get_profile_by_user_id("test-id")
    
    async def test_validation_error_propagation(self, async_session):
        """Test that validation errors are properly propagated."""
        service = ProfileService(db=async_session)
        
        invalid_data = {
            "skills": [],  # Empty skills should be invalid
            "dream_job": "",  # Empty dream job should be invalid
            "experience_years": -1  # Negative experience should be invalid
        }
        
        with pytest.raises(ValidationError):
            await service.create_profile(
                user_id="test-id",
                profile_data=invalid_data
            )