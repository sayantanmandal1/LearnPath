"""
Tests for dashboard service
"""
import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

from app.services.dashboard_service import DashboardService
from app.schemas.dashboard import DashboardSummary, UserProgressSummary, PersonalizedContent
from app.core.exceptions import ServiceError


class TestDashboardService:
    """Test dashboard service functionality"""
    
    @pytest.fixture
    def mock_db(self):
        """Mock database session"""
        return AsyncMock()
    
    @pytest.fixture
    def mock_profile(self):
        """Mock user profile"""
        profile = MagicMock()
        profile.id = "profile_123"
        profile.user_id = "user_123"
        profile.current_role = "Software Developer"
        profile.experience_years = 5
        profile.education_level = "Bachelor's"
        profile.location = "San Francisco"
        profile.skills = ["Python", "JavaScript", "React"]
        profile.career_goals = "Become a Senior Developer"
        profile.preferred_work_type = "Remote"
        profile.updated_at = datetime.utcnow()
        return profile
    
    @pytest.fixture
    def dashboard_service(self, mock_db):
        """Create dashboard service with mocked dependencies"""
        service = DashboardService(mock_db)
        
        # Mock the dependent services
        service.analytics_service = AsyncMock()
        service.recommendation_service = AsyncMock()
        service.profile_service = AsyncMock()
        
        return service
    
    @pytest.mark.asyncio
    async def test_get_dashboard_summary_success(self, dashboard_service, mock_profile):
        """Test successful dashboard summary generation"""
        user_id = "user_123"
        
        # Mock service responses
        dashboard_service.profile_service.get_profile.return_value = mock_profile
        dashboard_service.analytics_service.calculate_comprehensive_user_analytics.return_value = {
            "overall_career_score": 75.5,
            "skill_analytics": {"total_skills": 10},
            "experience_analytics": {"experience_score": 8.2},
            "market_analytics": {"market_position_percentile": 65}
        }
        dashboard_service.analytics_service.generate_overall_career_score_and_recommendations.return_value = {
            "overall_career_score": 75.5,
            "comprehensive_recommendations": ["Learn Docker", "Improve leadership skills"],
            "priority_actions": ["Update LinkedIn", "Take Python course", "Apply to senior roles"]
        }
        
        # Call the method
        result = await dashboard_service.get_dashboard_summary(user_id)
        
        # Verify the result
        assert isinstance(result, DashboardSummary)
        assert result.user_id == user_id
        assert result.overall_career_score == 75.5
        assert result.profile_completion > 0  # Should calculate some completion percentage
        assert len(result.key_metrics) > 0
        assert len(result.active_milestones) >= 0
        assert len(result.top_recommendations) >= 0
        assert len(result.recent_activities) >= 0
        assert result.generated_at is not None
    
    @pytest.mark.asyncio
    async def test_get_dashboard_summary_no_profile(self, dashboard_service):
        """Test dashboard summary when profile doesn't exist"""
        user_id = "user_123"
        
        # Mock no profile found
        dashboard_service.profile_service.get_profile.return_value = None
        
        # Should raise ServiceError
        with pytest.raises(ServiceError, match="Profile not found"):
            await dashboard_service.get_dashboard_summary(user_id)
    
    @pytest.mark.asyncio
    async def test_get_user_progress_summary_success(self, dashboard_service, mock_profile):
        """Test successful user progress summary generation"""
        user_id = "user_123"
        tracking_days = 90
        
        # Mock service responses
        dashboard_service.analytics_service.track_historical_progress.return_value = {
            "overall_progress_percentage": 68.5,
            "new_skills_count": 3,
            "skills_mastered_count": 2
        }
        
        # Call the method
        result = await dashboard_service.get_user_progress_summary(user_id, tracking_days)
        
        # Verify the result
        assert isinstance(result, UserProgressSummary)
        assert result.user_id == user_id
        assert result.overall_progress == 68.5
        assert result.new_skills_added == 3
        assert result.skills_mastered == 2
        assert result.tracking_period_days == tracking_days
        assert len(result.career_score_trend) > 0
        assert result.generated_at is not None
    
    @pytest.mark.asyncio
    async def test_get_personalized_content_success(self, dashboard_service, mock_profile):
        """Test successful personalized content generation"""
        user_id = "user_123"
        
        # Mock service responses
        dashboard_service.profile_service.get_profile.return_value = mock_profile
        dashboard_service.recommendation_service.get_recommendations.return_value = {
            "recommendations": [
                {"title": "Learn Docker", "description": "Container technology", "confidence_score": 8.5}
            ]
        }
        
        # Call the method
        result = await dashboard_service.get_personalized_content(user_id)
        
        # Verify the result
        assert isinstance(result, PersonalizedContent)
        assert result.user_id == user_id
        assert len(result.featured_jobs) >= 0
        assert len(result.recommended_skills) >= 0
        assert len(result.suggested_learning_paths) >= 0
        assert len(result.market_trends) >= 0
        assert result.personalization_score > 0
        assert result.generated_at is not None
    
    @pytest.mark.asyncio
    async def test_calculate_profile_completion(self, dashboard_service, mock_profile):
        """Test profile completion calculation"""
        # Test with complete profile
        completion = await dashboard_service._calculate_profile_completion(mock_profile)
        assert completion > 0
        assert completion <= 100
        
        # Test with incomplete profile
        incomplete_profile = MagicMock()
        incomplete_profile.current_role = None
        incomplete_profile.experience_years = 0
        incomplete_profile.education_level = ""
        incomplete_profile.location = ""
        incomplete_profile.skills = []
        incomplete_profile.career_goals = ""
        incomplete_profile.preferred_work_type = ""
        
        completion = await dashboard_service._calculate_profile_completion(incomplete_profile)
        assert completion == 0.0
    
    @pytest.mark.asyncio
    async def test_generate_key_metrics(self, dashboard_service):
        """Test key metrics generation"""
        user_id = "user_123"
        analytics = {
            "overall_career_score": 75.5,
            "skill_analytics": {"total_skills": 12},
            "market_analytics": {"market_position_percentile": 68},
            "experience_analytics": {"experience_score": 8.2}
        }
        
        metrics = await dashboard_service._generate_key_metrics(user_id, analytics)
        
        assert len(metrics) == 4
        assert any(m.name == "Career Score" for m in metrics)
        assert any(m.name == "Skills" for m in metrics)
        assert any(m.name == "Market Position" for m in metrics)
        assert any(m.name == "Experience Score" for m in metrics)
    
    @pytest.mark.asyncio
    async def test_get_active_milestones(self, dashboard_service):
        """Test active milestones retrieval"""
        user_id = "user_123"
        
        milestones = await dashboard_service._get_active_milestones(user_id)
        
        assert len(milestones) >= 0
        for milestone in milestones:
            assert hasattr(milestone, 'id')
            assert hasattr(milestone, 'title')
            assert hasattr(milestone, 'progress_percentage')
            assert 0 <= milestone.progress_percentage <= 100
    
    @pytest.mark.asyncio
    async def test_get_top_recommendations_with_service_failure(self, dashboard_service):
        """Test recommendations when service fails"""
        user_id = "user_123"
        
        # Mock service failure
        dashboard_service.recommendation_service.get_recommendations.side_effect = Exception("Service unavailable")
        
        recommendations = await dashboard_service._get_top_recommendations(user_id)
        
        # Should return default recommendations instead of failing
        assert len(recommendations) > 0
        assert recommendations[0].id == "default_1"
        assert "Update Your Skills" in recommendations[0].title
    
    @pytest.mark.asyncio
    async def test_service_error_handling(self, dashboard_service):
        """Test service error handling"""
        user_id = "user_123"
        
        # Mock analytics service failure
        dashboard_service.profile_service.get_profile.side_effect = Exception("Database error")
        
        with pytest.raises(ServiceError, match="Failed to generate dashboard summary"):
            await dashboard_service.get_dashboard_summary(user_id)