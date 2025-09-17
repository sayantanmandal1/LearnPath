"""
Tests for comprehensive analytics backend functionality
"""
import pytest
from datetime import datetime
from unittest.mock import AsyncMock, patch
from sqlalchemy.ext.asyncio import AsyncSession

from app.services.analytics_service import AnalyticsService
from app.models.user import User
from app.models.profile import UserProfile
from app.models.skill import Skill, UserSkill
from app.core.exceptions import AnalyticsError


@pytest.fixture
def mock_user():
    """Mock user for testing"""
    return User(
        id="test-user-id",
        email="test@example.com",
        first_name="Test",
        last_name="User"
    )


@pytest.fixture
def mock_user_profile():
    """Mock user profile for testing"""
    return UserProfile(
        id="profile-id",
        user_id="test-user-id",
        current_role="Software Engineer",
        experience_years=3,
        education="Bachelor's in Computer Science",
        location="San Francisco, CA"
    )


@pytest.fixture
def mock_user_skills():
    """Mock user skills for testing"""
    skills = []
    skill_data = [
        ("Python", "programming_languages", 0.85),
        ("React", "frameworks_libraries", 0.75),
        ("AWS", "cloud_devops", 0.60),
        ("Docker", "tools_technologies", 0.70),
        ("JavaScript", "programming_languages", 0.80)
    ]
    
    for name, category, confidence in skill_data:
        skill = Skill(id=f"skill-{name.lower()}", name=name, category=category)
        user_skill = UserSkill(
            user_id="test-user-id",
            skill_id=skill.id,
            skill=skill,
            confidence_score=confidence
        )
        skills.append(user_skill)
    
    return skills


@pytest.fixture
def mock_db_session():
    """Mock database session for testing"""
    return AsyncMock()


@pytest.fixture
def analytics_service(mock_db_session):
    """Analytics service fixture"""
    return AnalyticsService(mock_db_session)


class TestComprehensiveAnalytics:
    """Test comprehensive analytics functionality"""
    
    @pytest.mark.asyncio
    async def test_calculate_comprehensive_user_analytics(
        self, analytics_service, mock_user_profile, mock_user_skills
    ):
        """Test comprehensive user analytics calculation"""
        # Mock database queries
        with patch.object(analytics_service, '_get_user_profile', return_value=mock_user_profile), \
             patch.object(analytics_service, '_get_user_skills', return_value=mock_user_skills), \
             patch.object(analytics_service, '_get_skill_market_demand', return_value=0.7):
            
            result = await analytics_service.calculate_comprehensive_user_analytics("test-user-id")
            
            # Verify structure
            assert "user_id" in result
            assert "overall_career_score" in result
            assert "skill_analytics" in result
            assert "experience_analytics" in result
            assert "market_analytics" in result
            assert "progression_analytics" in result
            assert "calculated_at" in result
            
            # Verify skill analytics
            skill_analytics = result["skill_analytics"]
            assert skill_analytics["total_skills"] == 5
            assert skill_analytics["average_confidence"] > 0
            assert "skill_distribution" in skill_analytics
            assert "top_skills" in skill_analytics
            assert len(skill_analytics["top_skills"]) <= 10
            
            # Verify experience analytics
            experience_analytics = result["experience_analytics"]
            assert experience_analytics["experience_years"] == 3
            assert experience_analytics["current_role"] == "Software Engineer"
            assert experience_analytics["score"] > 0
            assert "role_level" in experience_analytics
            assert "career_stage" in experience_analytics
            
            # Verify market analytics
            market_analytics = result["market_analytics"]
            assert "position_score" in market_analytics
            assert "high_demand_skills" in market_analytics
            assert "market_competitiveness" in market_analytics
            
            # Verify overall score is calculated
            assert isinstance(result["overall_career_score"], float)
            assert 0 <= result["overall_career_score"] <= 100
    
    @pytest.mark.asyncio
    async def test_analyze_strengths_and_improvements(
        self, analytics_service, mock_user_profile, mock_user_skills
    ):
        """Test strengths and improvements analysis"""
        with patch.object(analytics_service, '_get_user_profile', return_value=mock_user_profile), \
             patch.object(analytics_service, '_get_user_skills', return_value=mock_user_skills), \
             patch.object(analytics_service, '_get_skill_market_demand', return_value=0.7):
            
            result = await analytics_service.analyze_strengths_and_improvements("test-user-id")
            
            # Verify structure
            assert "user_id" in result
            assert "strengths" in result
            assert "improvement_areas" in result
            assert "balance_analysis" in result
            assert "analyzed_at" in result
            
            # Verify strengths analysis
            strengths = result["strengths"]
            assert "skills" in strengths
            assert "experience" in strengths
            assert "overall_strength_score" in strengths
            
            # Verify improvement areas
            improvement_areas = result["improvement_areas"]
            assert "areas" in improvement_areas
            assert "recommendations" in improvement_areas
            assert "urgency_score" in improvement_areas
            
            # Verify balance analysis
            balance_analysis = result["balance_analysis"]
            assert "strength_to_improvement_ratio" in balance_analysis
            assert "development_focus" in balance_analysis
            assert balance_analysis["development_focus"] in ["strengths", "improvements"]
    
    @pytest.mark.asyncio
    async def test_generate_overall_career_score_and_recommendations(
        self, analytics_service, mock_user_profile, mock_user_skills
    ):
        """Test overall career score and recommendations generation"""
        target_role = "Senior Software Engineer"
        
        # Mock the dependencies
        mock_analytics = {
            "overall_career_score": 75.5,
            "skill_analytics": {"overall_score": 74.0},
            "experience_analytics": {"score": 70.0},
            "market_analytics": {"position_score": 80.0},
            "progression_analytics": {"progression_score": 65.0}
        }
        
        mock_strengths = {
            "strengths": {"overall_strength_score": 80.0},
            "improvement_areas": {"areas": [], "urgency_score": 40.0}
        }
        
        with patch.object(analytics_service, 'calculate_comprehensive_user_analytics', return_value=mock_analytics), \
             patch.object(analytics_service, 'analyze_strengths_and_improvements', return_value=mock_strengths), \
             patch.object(analytics_service, '_calculate_role_specific_score', return_value=78.0), \
             patch.object(analytics_service, '_generate_role_specific_recommendations', return_value=["Learn advanced Python"]):
            
            result = await analytics_service.generate_overall_career_score_and_recommendations(
                "test-user-id", target_role
            )
            
            # Verify structure
            assert "user_id" in result
            assert "overall_career_score" in result
            assert "role_specific_score" in result
            assert "target_role" in result
            assert "comprehensive_recommendations" in result
            assert "priority_actions" in result
            assert "trajectory_predictions" in result
            assert "score_breakdown" in result
            assert "generated_at" in result
            
            # Verify values
            assert result["overall_career_score"] == 75.5
            assert result["role_specific_score"] == 78.0
            assert result["target_role"] == target_role
            
            # Verify score breakdown
            score_breakdown = result["score_breakdown"]
            assert "skills" in score_breakdown
            assert "experience" in score_breakdown
            assert "market_position" in score_breakdown
            assert "progression" in score_breakdown
            
            # Verify trajectory predictions
            trajectory = result["trajectory_predictions"]
            assert "growth_potential" in trajectory
            assert "timeline_to_next_level" in trajectory
            assert "predicted_salary_growth" in trajectory
            assert "career_stability" in trajectory
    
    @pytest.mark.asyncio
    async def test_skill_analytics_calculation(self, analytics_service, mock_user_skills):
        """Test skill analytics calculation"""
        result = await analytics_service._calculate_skill_analytics(mock_user_skills)
        
        assert result["total_skills"] == 5
        assert result["average_confidence"] > 0
        assert "skill_distribution" in result
        assert "top_skills" in result
        assert len(result["top_skills"]) == 5  # All skills should be included
        
        # Verify top skills are sorted by confidence
        top_skills = result["top_skills"]
        for i in range(len(top_skills) - 1):
            assert top_skills[i]["confidence"] >= top_skills[i + 1]["confidence"]
    
    @pytest.mark.asyncio
    async def test_experience_analytics_calculation(self, analytics_service, mock_user_profile):
        """Test experience analytics calculation"""
        result = await analytics_service._calculate_experience_analytics(mock_user_profile)
        
        assert result["experience_years"] == 3
        assert result["current_role"] == "Software Engineer"
        assert result["education"] == "Bachelor's in Computer Science"
        assert result["score"] > 0
        assert result["role_level"] in ["junior", "mid", "senior", "lead", "executive"]
        assert result["career_stage"] in ["early_career", "mid_career", "senior_career", "executive_career"]
    
    @pytest.mark.asyncio
    async def test_market_position_analytics(self, analytics_service, mock_user_skills):
        """Test market position analytics calculation"""
        with patch.object(analytics_service, '_get_skill_market_demand', return_value=0.7):
            result = await analytics_service._calculate_market_position_analytics("test-user-id", mock_user_skills)
            
            assert "position_score" in result
            assert "high_demand_skills" in result
            assert "medium_demand_skills" in result
            assert "low_demand_skills" in result
            assert "market_competitiveness" in result
            
            assert result["market_competitiveness"] in ["high", "medium", "low"]
            assert 0 <= result["position_score"] <= 100
    
    @pytest.mark.asyncio
    async def test_improvement_areas_identification(self, analytics_service, mock_user_skills, mock_user_profile):
        """Test improvement areas identification"""
        # Add a low-confidence skill for testing
        low_skill = UserSkill(
            user_id="test-user-id",
            skill_id="skill-low",
            skill=Skill(id="skill-low", name="Kubernetes", category="cloud_devops"),
            confidence_score=0.3  # Low confidence
        )
        mock_user_skills.append(low_skill)
        
        with patch.object(analytics_service, '_get_skill_market_demand', return_value=0.8):
            result = await analytics_service._identify_improvement_areas(
                "test-user-id", mock_user_skills, mock_user_profile
            )
            
            assert isinstance(result, list)
            assert len(result) > 0
            
            # Should identify the low-confidence skill
            skill_improvements = [area for area in result if area["type"] == "skill_improvement"]
            assert len(skill_improvements) > 0
            
            # Should identify missing high-demand skills
            missing_skills = [area for area in result if area["type"] == "missing_skill"]
            assert len(missing_skills) > 0
    
    @pytest.mark.asyncio
    async def test_overall_career_score_calculation(self, analytics_service):
        """Test overall career score calculation"""
        skill_analytics = {"overall_score": 80.0}
        experience_analytics = {"score": 70.0}
        market_analytics = {"position_score": 75.0}
        progression_analytics = {"progression_score": 65.0}
        
        result = await analytics_service._calculate_overall_career_score(
            skill_analytics, experience_analytics, market_analytics, progression_analytics
        )
        
        assert isinstance(result, float)
        assert 0 <= result <= 100
        # Should be weighted average: 80*0.35 + 70*0.25 + 75*0.25 + 65*0.15 = 74.0
        assert abs(result - 74.0) < 0.1
    
    @pytest.mark.asyncio
    async def test_role_level_determination(self, analytics_service):
        """Test role level determination"""
        assert analytics_service._determine_role_level("Junior Software Engineer") == "junior"
        assert analytics_service._determine_role_level("Software Engineer") == "mid"
        assert analytics_service._determine_role_level("Senior Software Engineer") == "senior"
        assert analytics_service._determine_role_level("Lead Developer") == "lead"
        assert analytics_service._determine_role_level("Engineering Manager") == "lead"
        assert analytics_service._determine_role_level("Director of Engineering") == "executive"
    
    @pytest.mark.asyncio
    async def test_career_stage_determination(self, analytics_service):
        """Test career stage determination"""
        assert analytics_service._determine_career_stage(1) == "early_career"
        assert analytics_service._determine_career_stage(3) == "mid_career"
        assert analytics_service._determine_career_stage(7) == "senior_career"
        assert analytics_service._determine_career_stage(12) == "executive_career"
    
    @pytest.mark.asyncio
    async def test_skill_market_demand(self, analytics_service):
        """Test skill market demand calculation"""
        # High demand skills
        assert await analytics_service._get_skill_market_demand("Python") == 0.8
        assert await analytics_service._get_skill_market_demand("React") == 0.8
        assert await analytics_service._get_skill_market_demand("AWS") == 0.8
        
        # Medium demand skills
        assert await analytics_service._get_skill_market_demand("Java") == 0.6
        assert await analytics_service._get_skill_market_demand("Angular") == 0.6
        
        # Default demand for unknown skills
        assert await analytics_service._get_skill_market_demand("UnknownSkill") == 0.4
    
    @pytest.mark.asyncio
    async def test_error_handling(self, analytics_service):
        """Test error handling in analytics service"""
        # Test with invalid user ID
        with patch.object(analytics_service, '_get_user_profile', side_effect=Exception("Database error")):
            with pytest.raises(AnalyticsError):
                await analytics_service.calculate_comprehensive_user_analytics("invalid-user-id")
    
    @pytest.mark.asyncio
    async def test_empty_skills_handling(self, analytics_service, mock_user_profile):
        """Test handling of users with no skills"""
        with patch.object(analytics_service, '_get_user_profile', return_value=mock_user_profile), \
             patch.object(analytics_service, '_get_user_skills', return_value=[]):
            
            result = await analytics_service.calculate_comprehensive_user_analytics("test-user-id")
            
            # Should handle empty skills gracefully
            assert result["skill_analytics"]["total_skills"] == 0
            assert result["skill_analytics"]["average_confidence"] == 0
            assert result["overall_career_score"] >= 0  # Should still calculate based on other factors


class TestAnalyticsEndpoints:
    """Test analytics API endpoints"""
    
    @pytest.mark.asyncio
    async def test_comprehensive_analytics_endpoint(self, client, mock_user, mock_db_session):
        """Test comprehensive analytics endpoint"""
        # This would require setting up the full FastAPI test client
        # For now, we'll test the service layer which is the core functionality
        pass
    
    @pytest.mark.asyncio
    async def test_strengths_improvements_endpoint(self, client, mock_user, mock_db_session):
        """Test strengths and improvements endpoint"""
        # This would require setting up the full FastAPI test client
        pass
    
    @pytest.mark.asyncio
    async def test_career_score_endpoint(self, client, mock_user, mock_db_session):
        """Test career score endpoint"""
        # This would require setting up the full FastAPI test client
        pass