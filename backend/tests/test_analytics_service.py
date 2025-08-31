"""
Tests for analytics service functionality
"""
import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
from sqlalchemy.ext.asyncio import AsyncSession

from app.services.analytics_service import AnalyticsService
from app.schemas.analytics import (
    AnalyticsRequest, SkillRadarChart, CareerRoadmapVisualization,
    SkillGapReport, JobCompatibilityReport, HistoricalProgressReport
)
from app.models.user import User
from app.models.profile import Profile
from app.models.skill import Skill, UserSkill
from app.models.job import JobPosting
from app.core.exceptions import AnalyticsError, DataNotFoundError


@pytest.fixture
def mock_db():
    """Mock database session"""
    return Mock(spec=AsyncSession)


@pytest.fixture
def analytics_service(mock_db):
    """Analytics service instance with mocked database"""
    return AnalyticsService(mock_db)


@pytest.fixture
def sample_user():
    """Sample user for testing"""
    return User(
        id="test-user-123",
        email="test@example.com",
        is_active=True
    )


@pytest.fixture
def sample_profile():
    """Sample user profile for testing"""
    return Profile(
        user_id="test-user-123",
        full_name="Test User",
        current_role="Software Developer",
        experience_years=3,
        location="San Francisco, CA",
        bio="Experienced developer"
    )


@pytest.fixture
def sample_skills():
    """Sample user skills for testing"""
    skills = [
        Mock(spec=UserSkill, skill=Mock(spec=Skill, name="Python", category="programming"), confidence_score=0.8),
        Mock(spec=UserSkill, skill=Mock(spec=Skill, name="JavaScript", category="programming"), confidence_score=0.7),
        Mock(spec=UserSkill, skill=Mock(spec=Skill, name="React", category="frameworks"), confidence_score=0.6),
        Mock(spec=UserSkill, skill=Mock(spec=Skill, name="AWS", category="cloud"), confidence_score=0.5),
    ]
    return skills


class TestAnalyticsService:
    """Test cases for AnalyticsService"""
    
    @pytest.mark.asyncio
    async def test_generate_skill_radar_chart_success(self, analytics_service, sample_profile, sample_skills):
        """Test successful skill radar chart generation"""
        # Mock database queries
        analytics_service._get_user_profile = AsyncMock(return_value=sample_profile)
        analytics_service._get_user_skills = AsyncMock(return_value=sample_skills)
        analytics_service._get_skills_by_category = AsyncMock(return_value=[])
        analytics_service._calculate_category_score = AsyncMock(return_value=75.0)
        analytics_service._get_market_average_score = AsyncMock(return_value=65.0)
        analytics_service._calculate_target_score = AsyncMock(return_value=85.0)
        
        # Test radar chart generation
        result = await analytics_service.generate_skill_radar_chart("test-user-123", "Senior Developer")
        
        assert isinstance(result, SkillRadarChart)
        assert result.user_id == "test-user-123"
        assert len(result.categories) == 6  # Expected number of categories
        assert len(result.user_scores) == 6
        assert len(result.market_average) == 6
        assert result.target_scores is not None
        assert len(result.target_scores) == 6
    
    @pytest.mark.asyncio
    async def test_generate_skill_radar_chart_no_skills(self, analytics_service, sample_profile):
        """Test radar chart generation with no skills"""
        analytics_service._get_user_profile = AsyncMock(return_value=sample_profile)
        analytics_service._get_user_skills = AsyncMock(return_value=[])
        
        with pytest.raises(DataNotFoundError, match="No skills found for user"):
            await analytics_service.generate_skill_radar_chart("test-user-123")
    
    @pytest.mark.asyncio
    async def test_generate_career_roadmap_success(self, analytics_service, sample_profile):
        """Test successful career roadmap generation"""
        # Mock dependencies
        analytics_service._get_user_profile = AsyncMock(return_value=sample_profile)
        analytics_service.analyze_skill_gaps = AsyncMock(return_value=Mock(
            skill_gaps=[
                Mock(skill_name="Python", gap_size=20, priority="high"),
                Mock(skill_name="React", gap_size=15, priority="medium")
            ]
        ))
        analytics_service._generate_career_milestones = AsyncMock(return_value=[
            {
                "title": "Master Python",
                "description": "Become proficient in Python",
                "timeline_months": 6,
                "required_skills": ["Python", "Django"],
                "difficulty": 0.7,
                "required_actions": ["Take course", "Build project"]
            }
        ])
        analytics_service._get_alternative_career_paths = AsyncMock(return_value=[
            {"title": "Data Scientist", "description": "Data analysis role", "timeline_months": 12, "difficulty": 0.8}
        ])
        
        result = await analytics_service.generate_career_roadmap("test-user-123", "Senior Developer")
        
        assert isinstance(result, CareerRoadmapVisualization)
        assert result.user_id == "test-user-123"
        assert len(result.nodes) >= 3  # At least current, milestone, and target nodes
        assert len(result.edges) >= 2  # At least connections between nodes
        assert result.metadata["target_role"] == "Senior Developer"
    
    @pytest.mark.asyncio
    async def test_analyze_skill_gaps_success(self, analytics_service, sample_skills):
        """Test successful skill gap analysis"""
        # Mock dependencies
        analytics_service._get_user_skills = AsyncMock(return_value=sample_skills)
        analytics_service._get_target_role_skills = AsyncMock(return_value=[
            {"name": "Python", "required_level": 90, "importance": "high", "market_demand": 0.9},
            {"name": "React", "required_level": 85, "importance": "high", "market_demand": 0.8},
            {"name": "Docker", "required_level": 70, "importance": "medium", "market_demand": 0.7}
        ])
        analytics_service._get_learning_resources = AsyncMock(return_value=[
            {"title": "Python Course", "type": "course", "provider": "Coursera", "rating": 4.5}
        ])
        
        result = await analytics_service.analyze_skill_gaps("test-user-123", "Senior Developer")
        
        assert isinstance(result, SkillGapReport)
        assert result.user_id == "test-user-123"
        assert result.target_role == "Senior Developer"
        assert result.overall_match_score >= 0
        assert len(result.skill_gaps) > 0
        assert result.total_learning_hours > 0
    
    @pytest.mark.asyncio
    async def test_generate_job_compatibility_scores_success(self, analytics_service, sample_skills, sample_profile):
        """Test successful job compatibility scoring"""
        # Mock job postings
        mock_jobs = [
            Mock(
                spec=JobPosting,
                id="job-1",
                title="Python Developer",
                company="Tech Corp",
                location="San Francisco",
                experience_level="mid",
                is_active=True
            ),
            Mock(
                spec=JobPosting,
                id="job-2",
                title="Full Stack Developer",
                company="Startup Inc",
                location="Remote",
                experience_level="senior",
                is_active=True
            )
        ]
        
        # Mock dependencies
        analytics_service._get_user_skills = AsyncMock(return_value=sample_skills)
        analytics_service._get_user_profile = AsyncMock(return_value=sample_profile)
        analytics_service._get_filtered_jobs = AsyncMock(return_value=mock_jobs)
        analytics_service._calculate_job_compatibility = AsyncMock(return_value=Mock(
            job_id="job-1",
            job_title="Python Developer",
            company="Tech Corp",
            overall_score=85.0,
            skill_match_score=80.0,
            experience_match_score=90.0,
            matched_skills=["Python", "JavaScript"],
            missing_skills=["Docker"],
            recommendation="apply"
        ))
        
        result = await analytics_service.generate_job_compatibility_scores("test-user-123")
        
        assert isinstance(result, JobCompatibilityReport)
        assert result.user_id == "test-user-123"
        assert len(result.job_matches) > 0
        assert result.total_jobs_analyzed == 2
    
    @pytest.mark.asyncio
    async def test_track_historical_progress_success(self, analytics_service):
        """Test successful historical progress tracking"""
        # Mock dependencies
        analytics_service._calculate_skill_improvements = AsyncMock(return_value=[
            Mock(
                user_id="test-user-123",
                skill_name="Python",
                previous_score=70.0,
                current_score=80.0,
                improvement=10.0,
                tracking_period_days=90
            )
        ])
        analytics_service._get_achieved_milestones = AsyncMock(return_value=[
            "Completed Python Course",
            "Built first web app"
        ])
        analytics_service._analyze_progress_trends = AsyncMock(return_value={
            "trend": "improving",
            "velocity": 8.5,
            "total_skills_improved": 3
        })
        
        result = await analytics_service.track_historical_progress("test-user-123", 90)
        
        assert isinstance(result, HistoricalProgressReport)
        assert result.user_id == "test-user-123"
        assert result.tracking_period_days == 90
        assert len(result.skill_improvements) > 0
        assert result.overall_improvement_score > 0
        assert len(result.milestones_achieved) > 0
    
    @pytest.mark.asyncio
    async def test_generate_comprehensive_report_success(self, analytics_service):
        """Test successful comprehensive report generation"""
        # Mock all component methods
        analytics_service.generate_skill_radar_chart = AsyncMock(return_value=Mock(spec=SkillRadarChart))
        analytics_service.generate_career_roadmap = AsyncMock(return_value=Mock(spec=CareerRoadmapVisualization))
        analytics_service.analyze_skill_gaps = AsyncMock(return_value=Mock(spec=SkillGapReport))
        analytics_service.generate_job_compatibility_scores = AsyncMock(return_value=Mock(spec=JobCompatibilityReport))
        analytics_service.track_historical_progress = AsyncMock(return_value=Mock(spec=HistoricalProgressReport))
        analytics_service._get_profile_summary = AsyncMock(return_value={
            "name": "Test User",
            "current_role": "Developer",
            "experience_years": 3
        })
        analytics_service._generate_recommendations = AsyncMock(return_value=[
            "Focus on Python development",
            "Build more projects"
        ])
        analytics_service._generate_next_steps = AsyncMock(return_value=[
            "Take advanced Python course",
            "Contribute to open source"
        ])
        
        request = AnalyticsRequest(
            user_id="test-user-123",
            analysis_types=["full_report"],
            target_role="Senior Developer",
            include_job_matches=True,
            include_progress_tracking=True
        )
        
        result = await analytics_service.generate_comprehensive_report(request)
        
        assert result.user_id == "test-user-123"
        assert result.skill_radar_chart is not None
        assert result.career_roadmap is not None
        assert result.skill_gap_report is not None
        assert result.job_compatibility_report is not None
        assert result.progress_report is not None
        assert len(result.recommendations) > 0
        assert len(result.next_steps) > 0
    
    @pytest.mark.asyncio
    async def test_analytics_error_handling(self, analytics_service):
        """Test error handling in analytics service"""
        # Mock database error
        analytics_service._get_user_profile = AsyncMock(side_effect=Exception("Database error"))
        
        with pytest.raises(AnalyticsError, match="Failed to generate skill radar chart"):
            await analytics_service.generate_skill_radar_chart("test-user-123")
    
    def test_calculate_skill_priority(self, analytics_service):
        """Test skill priority calculation"""
        # High importance, large gap
        priority = analytics_service._calculate_skill_priority(60, "high")
        assert priority == "high"
        
        # Medium importance, medium gap
        priority = analytics_service._calculate_skill_priority(50, "medium")
        assert priority == "medium"
        
        # Low importance, small gap
        priority = analytics_service._calculate_skill_priority(20, "low")
        assert priority == "low"
    
    def test_estimate_learning_hours(self, analytics_service):
        """Test learning hours estimation"""
        # Test basic calculation
        hours = analytics_service._estimate_learning_hours(50, "Python")
        assert hours > 0
        assert isinstance(hours, int)
        
        # Test with complex skill
        ml_hours = analytics_service._estimate_learning_hours(50, "Machine Learning")
        python_hours = analytics_service._estimate_learning_hours(50, "Python")
        assert ml_hours > python_hours  # ML should take more time
    
    def test_parse_experience_level(self, analytics_service):
        """Test experience level parsing"""
        assert analytics_service._parse_experience_level("entry") == 1
        assert analytics_service._parse_experience_level("mid") == 3
        assert analytics_service._parse_experience_level("senior") == 5
        assert analytics_service._parse_experience_level("lead") == 7
        assert analytics_service._parse_experience_level("executive") == 10
        assert analytics_service._parse_experience_level("unknown") == 0
        assert analytics_service._parse_experience_level(None) == 0


class TestAnalyticsHelperMethods:
    """Test helper methods in analytics service"""
    
    def test_calculate_profile_completeness(self, analytics_service, sample_profile):
        """Test profile completeness calculation"""
        completeness = analytics_service._calculate_profile_completeness(sample_profile)
        assert 0 <= completeness <= 100
        assert isinstance(completeness, float)
    
    def test_find_common_missing_skills(self, analytics_service):
        """Test finding common missing skills across jobs"""
        jobs = [
            Mock(missing_skills=["Docker", "Kubernetes", "AWS"]),
            Mock(missing_skills=["Docker", "React", "TypeScript"]),
            Mock(missing_skills=["Docker", "Python", "PostgreSQL"])
        ]
        
        common_skills = analytics_service._find_common_missing_skills(jobs)
        assert "Docker" in common_skills  # Appears in all jobs
        assert len(common_skills) >= 1