"""
Test enhanced profile management functionality for analyze page integration.
"""

import pytest
from datetime import datetime
from unittest.mock import AsyncMock, patch

from app.services.profile_service import UserProfileService
from app.schemas.profile import ProfileCreate, ProfileUpdate
from app.models.profile import UserProfile


@pytest.fixture
def profile_service():
    """Create profile service instance for testing."""
    return UserProfileService()


@pytest.fixture
def sample_analyze_page_data():
    """Sample data from frontend analyze page."""
    return {
        # Personal Information
        "current_role": "Software Developer",
        "experience_years": 3,
        "industry": "Technology",
        "location": "San Francisco, CA",
        
        # Career Goals
        "desired_role": "Senior Software Engineer",
        "career_goals": "I want to become a technical lead and work on scalable systems",
        "timeframe": "medium",
        "salary_expectation": "$120,000 - $150,000",
        
        # Skills & Education
        "skills": {"Python": 0.8, "JavaScript": 0.7, "React": 0.6},
        "education": "Bachelor's in Computer Science",
        "certifications": "AWS Certified Developer",
        "languages": "English (Native), Spanish (Conversational)",
        
        # Work Preferences
        "work_type": "hybrid",
        "company_size": "medium",
        "work_culture": "Collaborative environment with focus on innovation",
        "benefits": ["Health Insurance", "401(k) Matching", "Remote Work", "Professional Development"]
    }


@pytest.fixture
def mock_profile():
    """Mock profile for testing."""
    profile = UserProfile()
    profile.id = "test-profile-id"
    profile.user_id = "test-user-id"
    profile.current_role = "Software Developer"
    profile.experience_years = 3
    profile.industry = "Technology"
    profile.location = "San Francisco, CA"
    profile.desired_role = "Senior Software Engineer"
    profile.career_goals = "I want to become a technical lead"
    profile.timeframe = "medium"
    profile.salary_expectation = "$120,000 - $150,000"
    profile.education = "Bachelor's in Computer Science"
    profile.certifications = "AWS Certified Developer"
    profile.languages = "English (Native), Spanish (Conversational)"
    profile.work_type = "hybrid"
    profile.company_size = "medium"
    profile.work_culture = "Collaborative environment"
    profile.benefits = ["Health Insurance", "401(k) Matching"]
    profile.skills = {"Python": 0.8, "JavaScript": 0.7, "React": 0.6}
    profile.data_last_updated = datetime.utcnow()
    profile.created_at = datetime.utcnow()
    profile.updated_at = datetime.utcnow()
    return profile


class TestProfileCompleteness:
    """Test profile completeness calculation."""
    
    def test_calculate_profile_completeness_full_profile(self, profile_service, mock_profile):
        """Test completeness calculation for a complete profile."""
        completeness = profile_service._calculate_profile_completeness(mock_profile)
        
        # Should be high since most fields are filled
        assert completeness > 80
        assert completeness <= 100
    
    def test_calculate_profile_completeness_minimal_profile(self, profile_service):
        """Test completeness calculation for minimal profile."""
        minimal_profile = UserProfile()
        minimal_profile.current_role = "Developer"
        
        completeness = profile_service._calculate_profile_completeness(minimal_profile)
        
        # Should be low since most fields are empty
        assert completeness < 30
    
    def test_calculate_profile_completeness_empty_profile(self, profile_service):
        """Test completeness calculation for empty profile."""
        empty_profile = UserProfile()
        
        completeness = profile_service._calculate_profile_completeness(empty_profile)
        
        # Should be 0 for completely empty profile
        assert completeness == 0


class TestProfileScoring:
    """Test profile scoring functionality."""
    
    def test_calculate_profile_score(self, profile_service, mock_profile):
        """Test overall profile score calculation."""
        score = profile_service._calculate_profile_score(mock_profile)
        
        assert 0 <= score <= 100
        # Should be reasonably high for a well-filled profile
        assert score > 60
    
    def test_calculate_skills_score(self, profile_service):
        """Test skills quality scoring."""
        # Test with good skills
        good_skills = {"Python": 0.9, "JavaScript": 0.8, "React": 0.7, "Node.js": 0.6}
        score = profile_service._calculate_skills_score(good_skills)
        assert score > 50
        
        # Test with no skills
        no_skills = {}
        score = profile_service._calculate_skills_score(no_skills)
        assert score == 0
        
        # Test with low confidence skills
        low_skills = {"Python": 0.2, "JavaScript": 0.1}
        score = profile_service._calculate_skills_score(low_skills)
        assert score < 30
    
    def test_calculate_platform_integration_score(self, profile_service, mock_profile):
        """Test platform integration scoring."""
        # Test with all platforms connected
        mock_profile.github_username = "testuser"
        mock_profile.leetcode_id = "testuser"
        mock_profile.linkedin_url = "https://linkedin.com/in/testuser"
        mock_profile.platform_data = {
            "github": {"repositories": []},
            "leetcode": {"problems_solved": 100},
            "linkedin": {"connections": 500}
        }
        
        score = profile_service._calculate_platform_integration_score(mock_profile)
        assert score > 80
        
        # Test with no platforms
        mock_profile.github_username = None
        mock_profile.leetcode_id = None
        mock_profile.linkedin_url = None
        mock_profile.platform_data = {}
        
        score = profile_service._calculate_platform_integration_score(mock_profile)
        assert score < 30


class TestSkillAnalysis:
    """Test skill distribution analysis."""
    
    def test_analyze_skill_distribution(self, profile_service):
        """Test skill categorization and analysis."""
        skills = {
            "Python": 0.9,
            "JavaScript": 0.8,
            "React": 0.7,
            "Docker": 0.6,
            "AWS": 0.5,
            "Leadership": 0.4,
            "Communication": 0.8
        }
        
        analysis = profile_service._analyze_skill_distribution(skills)
        
        assert analysis['total_skills'] == len(skills)
        assert 'categories' in analysis
        assert 'confidence_distribution' in analysis
        assert 'top_skills' in analysis
        
        # Check confidence distribution
        assert analysis['confidence_distribution']['high'] > 0  # Python, JavaScript, Communication
        assert analysis['confidence_distribution']['medium'] > 0  # React, Docker, AWS
        assert analysis['confidence_distribution']['low'] > 0   # Leadership
    
    def test_analyze_skill_distribution_empty(self, profile_service):
        """Test skill analysis with no skills."""
        analysis = profile_service._analyze_skill_distribution({})
        
        assert analysis['total_skills'] == 0
        assert analysis['categories'] == {}
        assert analysis['top_skills'] == []


class TestProfileValidation:
    """Test profile data validation."""
    
    @pytest.mark.asyncio
    async def test_validate_profile_consistency_valid_data(self, profile_service, mock_profile):
        """Test validation with valid profile data."""
        update_data = ProfileUpdate(
            current_role="Senior Developer",
            experience_years=5,
            timeframe="short",
            work_type="remote",
            company_size="large",
            benefits=["Health Insurance", "401(k)"]
        )
        
        result = await profile_service._validate_profile_consistency(mock_profile, update_data)
        
        assert result['is_valid'] is True
        assert len(result['errors']) == 0
    
    @pytest.mark.asyncio
    async def test_validate_profile_consistency_invalid_data(self, profile_service, mock_profile):
        """Test validation with invalid profile data."""
        update_data = ProfileUpdate(
            experience_years=-1,  # Invalid: negative
            skills={"Python": 1.5},  # Invalid: confidence > 1
            benefits="not a list"  # Invalid: should be list
        )
        
        result = await profile_service._validate_profile_consistency(mock_profile, update_data)
        
        assert result['is_valid'] is False
        assert len(result['errors']) > 0


class TestProfileRecommendations:
    """Test profile improvement recommendations."""
    
    def test_generate_profile_recommendations_incomplete(self, profile_service):
        """Test recommendations for incomplete profile."""
        incomplete_profile = UserProfile()
        incomplete_profile.current_role = "Developer"
        # Missing many fields
        
        recommendations = profile_service._generate_profile_recommendations(incomplete_profile, 30.0)
        
        assert len(recommendations) > 0
        
        # Should have completeness recommendation
        completeness_recs = [r for r in recommendations if r['type'] == 'completeness']
        assert len(completeness_recs) > 0
        
        # Should have skills recommendation
        skills_recs = [r for r in recommendations if r['type'] == 'skills']
        assert len(skills_recs) > 0
    
    def test_generate_profile_recommendations_complete(self, profile_service, mock_profile):
        """Test recommendations for complete profile."""
        # Set up a complete profile
        mock_profile.github_username = "testuser"
        mock_profile.leetcode_id = "testuser"
        mock_profile.linkedin_url = "https://linkedin.com/in/testuser"
        
        recommendations = profile_service._generate_profile_recommendations(mock_profile, 95.0)
        
        # Should have fewer recommendations for complete profile
        high_priority_recs = [r for r in recommendations if r['priority'] == 'high']
        assert len(high_priority_recs) == 0  # No high priority recommendations for complete profile


class TestAnalyzePageIntegration:
    """Test integration with analyze page data format."""
    
    def test_profile_create_schema_analyze_page_fields(self, sample_analyze_page_data):
        """Test that ProfileCreate schema handles all analyze page fields."""
        profile_create = ProfileCreate(**sample_analyze_page_data)
        
        # Verify all fields are properly set
        assert profile_create.current_role == "Software Developer"
        assert profile_create.industry == "Technology"
        assert profile_create.desired_role == "Senior Software Engineer"
        assert profile_create.career_goals == "I want to become a technical lead and work on scalable systems"
        assert profile_create.timeframe == "medium"
        assert profile_create.salary_expectation == "$120,000 - $150,000"
        assert profile_create.education == "Bachelor's in Computer Science"
        assert profile_create.certifications == "AWS Certified Developer"
        assert profile_create.languages == "English (Native), Spanish (Conversational)"
        assert profile_create.work_type == "hybrid"
        assert profile_create.company_size == "medium"
        assert profile_create.work_culture == "Collaborative environment with focus on innovation"
        assert profile_create.benefits == ["Health Insurance", "401(k) Matching", "Remote Work", "Professional Development"]
        assert profile_create.skills == {"Python": 0.8, "JavaScript": 0.7, "React": 0.6}
    
    def test_profile_update_schema_analyze_page_fields(self, sample_analyze_page_data):
        """Test that ProfileUpdate schema handles all analyze page fields."""
        profile_update = ProfileUpdate(**sample_analyze_page_data)
        
        # Verify all fields are properly set
        assert profile_update.current_role == "Software Developer"
        assert profile_update.industry == "Technology"
        assert profile_update.desired_role == "Senior Software Engineer"
        assert profile_update.work_type == "hybrid"
        assert profile_update.company_size == "medium"
        assert len(profile_update.benefits) == 4


if __name__ == "__main__":
    pytest.main([__file__])