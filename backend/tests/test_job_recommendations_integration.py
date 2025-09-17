"""
Integration tests for job recommendation functionality
"""
import pytest
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime

from app.services.recommendation_service import RecommendationService
from app.models.profile import UserProfile
from app.models.job import JobPosting


@pytest.fixture
def recommendation_service():
    """Create recommendation service instance for testing."""
    return RecommendationService()


@pytest.fixture
def sample_user_profile():
    """Create sample user profile for testing."""
    profile = Mock(spec=UserProfile)
    profile.user_id = "test-user-123"
    profile.current_role = "Software Developer"
    profile.dream_job = "Senior Software Engineer"
    profile.experience_years = 3
    profile.location = "San Francisco"
    profile.remote_work_preference = True
    profile.skills = {
        "python": 0.8,
        "javascript": 0.7,
        "react": 0.6,
        "sql": 0.5
    }
    return profile


@pytest.fixture
def sample_job_postings():
    """Create sample job postings for testing."""
    jobs = []
    
    # High match job
    job1 = Mock(spec=JobPosting)
    job1.id = "job-1"
    job1.title = "Senior Python Developer"
    job1.company = "Tech Corp"
    job1.location = "San Francisco"
    job1.remote_type = "remote"
    job1.employment_type = "full-time"
    job1.experience_level = "senior"
    job1.salary_min = 120000
    job1.salary_max = 150000
    job1.salary_currency = "USD"
    job1.posted_date = datetime.now()
    job1.source_url = "https://example.com/job1"
    job1.description = "We are looking for a senior Python developer with React experience."
    job1.processed_skills = {
        "python": 0.9,
        "javascript": 0.7,
        "react": 0.8,
        "sql": 0.6
    }
    jobs.append(job1)
    
    return jobs


class TestJobRecommendationIntegration:
    """Integration tests for job recommendation functionality."""
    
    @pytest.mark.asyncio
    async def test_simple_match_score_calculation(self, recommendation_service):
        """Test the basic match score calculation works correctly."""
        user_skills = {"python": 0.8, "javascript": 0.7, "react": 0.6}
        job_skills = {"python": 0.9, "javascript": 0.8, "sql": 0.7}
        
        score = recommendation_service._calculate_simple_match_score(user_skills, job_skills)
        
        # Should have some overlap between python and javascript
        assert 0.0 <= score <= 1.0
        assert score > 0  # Should have some match
    
    @pytest.mark.asyncio
    async def test_match_score_perfect_overlap(self, recommendation_service):
        """Test match score with perfect skill overlap."""
        user_skills = {"python": 0.8, "javascript": 0.7}
        job_skills = {"python": 0.9, "javascript": 0.8}
        
        score = recommendation_service._calculate_simple_match_score(user_skills, job_skills)
        
        # Should have high score due to perfect overlap
        assert score > 0.5
    
    @pytest.mark.asyncio
    async def test_match_score_no_overlap(self, recommendation_service):
        """Test match score with no skill overlap."""
        user_skills = {"python": 0.8, "javascript": 0.7}
        job_skills = {"java": 0.9, "c++": 0.8}
        
        score = recommendation_service._calculate_simple_match_score(user_skills, job_skills)
        
        # Should have zero score due to no overlap
        assert score == 0.0
    
    @pytest.mark.asyncio
    async def test_ranking_with_preferences(self, recommendation_service, sample_user_profile):
        """Test job ranking based on user preferences."""
        job_matches = [
            {
                "job_id": "job-1",
                "job_title": "Senior Software Engineer",  # Matches dream job
                "company": "Tech Corp",
                "location": "San Francisco",  # Matches user location
                "remote_type": "remote",  # Matches remote preference
                "match_score": 0.7
            },
            {
                "job_id": "job-2",
                "job_title": "Data Scientist",
                "company": "Analytics Co",
                "location": "New York",
                "remote_type": "onsite",
                "match_score": 0.8
            }
        ]
        
        ranked_matches = recommendation_service._rank_job_recommendations(job_matches, sample_user_profile)
        
        # Should have preference boost applied
        assert len(ranked_matches) == 2
        
        # Check that preference boost was calculated
        for match in ranked_matches:
            assert "preference_boost" in match
        
        # Job matching dream job should get boost
        dream_job_match = next((m for m in ranked_matches if "Senior Software Engineer" in m["job_title"]), None)
        assert dream_job_match is not None
        assert dream_job_match["preference_boost"] > 0
    
    @pytest.mark.asyncio
    async def test_job_recommendation_data_structure(self, recommendation_service, sample_user_profile, sample_job_postings):
        """Test that job recommendation returns correct data structure."""
        mock_db = AsyncMock()
        
        # Mock the repository to return our sample jobs
        with patch('app.repositories.job.JobRepository') as mock_job_repo_class:
            mock_job_repo = AsyncMock()
            mock_job_repo_class.return_value = mock_job_repo
            mock_job_repo.search_jobs.return_value = sample_job_postings
            
            # Mock skill gap analyzer
            with patch.object(recommendation_service, 'skill_gap_analyzer') as mock_analyzer:
                mock_gap_analysis = Mock()
                mock_gap_analysis.missing_skills = {"docker": 0.5}
                mock_gap_analysis.weak_skills = {"sql": 0.3}
                mock_gap_analysis.strong_skills = ["python", "javascript"]
                mock_gap_analysis.overall_readiness = 0.75
                mock_gap_analysis.priority_skills = ["docker"]
                mock_analyzer.analyze_skill_gaps.return_value = mock_gap_analysis
                
                matches = await recommendation_service.get_advanced_job_matches(
                    user_profile=sample_user_profile,
                    filters={},
                    match_threshold=0.0,  # Low threshold to ensure we get results
                    include_skill_gaps=True,
                    db=mock_db
                )
                
                # Should return at least one match
                assert len(matches) > 0
                
                # Check the structure of the first match
                first_match = matches[0]
                required_fields = [
                    'job_id', 'job_title', 'company', 'location', 'match_score', 
                    'match_percentage', 'skill_gaps', 'strong_skills', 'required_skills'
                ]
                
                for field in required_fields:
                    assert field in first_match, f"Missing required field: {field}"
                
                # Check data types
                assert isinstance(first_match['match_score'], float)
                assert isinstance(first_match['match_percentage'], float)
                assert isinstance(first_match['required_skills'], list)
                assert isinstance(first_match['strong_skills'], list)
    
    @pytest.mark.asyncio
    async def test_empty_skills_handling(self, recommendation_service):
        """Test handling of empty skills."""
        # Test with empty user skills
        user_skills = {}
        job_skills = {"python": 0.9, "javascript": 0.8}
        
        score = recommendation_service._calculate_simple_match_score(user_skills, job_skills)
        assert score == 0.0
        
        # Test with empty job skills
        user_skills = {"python": 0.8, "javascript": 0.7}
        job_skills = {}
        
        score = recommendation_service._calculate_simple_match_score(user_skills, job_skills)
        assert score == 0.0
        
        # Test with both empty
        user_skills = {}
        job_skills = {}
        
        score = recommendation_service._calculate_simple_match_score(user_skills, job_skills)
        assert score == 0.0


class TestJobRecommendationEndToEnd:
    """End-to-end tests for job recommendation workflow."""
    
    @pytest.mark.asyncio
    async def test_recommendation_workflow_components(self, recommendation_service):
        """Test that all components of the recommendation workflow exist and are callable."""
        # Test that key methods exist
        assert hasattr(recommendation_service, '_calculate_simple_match_score')
        assert hasattr(recommendation_service, 'get_advanced_job_matches')
        assert hasattr(recommendation_service, '_rank_job_recommendations')
        assert hasattr(recommendation_service, 'get_job_recommendations_with_ml')
        
        # Test that methods are callable
        assert callable(recommendation_service._calculate_simple_match_score)
        assert callable(recommendation_service.get_advanced_job_matches)
        assert callable(recommendation_service._rank_job_recommendations)
        assert callable(recommendation_service.get_job_recommendations_with_ml)
    
    @pytest.mark.asyncio
    async def test_recommendation_service_initialization(self, recommendation_service):
        """Test that recommendation service initializes correctly."""
        # Check that service has required attributes
        assert hasattr(recommendation_service, 'recommendation_engine')
        assert hasattr(recommendation_service, 'skill_gap_analyzer')
        assert hasattr(recommendation_service, 'model_trained')
        
        # Check initial state
        assert isinstance(recommendation_service.model_trained, bool)