"""
Tests for job recommendation functionality
"""
import pytest
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession

from app.services.recommendation_service import RecommendationService
from app.models.profile import UserProfile
from app.models.job import JobPosting
from app.core.exceptions import ServiceException


@pytest.fixture
def recommendation_service():
    """Create recommendation service instance for testing."""
    return RecommendationService()


@pytest.fixture
def mock_user_profile():
    """Create mock user profile for testing."""
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
def mock_job_postings():
    """Create mock job postings for testing."""
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
    job1.posted_date = datetime.utcnow()
    job1.source_url = "https://example.com/job1"
    job1.description = "We are looking for a senior Python developer with React experience."
    job1.processed_skills = {
        "python": 0.9,
        "javascript": 0.7,
        "react": 0.8,
        "sql": 0.6,
        "docker": 0.5
    }
    jobs.append(job1)
    
    # Medium match job
    job2 = Mock(spec=JobPosting)
    job2.id = "job-2"
    job2.title = "Full Stack Developer"
    job2.company = "Startup Inc"
    job2.location = "Remote"
    job2.remote_type = "remote"
    job2.employment_type = "full-time"
    job2.experience_level = "mid"
    job2.salary_min = 90000
    job2.salary_max = 120000
    job2.salary_currency = "USD"
    job2.posted_date = datetime.utcnow()
    job2.source_url = "https://example.com/job2"
    job2.description = "Full stack developer position with modern technologies."
    job2.processed_skills = {
        "javascript": 0.8,
        "react": 0.7,
        "node.js": 0.6,
        "mongodb": 0.5
    }
    jobs.append(job2)
    
    # Low match job
    job3 = Mock(spec=JobPosting)
    job3.id = "job-3"
    job3.title = "Data Scientist"
    job3.company = "Analytics Co"
    job3.location = "New York"
    job3.remote_type = "onsite"
    job3.employment_type = "full-time"
    job3.experience_level = "senior"
    job3.salary_min = 130000
    job3.salary_max = 160000
    job3.salary_currency = "USD"
    job3.posted_date = datetime.utcnow()
    job3.source_url = "https://example.com/job3"
    job3.description = "Data scientist role focusing on machine learning and statistics."
    job3.processed_skills = {
        "python": 0.8,
        "machine learning": 0.9,
        "statistics": 0.8,
        "r": 0.6,
        "tensorflow": 0.7
    }
    jobs.append(job3)
    
    return jobs


class TestJobRecommendations:
    """Test job recommendation functionality."""
    
    @pytest.mark.asyncio
    async def test_calculate_simple_match_score(self, recommendation_service):
        """Test simple match score calculation."""
        user_skills = {"python": 0.8, "javascript": 0.7, "react": 0.6}
        job_skills = {"python": 0.9, "javascript": 0.8, "sql": 0.7}
        
        score = recommendation_service._calculate_simple_match_score(user_skills, job_skills)
        
        assert 0.0 <= score <= 1.0
        assert score > 0  # Should have some overlap
    
    @pytest.mark.asyncio
    async def test_calculate_match_score_no_overlap(self, recommendation_service):
        """Test match score with no skill overlap."""
        user_skills = {"python": 0.8, "javascript": 0.7}
        job_skills = {"java": 0.9, "c++": 0.8}
        
        score = recommendation_service._calculate_simple_match_score(user_skills, job_skills)
        
        assert score == 0.0
    
    @pytest.mark.asyncio
    async def test_calculate_match_score_empty_skills(self, recommendation_service):
        """Test match score with empty skills."""
        user_skills = {}
        job_skills = {"python": 0.9}
        
        score = recommendation_service._calculate_simple_match_score(user_skills, job_skills)
        
        assert score == 0.0
    
    @pytest.mark.asyncio
    async def test_get_advanced_job_matches(self, recommendation_service, mock_user_profile, mock_job_postings):
        """Test advanced job matching functionality."""
        mock_db = AsyncMock(spec=AsyncSession)
        
        with patch('app.repositories.job.JobRepository') as mock_job_repo_class:
            mock_job_repo = AsyncMock()
            mock_job_repo_class.return_value = mock_job_repo
            mock_job_repo.search_jobs.return_value = mock_job_postings
            
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
                    user_profile=mock_user_profile,
                    filters={"location": "San Francisco"},
                    match_threshold=0.1,
                    include_skill_gaps=True,
                    db=mock_db
                )
                
                assert len(matches) > 0
                
                # Check first match (should be highest scoring)
                first_match = matches[0]
                assert "job_id" in first_match
                assert "job_title" in first_match
                assert "company" in first_match
                assert "match_score" in first_match
                assert "match_percentage" in first_match
                assert "skill_gaps" in first_match
                assert "strong_skills" in first_match
                
                # Verify matches are sorted by score
                for i in range(len(matches) - 1):
                    assert matches[i]["match_score"] >= matches[i + 1]["match_score"]
    
    @pytest.mark.asyncio
    async def test_get_advanced_job_matches_with_filters(self, recommendation_service, mock_user_profile, mock_job_postings):
        """Test job matching with various filters."""
        mock_db = AsyncMock(spec=AsyncSession)
        
        with patch('app.repositories.job.JobRepository') as mock_job_repo_class:
            mock_job_repo = AsyncMock()
            mock_job_repo_class.return_value = mock_job_repo
            mock_job_repo.search_jobs.return_value = mock_job_postings
            
            filters = {
                "location": "San Francisco",
                "experience_level": "senior",
                "remote_type": "remote",
                "min_salary": 100000,
                "max_salary": 200000
            }
            
            matches = await recommendation_service.get_advanced_job_matches(
                user_profile=mock_user_profile,
                filters=filters,
                match_threshold=0.0,
                include_skill_gaps=False,
                db=mock_db
            )
            
            # Verify filters were passed to repository
            mock_job_repo.search_jobs.assert_called_once_with(
                db=mock_db,
                location="San Francisco",
                experience_level="senior",
                remote_type="remote",
                min_salary=100000,
                max_salary=200000,
                limit=100
            )
    
    @pytest.mark.asyncio
    async def test_rank_job_recommendations(self, recommendation_service, mock_user_profile):
        """Test job recommendation ranking based on user preferences."""
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
        
        ranked_matches = recommendation_service._rank_job_recommendations(job_matches, mock_user_profile)
        
        # First job should get preference boost and potentially rank higher
        first_match = ranked_matches[0]
        assert "preference_boost" in first_match
        
        # Job matching dream job and preferences should get boost
        dream_job_match = next((m for m in ranked_matches if "Senior Software Engineer" in m["job_title"]), None)
        assert dream_job_match is not None
        assert dream_job_match["preference_boost"] > 0
    
    @pytest.mark.asyncio
    async def test_get_job_recommendations_with_ml_no_profile(self, recommendation_service):
        """Test ML job recommendations when user profile doesn't exist."""
        mock_db = AsyncMock(spec=AsyncSession)
        
        with patch('app.repositories.profile.ProfileRepository') as mock_profile_repo_class:
            mock_profile_repo = AsyncMock()
            mock_profile_repo_class.return_value = mock_profile_repo
            mock_profile_repo.get_by_user_id.return_value = None
            
            with pytest.raises(ServiceException, match="User profile not found"):
                await recommendation_service.get_job_recommendations_with_ml(
                    user_id="nonexistent-user",
                    db=mock_db
                )
    
    @pytest.mark.asyncio
    async def test_get_job_recommendations_with_ml_success(self, recommendation_service, mock_user_profile, mock_job_postings):
        """Test successful ML job recommendations."""
        mock_db = AsyncMock(spec=AsyncSession)
        
        with patch('app.repositories.profile.ProfileRepository') as mock_profile_repo_class:
            mock_profile_repo = AsyncMock()
            mock_profile_repo_class.return_value = mock_profile_repo
            mock_profile_repo.get_by_user_id.return_value = mock_user_profile
            
            with patch.object(recommendation_service, 'initialize_and_train_models') as mock_init:
                mock_init.return_value = None
                recommendation_service.model_trained = True
                
                with patch.object(recommendation_service, 'get_advanced_job_matches') as mock_get_matches:
                    mock_matches = [
                        {
                            "job_id": "job-1",
                            "job_title": "Senior Python Developer",
                            "match_score": 0.85,
                            "match_percentage": 85.0
                        }
                    ]
                    mock_get_matches.return_value = mock_matches
                    
                    with patch.object(recommendation_service, '_rank_job_recommendations') as mock_rank:
                        mock_rank.return_value = mock_matches
                        
                        recommendations = await recommendation_service.get_job_recommendations_with_ml(
                            user_id="test-user-123",
                            db=mock_db,
                            n_recommendations=5
                        )
                        
                        assert len(recommendations) > 0
                        assert recommendations[0]["job_id"] == "job-1"
                        mock_init.assert_called_once()
                        mock_get_matches.assert_called_once()
                        mock_rank.assert_called_once()


class TestJobRecommendationEdgeCases:
    """Test edge cases and error handling."""
    
    @pytest.mark.asyncio
    async def test_empty_job_database(self, recommendation_service, mock_user_profile):
        """Test behavior when no jobs are available."""
        mock_db = AsyncMock(spec=AsyncSession)
        
        with patch('app.repositories.job.JobRepository') as mock_job_repo_class:
            mock_job_repo = AsyncMock()
            mock_job_repo_class.return_value = mock_job_repo
            mock_job_repo.search_jobs.return_value = []
            
            matches = await recommendation_service.get_advanced_job_matches(
                user_profile=mock_user_profile,
                db=mock_db
            )
            
            assert matches == []
    
    @pytest.mark.asyncio
    async def test_high_match_threshold(self, recommendation_service, mock_user_profile, mock_job_postings):
        """Test with very high match threshold."""
        mock_db = AsyncMock(spec=AsyncSession)
        
        with patch('app.repositories.job.JobRepository') as mock_job_repo_class:
            mock_job_repo = AsyncMock()
            mock_job_repo_class.return_value = mock_job_repo
            mock_job_repo.search_jobs.return_value = mock_job_postings
            
            matches = await recommendation_service.get_advanced_job_matches(
                user_profile=mock_user_profile,
                match_threshold=0.99,  # Very high threshold
                db=mock_db
            )
            
            # Should return fewer or no matches
            assert len(matches) <= len(mock_job_postings)
    
    @pytest.mark.asyncio
    async def test_user_with_no_skills(self, recommendation_service, mock_job_postings):
        """Test recommendations for user with no skills."""
        mock_db = AsyncMock(spec=AsyncSession)
        
        # Create user profile with no skills
        profile_no_skills = Mock(spec=UserProfile)
        profile_no_skills.user_id = "test-user-no-skills"
        profile_no_skills.skills = {}
        profile_no_skills.current_role = None
        profile_no_skills.dream_job = None
        profile_no_skills.location = None
        profile_no_skills.remote_work_preference = False
        
        with patch('app.repositories.job.JobRepository') as mock_job_repo_class:
            mock_job_repo = AsyncMock()
            mock_job_repo_class.return_value = mock_job_repo
            mock_job_repo.search_jobs.return_value = mock_job_postings
            
            matches = await recommendation_service.get_advanced_job_matches(
                user_profile=profile_no_skills,
                match_threshold=0.0,  # Low threshold to allow matches
                db=mock_db
            )
            
            # Should still return jobs but with low match scores
            for match in matches:
                assert match["match_score"] == 0.0
                assert match["match_percentage"] == 0.0