"""
Tests for career trajectory recommendation service.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime
from typing import Dict, List

from app.services.career_trajectory_service import (
    CareerTrajectoryService,
    CareerTrajectoryRecommendation
)
from app.models.profile import UserProfile
from app.models.job import JobPosting
from app.core.exceptions import ServiceException


@pytest.fixture
def career_trajectory_service():
    """Create career trajectory service instance."""
    return CareerTrajectoryService()


@pytest.fixture
def mock_user_profile():
    """Create mock user profile."""
    profile = Mock(spec=UserProfile)
    profile.user_id = "test-user-123"
    profile.dream_job = "Senior Software Engineer"
    profile.current_role = "Software Engineer"
    profile.experience_years = 3
    profile.location = "San Francisco, CA"
    profile.skills = {
        "python": 0.8,
        "javascript": 0.7,
        "sql": 0.6,
        "git": 0.9,
        "react": 0.5
    }
    return profile


@pytest.fixture
def mock_job_postings():
    """Create mock job postings."""
    jobs = []
    
    # Senior Software Engineer job
    job1 = Mock(spec=JobPosting)
    job1.id = "job-1"
    job1.title = "Senior Software Engineer"
    job1.company = "Tech Corp"
    job1.description = "Senior software engineer role requiring Python, JavaScript, and system design skills"
    job1.processed_skills = {
        "python": 0.9,
        "javascript": 0.8,
        "system design": 0.7,
        "sql": 0.6,
        "docker": 0.5
    }
    job1.salary_min = 120000
    job1.salary_max = 160000
    job1.posted_date = datetime.utcnow()
    job1.is_active = True
    jobs.append(job1)
    
    # Data Scientist job
    job2 = Mock(spec=JobPosting)
    job2.id = "job-2"
    job2.title = "Data Scientist"
    job2.company = "Data Inc"
    job2.description = "Data scientist role requiring Python, machine learning, and statistics"
    job2.processed_skills = {
        "python": 0.9,
        "machine learning": 0.9,
        "statistics": 0.8,
        "pandas": 0.7,
        "sql": 0.6
    }
    job2.salary_min = 110000
    job2.salary_max = 150000
    job2.posted_date = datetime.utcnow()
    job2.is_active = True
    jobs.append(job2)
    
    return jobs


@pytest.mark.asyncio
class TestCareerTrajectoryService:
    """Test career trajectory service functionality."""
    
    async def test_get_career_trajectory_recommendations_success(
        self, career_trajectory_service, mock_user_profile, mock_job_postings
    ):
        """Test successful career trajectory recommendation generation."""
        # Mock dependencies
        with patch.object(career_trajectory_service, '_load_embedding_model') as mock_load_model, \
             patch('app.repositories.profile.ProfileRepository') as mock_profile_repo, \
             patch('app.repositories.job.JobRepository') as mock_job_repo, \
             patch.object(career_trajectory_service, 'embedding_model') as mock_embedding_model:
            
            # Setup mocks
            mock_load_model.return_value = None
            mock_profile_repo.return_value.get_by_user_id = AsyncMock(return_value=mock_user_profile)
            mock_job_repo.return_value.search_jobs = AsyncMock(return_value=mock_job_postings)
            mock_job_repo.return_value.get_recent_jobs = AsyncMock(return_value=mock_job_postings)
            
            # Mock embedding model
            mock_embedding_model.encode.return_value = [[0.1, 0.2, 0.3]]
            
            # Mock database session
            mock_db = AsyncMock()
            
            # Call service method
            trajectories = await career_trajectory_service.get_career_trajectory_recommendations(
                user_id="test-user-123",
                db=mock_db,
                n_recommendations=3,
                include_alternatives=True
            )
            
            # Assertions
            assert len(trajectories) > 0
            assert all(isinstance(t, CareerTrajectoryRecommendation) for t in trajectories)
            
            # Check first trajectory
            first_trajectory = trajectories[0]
            assert first_trajectory.target_role == "Senior Software Engineer"
            assert first_trajectory.confidence_score > 0
            assert first_trajectory.match_score >= 0
            assert len(first_trajectory.progression_steps) > 0
            assert len(first_trajectory.required_skills) > 0
    
    async def test_get_career_trajectory_recommendations_no_profile(
        self, career_trajectory_service
    ):
        """Test trajectory recommendations when user profile not found."""
        with patch('app.repositories.profile.ProfileRepository') as mock_profile_repo:
            mock_profile_repo.return_value.get_by_user_id = AsyncMock(return_value=None)
            mock_db = AsyncMock()
            
            with pytest.raises(ServiceException, match="User profile not found"):
                await career_trajectory_service.get_career_trajectory_recommendations(
                    user_id="nonexistent-user",
                    db=mock_db
                )
    
    async def test_analyze_skill_gaps_success(
        self, career_trajectory_service, mock_user_profile
    ):
        """Test successful skill gap analysis."""
        with patch('app.repositories.profile.ProfileRepository') as mock_profile_repo, \
             patch.object(career_trajectory_service, '_get_role_skill_requirements') as mock_get_skills:
            
            # Setup mocks
            mock_profile_repo.return_value.get_by_user_id = AsyncMock(return_value=mock_user_profile)
            mock_get_skills.return_value = {
                "python": 0.9,
                "system design": 0.8,
                "docker": 0.7,
                "kubernetes": 0.6
            }
            
            mock_db = AsyncMock()
            
            # Call service method
            gap_analysis = await career_trajectory_service.analyze_skill_gaps(
                user_id="test-user-123",
                target_role="Senior Software Engineer",
                db=mock_db
            )
            
            # Assertions
            assert gap_analysis['target_role'] == "Senior Software Engineer"
            assert 'missing_skills' in gap_analysis
            assert 'weak_skills' in gap_analysis
            assert 'strong_skills' in gap_analysis
            assert 'overall_readiness' in gap_analysis
            assert 'learning_time_estimate_weeks' in gap_analysis
            assert 'priority_skills' in gap_analysis
            assert gap_analysis['readiness_percentage'] >= 0
    
    async def test_get_job_match_score_success(
        self, career_trajectory_service, mock_user_profile, mock_job_postings
    ):
        """Test successful job match score calculation."""
        job_posting = mock_job_postings[0]
        
        with patch('app.repositories.profile.ProfileRepository') as mock_profile_repo, \
             patch('app.repositories.job.JobRepository') as mock_job_repo, \
             patch.object(career_trajectory_service, '_load_embedding_model') as mock_load_model, \
             patch.object(career_trajectory_service, 'embedding_model') as mock_embedding_model:
            
            # Setup mocks
            mock_profile_repo.return_value.get_by_user_id = AsyncMock(return_value=mock_user_profile)
            mock_job_repo.return_value.get_by_id = AsyncMock(return_value=job_posting)
            mock_load_model.return_value = None
            mock_embedding_model.encode.return_value = [[0.1, 0.2, 0.3]]
            
            mock_db = AsyncMock()
            
            # Call service method
            match_analysis = await career_trajectory_service.get_job_match_score(
                user_id="test-user-123",
                job_id="job-1",
                db=mock_db
            )
            
            # Assertions
            assert match_analysis['job_id'] == "job-1"
            assert match_analysis['job_title'] == "Senior Software Engineer"
            assert match_analysis['company'] == "Tech Corp"
            assert 'match_score' in match_analysis
            assert 'match_percentage' in match_analysis
            assert 'skill_gaps' in match_analysis
            assert 'strong_skills' in match_analysis
            assert match_analysis['match_percentage'] >= 0
    
    async def test_dream_job_trajectory_generation(
        self, career_trajectory_service, mock_user_profile, mock_job_postings
    ):
        """Test dream job trajectory generation."""
        with patch.object(career_trajectory_service, '_load_embedding_model') as mock_load_model, \
             patch.object(career_trajectory_service, 'embedding_model') as mock_embedding_model, \
             patch('app.repositories.job.JobRepository') as mock_job_repo:
            
            # Setup mocks
            mock_load_model.return_value = None
            mock_embedding_model.encode.return_value = [[0.1, 0.2, 0.3]]
            mock_job_repo.return_value.search_jobs = AsyncMock(return_value=mock_job_postings)
            
            mock_db = AsyncMock()
            
            # Call private method
            trajectory = await career_trajectory_service._generate_dream_job_trajectory(
                user_profile=mock_user_profile,
                user_skills=mock_user_profile.skills,
                dream_job="Senior Software Engineer",
                db=mock_db
            )
            
            # Assertions
            assert trajectory is not None
            assert trajectory.target_role == "Senior Software Engineer"
            assert trajectory.title == "Path to Senior Software Engineer"
            assert len(trajectory.progression_steps) > 0
            assert trajectory.confidence_score > 0
    
    async def test_market_demand_data_caching(
        self, career_trajectory_service, mock_job_postings
    ):
        """Test market demand data caching functionality."""
        with patch('app.repositories.job.JobRepository') as mock_job_repo:
            mock_job_repo.return_value.search_jobs = AsyncMock(return_value=mock_job_postings)
            mock_db = AsyncMock()
            
            # First call - should fetch from database
            market_data1 = await career_trajectory_service._get_market_demand_data(
                "Software Engineer", mock_db
            )
            
            # Second call - should use cache
            market_data2 = await career_trajectory_service._get_market_demand_data(
                "Software Engineer", mock_db
            )
            
            # Assertions
            assert market_data1 == market_data2
            assert 'demand_level' in market_data1
            assert 'growth_potential' in market_data1
            assert 'job_count' in market_data1
            
            # Verify cache was used (search_jobs should only be called once)
            mock_job_repo.return_value.search_jobs.assert_called_once()
    
    def test_calculate_confidence_score(self, career_trajectory_service):
        """Test confidence score calculation."""
        similarity = 0.8
        readiness = 0.7
        market_data = {
            'demand_level': 'high',
            'growth_potential': 0.8
        }
        
        confidence = career_trajectory_service._calculate_confidence_score(
            similarity, readiness, market_data
        )
        
        assert 0 <= confidence <= 1
        assert confidence > 0.5  # Should be reasonably high with good inputs
    
    def test_assess_difficulty_level(self, career_trajectory_service):
        """Test difficulty level assessment."""
        # Mock gap analysis
        gap_analysis = Mock()
        
        # Easy case
        gap_analysis.overall_readiness = 0.9
        gap_analysis.learning_time_estimate = 8
        difficulty = career_trajectory_service._assess_difficulty_level(gap_analysis)
        assert difficulty == "easy"
        
        # Moderate case
        gap_analysis.overall_readiness = 0.7
        gap_analysis.learning_time_estimate = 16
        difficulty = career_trajectory_service._assess_difficulty_level(gap_analysis)
        assert difficulty == "moderate"
        
        # Difficult case
        gap_analysis.overall_readiness = 0.3
        gap_analysis.learning_time_estimate = 60
        difficulty = career_trajectory_service._assess_difficulty_level(gap_analysis)
        assert difficulty == "difficult"
    
    def test_generate_trajectory_reasoning(self, career_trajectory_service):
        """Test trajectory reasoning generation."""
        gap_analysis = Mock()
        gap_analysis.overall_readiness = 0.8
        gap_analysis.strong_skills = ["python", "javascript", "sql"]
        gap_analysis.missing_skills = {"docker": 0.7, "kubernetes": 0.6}
        gap_analysis.learning_time_estimate = 16
        
        market_data = {
            'demand_level': 'high',
            'growth_potential': 0.8
        }
        
        reasoning = career_trajectory_service._generate_trajectory_reasoning(
            "Senior Software Engineer", gap_analysis, market_data, "dream_job"
        )
        
        assert isinstance(reasoning, str)
        assert len(reasoning) > 0
        assert "Senior Software Engineer" in reasoning
    
    def test_identify_success_factors(self, career_trajectory_service):
        """Test success factors identification."""
        gap_analysis = Mock()
        gap_analysis.priority_skills = ["system design", "docker", "kubernetes"]
        gap_analysis.strong_skills = ["python", "javascript", "sql", "git"]
        
        market_data = {'demand_level': 'high'}
        
        factors = career_trajectory_service._identify_success_factors(gap_analysis, market_data)
        
        assert isinstance(factors, list)
        assert len(factors) <= 5
        assert all(isinstance(factor, str) for factor in factors)
    
    def test_identify_challenges(self, career_trajectory_service):
        """Test challenges identification."""
        gap_analysis = Mock()
        gap_analysis.missing_skills = {f"skill_{i}": 0.5 for i in range(6)}  # 6 missing skills
        gap_analysis.learning_time_estimate = 50  # Long learning time
        gap_analysis.overall_readiness = 0.4  # Low readiness
        
        market_data = {'demand_level': 'low'}
        
        challenges = career_trajectory_service._identify_challenges(gap_analysis, market_data)
        
        assert isinstance(challenges, list)
        assert len(challenges) <= 4
        assert all(isinstance(challenge, str) for challenge in challenges)
    
    def test_get_default_role_skills(self, career_trajectory_service):
        """Test default role skills retrieval."""
        # Test known role
        skills = career_trajectory_service._get_default_role_skills("software engineer")
        assert isinstance(skills, dict)
        assert "python" in skills
        assert all(0 <= score <= 1 for score in skills.values())
        
        # Test unknown role
        skills = career_trajectory_service._get_default_role_skills("unknown role")
        assert isinstance(skills, dict)
        assert "communication" in skills
        assert all(0 <= score <= 1 for score in skills.values())
    
    def test_suggest_intermediate_role(self, career_trajectory_service):
        """Test intermediate role suggestion."""
        # Test known progression
        intermediate = career_trajectory_service._suggest_intermediate_role(
            "junior developer", "senior developer"
        )
        assert intermediate == "developer"
        
        # Test default case
        intermediate = career_trajectory_service._suggest_intermediate_role(
            "analyst", "senior manager"
        )
        assert "senior" in intermediate.lower()
    
    def test_estimate_trajectory_timeline(self, career_trajectory_service):
        """Test trajectory timeline estimation."""
        progression_steps = [
            {'duration_months': 6},
            {'duration_months': 12},
            {'duration_months': 18}
        ]
        learning_weeks = 24
        
        timeline = career_trajectory_service._estimate_trajectory_timeline(
            progression_steps, learning_weeks
        )
        
        assert timeline >= 36  # Sum of step durations
        assert timeline >= 6   # Minimum learning time
    
    async def test_alternative_routes_discovery(
        self, career_trajectory_service, mock_user_profile
    ):
        """Test alternative routes discovery."""
        # Create a mock trajectory
        trajectory = Mock()
        trajectory.target_role = "Senior Software Engineer"
        
        mock_db = AsyncMock()
        
        with patch.object(career_trajectory_service, '_discover_alternative_paths') as mock_discover, \
             patch.object(career_trajectory_service, '_plan_alternative_progression') as mock_plan:
            
            mock_discover.return_value = [
                {
                    'approach': 'Bootcamp Route',
                    'description': 'Intensive bootcamp program',
                    'advantages': ['Fast learning'],
                    'considerations': ['Intensive'],
                    'success_rate': 0.7
                }
            ]
            
            mock_plan.return_value = [
                {
                    'role': 'Bootcamp Student',
                    'duration_months': 6,
                    'description': 'Learning phase',
                    'key_activities': ['Study', 'Practice']
                }
            ]
            
            alternatives = await career_trajectory_service._find_alternative_routes(
                trajectory, mock_user_profile.skills, mock_db
            )
            
            assert len(alternatives) > 0
            assert alternatives[0]['approach'] == 'Bootcamp Route'
            assert 'progression_steps' in alternatives[0]


@pytest.mark.asyncio
class TestCareerTrajectoryServiceIntegration:
    """Integration tests for career trajectory service."""
    
    async def test_full_trajectory_generation_workflow(
        self, career_trajectory_service, mock_user_profile, mock_job_postings
    ):
        """Test complete trajectory generation workflow."""
        with patch.object(career_trajectory_service, '_load_embedding_model') as mock_load_model, \
             patch('app.repositories.profile.ProfileRepository') as mock_profile_repo, \
             patch('app.repositories.job.JobRepository') as mock_job_repo, \
             patch.object(career_trajectory_service, 'embedding_model') as mock_embedding_model:
            
            # Setup comprehensive mocks
            mock_load_model.return_value = None
            mock_profile_repo.return_value.get_by_user_id = AsyncMock(return_value=mock_user_profile)
            mock_job_repo.return_value.search_jobs = AsyncMock(return_value=mock_job_postings)
            mock_job_repo.return_value.get_recent_jobs = AsyncMock(return_value=mock_job_postings)
            mock_embedding_model.encode.return_value = [[0.1, 0.2, 0.3]]
            
            mock_db = AsyncMock()
            
            # Generate trajectories
            trajectories = await career_trajectory_service.get_career_trajectory_recommendations(
                user_id="test-user-123",
                db=mock_db,
                n_recommendations=5,
                include_alternatives=True
            )
            
            # Verify comprehensive trajectory structure
            assert len(trajectories) > 0
            
            for trajectory in trajectories:
                # Basic properties
                assert trajectory.trajectory_id
                assert trajectory.title
                assert trajectory.target_role
                assert 0 <= trajectory.match_score <= 1
                assert 0 <= trajectory.confidence_score <= 1
                
                # Progression details
                assert len(trajectory.progression_steps) > 0
                assert trajectory.estimated_timeline_months > 0
                assert trajectory.difficulty_level in ["easy", "moderate", "challenging", "difficult"]
                
                # Skills analysis
                assert isinstance(trajectory.required_skills, list)
                assert isinstance(trajectory.skill_gaps, dict)
                assert isinstance(trajectory.transferable_skills, list)
                
                # Market analysis
                assert trajectory.market_demand in ["low", "moderate", "high"]
                assert isinstance(trajectory.salary_progression, dict)
                assert 0 <= trajectory.growth_potential <= 1
                
                # Guidance
                assert trajectory.reasoning
                assert isinstance(trajectory.success_factors, list)
                assert isinstance(trajectory.potential_challenges, list)
                
                # Metadata
                assert trajectory.recommendation_date
                assert isinstance(trajectory.data_sources, list)
    
    async def test_error_handling_and_recovery(
        self, career_trajectory_service, mock_user_profile
    ):
        """Test error handling and graceful degradation."""
        with patch('app.repositories.profile.ProfileRepository') as mock_profile_repo, \
             patch('app.repositories.job.JobRepository') as mock_job_repo:
            
            # Setup profile repo to succeed
            mock_profile_repo.return_value.get_by_user_id = AsyncMock(return_value=mock_user_profile)
            
            # Setup job repo to fail
            mock_job_repo.return_value.search_jobs = AsyncMock(side_effect=Exception("Database error"))
            mock_job_repo.return_value.get_recent_jobs = AsyncMock(return_value=[])
            
            mock_db = AsyncMock()
            
            # Should still generate trajectories using default data
            trajectories = await career_trajectory_service.get_career_trajectory_recommendations(
                user_id="test-user-123",
                db=mock_db,
                n_recommendations=3
            )
            
            # Should have at least some trajectories despite errors
            assert len(trajectories) > 0