"""
Tests for the recommendation service.
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timedelta
import numpy as np

from app.services.recommendation_service import RecommendationService
from app.models.user import User
from app.models.profile import UserProfile
from app.models.job import JobPosting
from app.core.exceptions import ServiceException


class TestRecommendationService:
    """Test recommendation service functionality."""
    
    @pytest.fixture
    def recommendation_service(self):
        """Create recommendation service instance."""
        return RecommendationService()
    
    @pytest.fixture
    def mock_db(self):
        """Mock database session."""
        return AsyncMock()
    
    @pytest.fixture
    def sample_user_profile(self):
        """Sample user profile for testing."""
        profile = UserProfile()
        profile.user_id = "user123"
        profile.current_role = "Junior Developer"
        profile.dream_job = "Senior Python Developer"
        profile.experience_years = 3
        profile.skills = {
            "python": 0.8,
            "sql": 0.6,
            "javascript": 0.4,
            "git": 0.7
        }
        profile.location = "San Francisco"
        return profile
    
    @pytest.fixture
    def sample_job_posting(self):
        """Sample job posting for testing."""
        job = JobPosting()
        job.id = "job123"
        job.title = "Senior Python Developer"
        job.company = "Tech Corp"
        job.description = "Develop web applications using Python and Django"
        job.processed_skills = {
            "python": 0.9,
            "django": 0.8,
            "postgresql": 0.7,
            "redis": 0.5
        }
        job.is_active = True
        return job
    
    @pytest.mark.asyncio
    async def test_initialization(self, recommendation_service):
        """Test service initialization."""
        assert recommendation_service.recommendation_engine is not None
        assert recommendation_service.skill_gap_analyzer is not None
        assert recommendation_service.explainer is not None
        assert not recommendation_service.model_trained
        assert recommendation_service.last_training_time is None
    
    @pytest.mark.asyncio
    async def test_initialize_and_train_models_insufficient_data(self, recommendation_service, mock_db):
        """Test model training with insufficient data."""
        with patch.object(recommendation_service, '_fetch_user_data', return_value=[]):
            with patch.object(recommendation_service, '_fetch_job_data', return_value=[]):
                await recommendation_service.initialize_and_train_models(mock_db)
                assert not recommendation_service.model_trained
    
    @pytest.mark.asyncio
    async def test_initialize_and_train_models_success(self, recommendation_service, mock_db):
        """Test successful model training."""
        # Mock data
        user_data = [
            {'id': 'user1', 'current_role': 'Developer', 'skills': ['python', 'sql']},
            {'id': 'user2', 'current_role': 'Analyst', 'skills': ['sql', 'excel']}
        ]
        job_data = [
            {'id': 'job1', 'title': 'Developer', 'skills': ['python', 'django']},
            {'id': 'job2', 'title': 'Analyst', 'skills': ['sql', 'tableau']}
        ]
        user_item_matrix = np.array([[4, 2], [1, 5]], dtype=float)
        user_ids = ['user1', 'user2']
        
        with patch.object(recommendation_service, '_fetch_user_data', return_value=user_data):
            with patch.object(recommendation_service, '_fetch_job_data', return_value=job_data):
                with patch.object(recommendation_service, '_build_user_item_matrix', return_value=(user_item_matrix, user_ids)):
                    with patch.object(recommendation_service.recommendation_engine, 'fit') as mock_fit:
                        await recommendation_service.initialize_and_train_models(mock_db)
                        
                        mock_fit.assert_called_once_with(user_item_matrix, user_ids, job_data, user_data)
                        assert recommendation_service.model_trained
                        assert recommendation_service.last_training_time is not None
    
    @pytest.mark.asyncio
    async def test_get_career_recommendations_no_profile(self, recommendation_service, mock_db):
        """Test career recommendations with no user profile."""
        with patch('app.repositories.profile.ProfileRepository') as mock_repo_class:
            mock_repo = mock_repo_class.return_value
            mock_repo.get_by_user_id.return_value = None
            
            with pytest.raises(ServiceException, match="User profile not found"):
                await recommendation_service.get_career_recommendations("user123", mock_db)
    
    @pytest.mark.asyncio
    async def test_get_career_recommendations_success(self, recommendation_service, mock_db, sample_user_profile):
        """Test successful career recommendations."""
        # Mock dependencies
        with patch('app.repositories.profile.ProfileRepository') as mock_repo_class:
            mock_repo = mock_repo_class.return_value
            mock_repo.get_by_user_id.return_value = sample_user_profile
            
            with patch.object(recommendation_service, 'initialize_and_train_models'):
                recommendation_service.model_trained = True
                
                with patch.object(recommendation_service.recommendation_engine, 'recommend_careers') as mock_recommend:
                    from machinelearningmodel.recommendation_engine import CareerRecommendation
                    mock_recommendation = CareerRecommendation(
                        job_title="Senior Python Developer",
                        match_score=0.85,
                        required_skills=["python", "django"],
                        skill_gaps={"django": 0.3},
                        salary_range=(80000, 120000),
                        growth_potential=0.8,
                        market_demand="high",
                        reasoning="Good match",
                        confidence_score=0.85,
                        alternative_paths=[]
                    )
                    mock_recommend.return_value = [mock_recommendation]
                    
                    recommendations = await recommendation_service.get_career_recommendations("user123", mock_db)
                    
                    assert len(recommendations) == 1
                    assert recommendations[0]['job_title'] == "Senior Senior Python Developer"
                    assert recommendations[0]['match_score'] == 0.85
    
    @pytest.mark.asyncio
    async def test_analyze_skill_gaps(self, recommendation_service, mock_db, sample_user_profile):
        """Test skill gap analysis."""
        with patch('app.repositories.profile.ProfileRepository') as mock_repo_class:
            mock_repo = mock_repo_class.return_value
            mock_repo.get_by_user_id.return_value = sample_user_profile
            
            with patch.object(recommendation_service, '_get_target_role_skills') as mock_target_skills:
                mock_target_skills.return_value = {
                    "python": 0.9,
                    "django": 0.8,
                    "postgresql": 0.7
                }
                
                with patch.object(recommendation_service.skill_gap_analyzer, 'analyze_skill_gaps') as mock_analyze:
                    from machinelearningmodel.recommendation_engine import SkillGapAnalysis
                    mock_analysis = SkillGapAnalysis(
                        target_role="Senior Python Developer",
                        missing_skills={"django": 0.8, "postgresql": 0.7},
                        weak_skills={},
                        strong_skills=["python"],
                        overall_readiness=0.6,
                        learning_time_estimate=12,
                        priority_skills=["django", "postgresql"]
                    )
                    mock_analyze.return_value = mock_analysis
                    
                    analysis = await recommendation_service.analyze_skill_gaps("user123", "Senior Python Developer", mock_db)
                    
                    assert analysis['target_role'] == "Senior Python Developer"
                    assert analysis['overall_readiness'] == 0.6
                    assert analysis['readiness_percentage'] == 60.0
                    assert "django" in analysis['missing_skills']
                    assert "python" in analysis['strong_skills']
    
    @pytest.mark.asyncio
    async def test_get_job_match_score(self, recommendation_service, mock_db, sample_user_profile, sample_job_posting):
        """Test job match score calculation."""
        with patch('app.repositories.profile.ProfileRepository') as mock_profile_repo_class:
            with patch('app.repositories.job.JobRepository') as mock_job_repo_class:
                mock_profile_repo = mock_profile_repo_class.return_value
                mock_job_repo = mock_job_repo_class.return_value
                
                mock_profile_repo.get_by_user_id.return_value = sample_user_profile
                mock_job_repo.get_by_id.return_value = sample_job_posting
                
                with patch.object(recommendation_service.skill_gap_analyzer, 'analyze_skill_gaps') as mock_analyze:
                    from machinelearningmodel.recommendation_engine import SkillGapAnalysis
                    mock_analysis = SkillGapAnalysis(
                        target_role="Senior Python Developer",
                        missing_skills={"django": 0.8, "postgresql": 0.7},
                        weak_skills={},
                        strong_skills=["python"],
                        overall_readiness=0.7,
                        learning_time_estimate=8,
                        priority_skills=["django", "postgresql"]
                    )
                    mock_analyze.return_value = mock_analysis
                    
                    match_result = await recommendation_service.get_job_match_score("user123", "job123", mock_db)
                    
                    assert match_result['job_id'] == "job123"
                    assert match_result['job_title'] == "Senior Python Developer"
                    assert match_result['company'] == "Tech Corp"
                    assert 'match_score' in match_result
                    assert 'match_percentage' in match_result
                    assert match_result['overall_readiness'] == 0.7
    
    @pytest.mark.asyncio
    async def test_get_learning_path_recommendations(self, recommendation_service, mock_db, sample_user_profile):
        """Test learning path recommendations."""
        with patch('app.repositories.profile.ProfileRepository') as mock_repo_class:
            mock_repo = mock_repo_class.return_value
            mock_repo.get_by_user_id.return_value = sample_user_profile
            
            with patch.object(recommendation_service, '_get_target_role_skills') as mock_target_skills:
                mock_target_skills.return_value = {
                    "python": 0.9,
                    "django": 0.8,
                    "postgresql": 0.7
                }
                
                with patch.object(recommendation_service.recommendation_engine, 'recommend_learning_paths') as mock_recommend:
                    from machinelearningmodel.recommendation_engine import LearningPath
                    mock_path = LearningPath(
                        path_id="path1",
                        title="Master Django Development",
                        target_skills=["django"],
                        estimated_duration_weeks=8,
                        difficulty_level="intermediate",
                        resources=[],
                        milestones=[],
                        priority_score=0.9,
                        reasoning="Django is critical for your target role"
                    )
                    mock_recommend.return_value = [mock_path]
                    
                    with patch.object(recommendation_service, '_enrich_learning_path') as mock_enrich:
                        mock_enrich.return_value = {
                            'path_id': 'path1',
                            'title': 'Master Django Development',
                            'target_skills': ['django'],
                            'estimated_duration_weeks': 8,
                            'difficulty_level': 'intermediate',
                            'priority_score': 0.9,
                            'reasoning': 'Django is critical for your target role',
                            'resources': [],
                            'milestones': [],
                            'created_date': datetime.utcnow().isoformat()
                        }
                        
                        paths = await recommendation_service.get_learning_path_recommendations(
                            "user123", "Senior Python Developer", mock_db
                        )
                        
                        assert len(paths) == 1
                        assert paths[0]['title'] == "Master Django Development"
                        assert paths[0]['target_skills'] == ["django"]
    
    @pytest.mark.asyncio
    async def test_fetch_user_data(self, recommendation_service, mock_db):
        """Test fetching user data for training."""
        # Mock database query result
        mock_profiles = [
            Mock(
                user_id="user1",
                current_role="Developer",
                dream_job="Senior Developer",
                skills={"python": 0.8, "sql": 0.6},
                experience_years=3
            ),
            Mock(
                user_id="user2",
                current_role="Analyst",
                dream_job="Data Scientist",
                skills={"sql": 0.7, "excel": 0.5},
                experience_years=2
            )
        ]
        
        mock_result = Mock()
        mock_result.scalars.return_value.all.return_value = mock_profiles
        mock_db.execute.return_value = mock_result
        
        user_data = await recommendation_service._fetch_user_data(mock_db)
        
        assert len(user_data) == 2
        assert user_data[0]['id'] == "user1"
        assert user_data[0]['current_role'] == "Developer"
        assert user_data[0]['skills'] == ["python", "sql"]
    
    @pytest.mark.asyncio
    async def test_fetch_job_data(self, recommendation_service, mock_db):
        """Test fetching job data for training."""
        # Mock database query result
        mock_jobs = [
            Mock(
                id="job1",
                title="Python Developer",
                description="Develop web applications",
                processed_skills={"python": 0.9, "django": 0.8},
                company="Tech Corp",
                location="San Francisco",
                experience_level="mid"
            ),
            Mock(
                id="job2",
                title="Data Analyst",
                description="Analyze business data",
                processed_skills={"sql": 0.8, "excel": 0.7},
                company="Data Inc",
                location="Remote",
                experience_level="entry"
            )
        ]
        
        mock_result = Mock()
        mock_result.scalars.return_value.all.return_value = mock_jobs
        mock_db.execute.return_value = mock_result
        
        job_data = await recommendation_service._fetch_job_data(mock_db)
        
        assert len(job_data) == 2
        assert job_data[0]['id'] == "job1"
        assert job_data[0]['title'] == "Python Developer"
        assert job_data[0]['skills'] == ["python", "django"]
    
    @pytest.mark.asyncio
    async def test_build_user_item_matrix(self, recommendation_service, mock_db):
        """Test building user-item interaction matrix."""
        user_data = [
            {'id': 'user1', 'skills': ['python', 'sql']},
            {'id': 'user2', 'skills': ['sql', 'excel']}
        ]
        job_data = [
            {'id': 'job1', 'skills': ['python', 'django']},
            {'id': 'job2', 'skills': ['sql', 'tableau']}
        ]
        
        with patch.object(recommendation_service, '_fetch_user_data', return_value=user_data):
            with patch.object(recommendation_service, '_fetch_job_data', return_value=job_data):
                matrix, user_ids = await recommendation_service._build_user_item_matrix(mock_db)
                
                assert matrix.shape == (2, 2)  # 2 users, 2 jobs
                assert user_ids == ['user1', 'user2']
                assert np.all(matrix >= 1)  # All ratings should be >= 1
                assert np.all(matrix <= 5)  # All ratings should be <= 5
    
    def test_get_default_role_skills(self, recommendation_service):
        """Test getting default skills for common roles."""
        # Test known role
        skills = recommendation_service._get_default_role_skills("Software Engineer")
        assert "python" in skills
        assert "git" in skills
        assert skills["python"] > 0
        
        # Test data scientist role
        skills = recommendation_service._get_default_role_skills("Data Scientist")
        assert "python" in skills
        assert "machine learning" in skills
        assert "statistics" in skills
        
        # Test unknown role
        skills = recommendation_service._get_default_role_skills("Unknown Role")
        assert "communication" in skills
        assert "problem solving" in skills
    
    def test_calculate_simple_match_score(self, recommendation_service):
        """Test simple match score calculation."""
        user_skills = {"python": 0.8, "sql": 0.6, "javascript": 0.4}
        job_skills = {"python": 0.9, "sql": 0.7, "django": 0.8}
        
        score = recommendation_service._calculate_simple_match_score(user_skills, job_skills)
        
        assert 0 <= score <= 1
        assert score > 0  # Should have some overlap
        
        # Test with no overlap
        user_skills_no_overlap = {"html": 0.8, "css": 0.6}
        score_no_overlap = recommendation_service._calculate_simple_match_score(user_skills_no_overlap, job_skills)
        assert score_no_overlap == 0
        
        # Test with empty skills
        score_empty = recommendation_service._calculate_simple_match_score({}, job_skills)
        assert score_empty == 0
    
    @pytest.mark.asyncio
    async def test_get_target_role_skills_from_db(self, recommendation_service, mock_db):
        """Test getting target role skills from database."""
        # Mock job postings with similar titles
        mock_jobs = [
            Mock(processed_skills={"python": 0.9, "django": 0.8, "sql": 0.7}),
            Mock(processed_skills={"python": 0.8, "flask": 0.6, "sql": 0.8}),
            Mock(processed_skills={"python": 0.9, "fastapi": 0.7, "postgresql": 0.6})
        ]
        
        mock_result = Mock()
        mock_result.scalars.return_value.all.return_value = mock_jobs
        mock_db.execute.return_value = mock_result
        
        target_skills = await recommendation_service._get_target_role_skills("Python Developer", mock_db)
        
        assert "python" in target_skills
        assert "sql" in target_skills
        assert target_skills["python"] > target_skills.get("flask", 0)  # Python should be more important
    
    @pytest.mark.asyncio
    async def test_enrich_learning_path(self, recommendation_service, mock_db):
        """Test enriching learning path with resources."""
        from machinelearningmodel.recommendation_engine import LearningPath
        
        path = LearningPath(
            path_id="path1",
            title="Master Python",
            target_skills=["python"],
            estimated_duration_weeks=8,
            difficulty_level="intermediate",
            resources=[],
            milestones=[],
            priority_score=0.9,
            reasoning="Python is essential"
        )
        
        enriched_path = await recommendation_service._enrich_learning_path(path, mock_db)
        
        assert enriched_path['path_id'] == "path1"
        assert enriched_path['title'] == "Master Python"
        assert len(enriched_path['resources']) > 0
        assert len(enriched_path['milestones']) > 0
        assert 'created_date' in enriched_path
    
    @pytest.mark.asyncio
    async def test_service_exception_handling(self, recommendation_service, mock_db):
        """Test service exception handling."""
        with patch('app.repositories.profile.ProfileRepository') as mock_repo_class:
            mock_repo = mock_repo_class.return_value
            mock_repo.get_by_user_id.side_effect = Exception("Database error")
            
            with pytest.raises(ServiceException):
                await recommendation_service.get_career_recommendations("user123", mock_db)
    
    def test_training_interval_check(self, recommendation_service):
        """Test training interval checking."""
        # Set last training time to recent
        recommendation_service.model_trained = True
        recommendation_service.last_training_time = datetime.utcnow() - timedelta(hours=1)
        
        # Should not need retraining
        assert (datetime.utcnow() - recommendation_service.last_training_time) < recommendation_service.training_interval
        
        # Set last training time to old
        recommendation_service.last_training_time = datetime.utcnow() - timedelta(hours=25)
        
        # Should need retraining
        assert (datetime.utcnow() - recommendation_service.last_training_time) >= recommendation_service.training_interval


@pytest.mark.asyncio
async def test_integration_recommendation_flow():
    """Integration test for complete recommendation flow."""
    service = RecommendationService()
    mock_db = AsyncMock()
    
    # Mock user profile
    user_profile = UserProfile()
    user_profile.user_id = "user123"
    user_profile.skills = {"python": 0.8, "sql": 0.6}
    user_profile.dream_job = "Data Scientist"
    
    # Mock repositories
    with patch('app.repositories.profile.ProfileRepository') as mock_profile_repo:
        mock_profile_repo.return_value.get_by_user_id.return_value = user_profile
        
        with patch.object(service, 'initialize_and_train_models'):
            service.model_trained = True
            
            with patch.object(service.recommendation_engine, 'recommend_careers') as mock_recommend:
                from machinelearningmodel.recommendation_engine import CareerRecommendation
                mock_rec = CareerRecommendation(
                    job_title="Data Scientist",
                    match_score=0.85,
                    required_skills=["python", "machine learning"],
                    skill_gaps={"machine learning": 0.3},
                    salary_range=(90000, 130000),
                    growth_potential=0.9,
                    market_demand="high",
                    reasoning="Strong Python skills",
                    confidence_score=0.85,
                    alternative_paths=[]
                )
                mock_recommend.return_value = [mock_rec]
                
                recommendations = await service.get_career_recommendations("user123", mock_db)
                
                assert len(recommendations) == 1
                assert recommendations[0]['match_score'] == 0.85
                assert "python" in recommendations[0]['required_skills']