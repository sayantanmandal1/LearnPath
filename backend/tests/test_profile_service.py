"""
Unit tests for UserProfileService - comprehensive testing of profile aggregation accuracy.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timedelta
from typing import Dict, Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.services.profile_service import UserProfileService, SkillMergeResult, ProfileChangeTracker
from app.schemas.profile import ProfileCreate, ProfileUpdate
from app.models.profile import UserProfile
from app.core.exceptions import ValidationError, NotFoundError, ConflictError


class TestSkillMergeResult:
    """Test SkillMergeResult class."""
    
    def test_skill_merge_result_initialization(self):
        """Test SkillMergeResult initialization."""
        result = SkillMergeResult()
        
        assert result.merged_skills == {}
        assert result.conflicts == []
        assert result.sources == {}
        assert result.confidence_scores == {}


class TestProfileChangeTracker:
    """Test ProfileChangeTracker class."""
    
    def test_change_tracker_initialization(self):
        """Test ProfileChangeTracker initialization."""
        tracker = ProfileChangeTracker()
        assert tracker.changes == []
    
    def test_track_change(self):
        """Test tracking a field change."""
        tracker = ProfileChangeTracker()
        
        tracker.track_change('skills', {'python': 0.5}, {'python': 0.8}, 'manual_update')
        
        changes = tracker.get_changes()
        assert len(changes) == 1
        
        change = changes[0]
        assert change['field'] == 'skills'
        assert change['old_value'] == {'python': 0.5}
        assert change['new_value'] == {'python': 0.8}
        assert change['source'] == 'manual_update'
        assert isinstance(change['timestamp'], datetime)
    
    def test_multiple_changes(self):
        """Test tracking multiple changes."""
        tracker = ProfileChangeTracker()
        
        tracker.track_change('experience_years', 2, 3, 'manual_update')
        tracker.track_change('location', 'New York', 'San Francisco', 'manual_update')
        
        changes = tracker.get_changes()
        assert len(changes) == 2
        
        # Verify changes are independent
        assert changes[0]['field'] != changes[1]['field']


class TestUserProfileService:
    """Test UserProfileService class."""
    
    @pytest.fixture
    def mock_profile_repo(self):
        """Mock ProfileRepository."""
        return Mock()
    
    @pytest.fixture
    def mock_external_api_service(self):
        """Mock ExternalAPIIntegrationService."""
        return Mock()
    
    @pytest.fixture
    def mock_nlp_engine(self):
        """Mock NLPEngine."""
        return Mock()
    
    @pytest.fixture
    def profile_service(self, mock_profile_repo, mock_external_api_service, mock_nlp_engine):
        """Create UserProfileService with mocked dependencies."""
        service = UserProfileService()
        service.profile_repo = mock_profile_repo
        service.external_api_service = mock_external_api_service
        service.nlp_engine = mock_nlp_engine
        return service
    
    @pytest.fixture
    def sample_profile_data(self):
        """Sample profile creation data."""
        return ProfileCreate(
            dream_job="Software Engineer",
            experience_years=3,
            current_role="Junior Developer",
            location="New York",
            github_username="testuser",
            leetcode_id="testuser",
            linkedin_url="https://linkedin.com/in/testuser",
            skills={"Python": 0.8, "JavaScript": 0.6},
            career_interests={"web_development": 0.9}
        )
    
    @pytest.fixture
    def sample_user_profile(self):
        """Sample UserProfile model instance."""
        return UserProfile(
            id="profile-123",
            user_id="user-123",
            dream_job="Software Engineer",
            experience_years=3,
            current_role="Junior Developer",
            location="New York",
            github_username="testuser",
            leetcode_id="testuser",
            linkedin_url="https://linkedin.com/in/testuser",
            skills={"Python": 0.8, "JavaScript": 0.6},
            platform_data={},
            resume_data={},
            career_interests={"web_development": 0.9},
            skill_gaps={},
            data_last_updated=datetime.utcnow(),
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
    
    def test_service_initialization(self):
        """Test UserProfileService initialization."""
        service = UserProfileService(github_token="test-token")
        
        assert service.skill_confidence_weights is not None
        assert service.conflict_resolution_strategies is not None
        assert 'resume' in service.skill_confidence_weights
        assert 'github' in service.skill_confidence_weights
        assert 'skills' in service.conflict_resolution_strategies
    
    def test_skill_confidence_weights(self, profile_service):
        """Test skill confidence weights configuration."""
        weights = profile_service.skill_confidence_weights
        
        assert weights['manual'] == 1.0  # Highest confidence for manual input
        assert weights['resume'] == 0.9  # High confidence for resume
        assert weights['github'] > weights['leetcode']  # GitHub more reliable than LeetCode
        assert all(0 <= weight <= 1.0 for weight in weights.values())
    
    @pytest.mark.asyncio
    async def test_create_profile_with_integration_success(self, profile_service, sample_profile_data):
        """Test successful profile creation with multi-source integration."""
        # Mock database session
        mock_db = Mock(spec=AsyncSession)
        
        # Mock repository methods
        profile_service.profile_repo.get_by_user_id = AsyncMock(return_value=None)  # No existing profile
        profile_service.profile_repo.create = AsyncMock(return_value=Mock(id="profile-123"))
        profile_service.profile_repo.update = AsyncMock(return_value=Mock(
            id="profile-123",
            user_id="user-123",
            skills={"Python": 0.85, "JavaScript": 0.65, "React": 0.7}
        ))
        
        # Mock external API extraction
        mock_extraction_result = Mock()
        mock_extraction_result.success = True
        mock_extraction_result.github_profile = {"languages": {"Python": 1000, "JavaScript": 500}}
        mock_extraction_result.leetcode_profile = {"skill_tags": {"algorithms": 10, "data-structures": 8}}
        mock_extraction_result.linkedin_profile = {"skills": [{"name": "React", "endorsements": 5}]}
        
        profile_service._extract_external_platform_data = AsyncMock(return_value=mock_extraction_result)
        profile_service._generate_unified_profile = AsyncMock(return_value={
            'skills': {"Python": 0.85, "JavaScript": 0.65, "React": 0.7},
            'career_interests': {"web_development": 0.9},
            'skill_gaps': {}
        })
        
        # Execute
        result = await profile_service.create_profile_with_integration(
            db=mock_db,
            user_id="user-123",
            profile_data=sample_profile_data
        )
        
        # Verify
        assert result is not None
        profile_service.profile_repo.create.assert_called_once()
        profile_service.profile_repo.update.assert_called()
        profile_service._extract_external_platform_data.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_create_profile_existing_profile_conflict(self, profile_service, sample_profile_data, sample_user_profile):
        """Test profile creation when profile already exists."""
        mock_db = Mock(spec=AsyncSession)
        
        # Mock existing profile
        profile_service.profile_repo.get_by_user_id = AsyncMock(return_value=sample_user_profile)
        
        # Execute and verify exception
        with pytest.raises(ConflictError, match="Profile already exists"):
            await profile_service.create_profile_with_integration(
                db=mock_db,
                user_id="user-123",
                profile_data=sample_profile_data
            )
    
    def test_merge_skill_profiles_single_source(self, profile_service):
        """Test skill merging with single source."""
        skill_sources = {
            'manual': {'Python': 0.8, 'JavaScript': 0.6}
        }
        
        result = profile_service._merge_skill_profiles(skill_sources)
        
        assert len(result.merged_skills) == 2
        assert result.merged_skills['Python'] == 0.8  # Manual weight is 1.0
        assert result.merged_skills['JavaScript'] == 0.6
        assert len(result.conflicts) == 0
    
    def test_merge_skill_profiles_multiple_sources(self, profile_service):
        """Test skill merging with multiple sources."""
        skill_sources = {
            'manual': {'Python': 0.8},
            'github': {'Python': 0.6, 'JavaScript': 0.7},
            'resume': {'Python': 0.9, 'React': 0.5}
        }
        
        result = profile_service._merge_skill_profiles(skill_sources)
        
        # Python appears in all sources, should have high confidence
        assert 'Python' in result.merged_skills
        assert result.merged_skills['Python'] > 0.8  # Boosted by multiple sources
        
        # JavaScript only in GitHub
        assert 'JavaScript' in result.merged_skills
        assert result.merged_skills['JavaScript'] < 0.8  # GitHub weight is 0.8
        
        # React only in resume
        assert 'React' in result.merged_skills
        
        # Check sources tracking
        assert 'Python' in result.sources
        assert len(result.sources['Python']) == 3  # All three sources
    
    def test_merge_skill_profiles_with_conflicts(self, profile_service):
        """Test skill merging with confidence conflicts."""
        skill_sources = {
            'manual': {'Python': 0.9},  # High confidence
            'github': {'Python': 0.3},  # Low confidence
        }
        
        result = profile_service._merge_skill_profiles(skill_sources)
        
        # Should have conflicts due to significant difference
        assert len(result.conflicts) > 0
        
        conflict = result.conflicts[0]
        assert conflict['skill'] == 'Python'
        assert conflict['range'] > 0.4  # Significant difference
        assert 'resolved_confidence' in conflict
    
    def test_extract_skills_from_platforms_github(self, profile_service):
        """Test skill extraction from GitHub platform data."""
        platform_data = {
            'github': {
                'languages': {'Python': 2000, 'JavaScript': 1000, 'TypeScript': 500},
                'repositories': [
                    {'topics': ['web-development', 'react', 'nodejs']},
                    {'topics': ['machine-learning', 'python']}
                ]
            }
        }
        
        result = profile_service._extract_skills_from_platforms(platform_data)
        
        assert 'github' in result
        github_skills = result['github']
        
        # Check language skills
        assert 'Python' in github_skills
        assert 'JavaScript' in github_skills
        assert github_skills['Python'] > github_skills['JavaScript']  # More bytes
        
        # Check topic skills
        assert 'react' in github_skills
        assert 'machine-learning' in github_skills
    
    def test_extract_skills_from_platforms_leetcode(self, profile_service):
        """Test skill extraction from LeetCode platform data."""
        platform_data = {
            'leetcode': {
                'skill_tags': {'algorithms': 20, 'data-structures': 15, 'dynamic-programming': 10},
                'languages_used': {'Python': 25, 'Java': 10, 'C++': 5}
            }
        }
        
        result = profile_service._extract_skills_from_platforms(platform_data)
        
        assert 'leetcode' in result
        leetcode_skills = result['leetcode']
        
        # Check skill tags
        assert 'algorithms' in leetcode_skills
        assert 'data-structures' in leetcode_skills
        assert leetcode_skills['algorithms'] > leetcode_skills['dynamic-programming']
        
        # Check languages
        assert 'Python' in leetcode_skills
        assert 'Java' in leetcode_skills
        assert leetcode_skills['Python'] > leetcode_skills['Java']
    
    def test_extract_skills_from_platforms_linkedin(self, profile_service):
        """Test skill extraction from LinkedIn platform data."""
        platform_data = {
            'linkedin': {
                'skills': [
                    {'name': 'Python', 'endorsements': 20, 'is_top_skill': True},
                    {'name': 'JavaScript', 'endorsements': 10, 'is_top_skill': False},
                    {'name': 'React', 'endorsements': 5, 'is_top_skill': False}
                ]
            }
        }
        
        result = profile_service._extract_skills_from_platforms(platform_data)
        
        assert 'linkedin' in result
        linkedin_skills = result['linkedin']
        
        # Check skills with endorsements
        assert 'Python' in linkedin_skills
        assert 'JavaScript' in linkedin_skills
        assert 'React' in linkedin_skills
        
        # Higher endorsements should mean higher confidence
        assert linkedin_skills['Python'] > linkedin_skills['JavaScript']
        assert linkedin_skills['JavaScript'] > linkedin_skills['React']
    
    @pytest.mark.asyncio
    async def test_calculate_skill_gaps_software_engineer(self, profile_service):
        """Test skill gap calculation for software engineer role."""
        current_skills = {
            'Python': 0.8,
            'JavaScript': 0.6,
            'Git': 0.5
        }
        
        skill_gaps = await profile_service._calculate_skill_gaps(current_skills, "Software Engineer")
        
        # Should identify gaps for required skills
        assert 'SQL' in skill_gaps  # Required but missing
        assert 'React' in skill_gaps  # Required but missing
        assert 'Docker' in skill_gaps  # Required but missing
        
        # Should not have gaps for skills we already have at required level
        assert 'Python' not in skill_gaps  # We have 0.8, required 0.8
        
        # Should have gap for skills below required level
        if 'Git' in skill_gaps:
            assert skill_gaps['Git'] > 0  # We have 0.5, required 0.9
    
    @pytest.mark.asyncio
    async def test_calculate_skill_gaps_data_scientist(self, profile_service):
        """Test skill gap calculation for data scientist role."""
        current_skills = {
            'Python': 0.7,
            'SQL': 0.5
        }
        
        skill_gaps = await profile_service._calculate_skill_gaps(current_skills, "Data Scientist")
        
        # Should identify ML-specific gaps
        assert 'Machine Learning' in skill_gaps
        assert 'Pandas' in skill_gaps
        assert 'TensorFlow' in skill_gaps
        
        # Should have gaps for skills below required level
        assert 'Python' in skill_gaps  # We have 0.7, required 0.9
        assert 'SQL' in skill_gaps  # We have 0.5, required 0.8
    
    @pytest.mark.asyncio
    async def test_validate_profile_consistency_valid_data(self, profile_service, sample_user_profile):
        """Test profile validation with valid data."""
        update_data = ProfileUpdate(
            experience_years=5,
            github_username="valid-username",
            skills={"Python": 0.8, "JavaScript": 0.6}
        )
        
        result = await profile_service._validate_profile_consistency(sample_user_profile, update_data)
        
        assert result['is_valid'] is True
        assert len(result['errors']) == 0
    
    @pytest.mark.asyncio
    async def test_validate_profile_consistency_invalid_data(self, profile_service, sample_user_profile):
        """Test profile validation with invalid data."""
        update_data = ProfileUpdate(
            experience_years=-1,  # Invalid
            github_username="invalid@username",  # Invalid format
            skills={"Python": 1.5}  # Invalid confidence score
        )
        
        result = await profile_service._validate_profile_consistency(sample_user_profile, update_data)
        
        assert result['is_valid'] is False
        assert len(result['errors']) > 0
        assert any("negative" in error.lower() for error in result['errors'])
        assert any("invalid" in error.lower() for error in result['errors'])
    
    @pytest.mark.asyncio
    async def test_resolve_profile_conflicts_skills(self, profile_service, sample_user_profile):
        """Test conflict resolution for skills."""
        # Setup existing profile with skills
        sample_user_profile.skills = {'Python': 0.6, 'JavaScript': 0.5}
        
        update_data = {
            'skills': {'Python': 0.8, 'React': 0.7}  # Higher Python, new React
        }
        
        change_tracker = ProfileChangeTracker()
        
        resolved_data = await profile_service._resolve_profile_conflicts(
            sample_user_profile, update_data, change_tracker
        )
        
        # Should merge skills taking higher confidence
        expected_skills = {'Python': 0.8, 'JavaScript': 0.5, 'React': 0.7}
        assert resolved_data['skills'] == expected_skills
        
        # Should track the change
        changes = change_tracker.get_changes()
        assert len(changes) > 0
    
    @pytest.mark.asyncio
    async def test_resolve_profile_conflicts_experience_years(self, profile_service, sample_user_profile):
        """Test conflict resolution for experience years."""
        sample_user_profile.experience_years = 5
        
        update_data = {'experience_years': 3}  # Lower than existing
        change_tracker = ProfileChangeTracker()
        
        resolved_data = await profile_service._resolve_profile_conflicts(
            sample_user_profile, update_data, change_tracker
        )
        
        # Should take higher value (existing)
        assert resolved_data['experience_years'] == 5
        
        # Should track the change
        changes = change_tracker.get_changes()
        assert len(changes) > 0
    
    def test_calculate_profile_completeness_complete(self, profile_service, sample_user_profile):
        """Test profile completeness calculation for complete profile."""
        # Set all fields
        sample_user_profile.dream_job = "Software Engineer"
        sample_user_profile.experience_years = 3
        sample_user_profile.current_role = "Developer"
        sample_user_profile.location = "New York"
        sample_user_profile.github_username = "testuser"
        sample_user_profile.leetcode_id = "testuser"
        sample_user_profile.linkedin_url = "https://linkedin.com/in/testuser"
        sample_user_profile.skills = {"Python": 0.8}
        sample_user_profile.resume_data = {"skills": []}
        sample_user_profile.career_interests = {"web_dev": 0.9}
        
        result = profile_service._calculate_profile_completeness(sample_user_profile)
        
        assert result['score'] == 1.0  # 100% complete
        assert result['completed_fields'] == result['total_fields']
        assert len(result['missing_fields']) == 0
    
    def test_calculate_profile_completeness_partial(self, profile_service):
        """Test profile completeness calculation for partial profile."""
        partial_profile = UserProfile(
            id="profile-123",
            user_id="user-123",
            dream_job="Software Engineer",
            experience_years=3,
            # Missing other fields
        )
        
        result = profile_service._calculate_profile_completeness(partial_profile)
        
        assert result['score'] < 1.0  # Not complete
        assert result['completed_fields'] < result['total_fields']
        assert len(result['missing_fields']) > 0
        assert 'current_role' in result['missing_fields']
        assert 'location' in result['missing_fields']
    
    def test_analyze_skill_distribution(self, profile_service):
        """Test skill distribution analysis."""
        skills = {
            'Python': 0.9,
            'JavaScript': 0.8,
            'React': 0.7,
            'Docker': 0.6,
            'Git': 0.8,
            'Communication': 0.7
        }
        
        result = profile_service._analyze_skill_distribution(skills)
        
        assert result['total_skills'] == 6
        assert 'categories' in result
        assert 'top_skills' in result
        assert 'average_confidence' in result
        
        # Check categories
        categories = result['categories']
        assert 'programming_languages' in categories
        assert 'frameworks' in categories
        assert 'tools' in categories
        assert 'soft_skills' in categories
        
        # Check top skills are sorted by confidence
        top_skills = result['top_skills']
        assert len(top_skills) <= 10
        assert top_skills[0]['skill'] == 'Python'  # Highest confidence
        assert top_skills[0]['confidence'] == 0.9
    
    def test_analyze_data_freshness_recent(self, profile_service, sample_user_profile):
        """Test data freshness analysis for recent data."""
        sample_user_profile.data_last_updated = datetime.utcnow() - timedelta(hours=1)
        
        result = profile_service._analyze_data_freshness(sample_user_profile)
        
        assert result['freshness_score'] == 1.0  # Very fresh
        assert result['needs_refresh'] is False
        assert result['days_since_update'] == 0
    
    def test_analyze_data_freshness_stale(self, profile_service, sample_user_profile):
        """Test data freshness analysis for stale data."""
        sample_user_profile.data_last_updated = datetime.utcnow() - timedelta(days=45)
        
        result = profile_service._analyze_data_freshness(sample_user_profile)
        
        assert result['freshness_score'] < 0.5  # Stale
        assert result['needs_refresh'] is True
        assert result['days_since_update'] == 45
    
    def test_analyze_platform_coverage_full(self, profile_service, sample_user_profile):
        """Test platform coverage analysis with all platforms connected."""
        sample_user_profile.github_username = "testuser"
        sample_user_profile.leetcode_id = "testuser"
        sample_user_profile.linkedin_url = "https://linkedin.com/in/testuser"
        sample_user_profile.codeforces_id = "testuser"
        
        result = profile_service._analyze_platform_coverage(sample_user_profile)
        
        assert result['coverage_score'] == 1.0  # 100% coverage
        assert result['connected_count'] == 4
        assert len(result['missing_platforms']) == 0
    
    def test_analyze_platform_coverage_partial(self, profile_service, sample_user_profile):
        """Test platform coverage analysis with partial platform connection."""
        sample_user_profile.github_username = "testuser"
        sample_user_profile.leetcode_id = None
        sample_user_profile.linkedin_url = None
        sample_user_profile.codeforces_id = None
        
        result = profile_service._analyze_platform_coverage(sample_user_profile)
        
        assert result['coverage_score'] == 0.25  # 25% coverage (1/4)
        assert result['connected_count'] == 1
        assert len(result['missing_platforms']) == 3
        assert 'leetcode' in result['missing_platforms']
        assert 'linkedin' in result['missing_platforms']
        assert 'codeforces' in result['missing_platforms']
    
    def test_summarize_skill_gaps_empty(self, profile_service):
        """Test skill gaps summary with no gaps."""
        skill_gaps = {}
        
        result = profile_service._summarize_skill_gaps(skill_gaps)
        
        assert result['total_gaps'] == 0
        assert len(result['critical_gaps']) == 0
        assert len(result['moderate_gaps']) == 0
        assert len(result['minor_gaps']) == 0
    
    def test_summarize_skill_gaps_mixed(self, profile_service):
        """Test skill gaps summary with mixed gap levels."""
        skill_gaps = {
            'Machine Learning': 0.8,  # Critical
            'Docker': 0.5,  # Moderate
            'Git': 0.2,  # Minor
            'SQL': 0.6  # Moderate
        }
        
        result = profile_service._summarize_skill_gaps(skill_gaps)
        
        assert result['total_gaps'] == 4
        assert len(result['critical_gaps']) == 1
        assert len(result['moderate_gaps']) == 2
        assert len(result['minor_gaps']) == 1
        
        # Check categorization
        assert result['critical_gaps'][0]['skill'] == 'Machine Learning'
        assert result['critical_gaps'][0]['gap'] == 0.8
        
        moderate_skills = [gap['skill'] for gap in result['moderate_gaps']]
        assert 'Docker' in moderate_skills
        assert 'SQL' in moderate_skills
        
        assert result['minor_gaps'][0]['skill'] == 'Git'
    
    @pytest.mark.asyncio
    async def test_generate_profile_recommendations_missing_dream_job(self, profile_service, sample_user_profile):
        """Test recommendation generation for profile missing dream job."""
        sample_user_profile.dream_job = None
        
        recommendations = await profile_service._generate_profile_recommendations(sample_user_profile)
        
        # Should recommend adding dream job
        dream_job_rec = next((rec for rec in recommendations if 'Dream Job' in rec['title']), None)
        assert dream_job_rec is not None
        assert dream_job_rec['priority'] == 'high'
        assert dream_job_rec['type'] == 'completeness'
    
    @pytest.mark.asyncio
    async def test_generate_profile_recommendations_missing_resume(self, profile_service, sample_user_profile):
        """Test recommendation generation for profile missing resume."""
        sample_user_profile.resume_data = None
        
        recommendations = await profile_service._generate_profile_recommendations(sample_user_profile)
        
        # Should recommend uploading resume
        resume_rec = next((rec for rec in recommendations if 'Resume' in rec['title']), None)
        assert resume_rec is not None
        assert resume_rec['priority'] == 'high'
        assert resume_rec['type'] == 'completeness'
    
    @pytest.mark.asyncio
    async def test_generate_profile_recommendations_stale_data(self, profile_service, sample_user_profile):
        """Test recommendation generation for stale profile data."""
        sample_user_profile.data_last_updated = datetime.utcnow() - timedelta(days=45)
        
        recommendations = await profile_service._generate_profile_recommendations(sample_user_profile)
        
        # Should recommend refreshing data
        refresh_rec = next((rec for rec in recommendations if 'Refresh' in rec['title']), None)
        assert refresh_rec is not None
        assert refresh_rec['priority'] == 'medium'
        assert refresh_rec['type'] == 'freshness'
    
    @pytest.mark.asyncio
    async def test_update_profile_with_validation_success(self, profile_service, sample_user_profile):
        """Test successful profile update with validation."""
        mock_db = Mock(spec=AsyncSession)
        
        # Mock repository methods
        profile_service.profile_repo.get_by_user_id = AsyncMock(return_value=sample_user_profile)
        profile_service.profile_repo.update = AsyncMock(return_value=sample_user_profile)
        
        update_data = ProfileUpdate(
            experience_years=5,
            skills={"Python": 0.9, "React": 0.7}
        )
        
        result = await profile_service.update_profile_with_validation(
            db=mock_db,
            user_id="user-123",
            update_data=update_data
        )
        
        assert result is not None
        profile_service.profile_repo.update.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_update_profile_with_validation_not_found(self, profile_service):
        """Test profile update when profile doesn't exist."""
        mock_db = Mock(spec=AsyncSession)
        
        # Mock no existing profile
        profile_service.profile_repo.get_by_user_id = AsyncMock(return_value=None)
        
        update_data = ProfileUpdate(experience_years=5)
        
        with pytest.raises(NotFoundError, match="Profile not found"):
            await profile_service.update_profile_with_validation(
                db=mock_db,
                user_id="user-123",
                update_data=update_data
            )
    
    @pytest.mark.asyncio
    async def test_refresh_external_data_success(self, profile_service, sample_user_profile):
        """Test successful external data refresh."""
        mock_db = Mock(spec=AsyncSession)
        
        # Mock repository methods
        profile_service.profile_repo.get_by_user_id = AsyncMock(return_value=sample_user_profile)
        profile_service.profile_repo.update = AsyncMock(return_value=sample_user_profile)
        
        # Mock external API extraction
        mock_extraction_result = Mock()
        mock_extraction_result.success = True
        mock_extraction_result.github_profile = {"languages": {"Python": 2000}}
        mock_extraction_result.leetcode_profile = {"skill_tags": {"algorithms": 15}}
        mock_extraction_result.linkedin_profile = {"skills": [{"name": "React", "endorsements": 10}]}
        
        profile_service._extract_external_platform_data = AsyncMock(return_value=mock_extraction_result)
        
        result = await profile_service.refresh_external_data(
            db=mock_db,
            user_id="user-123",
            force_refresh=True
        )
        
        assert result is not None
        profile_service.profile_repo.update.assert_called_once()
        profile_service._extract_external_platform_data.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_refresh_external_data_recent_data_skip(self, profile_service, sample_user_profile):
        """Test skipping refresh for recent data."""
        mock_db = Mock(spec=AsyncSession)
        
        # Set recent update time
        sample_user_profile.data_last_updated = datetime.utcnow() - timedelta(hours=1)
        
        profile_service.profile_repo.get_by_user_id = AsyncMock(return_value=sample_user_profile)
        
        result = await profile_service.refresh_external_data(
            db=mock_db,
            user_id="user-123",
            force_refresh=False
        )
        
        assert result is not None
        # Should not call external API or update
        profile_service.profile_repo.update.assert_not_called()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])