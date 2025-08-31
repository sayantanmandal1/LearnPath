"""
Simple unit tests for UserProfileService to verify basic functionality.
"""

import pytest
from unittest.mock import Mock, AsyncMock
from datetime import datetime

from app.services.profile_service import UserProfileService, SkillMergeResult, ProfileChangeTracker
from app.schemas.profile import ProfileCreate


class TestBasicFunctionality:
    """Test basic functionality of profile service."""
    
    def test_skill_merge_result_initialization(self):
        """Test SkillMergeResult initialization."""
        result = SkillMergeResult()
        
        assert result.merged_skills == {}
        assert result.conflicts == []
        assert result.sources == {}
        assert result.confidence_scores == {}
    
    def test_profile_change_tracker(self):
        """Test ProfileChangeTracker functionality."""
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
    
    def test_service_initialization(self):
        """Test UserProfileService initialization."""
        service = UserProfileService(github_token="test-token")
        
        assert service.skill_confidence_weights is not None
        assert service.conflict_resolution_strategies is not None
        assert 'resume' in service.skill_confidence_weights
        assert 'github' in service.skill_confidence_weights
        assert 'skills' in service.conflict_resolution_strategies
    
    def test_skill_confidence_weights(self):
        """Test skill confidence weights configuration."""
        service = UserProfileService()
        weights = service.skill_confidence_weights
        
        assert weights['manual'] == 1.0  # Highest confidence for manual input
        assert weights['resume'] == 0.9  # High confidence for resume
        assert weights['github'] > weights['leetcode']  # GitHub more reliable than LeetCode
        assert all(0 <= weight <= 1.0 for weight in weights.values())
    
    def test_merge_skill_profiles_single_source(self):
        """Test skill merging with single source."""
        service = UserProfileService()
        skill_sources = {
            'manual': {'Python': 0.8, 'JavaScript': 0.6}
        }
        
        result = service._merge_skill_profiles(skill_sources)
        
        assert len(result.merged_skills) == 2
        assert result.merged_skills['Python'] == 0.8  # Manual weight is 1.0
        assert result.merged_skills['JavaScript'] == 0.6
        assert len(result.conflicts) == 0
    
    def test_merge_skill_profiles_multiple_sources(self):
        """Test skill merging with multiple sources."""
        service = UserProfileService()
        skill_sources = {
            'manual': {'Python': 0.8},
            'github': {'Python': 0.6, 'JavaScript': 0.7},
            'resume': {'Python': 0.9, 'React': 0.5}
        }
        
        result = service._merge_skill_profiles(skill_sources)
        
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
    
    def test_merge_skill_profiles_with_conflicts(self):
        """Test skill merging with confidence conflicts."""
        service = UserProfileService()
        skill_sources = {
            'manual': {'Python': 0.9},  # High confidence
            'github': {'Python': 0.3},  # Low confidence
        }
        
        result = service._merge_skill_profiles(skill_sources)
        
        # Should have conflicts due to significant difference
        assert len(result.conflicts) > 0
        
        conflict = result.conflicts[0]
        assert conflict['skill'] == 'Python'
        assert conflict['range'] > 0.4  # Significant difference
        assert 'resolved_confidence' in conflict
    
    def test_extract_skills_from_platforms_github(self):
        """Test skill extraction from GitHub platform data."""
        service = UserProfileService()
        platform_data = {
            'github': {
                'languages': {'Python': 2000, 'JavaScript': 1000, 'TypeScript': 500},
                'repositories': [
                    {'topics': ['web-development', 'react', 'nodejs']},
                    {'topics': ['machine-learning', 'python']}
                ]
            }
        }
        
        result = service._extract_skills_from_platforms(platform_data)
        
        assert 'github' in result
        github_skills = result['github']
        
        # Check language skills
        assert 'Python' in github_skills
        assert 'JavaScript' in github_skills
        assert github_skills['Python'] > github_skills['JavaScript']  # More bytes
        
        # Check topic skills
        assert 'react' in github_skills
        assert 'machine-learning' in github_skills
    
    @pytest.mark.asyncio
    async def test_calculate_skill_gaps_software_engineer(self):
        """Test skill gap calculation for software engineer role."""
        service = UserProfileService()
        current_skills = {
            'Python': 0.8,
            'JavaScript': 0.6,
            'Git': 0.5
        }
        
        skill_gaps = await service._calculate_skill_gaps(current_skills, "Software Engineer")
        
        # Should identify gaps for required skills
        assert 'SQL' in skill_gaps  # Required but missing
        assert 'React' in skill_gaps  # Required but missing
        assert 'Docker' in skill_gaps  # Required but missing
        
        # Should not have gaps for skills we already have at required level
        assert 'Python' not in skill_gaps  # We have 0.8, required 0.8
        
        # Should have gap for skills below required level
        if 'Git' in skill_gaps:
            assert skill_gaps['Git'] > 0  # We have 0.5, required 0.9


if __name__ == "__main__":
    pytest.main([__file__, "-v"])