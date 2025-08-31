"""
Tests for Learning Path Service.

This module tests the learning path generation functionality including:
- Skill gap identification and prioritization
- Learning resource integration
- Project recommendations
- Timeline estimation
- Resource quality scoring
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime

from app.services.learning_path_service import LearningPathService
from app.schemas.learning_path import (
    LearningPathRequest, LearningPathResponse, DifficultyLevel,
    ResourceProvider, SkillGap, LearningPath, ProjectRecommendation
)
from app.core.exceptions import ServiceException


class TestLearningPathService:
    """Test cases for LearningPathService."""
    
    @pytest.fixture
    def service(self):
        """Create a LearningPathService instance for testing."""
        return LearningPathService()
    
    @pytest.fixture
    def sample_request(self):
        """Create a sample learning path request."""
        return LearningPathRequest(
            user_id="test_user_123",
            target_role="software_engineer",
            target_skills=["python", "javascript", "react"],
            current_skills={"html": 0.8, "css": 0.7, "git": 0.6},
            time_commitment_hours_per_week=15,
            budget_limit=200.0,
            include_free_only=False,
            preferred_providers=[ResourceProvider.COURSERA, ResourceProvider.UDEMY],
            difficulty_preference=DifficultyLevel.INTERMEDIATE,
            include_certifications=True,
            include_projects=True
        )
    
    @pytest.mark.asyncio
    async def test_generate_learning_paths_success(self, service, sample_request):
        """Test successful learning path generation."""
        # Mock Redis client
        with patch.object(service, 'redis_client', Mock()):
            response = await service.generate_learning_paths(sample_request)
            
            # Verify response structure
            assert isinstance(response, LearningPathResponse)
            assert len(response.learning_paths) > 0
            assert len(response.learning_paths) <= 5  # Should return max 5 paths
            assert response.total_paths >= len(response.learning_paths)
            assert len(response.skill_gaps_identified) > 0
            
            # Verify learning paths have required fields
            for path in response.learning_paths:
                assert isinstance(path, LearningPath)
                assert path.title
                assert path.target_role == sample_request.target_role
                assert len(path.target_skills) > 0
                assert path.estimated_duration_weeks > 0
                assert path.estimated_duration_hours > 0
                assert path.confidence_score is not None
    
    @pytest.mark.asyncio
    async def test_identify_skill_gaps(self, service, sample_request):
        """Test skill gap identification and prioritization."""
        with patch.object(service, 'redis_client', Mock()):
            skill_gaps = await service._identify_skill_gaps(sample_request)
            
            # Verify skill gaps are identified
            assert len(skill_gaps) > 0
            
            # Verify skill gaps are properly structured
            for gap in skill_gaps:
                assert isinstance(gap, SkillGap)
                assert gap.skill_name
                assert 0 <= gap.current_level <= 1
                assert 0 <= gap.target_level <= 1
                assert gap.gap_size > 0  # Should only include skills with gaps
                assert 0 <= gap.priority <= 1
                assert gap.estimated_learning_hours > 0
                assert gap.difficulty in [level.value for level in DifficultyLevel]
            
            # Verify gaps are sorted by priority (highest first)
            priorities = [gap.priority for gap in skill_gaps]
            assert priorities == sorted(priorities, reverse=True)
    
    @pytest.mark.asyncio
    async def test_gather_learning_resources_coursera(self, service, sample_request):
        """Test gathering resources from Coursera."""
        skills = ["python", "machine_learning"]
        
        with patch.object(service, 'redis_client', Mock()):
            resources = await service._get_coursera_resources(skills, sample_request)
            
            # Verify resources are returned
            assert len(resources) > 0
            
            # Verify resource structure
            for resource in resources:
                assert resource.provider == ResourceProvider.COURSERA
                assert resource.title
                assert resource.url.startswith("https://")
                assert resource.type
                assert resource.difficulty_level
                assert len(resource.skills_taught) > 0
                assert resource.certificate_available is True  # Coursera courses have certificates
                assert resource.quality_score is not None
    
    @pytest.mark.asyncio
    async def test_gather_learning_resources_budget_filter(self, service):
        """Test resource filtering based on budget constraints."""
        request = LearningPathRequest(
            user_id="test_user",
            target_skills=["python"],
            budget_limit=50.0  # Low budget
        )
        
        with patch.object(service, 'redis_client', Mock()):
            resources = await service._get_coursera_resources(["python"], request)
            
            # Verify all resources are within budget
            for resource in resources:
                if resource.cost:
                    assert resource.cost <= request.budget_limit
    
    @pytest.mark.asyncio
    async def test_gather_learning_resources_free_only(self, service):
        """Test resource filtering for free resources only."""
        request = LearningPathRequest(
            user_id="test_user",
            target_skills=["python"],
            include_free_only=True
        )
        
        with patch.object(service, 'redis_client', Mock()):
            resources = await service._get_freecodecamp_resources(["python"], request)
            
            # Verify all resources are free
            for resource in resources:
                assert resource.cost == 0.0
    
    @pytest.mark.asyncio
    async def test_get_project_recommendations(self, service):
        """Test project recommendations from GitHub."""
        skills = ["python", "javascript"]
        difficulty = DifficultyLevel.INTERMEDIATE
        
        projects = await service.get_project_recommendations(skills, difficulty)
        
        # Verify projects are returned
        assert len(projects) > 0
        assert len(projects) <= 10  # Should return max 10 projects
        
        # Verify project structure
        for project in projects:
            assert isinstance(project, ProjectRecommendation)
            assert project.title
            assert project.description
            assert project.difficulty_level == difficulty
            assert len(project.skills_practiced) > 0
            assert len(project.technologies) > 0
            assert project.estimated_duration_hours > 0
            assert project.learning_value is not None
            assert project.market_relevance is not None
        
        # Verify projects are sorted by learning value and market relevance
        scores = [(p.learning_value or 0) * (p.market_relevance or 0) for p in projects]
        assert scores == sorted(scores, reverse=True)
    
    @pytest.mark.asyncio
    async def test_filter_and_score_resources(self, service, sample_request):
        """Test resource filtering and quality scoring."""
        from app.schemas.learning_path import LearningResource, ResourceType
        
        # Create sample resources
        resources = [
            LearningResource(
                title="High Quality Course",
                type=ResourceType.COURSE,
                provider=ResourceProvider.COURSERA,
                url="https://example.com/course1",
                rating=4.8,
                cost=49.0,
                difficulty_level=DifficultyLevel.INTERMEDIATE,
                certificate_available=True,
                hands_on_projects=True
            ),
            LearningResource(
                title="Low Quality Course",
                type=ResourceType.COURSE,
                provider=ResourceProvider.UDEMY,
                url="https://example.com/course2",
                rating=3.2,
                cost=150.0,  # Over budget
                difficulty_level=DifficultyLevel.BEGINNER,
                certificate_available=False,
                hands_on_projects=False
            ),
            LearningResource(
                title="Free Course",
                type=ResourceType.COURSE,
                provider=ResourceProvider.FREECODECAMP,
                url="https://example.com/course3",
                rating=4.5,
                cost=0.0,
                difficulty_level=DifficultyLevel.INTERMEDIATE,
                certificate_available=True,
                hands_on_projects=True
            )
        ]
        
        filtered_resources = await service._filter_and_score_resources(resources, sample_request)
        
        # Verify filtering (should exclude over-budget resource)
        assert len(filtered_resources) == 2
        
        # Verify quality scoring
        for resource in filtered_resources:
            assert resource.quality_score is not None
            assert 0 <= resource.quality_score <= 1
        
        # Verify sorting by quality score
        quality_scores = [r.quality_score for r in filtered_resources]
        assert quality_scores == sorted(quality_scores, reverse=True)
    
    @pytest.mark.asyncio
    async def test_create_milestones(self, service):
        """Test milestone creation with proper sequencing."""
        skills = ["html", "javascript", "react", "node.js"]
        skill_gaps = [
            SkillGap(
                skill_name=skill,
                current_level=0.2,
                target_level=0.8,
                gap_size=0.6,
                priority=0.8,
                market_demand=0.7,
                estimated_learning_hours=60,
                difficulty=DifficultyLevel.BEGINNER if skill == "html" else DifficultyLevel.INTERMEDIATE
            )
            for skill in skills
        ]
        
        milestones = await service._create_milestones(skills, skill_gaps)
        
        # Verify milestones are created
        assert len(milestones) > 0
        
        # Verify milestone structure
        for milestone in milestones:
            assert milestone.title
            assert milestone.description
            assert milestone.order >= 0
            assert len(milestone.skills_to_acquire) > 0
            assert milestone.estimated_duration_hours > 0
            assert len(milestone.completion_criteria) > 0
        
        # Verify milestones are ordered correctly
        orders = [m.order for m in milestones]
        assert orders == sorted(orders)
    
    def test_calculate_skill_priority(self, service):
        """Test skill priority calculation."""
        skill = "python"
        gap_size = 0.6
        target_role = "software_engineer"
        market_data = {"python": {"demand_score": 0.9}}
        
        priority = asyncio.run(service._calculate_skill_priority(
            skill, gap_size, target_role, market_data
        ))
        
        # Verify priority is calculated
        assert isinstance(priority, float)
        assert 0 <= priority <= 1
        
        # Test with role-critical skill (should have higher priority)
        role_critical_priority = asyncio.run(service._calculate_skill_priority(
            skill, gap_size, target_role, market_data
        ))
        
        # Test with non-critical skill
        non_critical_priority = asyncio.run(service._calculate_skill_priority(
            "obscure_skill", gap_size, None, None
        ))
        
        assert role_critical_priority >= non_critical_priority
    
    def test_estimate_learning_hours(self, service):
        """Test learning time estimation."""
        skill = "python"
        gap_size = 0.8
        current_level = 0.2
        
        hours = service._estimate_skill_learning_time(skill, gap_size, current_level)
        
        # Verify reasonable estimate
        assert isinstance(hours, int)
        assert hours >= 10  # Minimum hours
        assert hours <= 200  # Reasonable maximum
        
        # Test with higher current level (should be faster)
        hours_experienced = service._estimate_skill_learning_time(skill, gap_size, 0.6)
        assert hours_experienced < hours
    
    def test_get_skill_difficulty(self, service):
        """Test skill difficulty determination."""
        # Test known skills
        assert service._get_skill_difficulty("python") == DifficultyLevel.BEGINNER
        assert service._get_skill_difficulty("react") == DifficultyLevel.INTERMEDIATE
        assert service._get_skill_difficulty("machine_learning") == DifficultyLevel.ADVANCED
        
        # Test unknown skill (should default to intermediate)
        assert service._get_skill_difficulty("unknown_skill") == DifficultyLevel.INTERMEDIATE
    
    def test_determine_overall_difficulty(self, service):
        """Test overall difficulty determination for learning path."""
        skill_gaps = [
            SkillGap(
                skill_name="skill1",
                current_level=0.2,
                target_level=0.8,
                gap_size=0.6,
                priority=0.8,
                market_demand=0.7,
                estimated_learning_hours=60,
                difficulty=DifficultyLevel.BEGINNER
            ),
            SkillGap(
                skill_name="skill2",
                current_level=0.2,
                target_level=0.8,
                gap_size=0.6,
                priority=0.8,
                market_demand=0.7,
                estimated_learning_hours=60,
                difficulty=DifficultyLevel.INTERMEDIATE
            ),
            SkillGap(
                skill_name="skill3",
                current_level=0.2,
                target_level=0.8,
                gap_size=0.6,
                priority=0.8,
                market_demand=0.7,
                estimated_learning_hours=60,
                difficulty=DifficultyLevel.INTERMEDIATE
            )
        ]
        
        # Should return most common difficulty (intermediate)
        difficulty = service._determine_overall_difficulty(skill_gaps)
        assert difficulty == DifficultyLevel.INTERMEDIATE
        
        # Test with empty list
        difficulty_empty = service._determine_overall_difficulty([])
        assert difficulty_empty == DifficultyLevel.BEGINNER
    
    def test_calculate_total_duration_weeks(self, service):
        """Test total duration calculation in weeks."""
        skill_gaps = [
            SkillGap(
                skill_name="skill1",
                current_level=0.2,
                target_level=0.8,
                gap_size=0.6,
                priority=0.8,
                market_demand=0.7,
                estimated_learning_hours=60,
                difficulty=DifficultyLevel.INTERMEDIATE
            ),
            SkillGap(
                skill_name="skill2",
                current_level=0.2,
                target_level=0.8,
                gap_size=0.6,
                priority=0.8,
                market_demand=0.7,
                estimated_learning_hours=40,
                difficulty=DifficultyLevel.INTERMEDIATE
            )
        ]
        
        hours_per_week = 10
        weeks = service._calculate_total_duration_weeks(skill_gaps, hours_per_week)
        
        # Total hours = 100, at 10 hours/week = 10 weeks
        assert weeks == 10
        
        # Test minimum 1 week
        weeks_min = service._calculate_total_duration_weeks([], 10)
        assert weeks_min == 1
    
    @pytest.mark.asyncio
    async def test_error_handling(self, service):
        """Test error handling in learning path generation."""
        # Test with invalid request
        invalid_request = LearningPathRequest(
            user_id="",  # Empty user ID
            target_skills=[],  # No target skills
            time_commitment_hours_per_week=0  # Invalid time commitment
        )
        
        # Should handle gracefully and not crash
        try:
            response = await service.generate_learning_paths(invalid_request)
            # Should return some response even with invalid input
            assert isinstance(response, LearningPathResponse)
        except ServiceException:
            # Or raise a proper service exception
            pass
    
    @pytest.mark.asyncio
    async def test_concurrent_resource_fetching(self, service, sample_request):
        """Test that resource fetching works concurrently."""
        skills = ["python", "javascript", "react"]
        
        with patch.object(service, 'redis_client', Mock()):
            # This should execute multiple API calls concurrently
            resources = await service._gather_learning_resources(skills, sample_request)
            
            # Verify resources from multiple providers
            providers = {r.provider for r in resources}
            assert len(providers) > 1  # Should have resources from multiple providers
    
    def test_group_skills_by_level(self, service):
        """Test skill grouping by difficulty level."""
        skills = ["html", "python", "react", "machine_learning"]
        
        grouped = service._group_skills_by_level(skills)
        
        # Verify grouping structure
        assert "beginner" in grouped
        assert "intermediate" in grouped
        assert "advanced" in grouped
        
        # Verify skills are in correct groups
        assert "html" in grouped["beginner"] or "python" in grouped["beginner"]
        assert "react" in grouped["intermediate"]
        assert "machine_learning" in grouped["advanced"]
    
    def test_calculate_path_confidence(self, service, sample_request):
        """Test learning path confidence calculation."""
        from app.schemas.learning_path import LearningResource, ResourceType
        
        # Create a sample learning path
        path = LearningPath(
            title="Test Path",
            target_role="software_engineer",
            target_skills=["python", "javascript"],
            difficulty_level=DifficultyLevel.INTERMEDIATE,
            estimated_duration_weeks=10,
            estimated_duration_hours=100,
            resources=[
                LearningResource(
                    title="Test Resource",
                    type=ResourceType.COURSE,
                    provider=ResourceProvider.COURSERA,
                    url="https://example.com",
                    difficulty_level=DifficultyLevel.INTERMEDIATE,
                    quality_score=0.8
                )
            ]
        )
        
        confidence = service._calculate_path_confidence(path, sample_request)
        
        # Verify confidence score
        assert isinstance(confidence, float)
        assert 0 <= confidence <= 1
    
    @pytest.mark.asyncio
    async def test_alternative_path_generation(self, service, sample_request):
        """Test generation of alternative learning paths."""
        skill_gaps = [
            SkillGap(
                skill_name="python",
                current_level=0.2,
                target_level=0.8,
                gap_size=0.6,
                priority=0.9,
                market_demand=0.8,
                estimated_learning_hours=80,
                difficulty=DifficultyLevel.BEGINNER
            )
        ]
        
        with patch.object(service, 'redis_client', Mock()):
            # Test project-focused path
            project_path = await service._generate_project_focused_path(sample_request, skill_gaps)
            assert "project" in project_path.title.lower()
            
            # Test certification path
            cert_path = await service._generate_certification_path(sample_request, skill_gaps)
            assert "certification" in cert_path.title.lower()
            
            # Test free resources path
            free_path = await service._generate_free_resources_path(sample_request, skill_gaps)
            assert "free" in free_path.title.lower()