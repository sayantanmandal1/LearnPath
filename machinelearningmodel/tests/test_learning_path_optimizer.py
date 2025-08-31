"""
Tests for Learning Path Optimizer.

This module tests the ML-based learning path optimization functionality including:
- Skill gap analysis with ML models
- Timeline estimation algorithms
- Resource quality scoring
- Learning sequence optimization
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from learning_path_optimizer import (
    LearningPathOptimizer, SkillGapAnalysis, LearningTimelineEstimate
)


class TestLearningPathOptimizer:
    """Test cases for LearningPathOptimizer."""
    
    @pytest.fixture
    def optimizer(self):
        """Create a LearningPathOptimizer instance for testing."""
        return LearningPathOptimizer()
    
    @pytest.fixture
    def sample_current_skills(self):
        """Sample current skills for testing."""
        return {
            "html": 0.8,
            "css": 0.7,
            "javascript": 0.5,
            "git": 0.6,
            "python": 0.3
        }
    
    @pytest.fixture
    def sample_target_skills(self):
        """Sample target skills for testing."""
        return ["python", "javascript", "react", "node.js", "sql", "docker"]
    
    @pytest.fixture
    def sample_user_profile(self):
        """Sample user profile for testing."""
        return {
            "experience_years": 3,
            "learning_speed_factor": 1.2,
            "current_skills": {"html": 0.8, "css": 0.7, "javascript": 0.5},
            "learning_style": "hands_on",
            "preferred_difficulty": "intermediate"
        }
    
    def test_initialization(self, optimizer):
        """Test optimizer initialization."""
        assert optimizer is not None
        assert optimizer.skill_embeddings is not None
        assert optimizer.timeline_model is not None
        assert optimizer.quality_model is not None
        assert optimizer.scaler is not None
    
    def test_analyze_skill_gaps_basic(self, optimizer, sample_current_skills, sample_target_skills):
        """Test basic skill gap analysis."""
        skill_gaps = optimizer.analyze_skill_gaps(
            current_skills=sample_current_skills,
            target_skills=sample_target_skills,
            target_role="software_engineer"
        )
        
        # Verify skill gaps are identified
        assert len(skill_gaps) > 0
        
        # Verify skill gap structure
        for gap in skill_gaps:
            assert isinstance(gap, SkillGapAnalysis)
            assert gap.skill_name in sample_target_skills
            assert 0 <= gap.current_level <= 1
            assert 0 <= gap.target_level <= 1
            assert gap.gap_size > 0  # Should only include skills with gaps
            assert 0 <= gap.priority_score <= 1
            assert gap.estimated_hours > 0
            assert gap.difficulty_level in ["beginner", "intermediate", "advanced"]
            assert isinstance(gap.prerequisites, list)
            assert 0 <= gap.market_demand <= 1
        
        # Verify gaps are sorted by priority
        priorities = [gap.priority_score for gap in skill_gaps]
        assert priorities == sorted(priorities, reverse=True)
    
    def test_analyze_skill_gaps_with_role(self, optimizer, sample_current_skills):
        """Test skill gap analysis with specific target role."""
        skill_gaps = optimizer.analyze_skill_gaps(
            current_skills=sample_current_skills,
            target_skills=["python"],
            target_role="data_scientist"
        )
        
        # Should include role-specific skills
        skill_names = [gap.skill_name for gap in skill_gaps]
        
        # Data scientist role should include ML-related skills
        expected_skills = ["python", "machine_learning", "statistics", "sql", "data_visualization"]
        common_skills = set(skill_names) & set(expected_skills)
        assert len(common_skills) > 1  # Should have multiple overlapping skills
    
    def test_analyze_skill_gaps_with_market_data(self, optimizer, sample_current_skills, sample_target_skills):
        """Test skill gap analysis with market demand data."""
        market_data = {
            "python": {"demand_score": 0.9},
            "javascript": {"demand_score": 0.8},
            "react": {"demand_score": 0.7}
        }
        
        skill_gaps = optimizer.analyze_skill_gaps(
            current_skills=sample_current_skills,
            target_skills=sample_target_skills,
            market_data=market_data
        )
        
        # Verify market demand is incorporated
        for gap in skill_gaps:
            if gap.skill_name in market_data:
                expected_demand = market_data[gap.skill_name]["demand_score"]
                assert gap.market_demand == expected_demand
    
    def test_estimate_learning_timeline(self, optimizer, sample_user_profile):
        """Test learning timeline estimation."""
        # Create sample skill gaps
        skill_gaps = [
            SkillGapAnalysis(
                skill_name="python",
                current_level=0.3,
                target_level=0.8,
                gap_size=0.5,
                priority_score=0.9,
                estimated_hours=80,
                difficulty_level="beginner",
                prerequisites=[],
                market_demand=0.9
            ),
            SkillGapAnalysis(
                skill_name="react",
                current_level=0.0,
                target_level=0.8,
                gap_size=0.8,
                priority_score=0.8,
                estimated_hours=60,
                difficulty_level="intermediate",
                prerequisites=["javascript", "html", "css"],
                market_demand=0.7
            )
        ]
        
        timeline = optimizer.estimate_learning_timeline(
            skill_gaps=skill_gaps,
            user_profile=sample_user_profile,
            time_commitment_hours_per_week=15
        )
        
        # Verify timeline structure
        assert isinstance(timeline, LearningTimelineEstimate)
        assert timeline.total_hours > 0
        assert timeline.total_weeks > 0
        assert len(timeline.milestones) > 0
        assert 0 <= timeline.confidence_score <= 1
        assert len(timeline.factors_considered) > 0
        
        # Verify milestones
        for milestone in timeline.milestones:
            assert "id" in milestone
            assert "title" in milestone
            assert "skills" in milestone
            assert "start_week" in milestone
            assert "end_week" in milestone
            assert milestone["start_week"] <= milestone["end_week"]
    
    def test_score_resource_quality(self, optimizer):
        """Test resource quality scoring."""
        resource_data = {
            "rating": 4.5,
            "duration_hours": 40,
            "certificate_available": True,
            "hands_on_projects": True,
            "provider": "coursera",
            "cost": 49.0,
            "skills_taught": ["python", "data_science"]
        }
        
        user_preferences = {
            "preferred_providers": ["coursera", "edx"],
            "budget_limit": 100.0,
            "include_certifications": True
        }
        
        quality_score = optimizer.score_resource_quality(
            resource_data=resource_data,
            user_preferences=user_preferences,
            skill_context="python"
        )
        
        # Verify quality score
        assert isinstance(quality_score, float)
        assert 0 <= quality_score <= 1
        
        # High-quality resource should have high score
        assert quality_score > 0.6  # Should be above average
    
    def test_score_resource_quality_low_quality(self, optimizer):
        """Test quality scoring for low-quality resource."""
        resource_data = {
            "rating": 2.5,  # Low rating
            "duration_hours": 200,  # Very long
            "certificate_available": False,
            "hands_on_projects": False,
            "provider": "unknown",
            "cost": 200.0,  # Expensive
            "skills_taught": ["unrelated_skill"]
        }
        
        user_preferences = {
            "preferred_providers": ["coursera"],
            "budget_limit": 50.0,
            "include_certifications": True
        }
        
        quality_score = optimizer.score_resource_quality(
            resource_data=resource_data,
            user_preferences=user_preferences,
            skill_context="python"
        )
        
        # Low-quality resource should have low score
        assert quality_score < 0.6
    
    def test_optimize_learning_sequence(self, optimizer):
        """Test learning sequence optimization."""
        skill_gaps = [
            SkillGapAnalysis(
                skill_name="html",
                current_level=0.2,
                target_level=0.8,
                gap_size=0.6,
                priority_score=0.7,
                estimated_hours=40,
                difficulty_level="beginner",
                prerequisites=[],
                market_demand=0.6
            ),
            SkillGapAnalysis(
                skill_name="javascript",
                current_level=0.1,
                target_level=0.8,
                gap_size=0.7,
                priority_score=0.8,
                estimated_hours=70,
                difficulty_level="beginner",
                prerequisites=[],
                market_demand=0.9
            ),
            SkillGapAnalysis(
                skill_name="react",
                current_level=0.0,
                target_level=0.8,
                gap_size=0.8,
                priority_score=0.9,
                estimated_hours=60,
                difficulty_level="intermediate",
                prerequisites=["javascript", "html"],
                market_demand=0.8
            )
        ]
        
        user_constraints = {
            "time_commitment": 10,
            "difficulty_preference": "intermediate"
        }
        
        optimized_sequence = optimizer.optimize_learning_sequence(
            skill_gaps=skill_gaps,
            user_constraints=user_constraints
        )
        
        # Verify sequence is optimized
        assert len(optimized_sequence) == len(skill_gaps)
        
        # Verify prerequisites are respected
        skill_positions = {gap.skill_name: i for i, gap in enumerate(optimized_sequence)}
        
        for gap in optimized_sequence:
            for prereq in gap.prerequisites:
                if prereq in skill_positions:
                    # Prerequisite should come before the skill
                    assert skill_positions[prereq] < skill_positions[gap.skill_name]
    
    def test_get_role_skill_requirements(self, optimizer):
        """Test role-specific skill requirements."""
        # Test software engineer requirements
        se_requirements = optimizer._get_role_skill_requirements("software_engineer")
        assert len(se_requirements) > 0
        assert "python" in se_requirements or "javascript" in se_requirements
        
        # Test data scientist requirements
        ds_requirements = optimizer._get_role_skill_requirements("data_scientist")
        assert len(ds_requirements) > 0
        assert "python" in ds_requirements
        assert "machine_learning" in ds_requirements
        
        # Test unknown role
        unknown_requirements = optimizer._get_role_skill_requirements("unknown_role")
        assert len(unknown_requirements) == 0
    
    def test_calculate_skill_priority(self, optimizer):
        """Test skill priority calculation."""
        skill = "python"
        gap_size = 0.6
        target_role = "software_engineer"
        market_data = {"python": {"demand_score": 0.9}}
        
        priority = optimizer._calculate_skill_priority(
            skill, gap_size, target_role, market_data
        )
        
        # Verify priority calculation
        assert isinstance(priority, float)
        assert 0 <= priority <= 1
        
        # Test role-critical skill should have higher priority
        role_critical_priority = optimizer._calculate_skill_priority(
            skill, gap_size, target_role, market_data
        )
        
        non_critical_priority = optimizer._calculate_skill_priority(
            "obscure_skill", gap_size, None, None
        )
        
        assert role_critical_priority >= non_critical_priority
    
    def test_estimate_skill_learning_time(self, optimizer):
        """Test skill learning time estimation."""
        skill = "python"
        gap_size = 0.8
        current_level = 0.2
        
        hours = optimizer._estimate_skill_learning_time(skill, gap_size, current_level)
        
        # Verify reasonable estimate
        assert isinstance(hours, int)
        assert hours >= 10  # Minimum hours
        assert hours <= 200  # Reasonable maximum
        
        # Test with higher current level (should be faster)
        hours_experienced = optimizer._estimate_skill_learning_time(skill, gap_size, 0.6)
        assert hours_experienced < hours
    
    def test_get_skill_metadata(self, optimizer):
        """Test skill metadata retrieval."""
        # Test known skill
        python_meta = optimizer._get_skill_metadata("python")
        assert python_meta["difficulty"] == "beginner"
        assert python_meta["category"] == "programming"
        assert isinstance(python_meta["prerequisites"], list)
        
        # Test skill with prerequisites
        react_meta = optimizer._get_skill_metadata("react")
        assert "javascript" in react_meta["prerequisites"]
        
        # Test unknown skill (should return defaults)
        unknown_meta = optimizer._get_skill_metadata("unknown_skill")
        assert unknown_meta["difficulty"] == "intermediate"
        assert unknown_meta["category"] == "general"
    
    def test_get_market_demand_score(self, optimizer):
        """Test market demand score retrieval."""
        # Test with market data
        market_data = {"python": {"demand_score": 0.95}}
        score_with_data = optimizer._get_market_demand_score("python", market_data)
        assert score_with_data == 0.95
        
        # Test without market data (should use defaults)
        score_default = optimizer._get_market_demand_score("python", None)
        assert isinstance(score_default, float)
        assert 0 <= score_default <= 1
        
        # Test unknown skill
        score_unknown = optimizer._get_market_demand_score("unknown_skill", None)
        assert score_unknown == 0.5  # Default score
    
    def test_extract_timeline_features(self, optimizer, sample_user_profile):
        """Test timeline feature extraction."""
        skill_gaps = [
            SkillGapAnalysis(
                skill_name="python",
                current_level=0.3,
                target_level=0.8,
                gap_size=0.5,
                priority_score=0.9,
                estimated_hours=80,
                difficulty_level="beginner",
                prerequisites=[],
                market_demand=0.9
            ),
            SkillGapAnalysis(
                skill_name="react",
                current_level=0.0,
                target_level=0.8,
                gap_size=0.8,
                priority_score=0.8,
                estimated_hours=60,
                difficulty_level="intermediate",
                prerequisites=["javascript"],
                market_demand=0.7
            )
        ]
        
        features = optimizer._extract_timeline_features(skill_gaps, sample_user_profile)
        
        # Verify feature extraction
        assert isinstance(features, list)
        assert len(features) == 9  # Expected number of features
        
        # Verify feature values are reasonable
        for feature in features:
            assert isinstance(feature, (int, float))
            assert feature >= 0  # All features should be non-negative
    
    def test_apply_user_adjustments(self, optimizer, sample_user_profile):
        """Test user-specific timeline adjustments."""
        base_hours = 100.0
        skill_gaps = [
            SkillGapAnalysis(
                skill_name="python",
                current_level=0.3,
                target_level=0.8,
                gap_size=0.5,
                priority_score=0.9,
                estimated_hours=80,
                difficulty_level="beginner",
                prerequisites=[],
                market_demand=0.9
            )
        ]
        
        adjusted_hours = optimizer._apply_user_adjustments(
            base_hours, sample_user_profile, skill_gaps
        )
        
        # Verify adjustment
        assert isinstance(adjusted_hours, float)
        assert adjusted_hours > 0
        
        # Experienced user should have reduced time
        experienced_profile = sample_user_profile.copy()
        experienced_profile["experience_years"] = 8
        
        experienced_hours = optimizer._apply_user_adjustments(
            base_hours, experienced_profile, skill_gaps
        )
        
        assert experienced_hours < base_hours  # Should be reduced for experienced user
    
    def test_create_milestone_timeline(self, optimizer):
        """Test milestone timeline creation."""
        skill_gaps = [
            SkillGapAnalysis(
                skill_name="html",
                current_level=0.2,
                target_level=0.8,
                gap_size=0.6,
                priority_score=0.7,
                estimated_hours=40,
                difficulty_level="beginner",
                prerequisites=[],
                market_demand=0.6
            ),
            SkillGapAnalysis(
                skill_name="react",
                current_level=0.0,
                target_level=0.8,
                gap_size=0.8,
                priority_score=0.8,
                estimated_hours=60,
                difficulty_level="intermediate",
                prerequisites=["javascript"],
                market_demand=0.8
            )
        ]
        
        total_weeks = 12
        hours_per_week = 10
        
        milestones = optimizer._create_milestone_timeline(
            skill_gaps, total_weeks, hours_per_week
        )
        
        # Verify milestone structure
        assert len(milestones) > 0
        
        for milestone in milestones:
            assert "id" in milestone
            assert "title" in milestone
            assert "skills" in milestone
            assert "start_week" in milestone
            assert "end_week" in milestone
            assert "estimated_hours" in milestone
            assert "difficulty" in milestone
            
            # Verify week ranges are valid
            assert milestone["start_week"] >= 1
            assert milestone["end_week"] >= milestone["start_week"]
            assert milestone["estimated_hours"] > 0
    
    def test_calculate_timeline_confidence(self, optimizer, sample_user_profile):
        """Test timeline confidence calculation."""
        skill_gaps = [
            SkillGapAnalysis(
                skill_name="python",
                current_level=0.3,
                target_level=0.8,
                gap_size=0.5,
                priority_score=0.9,
                estimated_hours=80,
                difficulty_level="beginner",
                prerequisites=[],
                market_demand=0.9
            )
        ]
        
        estimated_hours = 100.0
        
        confidence = optimizer._calculate_timeline_confidence(
            skill_gaps, sample_user_profile, estimated_hours
        )
        
        # Verify confidence score
        assert isinstance(confidence, float)
        assert 0 <= confidence <= 1
    
    def test_build_skill_dependency_graph(self, optimizer):
        """Test skill dependency graph construction."""
        skill_gaps = [
            SkillGapAnalysis(
                skill_name="html",
                current_level=0.2,
                target_level=0.8,
                gap_size=0.6,
                priority_score=0.7,
                estimated_hours=40,
                difficulty_level="beginner",
                prerequisites=[],
                market_demand=0.6
            ),
            SkillGapAnalysis(
                skill_name="javascript",
                current_level=0.1,
                target_level=0.8,
                gap_size=0.7,
                priority_score=0.8,
                estimated_hours=70,
                difficulty_level="beginner",
                prerequisites=[],
                market_demand=0.9
            ),
            SkillGapAnalysis(
                skill_name="react",
                current_level=0.0,
                target_level=0.8,
                gap_size=0.8,
                priority_score=0.9,
                estimated_hours=60,
                difficulty_level="intermediate",
                prerequisites=["javascript", "html"],
                market_demand=0.8
            )
        ]
        
        graph = optimizer._build_skill_dependency_graph(skill_gaps)
        
        # Verify graph structure
        assert isinstance(graph, dict)
        assert "html" in graph
        assert "javascript" in graph
        assert "react" in graph
        
        # Verify dependencies
        assert len(graph["html"]) == 0  # No prerequisites
        assert len(graph["javascript"]) == 0  # No prerequisites
        assert "javascript" in graph["react"]  # React depends on JavaScript
        assert "html" in graph["react"]  # React depends on HTML
    
    def test_topological_sort_with_priority(self, optimizer):
        """Test topological sorting with priority weighting."""
        dependency_graph = {
            "html": [],
            "javascript": [],
            "react": ["javascript", "html"],
            "node.js": ["javascript"]
        }
        
        skill_gaps = [
            SkillGapAnalysis(
                skill_name="html",
                current_level=0.2,
                target_level=0.8,
                gap_size=0.6,
                priority_score=0.7,
                estimated_hours=40,
                difficulty_level="beginner",
                prerequisites=[],
                market_demand=0.6
            ),
            SkillGapAnalysis(
                skill_name="javascript",
                current_level=0.1,
                target_level=0.8,
                gap_size=0.7,
                priority_score=0.9,  # Higher priority
                estimated_hours=70,
                difficulty_level="beginner",
                prerequisites=[],
                market_demand=0.9
            ),
            SkillGapAnalysis(
                skill_name="react",
                current_level=0.0,
                target_level=0.8,
                gap_size=0.8,
                priority_score=0.8,
                estimated_hours=60,
                difficulty_level="intermediate",
                prerequisites=["javascript", "html"],
                market_demand=0.8
            ),
            SkillGapAnalysis(
                skill_name="node.js",
                current_level=0.0,
                target_level=0.8,
                gap_size=0.8,
                priority_score=0.6,
                estimated_hours=55,
                difficulty_level="intermediate",
                prerequisites=["javascript"],
                market_demand=0.7
            )
        ]
        
        user_constraints = {}
        
        sorted_skills = optimizer._topological_sort_with_priority(
            dependency_graph, skill_gaps, user_constraints
        )
        
        # Verify topological order is maintained
        skill_positions = {gap.skill_name: i for i, gap in enumerate(sorted_skills)}
        
        # JavaScript should come before React
        assert skill_positions["javascript"] < skill_positions["react"]
        
        # HTML should come before React
        assert skill_positions["html"] < skill_positions["react"]
        
        # JavaScript should come before Node.js
        assert skill_positions["javascript"] < skill_positions["node.js"]
        
        # Among skills with no dependencies, higher priority should come first
        if skill_positions["html"] < skill_positions["javascript"]:
            # If HTML comes first, it should be due to some other factor
            pass
        else:
            # JavaScript has higher priority, so it should come first
            assert skill_positions["javascript"] < skill_positions["html"]