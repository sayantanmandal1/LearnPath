"""
Learning Path Schemas for the AI Career Recommender System.

This module defines Pydantic models for learning paths, resources, milestones,
and related data structures used in the personalized learning path generation system.
"""

from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Union
from enum import Enum
from pydantic import BaseModel, Field, validator
from uuid import uuid4


class DifficultyLevel(str, Enum):
    """Difficulty levels for learning resources and paths."""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


class ResourceType(str, Enum):
    """Types of learning resources."""
    COURSE = "course"
    BOOK = "book"
    PROJECT = "project"
    TUTORIAL = "tutorial"
    VIDEO = "video"
    ARTICLE = "article"
    PRACTICE = "practice"
    CERTIFICATION = "certification"


class ResourceProvider(str, Enum):
    """Learning resource providers."""
    COURSERA = "coursera"
    UDEMY = "udemy"
    EDX = "edx"
    FREECODECAMP = "freecodecamp"
    GITHUB = "github"
    YOUTUBE = "youtube"
    PLURALSIGHT = "pluralsight"
    UDACITY = "udacity"
    CODECADEMY = "codecademy"
    KHAN_ACADEMY = "khan_academy"
    OTHER = "other"


class LearningResource(BaseModel):
    """Individual learning resource model."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    title: str = Field(..., description="Resource title")
    description: Optional[str] = Field(None, description="Resource description")
    type: ResourceType = Field(..., description="Type of resource")
    provider: ResourceProvider = Field(..., description="Resource provider")
    url: str = Field(..., description="Resource URL")
    rating: Optional[float] = Field(None, ge=0, le=5, description="Resource rating (0-5)")
    duration_hours: Optional[int] = Field(None, ge=0, description="Estimated duration in hours")
    cost: Optional[float] = Field(None, ge=0, description="Cost in USD, None for free")
    difficulty_level: DifficultyLevel = Field(..., description="Difficulty level")
    prerequisites: List[str] = Field(default_factory=list, description="Required skills/knowledge")
    skills_taught: List[str] = Field(default_factory=list, description="Skills taught by this resource")
    tags: List[str] = Field(default_factory=list, description="Resource tags")
    language: str = Field(default="en", description="Resource language")
    certificate_available: bool = Field(default=False, description="Whether certificate is available")
    hands_on_projects: bool = Field(default=False, description="Whether includes hands-on projects")
    quality_score: Optional[float] = Field(None, ge=0, le=1, description="Quality score (0-1)")
    popularity_score: Optional[float] = Field(None, ge=0, le=1, description="Popularity score (0-1)")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    @validator('url')
    def validate_url(cls, v):
        if not v.startswith(('http://', 'https://')):
            raise ValueError('URL must start with http:// or https://')
        return v


class Milestone(BaseModel):
    """Learning path milestone model."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    title: str = Field(..., description="Milestone title")
    description: Optional[str] = Field(None, description="Milestone description")
    order: int = Field(..., ge=0, description="Order in the learning path")
    skills_to_acquire: List[str] = Field(default_factory=list, description="Skills to acquire")
    resources: List[str] = Field(default_factory=list, description="Resource IDs for this milestone")
    estimated_duration_hours: int = Field(..., ge=0, description="Estimated duration in hours")
    completion_criteria: List[str] = Field(default_factory=list, description="Completion criteria")
    projects: List[str] = Field(default_factory=list, description="Project recommendations")
    is_mandatory: bool = Field(default=True, description="Whether milestone is mandatory")


class SkillGap(BaseModel):
    """Skill gap analysis model."""
    skill_name: str = Field(..., description="Name of the skill")
    current_level: float = Field(..., ge=0, le=1, description="Current skill level (0-1)")
    target_level: float = Field(..., ge=0, le=1, description="Target skill level (0-1)")
    gap_size: float = Field(..., ge=0, le=1, description="Size of the gap (0-1)")
    priority: float = Field(..., ge=0, le=1, description="Priority for learning (0-1)")
    market_demand: float = Field(..., ge=0, le=1, description="Market demand for skill (0-1)")
    estimated_learning_hours: int = Field(..., ge=0, description="Estimated hours to bridge gap")
    difficulty: DifficultyLevel = Field(..., description="Difficulty to learn this skill")


class LearningPath(BaseModel):
    """Complete learning path model."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    title: str = Field(..., description="Learning path title")
    description: Optional[str] = Field(None, description="Learning path description")
    target_role: Optional[str] = Field(None, description="Target job role")
    target_skills: List[str] = Field(default_factory=list, description="Skills to be acquired")
    skill_gaps: List[SkillGap] = Field(default_factory=list, description="Skill gaps to address")
    difficulty_level: DifficultyLevel = Field(..., description="Overall difficulty level")
    estimated_duration_weeks: int = Field(..., ge=0, description="Estimated duration in weeks")
    estimated_duration_hours: int = Field(..., ge=0, description="Estimated duration in hours")
    milestones: List[Milestone] = Field(default_factory=list, description="Learning milestones")
    resources: List[LearningResource] = Field(default_factory=list, description="Learning resources")
    prerequisites: List[str] = Field(default_factory=list, description="Prerequisites")
    learning_objectives: List[str] = Field(default_factory=list, description="Learning objectives")
    success_metrics: List[str] = Field(default_factory=list, description="Success metrics")
    tags: List[str] = Field(default_factory=list, description="Path tags")
    is_personalized: bool = Field(default=True, description="Whether path is personalized")
    confidence_score: Optional[float] = Field(None, ge=0, le=1, description="Confidence in recommendations")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    @validator('milestones')
    def validate_milestone_order(cls, v):
        if v:
            orders = [m.order for m in v]
            if len(orders) != len(set(orders)):
                raise ValueError('Milestone orders must be unique')
            if orders != sorted(orders):
                raise ValueError('Milestones must be ordered sequentially')
        return v


class LearningPathRequest(BaseModel):
    """Request model for generating learning paths."""
    user_id: str = Field(..., description="User ID")
    target_role: Optional[str] = Field(None, description="Target job role")
    target_skills: List[str] = Field(default_factory=list, description="Specific skills to learn")
    current_skills: Dict[str, float] = Field(default_factory=dict, description="Current skill levels")
    time_commitment_hours_per_week: int = Field(default=10, ge=1, le=168, description="Weekly time commitment")
    preferred_learning_style: Optional[str] = Field(None, description="Preferred learning style")
    budget_limit: Optional[float] = Field(None, ge=0, description="Budget limit in USD")
    include_free_only: bool = Field(default=False, description="Include only free resources")
    preferred_providers: List[ResourceProvider] = Field(default_factory=list, description="Preferred providers")
    difficulty_preference: Optional[DifficultyLevel] = Field(None, description="Preferred difficulty level")
    include_certifications: bool = Field(default=True, description="Include certification courses")
    include_projects: bool = Field(default=True, description="Include hands-on projects")


class LearningPathResponse(BaseModel):
    """Response model for learning path generation."""
    learning_paths: List[LearningPath] = Field(default_factory=list, description="Generated learning paths")
    total_paths: int = Field(..., description="Total number of paths generated")
    skill_gaps_identified: List[SkillGap] = Field(default_factory=list, description="Identified skill gaps")
    recommendations_metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    generation_timestamp: datetime = Field(default_factory=datetime.utcnow)


class ProjectRecommendation(BaseModel):
    """Project recommendation model."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    title: str = Field(..., description="Project title")
    description: str = Field(..., description="Project description")
    repository_url: Optional[str] = Field(None, description="GitHub repository URL")
    difficulty_level: DifficultyLevel = Field(..., description="Project difficulty")
    skills_practiced: List[str] = Field(default_factory=list, description="Skills practiced")
    technologies: List[str] = Field(default_factory=list, description="Technologies used")
    estimated_duration_hours: int = Field(..., ge=0, description="Estimated completion time")
    stars: Optional[int] = Field(None, ge=0, description="GitHub stars")
    forks: Optional[int] = Field(None, ge=0, description="GitHub forks")
    last_updated: Optional[datetime] = Field(None, description="Last updated date")
    trending_score: Optional[float] = Field(None, ge=0, le=1, description="Trending score")
    learning_value: Optional[float] = Field(None, ge=0, le=1, description="Learning value score")
    market_relevance: Optional[float] = Field(None, ge=0, le=1, description="Market relevance score")


class LearningProgress(BaseModel):
    """Learning progress tracking model."""
    user_id: str = Field(..., description="User ID")
    learning_path_id: str = Field(..., description="Learning path ID")
    milestone_id: str = Field(..., description="Milestone ID")
    resource_id: Optional[str] = Field(None, description="Resource ID")
    progress_percentage: float = Field(..., ge=0, le=100, description="Progress percentage")
    time_spent_hours: float = Field(default=0, ge=0, description="Time spent in hours")
    completed: bool = Field(default=False, description="Whether completed")
    completion_date: Optional[datetime] = Field(None, description="Completion date")
    notes: Optional[str] = Field(None, description="User notes")
    rating: Optional[int] = Field(None, ge=1, le=5, description="User rating")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class LearningPathUpdate(BaseModel):
    """Model for updating learning paths."""
    title: Optional[str] = Field(None, description="Updated title")
    description: Optional[str] = Field(None, description="Updated description")
    target_skills: Optional[List[str]] = Field(None, description="Updated target skills")
    estimated_duration_weeks: Optional[int] = Field(None, ge=0, description="Updated duration")
    milestones: Optional[List[Milestone]] = Field(None, description="Updated milestones")
    resources: Optional[List[LearningResource]] = Field(None, description="Updated resources")