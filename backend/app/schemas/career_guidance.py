"""
Career Guidance Schema Definitions
"""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum


class DifficultyLevel(str, Enum):
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


class ResourceType(str, Enum):
    COURSE = "course"
    BOOK = "book"
    TUTORIAL = "tutorial"
    DOCUMENTATION = "documentation"
    PROJECT = "project"
    PRACTICE = "practice"
    CERTIFICATION = "certification"


class FocusArea(BaseModel):
    """Focus area recommendation for career development"""
    id: str
    name: str
    description: str
    importance_score: float = Field(..., ge=0, le=10)
    current_level: DifficultyLevel
    target_level: DifficultyLevel
    skills_required: List[str]
    estimated_time_weeks: int
    priority_rank: int


class LearningOutcome(BaseModel):
    """Expected learning outcome from a project or resource"""
    id: str
    description: str
    skills_gained: List[str]
    competency_level: DifficultyLevel
    measurable_criteria: List[str]


class ProjectSpecification(BaseModel):
    """Detailed project specification with learning outcomes"""
    id: str
    title: str
    description: str
    difficulty_level: DifficultyLevel
    estimated_duration_weeks: int
    technologies: List[str]
    learning_outcomes: List[LearningOutcome]
    prerequisites: List[str]
    deliverables: List[str]
    success_metrics: List[str]
    github_template_url: Optional[str] = None


class Milestone(BaseModel):
    """Milestone in preparation roadmap"""
    id: str
    title: str
    description: str
    target_date: datetime
    completion_criteria: List[str]
    dependencies: List[str] = []
    estimated_effort_hours: int
    resources_needed: List[str]


class PreparationRoadmap(BaseModel):
    """Comprehensive preparation roadmap with timelines"""
    id: str
    target_role: str
    total_duration_weeks: int
    phases: List[Dict[str, Any]]
    milestones: List[Milestone]
    critical_path: List[str]
    buffer_time_weeks: int
    success_probability: float = Field(..., ge=0, le=1)


class ResourceRating(BaseModel):
    """Quality rating for learning resources"""
    overall_score: float = Field(..., ge=0, le=5)
    content_quality: float = Field(..., ge=0, le=5)
    difficulty_accuracy: float = Field(..., ge=0, le=5)
    practical_relevance: float = Field(..., ge=0, le=5)
    community_rating: float = Field(..., ge=0, le=5)
    last_updated: datetime


class CuratedResource(BaseModel):
    """Curated learning resource with quality ratings"""
    id: str
    title: str
    description: str
    resource_type: ResourceType
    url: str
    provider: str
    difficulty_level: DifficultyLevel
    estimated_time_hours: int
    cost: Optional[float] = None
    currency: str = "USD"
    rating: ResourceRating
    tags: List[str]
    prerequisites: List[str]
    learning_outcomes: List[str]
    is_free: bool
    certification_available: bool = False


class CareerGuidanceResponse(BaseModel):
    """Complete career guidance response"""
    user_id: str
    target_role: str
    generated_at: datetime
    focus_areas: List[FocusArea]
    project_specifications: List[ProjectSpecification]
    preparation_roadmap: PreparationRoadmap
    curated_resources: List[CuratedResource]
    personalization_factors: Dict[str, Any]
    confidence_score: float = Field(..., ge=0, le=1)


class CareerGuidanceRequest(BaseModel):
    """Request for career guidance generation"""
    user_id: str
    target_role: str
    current_experience_years: int
    preferred_learning_style: Optional[str] = None
    time_commitment_hours_per_week: int = 10
    budget_limit: Optional[float] = None
    specific_interests: List[str] = []
    career_timeline_months: int = 12