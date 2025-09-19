"""
Job application tracking schemas.
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, validator


class JobApplicationCreate(BaseModel):
    """Schema for creating a job application."""
    job_posting_id: str = Field(..., description="External job posting ID")
    job_title: str = Field(..., max_length=255)
    company_name: str = Field(..., max_length=255)
    job_url: Optional[str] = None
    match_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    skill_matches: Optional[List[str]] = None
    skill_gaps: Optional[List[str]] = None
    application_method: Optional[str] = Field(None, max_length=100)
    cover_letter: Optional[str] = None
    notes: Optional[str] = None


class JobApplicationUpdate(BaseModel):
    """Schema for updating a job application."""
    status: Optional[str] = Field(None, max_length=50)
    applied_date: Optional[datetime] = None
    application_method: Optional[str] = Field(None, max_length=100)
    cover_letter: Optional[str] = None
    notes: Optional[str] = None
    interview_scheduled: Optional[bool] = None
    interview_date: Optional[datetime] = None
    interview_notes: Optional[str] = None
    feedback_received: Optional[bool] = None
    feedback_text: Optional[str] = None
    rejection_reason: Optional[str] = Field(None, max_length=255)
    
    @validator("status")
    def validate_status(cls, v):
        if v and v not in ["interested", "applied", "interviewing", "rejected", "accepted", "withdrawn"]:
            raise ValueError("Invalid application status")
        return v


class JobApplicationResponse(BaseModel):
    """Schema for job application response."""
    id: str
    user_id: str
    job_posting_id: str
    job_title: str
    company_name: str
    job_url: Optional[str]
    status: str
    applied_date: Optional[datetime]
    last_updated: datetime
    match_score: Optional[float]
    skill_matches: Optional[List[str]]
    skill_gaps: Optional[List[str]]
    application_method: Optional[str]
    cover_letter: Optional[str]
    notes: Optional[str]
    interview_scheduled: bool
    interview_date: Optional[datetime]
    interview_notes: Optional[str]
    feedback_received: bool
    feedback_text: Optional[str]
    rejection_reason: Optional[str]
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


class JobApplicationFeedbackCreate(BaseModel):
    """Schema for creating job application feedback."""
    feedback_type: str = Field(..., max_length=50)
    rating: Optional[int] = Field(None, ge=1, le=5)
    feedback_text: Optional[str] = None
    match_accuracy_rating: Optional[int] = Field(None, ge=1, le=5)
    recommendation_helpfulness: Optional[int] = Field(None, ge=1, le=5)
    gap_analysis_accuracy: Optional[int] = Field(None, ge=1, le=5)
    suggested_improvements: Optional[Dict[str, Any]] = None
    
    @validator("feedback_type")
    def validate_feedback_type(cls, v):
        allowed_types = ["recommendation_quality", "match_accuracy", "application_outcome", "general"]
        if v not in allowed_types:
            raise ValueError(f"Feedback type must be one of: {', '.join(allowed_types)}")
        return v


class JobApplicationFeedbackResponse(BaseModel):
    """Schema for job application feedback response."""
    id: str
    application_id: str
    feedback_type: str
    rating: Optional[int]
    feedback_text: Optional[str]
    match_accuracy_rating: Optional[int]
    recommendation_helpfulness: Optional[int]
    gap_analysis_accuracy: Optional[int]
    suggested_improvements: Optional[Dict[str, Any]]
    created_at: datetime
    
    class Config:
        from_attributes = True


class JobRecommendationFeedbackCreate(BaseModel):
    """Schema for creating job recommendation feedback."""
    job_posting_id: str
    user_interested: Optional[bool] = None
    user_applied: bool = False
    match_score_feedback: Optional[str] = Field(None, max_length=50)
    skill_match_feedback: Optional[str] = Field(None, max_length=50)
    location_feedback: Optional[str] = Field(None, max_length=50)
    feedback_text: Optional[str] = None
    improvement_suggestions: Optional[Dict[str, Any]] = None
    
    @validator("match_score_feedback")
    def validate_match_score_feedback(cls, v):
        if v and v not in ["too_high", "accurate", "too_low"]:
            raise ValueError("Match score feedback must be: too_high, accurate, or too_low")
        return v
    
    @validator("skill_match_feedback")
    def validate_skill_match_feedback(cls, v):
        if v and v not in ["accurate", "missing_skills", "wrong_skills"]:
            raise ValueError("Skill match feedback must be: accurate, missing_skills, or wrong_skills")
        return v
    
    @validator("location_feedback")
    def validate_location_feedback(cls, v):
        if v and v not in ["good", "not_preferred", "wrong_location"]:
            raise ValueError("Location feedback must be: good, not_preferred, or wrong_location")
        return v


class JobRecommendationFeedbackResponse(BaseModel):
    """Schema for job recommendation feedback response."""
    id: str
    user_id: str
    job_posting_id: str
    recommendation_shown: bool
    user_interested: Optional[bool]
    user_applied: bool
    match_score_feedback: Optional[str]
    skill_match_feedback: Optional[str]
    location_feedback: Optional[str]
    feedback_text: Optional[str]
    improvement_suggestions: Optional[Dict[str, Any]]
    created_at: datetime
    
    class Config:
        from_attributes = True


class JobApplicationStats(BaseModel):
    """Schema for job application statistics."""
    total_applications: int
    status_breakdown: Dict[str, int]
    average_match_score: Optional[float]
    applications_this_month: int
    interviews_scheduled: int
    success_rate: float
    top_companies: List[Dict[str, Any]]
    application_timeline: List[Dict[str, Any]]


class JobRecommendationMetrics(BaseModel):
    """Schema for job recommendation metrics."""
    total_recommendations: int
    user_engagement_rate: float
    application_conversion_rate: float
    average_match_accuracy: float
    feedback_summary: Dict[str, Any]
    improvement_areas: List[str]


class EnhancedJobMatch(BaseModel):
    """Enhanced job match with application tracking."""
    job_posting_id: str
    job_title: str
    company_name: str
    location: str
    job_url: Optional[str]
    match_score: float = Field(ge=0.0, le=1.0)
    skill_matches: List[str]
    skill_gaps: List[str]
    salary_range: Optional[str]
    experience_level: Optional[str]
    posted_date: Optional[datetime]
    source: str
    
    # Application tracking info
    application_status: Optional[str] = None  # If user has applied
    application_id: Optional[str] = None
    
    # Enhanced gap analysis
    gap_analysis: Optional[Dict[str, Any]] = None
    recommendation_reason: str = ""
    
    # Location-based scoring
    location_score: float = Field(ge=0.0, le=1.0, default=0.0)
    is_indian_tech_city: bool = False
    
    # Market insights
    market_demand: Optional[str] = None
    competition_level: Optional[str] = None


class LocationBasedJobSearch(BaseModel):
    """Schema for location-based job search."""
    target_role: str
    preferred_cities: List[str] = Field(default_factory=list)
    max_distance_km: Optional[int] = Field(None, ge=0, le=100)
    remote_acceptable: bool = False
    hybrid_acceptable: bool = True
    salary_min: Optional[int] = Field(None, ge=0)
    salary_max: Optional[int] = Field(None, ge=0)
    experience_level: Optional[str] = None
    limit: int = Field(50, ge=1, le=200)


class IndianTechJobsResponse(BaseModel):
    """Response schema for Indian tech jobs."""
    jobs: List[EnhancedJobMatch]
    total_count: int
    location_distribution: Dict[str, int]
    salary_insights: Dict[str, Any]
    market_trends: Dict[str, Any]
    search_metadata: Dict[str, Any]