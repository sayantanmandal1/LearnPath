"""
Job schemas for request/response models
"""
from datetime import datetime
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field, validator


class JobPostingCreate(BaseModel):
    """Job posting creation request schema"""
    external_id: Optional[str] = Field(None, max_length=255)
    title: str = Field(..., max_length=255)
    company: str = Field(..., max_length=255)
    location: Optional[str] = Field(None, max_length=255)
    remote_type: Optional[str] = Field(None, max_length=50)
    employment_type: Optional[str] = Field(None, max_length=50)
    experience_level: Optional[str] = Field(None, max_length=50)
    description: str
    requirements: Optional[str] = None
    salary_min: Optional[int] = Field(None, ge=0)
    salary_max: Optional[int] = Field(None, ge=0)
    salary_currency: Optional[str] = Field("USD", max_length=10)
    salary_period: Optional[str] = Field(None, max_length=20)
    source: str = Field(..., max_length=100)
    source_url: Optional[str] = Field(None, max_length=1000)
    posted_date: Optional[datetime] = None
    expires_date: Optional[datetime] = None
    
    @validator("remote_type")
    def validate_remote_type(cls, v):
        if v and v.lower() not in ["remote", "hybrid", "onsite"]:
            raise ValueError("Remote type must be one of: remote, hybrid, onsite")
        return v.lower() if v else v
    
    @validator("employment_type")
    def validate_employment_type(cls, v):
        if v and v.lower() not in ["full-time", "part-time", "contract", "internship"]:
            raise ValueError("Employment type must be one of: full-time, part-time, contract, internship")
        return v.lower() if v else v
    
    @validator("experience_level")
    def validate_experience_level(cls, v):
        if v and v.lower() not in ["entry", "mid", "senior", "lead", "executive"]:
            raise ValueError("Experience level must be one of: entry, mid, senior, lead, executive")
        return v.lower() if v else v
    
    @validator("salary_period")
    def validate_salary_period(cls, v):
        if v and v.lower() not in ["yearly", "monthly", "hourly"]:
            raise ValueError("Salary period must be one of: yearly, monthly, hourly")
        return v.lower() if v else v
    
    @validator("source")
    def validate_source(cls, v):
        allowed_sources = ["linkedin", "indeed", "glassdoor", "stackoverflow", "angel", "manual"]
        if v.lower() not in allowed_sources:
            raise ValueError(f"Source must be one of: {', '.join(allowed_sources)}")
        return v.lower()


class JobPostingUpdate(BaseModel):
    """Job posting update request schema"""
    title: Optional[str] = Field(None, max_length=255)
    company: Optional[str] = Field(None, max_length=255)
    location: Optional[str] = Field(None, max_length=255)
    remote_type: Optional[str] = Field(None, max_length=50)
    employment_type: Optional[str] = Field(None, max_length=50)
    experience_level: Optional[str] = Field(None, max_length=50)
    description: Optional[str] = None
    requirements: Optional[str] = None
    salary_min: Optional[int] = Field(None, ge=0)
    salary_max: Optional[int] = Field(None, ge=0)
    salary_currency: Optional[str] = Field(None, max_length=10)
    salary_period: Optional[str] = Field(None, max_length=20)
    source_url: Optional[str] = Field(None, max_length=1000)
    expires_date: Optional[datetime] = None
    processed_skills: Optional[Dict[str, Any]] = None
    market_analysis: Optional[Dict[str, Any]] = None
    is_active: Optional[bool] = None
    is_processed: Optional[bool] = None
    quality_score: Optional[float] = Field(None, ge=0.0, le=1.0)


class JobPostingResponse(BaseModel):
    """Job posting response schema"""
    id: str
    external_id: Optional[str]
    title: str
    company: str
    location: Optional[str]
    remote_type: Optional[str]
    employment_type: Optional[str]
    experience_level: Optional[str]
    description: str
    requirements: Optional[str]
    salary_min: Optional[int]
    salary_max: Optional[int]
    salary_currency: Optional[str]
    salary_period: Optional[str]
    source: str
    source_url: Optional[str]
    posted_date: Optional[datetime]
    expires_date: Optional[datetime]
    processed_skills: Optional[Dict[str, Any]]
    market_analysis: Optional[Dict[str, Any]]
    is_active: bool
    is_processed: bool
    quality_score: Optional[float]
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


class JobSearch(BaseModel):
    """Job search request schema"""
    title: Optional[str] = None
    company: Optional[str] = None
    location: Optional[str] = None
    remote_type: Optional[str] = None
    experience_level: Optional[str] = None
    min_salary: Optional[int] = Field(None, ge=0)
    max_salary: Optional[int] = Field(None, ge=0)
    skills: Optional[List[str]] = None
    skip: int = Field(0, ge=0)
    limit: int = Field(20, ge=1, le=100)


class JobSkillCreate(BaseModel):
    """Job skill creation request schema"""
    job_posting_id: str
    skill_id: str
    importance: str = Field(..., max_length=50)
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    years_required: Optional[int] = Field(None, ge=0, le=20)
    proficiency_level: Optional[str] = Field(None, max_length=50)
    context: Optional[str] = None
    
    @validator("importance")
    def validate_importance(cls, v):
        allowed_importance = ["required", "preferred", "nice-to-have"]
        if v.lower() not in allowed_importance:
            raise ValueError(f"Importance must be one of: {', '.join(allowed_importance)}")
        return v.lower()
    
    @validator("proficiency_level")
    def validate_proficiency_level(cls, v):
        if v and v.lower() not in ["beginner", "intermediate", "advanced", "expert"]:
            raise ValueError("Proficiency level must be one of: beginner, intermediate, advanced, expert")
        return v.lower() if v else v


class JobSkillResponse(BaseModel):
    """Job skill response schema"""
    id: str
    job_posting_id: str
    skill_id: str
    importance: str
    confidence_score: float
    years_required: Optional[int]
    proficiency_level: Optional[str]
    context: Optional[str]
    created_at: datetime
    
    class Config:
        from_attributes = True


class CompanyCreate(BaseModel):
    """Company creation request schema"""
    name: str = Field(..., max_length=255)
    domain: Optional[str] = Field(None, max_length=255)
    industry: Optional[str] = Field(None, max_length=100)
    size: Optional[str] = Field(None, max_length=50)
    description: Optional[str] = None
    headquarters: Optional[str] = Field(None, max_length=255)
    founded_year: Optional[int] = Field(None, ge=1800, le=2030)
    glassdoor_rating: Optional[float] = Field(None, ge=0.0, le=5.0)
    employee_count: Optional[int] = Field(None, ge=1)
    tech_stack: Optional[Dict[str, Any]] = None
    culture_keywords: Optional[Dict[str, Any]] = None
    
    @validator("size")
    def validate_size(cls, v):
        if v and v.lower() not in ["startup", "small", "medium", "large", "enterprise"]:
            raise ValueError("Size must be one of: startup, small, medium, large, enterprise")
        return v.lower() if v else v


class CompanyResponse(BaseModel):
    """Company response schema"""
    id: str
    name: str
    domain: Optional[str]
    industry: Optional[str]
    size: Optional[str]
    description: Optional[str]
    headquarters: Optional[str]
    founded_year: Optional[int]
    glassdoor_rating: Optional[float]
    employee_count: Optional[int]
    tech_stack: Optional[Dict[str, Any]]
    culture_keywords: Optional[Dict[str, Any]]
    is_active: bool
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


class JobMatchResult(BaseModel):
    """Job matching result schema"""
    job: JobPostingResponse
    match_score: float = Field(..., ge=0.0, le=1.0)
    skill_matches: List[str]
    skill_gaps: List[str]
    salary_fit: Optional[str] = None
    location_fit: Optional[str] = None
    experience_fit: Optional[str] = None


class MarketTrendData(BaseModel):
    """Market trend data schema"""
    skill_name: str
    demand_trend: List[Dict[str, Any]]
    salary_trend: List[Dict[str, Any]]
    growth_rate: float
    market_size: int
    top_companies: List[str]
    related_skills: List[str]


class BulkJobCreate(BaseModel):
    """Bulk job creation schema"""
    jobs: List[JobPostingCreate] = Field(..., min_items=1, max_items=100)

class SalaryRange(BaseModel):
    """Salary range schema"""
    min_amount: Optional[int] = Field(None, ge=0)
    max_amount: Optional[int] = Field(None, ge=0)
    currency: str = Field(default="INR", max_length=10)
    period: str = Field(default="annual", max_length=20)


class JobPosting(BaseModel):
    """Job posting schema for real-time job service"""
    job_id: str
    title: str
    company: str
    location: str
    description: str
    required_skills: List[str] = Field(default_factory=list)
    experience_level: Optional[str] = None
    salary_range: Optional[SalaryRange] = None
    posted_date: Optional[datetime] = None
    source: str
    url: str = ""


class SkillMatch(BaseModel):
    """Skill match schema for job matching"""
    skill: str
    user_level: float = Field(ge=0.0, le=1.0)
    required_level: float = Field(ge=0.0, le=1.0)
    match_score: float = Field(ge=0.0, le=1.0)


class SkillGap(BaseModel):
    """Skill gap schema for job matching"""
    skill: str
    gap_level: float = Field(ge=0.0, le=1.0)
    importance: str = Field(default="medium")
    learning_resources: List[str] = Field(default_factory=list)


class JobMatch(BaseModel):
    """Job match schema with compatibility scoring"""
    job_posting: JobPosting
    match_score: float = Field(ge=0.0, le=1.0)
    skill_matches: List[SkillMatch] = Field(default_factory=list)
    skill_gaps: List[SkillGap] = Field(default_factory=list)
    recommendation_reason: str = ""
    compatibility_factors: Dict[str, float] = Field(default_factory=dict)