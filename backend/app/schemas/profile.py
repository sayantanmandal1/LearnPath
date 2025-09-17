"""
Profile schemas for request/response models
"""
from datetime import datetime
from typing import Optional, Dict, Any, List

from pydantic import BaseModel, Field, validator


class ProfileCreate(BaseModel):
    """Profile creation request schema"""
    # Basic profile information
    dream_job: Optional[str] = Field(None, max_length=255)
    experience_years: Optional[int] = Field(None, ge=0, le=50)
    current_role: Optional[str] = Field(None, max_length=255)
    location: Optional[str] = Field(None, max_length=255)
    
    # Platform IDs
    github_username: Optional[str] = Field(None, max_length=100)
    leetcode_id: Optional[str] = Field(None, max_length=100)
    linkedin_url: Optional[str] = Field(None, max_length=500)
    codeforces_id: Optional[str] = Field(None, max_length=100)
    
    # Frontend analyze page fields
    industry: Optional[str] = Field(None, max_length=255)
    desired_role: Optional[str] = Field(None, max_length=255)
    career_goals: Optional[str] = Field(None, max_length=2000)
    timeframe: Optional[str] = Field(None, max_length=100)
    salary_expectation: Optional[str] = Field(None, max_length=100)
    education: Optional[str] = Field(None, max_length=500)
    certifications: Optional[str] = Field(None, max_length=1000)
    languages: Optional[str] = Field(None, max_length=500)
    work_type: Optional[str] = Field(None, max_length=100)
    company_size: Optional[str] = Field(None, max_length=100)
    work_culture: Optional[str] = Field(None, max_length=2000)
    benefits: Optional[List[str]] = None
    
    # Skills and analysis data
    skills: Optional[Dict[str, float]] = None
    career_interests: Optional[Dict[str, Any]] = None
    
    @validator("github_username")
    def validate_github_username(cls, v):
        if v and not v.replace("-", "").replace("_", "").isalnum():
            raise ValueError("Invalid GitHub username format")
        return v
    
    @validator("linkedin_url")
    def validate_linkedin_url(cls, v):
        if v and not (v.startswith("https://linkedin.com/") or v.startswith("https://www.linkedin.com/")):
            raise ValueError("Invalid LinkedIn URL format")
        return v


class ProfileUpdate(BaseModel):
    """Profile update request schema"""
    # Basic profile information
    dream_job: Optional[str] = Field(None, max_length=255)
    experience_years: Optional[int] = Field(None, ge=0, le=50)
    current_role: Optional[str] = Field(None, max_length=255)
    location: Optional[str] = Field(None, max_length=255)
    
    # Platform IDs
    github_username: Optional[str] = Field(None, max_length=100)
    leetcode_id: Optional[str] = Field(None, max_length=100)
    linkedin_url: Optional[str] = Field(None, max_length=500)
    codeforces_id: Optional[str] = Field(None, max_length=100)
    
    # Frontend analyze page fields
    industry: Optional[str] = Field(None, max_length=255)
    desired_role: Optional[str] = Field(None, max_length=255)
    career_goals: Optional[str] = Field(None, max_length=2000)
    timeframe: Optional[str] = Field(None, max_length=100)
    salary_expectation: Optional[str] = Field(None, max_length=100)
    education: Optional[str] = Field(None, max_length=500)
    certifications: Optional[str] = Field(None, max_length=1000)
    languages: Optional[str] = Field(None, max_length=500)
    work_type: Optional[str] = Field(None, max_length=100)
    company_size: Optional[str] = Field(None, max_length=100)
    work_culture: Optional[str] = Field(None, max_length=2000)
    benefits: Optional[List[str]] = None
    
    # Skills and analysis data
    skills: Optional[Dict[str, float]] = None
    career_interests: Optional[Dict[str, Any]] = None
    skill_gaps: Optional[Dict[str, Any]] = None


class ProfileResponse(BaseModel):
    """Profile response schema"""
    id: str
    user_id: str
    
    # Basic profile information
    dream_job: Optional[str]
    experience_years: Optional[int]
    current_role: Optional[str]
    location: Optional[str]
    
    # Platform IDs
    github_username: Optional[str]
    leetcode_id: Optional[str]
    linkedin_url: Optional[str]
    codeforces_id: Optional[str]
    
    # Frontend analyze page fields
    industry: Optional[str]
    desired_role: Optional[str]
    career_goals: Optional[str]
    timeframe: Optional[str]
    salary_expectation: Optional[str]
    education: Optional[str]
    certifications: Optional[str]
    languages: Optional[str]
    work_type: Optional[str]
    company_size: Optional[str]
    work_culture: Optional[str]
    benefits: Optional[List[str]]
    
    # Skills and analysis data
    skills: Optional[Dict[str, float]]
    platform_data: Optional[Dict[str, Any]]
    resume_data: Optional[Dict[str, Any]]
    career_interests: Optional[Dict[str, Any]]
    skill_gaps: Optional[Dict[str, Any]]
    
    # Profile analytics and scoring
    profile_score: Optional[float]
    completeness_score: Optional[float]
    
    # Metadata
    data_last_updated: Optional[datetime]
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


class ResumeUpload(BaseModel):
    """Resume upload request schema"""
    file_content: bytes
    filename: str
    content_type: str
    
    @validator("content_type")
    def validate_content_type(cls, v):
        allowed_types = [
            "application/pdf",
            "application/msword",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        ]
        if v not in allowed_types:
            raise ValueError("Only PDF and Word documents are supported")
        return v


class SkillExtraction(BaseModel):
    """Skill extraction result schema"""
    skills: Dict[str, float] = Field(..., description="Extracted skills with confidence scores")
    source: str = Field(..., description="Source of extraction")
    metadata: Optional[Dict[str, Any]] = None


class PlatformDataUpdate(BaseModel):
    """Platform data update schema"""
    platform: str = Field(..., description="Platform name (github, leetcode, linkedin)")
    data: Dict[str, Any] = Field(..., description="Platform-specific data")
    
    @validator("platform")
    def validate_platform(cls, v):
        allowed_platforms = ["github", "leetcode", "linkedin", "codeforces"]
        if v not in allowed_platforms:
            raise ValueError(f"Platform must be one of: {', '.join(allowed_platforms)}")
        return v