"""
Skill schemas for request/response models
"""
from datetime import datetime
from typing import Optional, List

from pydantic import BaseModel, Field, validator


class SkillCreate(BaseModel):
    """Skill creation request schema"""
    name: str = Field(..., max_length=255)
    category: str = Field(..., max_length=100)
    subcategory: Optional[str] = Field(None, max_length=100)
    description: Optional[str] = None
    aliases: Optional[str] = None
    market_demand: Optional[float] = Field(None, ge=0.0, le=1.0)
    average_salary_impact: Optional[float] = None
    
    @validator("name")
    def validate_name(cls, v):
        if not v.strip():
            raise ValueError("Skill name cannot be empty")
        return v.strip().title()
    
    @validator("category")
    def validate_category(cls, v):
        allowed_categories = [
            "programming", "soft_skills", "tools", "frameworks", 
            "databases", "cloud", "devops", "design", "analytics",
            "project_management", "languages", "certifications"
        ]
        if v.lower() not in allowed_categories:
            raise ValueError(f"Category must be one of: {', '.join(allowed_categories)}")
        return v.lower()


class SkillUpdate(BaseModel):
    """Skill update request schema"""
    name: Optional[str] = Field(None, max_length=255)
    category: Optional[str] = Field(None, max_length=100)
    subcategory: Optional[str] = Field(None, max_length=100)
    description: Optional[str] = None
    aliases: Optional[str] = None
    market_demand: Optional[float] = Field(None, ge=0.0, le=1.0)
    average_salary_impact: Optional[float] = None
    is_active: Optional[bool] = None


class SkillResponse(BaseModel):
    """Skill response schema"""
    id: str
    name: str
    category: str
    subcategory: Optional[str]
    description: Optional[str]
    aliases: Optional[str]
    market_demand: Optional[float]
    average_salary_impact: Optional[float]
    is_active: bool
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


class UserSkillCreate(BaseModel):
    """User skill creation request schema"""
    skill_id: str
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    proficiency_level: str = Field(..., max_length=50)
    source: str = Field(..., max_length=100)
    evidence: Optional[str] = None
    years_experience: Optional[float] = Field(None, ge=0.0, le=50.0)
    last_used: Optional[datetime] = None
    is_verified: bool = False
    
    @validator("proficiency_level")
    def validate_proficiency_level(cls, v):
        allowed_levels = ["beginner", "intermediate", "advanced", "expert"]
        if v.lower() not in allowed_levels:
            raise ValueError(f"Proficiency level must be one of: {', '.join(allowed_levels)}")
        return v.lower()
    
    @validator("source")
    def validate_source(cls, v):
        allowed_sources = ["resume", "github", "leetcode", "linkedin", "manual", "assessment"]
        if v.lower() not in allowed_sources:
            raise ValueError(f"Source must be one of: {', '.join(allowed_sources)}")
        return v.lower()


class UserSkillUpdate(BaseModel):
    """User skill update request schema"""
    confidence_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    proficiency_level: Optional[str] = Field(None, max_length=50)
    evidence: Optional[str] = None
    years_experience: Optional[float] = Field(None, ge=0.0, le=50.0)
    last_used: Optional[datetime] = None
    is_verified: Optional[bool] = None


class UserSkillResponse(BaseModel):
    """User skill response schema"""
    id: str
    user_id: str
    skill_id: str
    confidence_score: float
    proficiency_level: str
    source: str
    evidence: Optional[str]
    years_experience: Optional[float]
    last_used: Optional[datetime]
    is_verified: bool
    created_at: datetime
    updated_at: datetime
    skill: Optional[SkillResponse] = None
    
    class Config:
        from_attributes = True


class SkillSearch(BaseModel):
    """Skill search request schema"""
    query: str = Field(..., min_length=1, max_length=100)
    category: Optional[str] = None
    limit: int = Field(20, ge=1, le=100)


class SkillCategoryResponse(BaseModel):
    """Skill category response schema"""
    id: str
    name: str
    parent_id: Optional[str]
    description: Optional[str]
    display_order: int
    is_active: bool
    created_at: datetime
    children: Optional[List["SkillCategoryResponse"]] = None
    
    class Config:
        from_attributes = True


class SkillDemandStats(BaseModel):
    """Skill demand statistics schema"""
    skill_id: str
    total_demand: int
    importance_breakdown: dict
    period_days: int
    trend: Optional[str] = None


class BulkSkillCreate(BaseModel):
    """Bulk skill creation schema"""
    skills: List[SkillCreate] = Field(..., min_items=1, max_items=100)


class SkillSuggestion(BaseModel):
    """Skill suggestion schema"""
    skill_name: str
    confidence: float
    category: str
    reason: str
    market_demand: Optional[float] = None


# Update forward references
SkillCategoryResponse.model_rebuild()