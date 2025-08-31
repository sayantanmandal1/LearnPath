"""
Pydantic schemas for request/response validation
"""

from .auth import (
    UserRegister, UserLogin, TokenResponse, TokenRefresh, 
    UserResponse, PasswordChange, PasswordReset, PasswordResetConfirm,
    UserCreate, UserUpdate
)
from .profile import (
    ProfileCreate, ProfileUpdate, ProfileResponse, 
    ResumeUpload, SkillExtraction, PlatformDataUpdate
)
from .skill import (
    SkillCreate, SkillUpdate, SkillResponse,
    UserSkillCreate, UserSkillUpdate, UserSkillResponse,
    SkillSearch, SkillCategoryResponse, SkillDemandStats,
    BulkSkillCreate, SkillSuggestion
)
from .job import (
    JobPostingCreate, JobPostingUpdate, JobPostingResponse,
    JobSearch, JobSkillCreate, JobSkillResponse,
    CompanyCreate, CompanyResponse, JobMatchResult,
    MarketTrendData, BulkJobCreate
)

__all__ = [
    # Auth schemas
    "UserRegister", "UserLogin", "TokenResponse", "TokenRefresh",
    "UserResponse", "PasswordChange", "PasswordReset", "PasswordResetConfirm",
    "UserCreate", "UserUpdate",
    
    # Profile schemas
    "ProfileCreate", "ProfileUpdate", "ProfileResponse",
    "ResumeUpload", "SkillExtraction", "PlatformDataUpdate",
    
    # Skill schemas
    "SkillCreate", "SkillUpdate", "SkillResponse",
    "UserSkillCreate", "UserSkillUpdate", "UserSkillResponse",
    "SkillSearch", "SkillCategoryResponse", "SkillDemandStats",
    "BulkSkillCreate", "SkillSuggestion",
    
    # Job schemas
    "JobPostingCreate", "JobPostingUpdate", "JobPostingResponse",
    "JobSearch", "JobSkillCreate", "JobSkillResponse",
    "CompanyCreate", "CompanyResponse", "JobMatchResult",
    "MarketTrendData", "BulkJobCreate",
]