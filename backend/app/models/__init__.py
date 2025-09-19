"""
Database models package
"""

# Import all models to ensure they are registered with SQLAlchemy
from .user import User, RefreshToken
from .profile import UserProfile
from .skill import Skill, UserSkill, SkillCategory
from .job import JobPosting, JobSkill, Company
from .resume import ResumeData, ProcessingStatus
from .platform_account import PlatformAccount, PlatformScrapingLog, PlatformType, ScrapingStatus
from .analysis_result import AnalysisResult, JobRecommendation, AnalysisType, AnalysisStatus
from .job_application import JobApplication, JobApplicationFeedback, JobRecommendationFeedback

__all__ = [
    "User",
    "RefreshToken", 
    "UserProfile",
    "Skill",
    "UserSkill",
    "SkillCategory",
    "JobPosting",
    "JobSkill",
    "Company",
    "ResumeData",
    "ProcessingStatus",
    "PlatformAccount",
    "PlatformScrapingLog",
    "PlatformType",
    "ScrapingStatus",
    "AnalysisResult",
    "JobRecommendation",
    "AnalysisType",
    "AnalysisStatus",
    "JobApplication",
    "JobApplicationFeedback",
    "JobRecommendationFeedback",
]