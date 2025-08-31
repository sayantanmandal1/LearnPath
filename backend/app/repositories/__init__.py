"""
Repository pattern implementation for data access
"""

from .base import BaseRepository
from .user import UserRepository
from .profile import ProfileRepository
from .skill import SkillRepository, UserSkillRepository
from .job import JobRepository

__all__ = [
    "BaseRepository",
    "UserRepository",
    "ProfileRepository", 
    "SkillRepository",
    "UserSkillRepository",
    "JobRepository",
]