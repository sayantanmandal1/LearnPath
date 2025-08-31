"""
User profile model for storing comprehensive user data
"""
from datetime import datetime
from typing import Optional
from uuid import uuid4

from sqlalchemy import DateTime, ForeignKey, JSON, String, Text, Integer
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from app.core.database import Base


class UserProfile(Base):
    """User profile model for storing comprehensive user data"""
    
    __tablename__ = "user_profiles"
    
    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        primary_key=True,
        default=lambda: str(uuid4()),
        index=True
    )
    user_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
        index=True
    )
    
    # Basic profile information
    dream_job: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True
    )
    experience_years: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True
    )
    current_role: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True
    )
    location: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True
    )
    
    # Platform IDs
    github_username: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True
    )
    leetcode_id: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True
    )
    linkedin_url: Mapped[Optional[str]] = mapped_column(
        String(500),
        nullable=True
    )
    codeforces_id: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True
    )
    
    # Extracted data (stored as JSON)
    skills: Mapped[Optional[dict]] = mapped_column(
        JSON,
        nullable=True,
        comment="Unified skills with confidence scores"
    )
    platform_data: Mapped[Optional[dict]] = mapped_column(
        JSON,
        nullable=True,
        comment="Raw data from external platforms"
    )
    resume_data: Mapped[Optional[dict]] = mapped_column(
        JSON,
        nullable=True,
        comment="Parsed resume information"
    )
    
    # Analysis results
    career_interests: Mapped[Optional[dict]] = mapped_column(
        JSON,
        nullable=True,
        comment="Identified career interests and preferences"
    )
    skill_gaps: Mapped[Optional[dict]] = mapped_column(
        JSON,
        nullable=True,
        comment="Identified skill gaps for target roles"
    )
    
    # Metadata
    data_last_updated: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="Last time external data was refreshed"
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False
    )
    
    # Relationship to user
    user = relationship("User", back_populates="profile")
    
    def __repr__(self) -> str:
        return f"<UserProfile(id={self.id}, user_id={self.user_id})>"