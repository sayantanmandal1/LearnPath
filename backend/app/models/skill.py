"""
Skill models for skill taxonomy and user skill relationships
"""
from datetime import datetime
from typing import Optional
from uuid import uuid4

from sqlalchemy import DateTime, ForeignKey, String, Text, Float, Integer, Boolean
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from app.core.database import Base


class Skill(Base):
    """Master skill taxonomy model"""
    
    __tablename__ = "skills"
    
    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        primary_key=True,
        default=lambda: str(uuid4()),
        index=True
    )
    name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        unique=True,
        index=True
    )
    category: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        index=True,
        comment="Category like 'programming', 'soft_skills', 'tools', etc."
    )
    subcategory: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
        comment="Subcategory for more granular classification"
    )
    description: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True
    )
    aliases: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        comment="Comma-separated alternative names for this skill"
    )
    market_demand: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
        comment="Market demand score from 0.0 to 1.0"
    )
    average_salary_impact: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
        comment="Average salary impact percentage"
    )
    is_active: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        nullable=False
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
    
    # Relationships
    user_skills = relationship("UserSkill", back_populates="skill")
    job_skills = relationship("JobSkill", back_populates="skill")
    
    def __repr__(self) -> str:
        return f"<Skill(id={self.id}, name={self.name}, category={self.category})>"


class UserSkill(Base):
    """User-skill relationship with confidence scores and evidence"""
    
    __tablename__ = "user_skills"
    
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
        index=True
    )
    skill_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("skills.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    confidence_score: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        comment="Confidence score from 0.0 to 1.0"
    )
    proficiency_level: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        comment="beginner, intermediate, advanced, expert"
    )
    source: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        comment="Source of skill detection: resume, github, leetcode, manual, etc."
    )
    evidence: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        comment="Supporting evidence for this skill"
    )
    years_experience: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
        comment="Years of experience with this skill"
    )
    last_used: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="When this skill was last used"
    )
    is_verified: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
        comment="Whether user has verified this skill"
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
    
    # Relationships
    skill = relationship("Skill", back_populates="user_skills")
    
    def __repr__(self) -> str:
        return f"<UserSkill(user_id={self.user_id}, skill_id={self.skill_id}, confidence={self.confidence_score})>"


class SkillCategory(Base):
    """Skill category taxonomy for better organization"""
    
    __tablename__ = "skill_categories"
    
    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        primary_key=True,
        default=lambda: str(uuid4()),
        index=True
    )
    name: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        unique=True,
        index=True
    )
    parent_id: Mapped[Optional[str]] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("skill_categories.id"),
        nullable=True,
        index=True
    )
    description: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True
    )
    display_order: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False
    )
    is_active: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False
    )
    
    # Self-referential relationship for hierarchical categories
    parent = relationship("SkillCategory", remote_side=[id], back_populates="children")
    children = relationship("SkillCategory", back_populates="parent")
    
    def __repr__(self) -> str:
        return f"<SkillCategory(id={self.id}, name={self.name})>"