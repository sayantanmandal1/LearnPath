"""
Job posting models for market analysis and matching
"""
from datetime import datetime
from typing import Optional
from uuid import uuid4

from sqlalchemy import DateTime, ForeignKey, String, Text, Float, Integer, Boolean, JSON
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from app.core.database import Base


class JobPosting(Base):
    """Job posting model for market analysis"""
    
    __tablename__ = "job_postings"
    
    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        primary_key=True,
        default=lambda: str(uuid4()),
        index=True
    )
    external_id: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
        index=True,
        comment="External platform job ID"
    )
    title: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        index=True
    )
    company: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        index=True
    )
    location: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
        index=True
    )
    remote_type: Mapped[Optional[str]] = mapped_column(
        String(50),
        nullable=True,
        comment="remote, hybrid, onsite"
    )
    employment_type: Mapped[Optional[str]] = mapped_column(
        String(50),
        nullable=True,
        comment="full-time, part-time, contract, internship"
    )
    experience_level: Mapped[Optional[str]] = mapped_column(
        String(50),
        nullable=True,
        comment="entry, mid, senior, lead, executive"
    )
    description: Mapped[Text] = mapped_column(
        Text,
        nullable=False
    )
    requirements: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        comment="Extracted requirements text"
    )
    
    # Salary information
    salary_min: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True
    )
    salary_max: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True
    )
    salary_currency: Mapped[Optional[str]] = mapped_column(
        String(10),
        nullable=True,
        default="USD"
    )
    salary_period: Mapped[Optional[str]] = mapped_column(
        String(20),
        nullable=True,
        comment="yearly, monthly, hourly"
    )
    
    # Source information
    source: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        index=True,
        comment="linkedin, indeed, glassdoor, etc."
    )
    source_url: Mapped[Optional[str]] = mapped_column(
        String(1000),
        nullable=True
    )
    posted_date: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        index=True
    )
    expires_date: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True
    )
    
    # Analysis results
    processed_skills: Mapped[Optional[dict]] = mapped_column(
        JSON,
        nullable=True,
        comment="Extracted skills with confidence scores"
    )
    market_analysis: Mapped[Optional[dict]] = mapped_column(
        JSON,
        nullable=True,
        comment="Market trend analysis results"
    )
    
    # Status and metadata
    is_active: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        nullable=False
    )
    is_processed: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
        comment="Whether NLP processing has been completed"
    )
    quality_score: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
        comment="Quality score of job posting data"
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
    job_skills = relationship("JobSkill", back_populates="job_posting")
    
    def __repr__(self) -> str:
        return f"<JobPosting(id={self.id}, title={self.title}, company={self.company})>"


class JobSkill(Base):
    """Job posting skill requirements"""
    
    __tablename__ = "job_skills"
    
    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        primary_key=True,
        default=lambda: str(uuid4()),
        index=True
    )
    job_posting_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("job_postings.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    skill_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("skills.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    importance: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        comment="required, preferred, nice-to-have"
    )
    confidence_score: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        comment="Confidence in skill extraction from 0.0 to 1.0"
    )
    years_required: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
        comment="Years of experience required"
    )
    proficiency_level: Mapped[Optional[str]] = mapped_column(
        String(50),
        nullable=True,
        comment="beginner, intermediate, advanced, expert"
    )
    context: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        comment="Context where this skill was mentioned in job posting"
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False
    )
    
    # Relationships
    job_posting = relationship("JobPosting", back_populates="job_skills")
    skill = relationship("Skill", back_populates="job_skills")
    
    def __repr__(self) -> str:
        return f"<JobSkill(job_id={self.job_posting_id}, skill_id={self.skill_id}, importance={self.importance})>"


class Company(Base):
    """Company information for enhanced job matching"""
    
    __tablename__ = "companies"
    
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
    domain: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
        index=True
    )
    industry: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
        index=True
    )
    size: Mapped[Optional[str]] = mapped_column(
        String(50),
        nullable=True,
        comment="startup, small, medium, large, enterprise"
    )
    description: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True
    )
    headquarters: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True
    )
    founded_year: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True
    )
    
    # Company metrics
    glassdoor_rating: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True
    )
    employee_count: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True
    )
    
    # Additional data
    tech_stack: Mapped[Optional[dict]] = mapped_column(
        JSON,
        nullable=True,
        comment="Known technology stack"
    )
    culture_keywords: Mapped[Optional[dict]] = mapped_column(
        JSON,
        nullable=True,
        comment="Company culture keywords and values"
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
    
    def __repr__(self) -> str:
        return f"<Company(id={self.id}, name={self.name}, industry={self.industry})>"