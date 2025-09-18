"""
Analysis result models for storing AI-powered career analysis and recommendations
"""
from datetime import datetime
from typing import Optional
from uuid import uuid4

from sqlalchemy import DateTime, ForeignKey, JSON, String, Text, Float, Enum, Integer
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func
import enum

from app.core.database import Base


class AnalysisType(str, enum.Enum):
    """Types of analysis performed"""
    SKILL_ASSESSMENT = "skill_assessment"
    CAREER_TRAJECTORY = "career_trajectory"
    LEARNING_PATH = "learning_path"
    PROJECT_RECOMMENDATION = "project_recommendation"
    JOB_MATCHING = "job_matching"
    COMPREHENSIVE = "comprehensive"


class AnalysisStatus(str, enum.Enum):
    """Analysis processing status"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    EXPIRED = "expired"


class AnalysisResult(Base):
    """Model for storing AI analysis results and career insights"""
    
    __tablename__ = "analysis_results"
    
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
    
    # Analysis metadata
    analysis_type: Mapped[AnalysisType] = mapped_column(
        Enum(AnalysisType),
        nullable=False,
        index=True
    )
    analysis_version: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default="1.0",
        comment="Version of analysis algorithm used"
    )
    status: Mapped[AnalysisStatus] = mapped_column(
        Enum(AnalysisStatus),
        nullable=False,
        default=AnalysisStatus.PENDING
    )
    
    # Input data references
    resume_data_id: Mapped[Optional[str]] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("resume_data.id"),
        nullable=True,
        comment="Resume data used for analysis"
    )
    platform_data_snapshot: Mapped[Optional[dict]] = mapped_column(
        JSON,
        nullable=True,
        comment="Snapshot of platform data used for analysis"
    )
    
    # Analysis results
    skill_assessment: Mapped[Optional[dict]] = mapped_column(
        JSON,
        nullable=True,
        comment="Detailed skill analysis and ratings"
    )
    career_recommendations: Mapped[Optional[dict]] = mapped_column(
        JSON,
        nullable=True,
        comment="Career path and role recommendations"
    )
    learning_paths: Mapped[Optional[dict]] = mapped_column(
        JSON,
        nullable=True,
        comment="Personalized learning recommendations"
    )
    project_suggestions: Mapped[Optional[dict]] = mapped_column(
        JSON,
        nullable=True,
        comment="Project recommendations for skill development"
    )
    skill_gaps: Mapped[Optional[dict]] = mapped_column(
        JSON,
        nullable=True,
        comment="Identified skill gaps for target roles"
    )
    market_insights: Mapped[Optional[dict]] = mapped_column(
        JSON,
        nullable=True,
        comment="Market trends and salary insights"
    )
    
    # Scoring and metrics
    overall_score: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
        comment="Overall profile score (0-100)"
    )
    skill_diversity_score: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
        comment="Score indicating skill diversity (0-100)"
    )
    experience_relevance_score: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
        comment="Relevance of experience to target role (0-100)"
    )
    market_readiness_score: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
        comment="Readiness for job market (0-100)"
    )
    
    # AI processing information
    gemini_request_id: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
        comment="Gemini API request ID for tracking"
    )
    processing_time_seconds: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
        comment="Time taken to complete analysis"
    )
    confidence_score: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
        comment="AI confidence in analysis results (0-1)"
    )
    
    # Error handling
    error_message: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True
    )
    retry_count: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
        comment="Number of retry attempts"
    )
    
    # Validity and expiration
    expires_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="When this analysis expires and needs refresh"
    )
    is_current: Mapped[bool] = mapped_column(
        default=True,
        nullable=False,
        comment="Whether this is the current analysis for the user"
    )
    
    # Timestamps
    analysis_started_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True
    )
    analysis_completed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True
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
    user = relationship("User", backref="analysis_results")
    resume_data = relationship("ResumeData", backref="analysis_results")
    
    def __repr__(self) -> str:
        return f"<AnalysisResult(id={self.id}, type={self.analysis_type}, status={self.status})>"


class JobRecommendation(Base):
    """Model for storing job recommendations and matching results"""
    
    __tablename__ = "job_recommendations"
    
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
    analysis_result_id: Mapped[Optional[str]] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("analysis_results.id"),
        nullable=True,
        comment="Analysis result that generated this recommendation"
    )
    
    # Job information
    job_title: Mapped[str] = mapped_column(
        String(255),
        nullable=False
    )
    company_name: Mapped[str] = mapped_column(
        String(255),
        nullable=False
    )
    job_description: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True
    )
    location: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True
    )
    salary_range: Mapped[Optional[dict]] = mapped_column(
        JSON,
        nullable=True,
        comment="Salary range information"
    )
    required_skills: Mapped[Optional[dict]] = mapped_column(
        JSON,
        nullable=True,
        comment="Skills required for the job"
    )
    experience_level: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True
    )
    
    # Job source information
    source_platform: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        comment="Platform where job was found (LinkedIn, Naukri, etc.)"
    )
    job_url: Mapped[Optional[str]] = mapped_column(
        String(500),
        nullable=True,
        comment="URL to original job posting"
    )
    external_job_id: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
        comment="Job ID from source platform"
    )
    posted_date: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True
    )
    
    # Matching information
    match_score: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        comment="AI-calculated match score (0-100)"
    )
    skill_match_percentage: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
        comment="Percentage of required skills user has"
    )
    skill_gaps: Mapped[Optional[dict]] = mapped_column(
        JSON,
        nullable=True,
        comment="Skills user needs to develop for this job"
    )
    recommendation_reason: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        comment="AI-generated explanation for recommendation"
    )
    
    # User interaction
    is_viewed: Mapped[bool] = mapped_column(
        default=False,
        nullable=False
    )
    is_saved: Mapped[bool] = mapped_column(
        default=False,
        nullable=False
    )
    is_applied: Mapped[bool] = mapped_column(
        default=False,
        nullable=False
    )
    user_rating: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
        comment="User rating of recommendation (1-5)"
    )
    user_feedback: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        comment="User feedback on recommendation"
    )
    
    # Timestamps
    recommended_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False
    )
    viewed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True
    )
    applied_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True
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
    user = relationship("User", backref="job_recommendations")
    analysis_result = relationship("AnalysisResult", backref="job_recommendations")
    
    def __repr__(self) -> str:
        return f"<JobRecommendation(id={self.id}, job_title={self.job_title}, match_score={self.match_score})>"