"""
Job application tracking models.
"""

from sqlalchemy import Column, String, DateTime, Text, Boolean, Float, ForeignKey, Integer, JSON
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid

from ..core.database import Base


class JobApplication(Base):
    """Job application tracking model."""
    
    __tablename__ = "job_applications"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    job_posting_id = Column(String(255), nullable=False)  # External job ID
    job_title = Column(String(255), nullable=False)
    company_name = Column(String(255), nullable=False)
    job_url = Column(Text, nullable=True)
    
    # Application status tracking
    status = Column(String(50), nullable=False, default="interested")  # interested, applied, interviewing, rejected, accepted
    applied_date = Column(DateTime(timezone=True), nullable=True)
    last_updated = Column(DateTime(timezone=True), nullable=False, default=func.now())
    
    # Match information
    match_score = Column(Float, nullable=True)
    skill_matches = Column(JSON, nullable=True)
    skill_gaps = Column(JSON, nullable=True)
    
    # Application details
    application_method = Column(String(100), nullable=True)  # direct, linkedin, naukri, etc.
    cover_letter = Column(Text, nullable=True)
    notes = Column(Text, nullable=True)
    
    # Interview tracking
    interview_scheduled = Column(Boolean, default=False)
    interview_date = Column(DateTime(timezone=True), nullable=True)
    interview_notes = Column(Text, nullable=True)
    
    # Feedback and outcome
    feedback_received = Column(Boolean, default=False)
    feedback_text = Column(Text, nullable=True)
    rejection_reason = Column(String(255), nullable=True)
    
    # Metadata
    created_at = Column(DateTime(timezone=True), nullable=False, default=func.now())
    updated_at = Column(DateTime(timezone=True), nullable=False, default=func.now(), onupdate=func.now())
    
    # Relationships
    user = relationship("User", back_populates="job_applications")
    feedback_entries = relationship("JobApplicationFeedback", back_populates="application", cascade="all, delete-orphan")


class JobApplicationFeedback(Base):
    """Job application feedback and recommendation tracking."""
    
    __tablename__ = "job_application_feedback"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    application_id = Column(UUID(as_uuid=True), ForeignKey("job_applications.id"), nullable=False)
    
    # Feedback type
    feedback_type = Column(String(50), nullable=False)  # recommendation_quality, match_accuracy, application_outcome
    
    # Rating and feedback
    rating = Column(Integer, nullable=True)  # 1-5 scale
    feedback_text = Column(Text, nullable=True)
    
    # Specific feedback categories
    match_accuracy_rating = Column(Integer, nullable=True)  # How accurate was the match score
    recommendation_helpfulness = Column(Integer, nullable=True)  # How helpful was the recommendation
    gap_analysis_accuracy = Column(Integer, nullable=True)  # How accurate was the gap analysis
    
    # System improvement data
    suggested_improvements = Column(JSON, nullable=True)
    
    # Metadata
    created_at = Column(DateTime(timezone=True), nullable=False, default=func.now())
    
    # Relationships
    application = relationship("JobApplication", back_populates="feedback_entries")


class JobRecommendationFeedback(Base):
    """Feedback on job recommendations to improve matching algorithm."""
    
    __tablename__ = "job_recommendation_feedback"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    job_posting_id = Column(String(255), nullable=False)
    
    # Recommendation feedback
    recommendation_shown = Column(Boolean, default=True)
    user_interested = Column(Boolean, nullable=True)
    user_applied = Column(Boolean, default=False)
    
    # Feedback on match quality
    match_score_feedback = Column(String(50), nullable=True)  # too_high, accurate, too_low
    skill_match_feedback = Column(String(50), nullable=True)  # accurate, missing_skills, wrong_skills
    location_feedback = Column(String(50), nullable=True)  # good, not_preferred, wrong_location
    
    # Detailed feedback
    feedback_text = Column(Text, nullable=True)
    improvement_suggestions = Column(JSON, nullable=True)
    
    # Metadata
    created_at = Column(DateTime(timezone=True), nullable=False, default=func.now())
    
    # Relationships
    user = relationship("User")