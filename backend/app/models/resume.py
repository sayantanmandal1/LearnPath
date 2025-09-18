"""
Resume data models for storing uploaded resume information and processing results
"""
from datetime import datetime
from typing import Optional
from uuid import uuid4

from sqlalchemy import DateTime, ForeignKey, JSON, String, Text, Float, Enum
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func
import enum

from app.core.database import Base


class ProcessingStatus(str, enum.Enum):
    """Resume processing status enumeration"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    MANUAL_ENTRY = "manual_entry"


class ResumeData(Base):
    """Model for storing resume files and extracted data"""
    
    __tablename__ = "resume_data"
    
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
    
    # File information
    original_filename: Mapped[str] = mapped_column(
        String(255),
        nullable=False
    )
    file_path: Mapped[str] = mapped_column(
        String(500),
        nullable=False,
        comment="Path to stored resume file"
    )
    file_size: Mapped[int] = mapped_column(
        nullable=False,
        comment="File size in bytes"
    )
    file_type: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        comment="MIME type of uploaded file"
    )
    
    # Processing information
    processing_status: Mapped[ProcessingStatus] = mapped_column(
        Enum(ProcessingStatus),
        nullable=False,
        default=ProcessingStatus.PENDING
    )
    extracted_text: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        comment="Raw text extracted from resume"
    )
    extraction_confidence: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
        comment="Confidence score for text extraction (0-1)"
    )
    
    # Parsed resume sections
    parsed_sections: Mapped[Optional[dict]] = mapped_column(
        JSON,
        nullable=True,
        comment="Structured data extracted from resume sections"
    )
    contact_info: Mapped[Optional[dict]] = mapped_column(
        JSON,
        nullable=True,
        comment="Extracted contact information"
    )
    work_experience: Mapped[Optional[dict]] = mapped_column(
        JSON,
        nullable=True,
        comment="Parsed work experience data"
    )
    education_data: Mapped[Optional[dict]] = mapped_column(
        JSON,
        nullable=True,
        comment="Extracted education information"
    )
    skills_extracted: Mapped[Optional[dict]] = mapped_column(
        JSON,
        nullable=True,
        comment="Skills identified from resume"
    )
    certifications_data: Mapped[Optional[dict]] = mapped_column(
        JSON,
        nullable=True,
        comment="Certifications found in resume"
    )
    
    # Processing metadata
    processing_started_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True
    )
    processing_completed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True
    )
    error_message: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        comment="Error message if processing failed"
    )
    gemini_request_id: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
        comment="Gemini API request ID for tracking"
    )
    
    # Timestamps
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
    user = relationship("User", backref="resume_data")
    
    def __repr__(self) -> str:
        return f"<ResumeData(id={self.id}, user_id={self.user_id}, status={self.processing_status})>"