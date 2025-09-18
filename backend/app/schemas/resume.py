"""
Resume processing schemas for API requests and responses
"""
from datetime import datetime
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, validator
from enum import Enum


class ProcessingStatus(str, Enum):
    """Resume processing status enumeration"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    MANUAL_ENTRY = "manual_entry"


class SupportedFileType(str, Enum):
    """Supported resume file types"""
    PDF = "application/pdf"
    DOC = "application/msword"
    DOCX = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"


class ContactInfo(BaseModel):
    """Contact information extracted from resume"""
    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    location: Optional[str] = None
    linkedin: Optional[str] = None
    github: Optional[str] = None
    portfolio: Optional[str] = None


class WorkExperience(BaseModel):
    """Work experience entry"""
    company: Optional[str] = None
    position: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    description: Optional[str] = None
    technologies: Optional[List[str]] = None
    achievements: Optional[List[str]] = None


class Education(BaseModel):
    """Education entry"""
    institution: Optional[str] = None
    degree: Optional[str] = None
    field_of_study: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    gpa: Optional[str] = None
    achievements: Optional[List[str]] = None


class Certification(BaseModel):
    """Certification entry"""
    name: str
    issuer: Optional[str] = None
    issue_date: Optional[str] = None
    expiry_date: Optional[str] = None
    credential_id: Optional[str] = None
    url: Optional[str] = None


class SkillCategory(BaseModel):
    """Skill category with skills"""
    category: str
    skills: List[str]
    proficiency_level: Optional[str] = None


class ParsedResumeData(BaseModel):
    """Structured resume data after parsing"""
    contact_info: Optional[ContactInfo] = None
    summary: Optional[str] = None
    work_experience: Optional[List[WorkExperience]] = None
    education: Optional[List[Education]] = None
    skills: Optional[List[SkillCategory]] = None
    certifications: Optional[List[Certification]] = None
    projects: Optional[List[Dict[str, Any]]] = None
    languages: Optional[List[str]] = None
    awards: Optional[List[str]] = None


class ResumeUploadResponse(BaseModel):
    """Response after resume upload"""
    id: str
    user_id: str
    original_filename: str
    file_size: int
    file_type: str
    processing_status: ProcessingStatus
    message: str
    created_at: datetime


class ResumeProcessingResult(BaseModel):
    """Complete resume processing result"""
    id: str
    user_id: str
    original_filename: str
    processing_status: ProcessingStatus
    extracted_text: Optional[str] = None
    extraction_confidence: Optional[float] = None
    parsed_data: Optional[ParsedResumeData] = None
    error_message: Optional[str] = None
    processing_started_at: Optional[datetime] = None
    processing_completed_at: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime


class ResumeValidationError(BaseModel):
    """Resume validation error details"""
    field: str
    message: str
    suggested_value: Optional[str] = None


class ResumeValidationResult(BaseModel):
    """Resume data validation result"""
    is_valid: bool
    errors: List[ResumeValidationError] = []
    warnings: List[ResumeValidationError] = []
    confidence_score: float = Field(ge=0.0, le=1.0)


class ManualResumeEntry(BaseModel):
    """Manual resume data entry when processing fails"""
    contact_info: ContactInfo
    summary: Optional[str] = None
    work_experience: List[WorkExperience] = []
    education: List[Education] = []
    skills: List[SkillCategory] = []
    certifications: List[Certification] = []
    
    @validator('contact_info')
    def validate_contact_info(cls, v):
        if not v.name and not v.email:
            raise ValueError('Either name or email must be provided')
        return v


class ResumeProcessingStats(BaseModel):
    """Resume processing statistics"""
    total_processed: int
    successful_extractions: int
    failed_extractions: int
    manual_entries: int
    average_processing_time: float
    average_confidence_score: float