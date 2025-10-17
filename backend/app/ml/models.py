"""
Data models for the NLP processing engine.
"""

from enum import Enum
from typing import List, Dict, Optional, Any
from datetime import datetime
from pydantic import BaseModel, Field


class SkillCategory(str, Enum):
    """Enumeration of skill categories."""
    PROGRAMMING_LANGUAGES = "programming_languages"
    FRAMEWORKS_LIBRARIES = "frameworks_libraries"
    DATABASES = "databases"
    CLOUD_PLATFORMS = "cloud_platforms"
    DEVOPS_TOOLS = "devops_tools"
    OPERATING_SYSTEMS = "operating_systems"
    SOFT_SKILLS = "soft_skills"
    TECHNICAL = "technical"
    OTHER = "other"


class SkillExtraction(BaseModel):
    """Model for extracted skill information."""
    skill_name: str = Field(..., description="Name of the extracted skill")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Confidence score for the extraction")
    source: str = Field(..., description="Source method used for extraction")
    evidence: List[str] = Field(default_factory=list, description="Text evidence supporting the extraction")
    category: SkillCategory = Field(default=SkillCategory.OTHER, description="Skill category")
    
    class Config:
        use_enum_values = True


class ExperienceEntry(BaseModel):
    """Model for work experience entry."""
    raw_text: str = Field(..., description="Raw text of the experience entry")
    first_line: str = Field(..., description="First line containing position/company")
    dates: Optional[str] = Field(None, description="Date range for the position")
    description: str = Field(default="", description="Job description and responsibilities")
    extracted_skills: List[str] = Field(default_factory=list, description="Skills extracted from this experience")


class EducationEntry(BaseModel):
    """Model for education entry."""
    raw_text: str = Field(..., description="Raw text of the education entry")
    degree_institution: str = Field(..., description="Degree and institution information")
    graduation_year: Optional[str] = Field(None, description="Graduation year")
    details: str = Field(default="", description="Additional education details")


class ResumeData(BaseModel):
    """Model for parsed resume data."""
    file_path: Optional[str] = Field(None, description="Path to the original resume file")
    file_type: Optional[str] = Field(None, description="Type of the resume file")
    raw_text: str = Field(..., description="Raw extracted text from resume")
    cleaned_text: str = Field(..., description="Cleaned and preprocessed text")
    sections: Dict[str, str] = Field(default_factory=dict, description="Extracted resume sections")
    skills: List[SkillExtraction] = Field(default_factory=list, description="Extracted skills")
    experience: List[ExperienceEntry] = Field(default_factory=list, description="Work experience entries")
    education: List[EducationEntry] = Field(default_factory=list, description="Education entries")
    certifications: List[str] = Field(default_factory=list, description="Extracted certifications")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Timestamp of creation")
    
    def get_skills_by_category(self, category: SkillCategory) -> List[SkillExtraction]:
        """Get skills filtered by category."""
        return [skill for skill in self.skills if skill.category == category]
    
    def get_skill_names(self) -> List[str]:
        """Get list of skill names."""
        return [skill.skill_name for skill in self.skills]
    
    def get_high_confidence_skills(self, threshold: float = 0.7) -> List[SkillExtraction]:
        """Get skills with confidence above threshold."""
        return [skill for skill in self.skills if skill.confidence_score >= threshold]


class TextAnalysisResult(BaseModel):
    """Model for text analysis results."""
    original_text: str = Field(..., description="Original input text")
    cleaned_text: str = Field(..., description="Cleaned text")
    sentences: List[str] = Field(default_factory=list, description="Extracted sentences")
    tokens: List[str] = Field(default_factory=list, description="Extracted tokens")
    technical_terms: List[str] = Field(default_factory=list, description="Extracted technical terms")
    embeddings: Optional[List[float]] = Field(None, description="Text embeddings")
    processing_time: float = Field(..., description="Processing time in seconds")


class SkillMatchResult(BaseModel):
    """Model for skill matching results."""
    query_skill: str = Field(..., description="Original query skill")
    matched_skills: List[str] = Field(default_factory=list, description="Matched skill names")
    similarity_scores: List[float] = Field(default_factory=list, description="Similarity scores")
    categories: List[SkillCategory] = Field(default_factory=list, description="Skill categories")
    
    class Config:
        use_enum_values = True


class NLPProcessingConfig(BaseModel):
    """Configuration for NLP processing."""
    model_cache_dir: str = Field(default="./models", description="Directory for model cache")
    max_text_length: int = Field(default=50000, description="Maximum text length to process")
    confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0, description="Minimum confidence threshold")
    enable_gpu: bool = Field(default=False, description="Enable GPU acceleration if available")
    batch_size: int = Field(default=32, description="Batch size for processing")
    
    # Model-specific settings
    sentence_transformer_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Sentence transformer model name"
    )
    spacy_model: str = Field(default="en_core_web_sm", description="spaCy model name")
    
    # Processing settings
    remove_personal_info: bool = Field(default=True, description="Remove personal information from text")
    normalize_skills: bool = Field(default=True, description="Normalize skill names")
    extract_context: bool = Field(default=True, description="Extract context for skills")
    context_window_size: int = Field(default=100, description="Context window size in characters")


class ProcessingStats(BaseModel):
    """Statistics for processing operations."""
    total_documents_processed: int = Field(default=0)
    total_skills_extracted: int = Field(default=0)
    average_processing_time: float = Field(default=0.0)
    success_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    error_count: int = Field(default=0)
    last_updated: datetime = Field(default_factory=datetime.utcnow)
    
    def update_stats(self, processing_time: float, skills_count: int, success: bool):
        """Update processing statistics."""
        self.total_documents_processed += 1
        if success:
            self.total_skills_extracted += skills_count
            # Update average processing time
            if self.total_documents_processed == 1:
                self.average_processing_time = processing_time
            else:
                self.average_processing_time = (
                    (self.average_processing_time * (self.total_documents_processed - 1) + processing_time) /
                    self.total_documents_processed
                )
        else:
            self.error_count += 1
        
        # Update success rate
        self.success_rate = (self.total_documents_processed - self.error_count) / self.total_documents_processed
        self.last_updated = datetime.utcnow()