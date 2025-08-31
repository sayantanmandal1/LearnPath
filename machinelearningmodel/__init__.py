"""
Machine Learning Model Package for AI Career Recommender.
"""

from .text_preprocessor import TextPreprocessor
from .skill_classifier import SkillClassifier
from .models import (
    SkillCategory,
    SkillExtraction,
    ResumeData,
    ExperienceEntry,
    EducationEntry,
    TextAnalysisResult,
    SkillMatchResult,
    NLPProcessingConfig,
    ProcessingStats
)

__version__ = "1.0.0"
__all__ = [
    "TextPreprocessor",
    "SkillClassifier", 
    "SkillCategory",
    "SkillExtraction",
    "ResumeData",
    "ExperienceEntry",
    "EducationEntry",
    "TextAnalysisResult",
    "SkillMatchResult",
    "NLPProcessingConfig",
    "ProcessingStats"
]