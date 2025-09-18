"""
AI Analysis schemas for request/response models
"""
from datetime import datetime
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
from enum import Enum


class AnalysisTypeEnum(str, Enum):
    """Analysis type enumeration"""
    SKILL_ASSESSMENT = "skill_assessment"
    CAREER_RECOMMENDATION = "career_recommendation"
    LEARNING_PATH = "learning_path"
    PROJECT_SUGGESTION = "project_suggestion"
    MARKET_ANALYSIS = "market_analysis"
    COMPLETE_ANALYSIS = "complete_analysis"


class SkillAssessmentResponse(BaseModel):
    """Skill assessment response schema"""
    technical_skills: Dict[str, float] = Field(..., description="Technical skills with proficiency scores (0-1)")
    soft_skills: Dict[str, float] = Field(..., description="Soft skills with proficiency scores (0-1)")
    skill_strengths: List[str] = Field(..., description="Top skill strengths")
    skill_gaps: List[str] = Field(..., description="Identified skill gaps")
    improvement_areas: List[str] = Field(..., description="Areas needing improvement")
    market_relevance_score: float = Field(..., ge=0.0, le=1.0, description="Market relevance score")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Assessment confidence score")


class CareerRecommendationResponse(BaseModel):
    """Career recommendation response schema"""
    recommended_role: str = Field(..., description="Recommended job role")
    match_score: float = Field(..., ge=0.0, le=1.0, description="Match score with current profile")
    reasoning: str = Field(..., description="Reasoning for the recommendation")
    required_skills: List[str] = Field(..., description="Skills required for the role")
    skill_gaps: List[str] = Field(..., description="Skills gaps to address")
    preparation_timeline: str = Field(..., description="Estimated preparation time")
    salary_range: Optional[str] = Field(None, description="Expected salary range")
    market_demand: str = Field(..., description="Market demand assessment")


class LearningModuleResponse(BaseModel):
    """Learning module response schema"""
    module_name: str = Field(..., description="Module name")
    topics: List[str] = Field(..., description="Topics covered in the module")
    duration: str = Field(..., description="Estimated duration")


class LearningResourceResponse(BaseModel):
    """Learning resource response schema"""
    type: str = Field(..., description="Resource type (course, book, tutorial, etc.)")
    title: str = Field(..., description="Resource title")
    provider: str = Field(..., description="Resource provider")
    url: Optional[str] = Field(None, description="Resource URL")


class LearningPathResponse(BaseModel):
    """Learning path response schema"""
    title: str = Field(..., description="Learning path title")
    description: str = Field(..., description="Path description")
    target_skills: List[str] = Field(..., description="Skills targeted by this path")
    learning_modules: List[LearningModuleResponse] = Field(..., description="Learning modules")
    estimated_duration: str = Field(..., description="Total estimated duration")
    difficulty_level: str = Field(..., description="Difficulty level")
    resources: List[LearningResourceResponse] = Field(..., description="Recommended resources")


class ProjectSuggestionResponse(BaseModel):
    """Project suggestion response schema"""
    title: str = Field(..., description="Project title")
    description: str = Field(..., description="Project description")
    technologies: List[str] = Field(..., description="Technologies to use")
    difficulty_level: str = Field(..., description="Project difficulty level")
    estimated_duration: str = Field(..., description="Estimated completion time")
    learning_outcomes: List[str] = Field(..., description="Expected learning outcomes")
    portfolio_value: str = Field(..., description="Value for portfolio/career")


class MarketInsightsResponse(BaseModel):
    """Market insights response schema"""
    industry_trends: List[str] = Field(..., description="Current industry trends")
    in_demand_skills: List[str] = Field(..., description="Skills in high demand")
    salary_trends: Dict[str, str] = Field(..., description="Salary trends by role")
    city_opportunities: Dict[str, str] = Field(..., description="Opportunities by city")
    emerging_technologies: List[str] = Field(..., description="Emerging technologies to watch")
    market_forecast: str = Field(..., description="Overall market forecast")
    actionable_insights: List[str] = Field(..., description="Actionable career insights")


class CompleteAnalysisResponse(BaseModel):
    """Complete AI analysis response schema"""
    user_id: str = Field(..., description="User ID")
    skill_assessment: SkillAssessmentResponse = Field(..., description="Skill assessment results")
    career_recommendations: List[CareerRecommendationResponse] = Field(..., description="Career recommendations")
    learning_paths: List[LearningPathResponse] = Field(..., description="Personalized learning paths")
    project_suggestions: List[ProjectSuggestionResponse] = Field(..., description="Project suggestions")
    market_insights: MarketInsightsResponse = Field(..., description="Market insights")
    analysis_timestamp: datetime = Field(..., description="When the analysis was performed")
    gemini_request_id: Optional[str] = Field(None, description="Gemini API request ID for tracking")


class AnalysisRequest(BaseModel):
    """Analysis request schema"""
    analysis_type: AnalysisTypeEnum = Field(..., description="Type of analysis to perform")
    force_refresh: bool = Field(False, description="Force refresh even if cached data exists")
    include_fallback: bool = Field(True, description="Use fallback analysis if AI is unavailable")


class AnalysisStatusResponse(BaseModel):
    """Analysis status response schema"""
    user_id: str = Field(..., description="User ID")
    analysis_type: AnalysisTypeEnum = Field(..., description="Analysis type")
    status: str = Field(..., description="Analysis status (pending, processing, completed, failed)")
    last_updated: Optional[datetime] = Field(None, description="Last update timestamp")
    confidence_score: Optional[float] = Field(None, description="Analysis confidence score")
    error_message: Optional[str] = Field(None, description="Error message if failed")


class BulkAnalysisRequest(BaseModel):
    """Bulk analysis request schema"""
    analysis_types: List[AnalysisTypeEnum] = Field(..., description="Types of analysis to perform")
    force_refresh: bool = Field(False, description="Force refresh even if cached data exists")
    include_fallback: bool = Field(True, description="Use fallback analysis if AI is unavailable")


class BulkAnalysisResponse(BaseModel):
    """Bulk analysis response schema"""
    user_id: str = Field(..., description="User ID")
    results: Dict[str, Any] = Field(..., description="Analysis results by type")
    status: Dict[str, str] = Field(..., description="Status of each analysis type")
    errors: Dict[str, str] = Field(default_factory=dict, description="Errors by analysis type")
    overall_confidence: float = Field(..., description="Overall confidence score")
    analysis_timestamp: datetime = Field(..., description="Analysis completion timestamp")


class AnalysisHistoryResponse(BaseModel):
    """Analysis history response schema"""
    user_id: str = Field(..., description="User ID")
    analysis_type: AnalysisTypeEnum = Field(..., description="Analysis type")
    results: List[Dict[str, Any]] = Field(..., description="Historical analysis results")
    timestamps: List[datetime] = Field(..., description="Analysis timestamps")
    confidence_scores: List[float] = Field(..., description="Confidence scores over time")


class AnalysisComparisonResponse(BaseModel):
    """Analysis comparison response schema"""
    user_id: str = Field(..., description="User ID")
    analysis_type: AnalysisTypeEnum = Field(..., description="Analysis type")
    current_analysis: Dict[str, Any] = Field(..., description="Current analysis results")
    previous_analysis: Optional[Dict[str, Any]] = Field(None, description="Previous analysis results")
    changes: Dict[str, Any] = Field(..., description="Changes between analyses")
    improvement_score: float = Field(..., description="Overall improvement score")
    timestamp_current: datetime = Field(..., description="Current analysis timestamp")
    timestamp_previous: Optional[datetime] = Field(None, description="Previous analysis timestamp")


class AnalysisMetricsResponse(BaseModel):
    """Analysis metrics response schema"""
    total_analyses: int = Field(..., description="Total number of analyses performed")
    success_rate: float = Field(..., description="Analysis success rate")
    average_confidence: float = Field(..., description="Average confidence score")
    most_common_recommendations: List[str] = Field(..., description="Most common career recommendations")
    skill_improvement_trends: Dict[str, float] = Field(..., description="Skill improvement trends")
    analysis_frequency: Dict[str, int] = Field(..., description="Analysis frequency by type")


class GeminiAPIStatus(BaseModel):
    """Gemini API status response schema"""
    is_available: bool = Field(..., description="Whether Gemini API is available")
    last_successful_request: Optional[datetime] = Field(None, description="Last successful API request")
    error_rate: float = Field(..., description="Recent error rate")
    average_response_time: float = Field(..., description="Average response time in seconds")
    rate_limit_status: Dict[str, Any] = Field(..., description="Rate limit information")


class AnalysisConfigResponse(BaseModel):
    """Analysis configuration response schema"""
    gemini_api_enabled: bool = Field(..., description="Whether Gemini API is enabled")
    fallback_enabled: bool = Field(..., description="Whether fallback analysis is enabled")
    cache_duration_hours: int = Field(..., description="Cache duration in hours")
    max_concurrent_analyses: int = Field(..., description="Maximum concurrent analyses")
    supported_analysis_types: List[AnalysisTypeEnum] = Field(..., description="Supported analysis types")