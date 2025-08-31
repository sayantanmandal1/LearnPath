"""
Analytics schemas for request/response models
"""
from datetime import datetime
from typing import Optional, List, Dict, Any, Union
from enum import Enum

from pydantic import BaseModel, Field, validator


class ChartType(str, Enum):
    """Chart type enumeration"""
    RADAR = "radar"
    BAR = "bar"
    LINE = "line"
    PIE = "pie"
    SCATTER = "scatter"
    HEATMAP = "heatmap"


class SkillRadarChart(BaseModel):
    """Skill radar chart data schema"""
    user_id: str
    categories: List[str] = Field(..., description="Skill categories for radar axes")
    user_scores: List[float] = Field(..., description="User's scores for each category (0-100)")
    market_average: List[float] = Field(..., description="Market average scores for comparison")
    target_scores: Optional[List[float]] = Field(None, description="Target scores for dream job")
    max_score: float = Field(100.0, description="Maximum score for scaling")
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    
    @validator("user_scores", "market_average", "target_scores")
    def validate_scores(cls, v):
        if v and any(score < 0 or score > 100 for score in v):
            raise ValueError("All scores must be between 0 and 100")
        return v


class CareerRoadmapNode(BaseModel):
    """Career roadmap node schema"""
    id: str
    title: str
    description: str
    position: Dict[str, float] = Field(..., description="X, Y coordinates for visualization")
    node_type: str = Field(..., description="current, target, milestone, alternative")
    timeline_months: Optional[int] = Field(None, description="Months from current position")
    required_skills: List[str] = Field(default_factory=list)
    salary_range: Optional[Dict[str, int]] = None
    completion_status: Optional[str] = Field(None, description="not_started, in_progress, completed")
    
    @validator("node_type")
    def validate_node_type(cls, v):
        allowed_types = ["current", "target", "milestone", "alternative"]
        if v not in allowed_types:
            raise ValueError(f"Node type must be one of: {', '.join(allowed_types)}")
        return v


class CareerRoadmapEdge(BaseModel):
    """Career roadmap edge schema"""
    id: str
    source_id: str
    target_id: str
    edge_type: str = Field(..., description="direct, alternative, prerequisite")
    difficulty: float = Field(..., ge=0.0, le=1.0, description="Transition difficulty")
    estimated_duration_months: Optional[int] = None
    required_actions: List[str] = Field(default_factory=list)
    
    @validator("edge_type")
    def validate_edge_type(cls, v):
        allowed_types = ["direct", "alternative", "prerequisite"]
        if v not in allowed_types:
            raise ValueError(f"Edge type must be one of: {', '.join(allowed_types)}")
        return v


class CareerRoadmapVisualization(BaseModel):
    """Career roadmap visualization schema"""
    user_id: str
    nodes: List[CareerRoadmapNode]
    edges: List[CareerRoadmapEdge]
    metadata: Dict[str, Any] = Field(default_factory=dict)
    generated_at: datetime = Field(default_factory=datetime.utcnow)


class SkillGapAnalysis(BaseModel):
    """Skill gap analysis schema"""
    skill_name: str
    current_level: float = Field(..., ge=0.0, le=100.0)
    target_level: float = Field(..., ge=0.0, le=100.0)
    gap_size: float = Field(..., description="Calculated gap size")
    priority: str = Field(..., description="high, medium, low")
    estimated_learning_hours: Optional[int] = None
    recommended_resources: List[Dict[str, Any]] = Field(default_factory=list)
    market_demand: Optional[float] = Field(None, ge=0.0, le=1.0)
    salary_impact: Optional[float] = None
    
    @validator("priority")
    def validate_priority(cls, v):
        allowed_priorities = ["high", "medium", "low"]
        if v not in allowed_priorities:
            raise ValueError(f"Priority must be one of: {', '.join(allowed_priorities)}")
        return v


class SkillGapReport(BaseModel):
    """Comprehensive skill gap report schema"""
    user_id: str
    target_role: str
    overall_match_score: float = Field(..., ge=0.0, le=100.0)
    skill_gaps: List[SkillGapAnalysis]
    strengths: List[str] = Field(default_factory=list)
    total_learning_hours: int = Field(0, description="Total estimated learning time")
    priority_skills: List[str] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=datetime.utcnow)


class JobCompatibilityScore(BaseModel):
    """Job compatibility scoring schema"""
    job_id: str
    job_title: str
    company: str
    overall_score: float = Field(..., ge=0.0, le=100.0)
    skill_match_score: float = Field(..., ge=0.0, le=100.0)
    experience_match_score: float = Field(..., ge=0.0, le=100.0)
    location_match_score: Optional[float] = Field(None, ge=0.0, le=100.0)
    salary_match_score: Optional[float] = Field(None, ge=0.0, le=100.0)
    matched_skills: List[str] = Field(default_factory=list)
    missing_skills: List[str] = Field(default_factory=list)
    skill_gaps: List[SkillGapAnalysis] = Field(default_factory=list)
    recommendation: str = Field(..., description="apply, consider, improve_first")
    
    @validator("recommendation")
    def validate_recommendation(cls, v):
        allowed_recommendations = ["apply", "consider", "improve_first"]
        if v not in allowed_recommendations:
            raise ValueError(f"Recommendation must be one of: {', '.join(allowed_recommendations)}")
        return v


class JobCompatibilityReport(BaseModel):
    """Job compatibility report schema"""
    user_id: str
    job_matches: List[JobCompatibilityScore]
    filters_applied: Dict[str, Any] = Field(default_factory=dict)
    total_jobs_analyzed: int
    generated_at: datetime = Field(default_factory=datetime.utcnow)


class ProgressTrackingEntry(BaseModel):
    """Progress tracking entry schema"""
    user_id: str
    skill_name: str
    previous_score: float = Field(..., ge=0.0, le=100.0)
    current_score: float = Field(..., ge=0.0, le=100.0)
    improvement: float = Field(..., description="Score improvement")
    tracking_period_days: int = Field(..., gt=0)
    evidence: Optional[str] = None
    milestone_achieved: Optional[str] = None
    recorded_at: datetime = Field(default_factory=datetime.utcnow)


class HistoricalProgressReport(BaseModel):
    """Historical progress report schema"""
    user_id: str
    tracking_period_days: int
    skill_improvements: List[ProgressTrackingEntry]
    overall_improvement_score: float
    milestones_achieved: List[str] = Field(default_factory=list)
    trend_analysis: Dict[str, Any] = Field(default_factory=dict)
    generated_at: datetime = Field(default_factory=datetime.utcnow)


class CareerAnalysisReport(BaseModel):
    """Comprehensive career analysis report schema"""
    user_id: str
    profile_summary: Dict[str, Any]
    skill_radar_chart: SkillRadarChart
    career_roadmap: CareerRoadmapVisualization
    skill_gap_report: SkillGapReport
    job_compatibility_report: JobCompatibilityReport
    progress_report: HistoricalProgressReport
    recommendations: List[str] = Field(default_factory=list)
    next_steps: List[str] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    report_version: str = Field("1.0", description="Report format version")


class AnalyticsRequest(BaseModel):
    """Analytics generation request schema"""
    user_id: str
    analysis_types: List[str] = Field(..., description="Types of analysis to perform")
    target_role: Optional[str] = None
    include_job_matches: bool = Field(True)
    include_progress_tracking: bool = Field(True)
    tracking_period_days: int = Field(90, gt=0, le=365)
    job_search_filters: Optional[Dict[str, Any]] = None
    
    @validator("analysis_types")
    def validate_analysis_types(cls, v):
        allowed_types = [
            "skill_radar", "career_roadmap", "skill_gaps", 
            "job_compatibility", "progress_tracking", "full_report"
        ]
        for analysis_type in v:
            if analysis_type not in allowed_types:
                raise ValueError(f"Analysis type must be one of: {', '.join(allowed_types)}")
        return v


class ChartConfiguration(BaseModel):
    """Chart configuration schema"""
    chart_type: ChartType
    title: str
    width: int = Field(800, gt=0)
    height: int = Field(600, gt=0)
    color_scheme: str = Field("default")
    interactive: bool = Field(True)
    export_format: str = Field("svg", description="svg, png, pdf")
    custom_options: Dict[str, Any] = Field(default_factory=dict)
    
    @validator("export_format")
    def validate_export_format(cls, v):
        allowed_formats = ["svg", "png", "pdf", "json"]
        if v not in allowed_formats:
            raise ValueError(f"Export format must be one of: {', '.join(allowed_formats)}")
        return v


class VisualizationResponse(BaseModel):
    """Visualization response schema"""
    chart_id: str
    chart_type: ChartType
    data: Dict[str, Any]
    configuration: ChartConfiguration
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None


class PDFReportRequest(BaseModel):
    """PDF report generation request schema"""
    user_id: str
    report_type: str = Field("comprehensive", description="comprehensive, skills_only, career_only")
    include_charts: bool = Field(True)
    include_recommendations: bool = Field(True)
    custom_sections: Optional[List[str]] = None
    branding: Optional[Dict[str, Any]] = None
    
    @validator("report_type")
    def validate_report_type(cls, v):
        allowed_types = ["comprehensive", "skills_only", "career_only", "progress_only"]
        if v not in allowed_types:
            raise ValueError(f"Report type must be one of: {', '.join(allowed_types)}")
        return v


class PDFReportResponse(BaseModel):
    """PDF report response schema"""
    report_id: str
    file_url: str
    file_size_bytes: int
    page_count: int
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: datetime
    download_count: int = Field(0)