"""
Dashboard schemas for API responses
"""
from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class DashboardMetric(BaseModel):
    """Individual dashboard metric"""
    title: str
    value: str
    change: Optional[str] = None
    trend: Optional[str] = Field(None, description="positive, negative, or neutral")
    description: Optional[str] = None


class ProgressMilestone(BaseModel):
    """User progress milestone"""
    id: str
    title: str
    description: str
    category: str  # skill, career, learning, etc.
    completed: bool
    completion_date: Optional[datetime] = None
    target_date: Optional[datetime] = None
    progress_percentage: float = Field(ge=0, le=100)
    priority: str = Field(default="medium", description="low, medium, high")


class DashboardRecommendation(BaseModel):
    """Dashboard recommendation item"""
    title: str
    description: str
    confidence_score: float = Field(ge=0, le=10)
    type: str  # job, learning, skill, career, profile_setup
    priority: Optional[str] = Field(default="medium", description="low, medium, high")
    action_url: Optional[str] = None
    estimated_time: Optional[str] = None


class DashboardActivity(BaseModel):
    """Recent user activity"""
    title: str
    description: str
    timestamp: datetime
    type: str  # profile_update, analysis_completed, skill_added, etc.
    status: Optional[str] = Field(default="completed", description="pending, in_progress, completed, failed")
    metadata: Optional[Dict[str, Any]] = None


class DashboardSummary(BaseModel):
    """Main dashboard summary data"""
    user_id: str
    overall_career_score: float = Field(ge=0, le=100)
    profile_completion: float = Field(ge=0, le=100)
    
    # Key metrics
    key_metrics: List[DashboardMetric]
    
    # Progress tracking
    active_milestones: List[ProgressMilestone]
    completed_milestones_count: int
    total_milestones_count: int
    
    # Recommendations
    top_recommendations: List[DashboardRecommendation]
    
    # Recent activity
    recent_activities: List[DashboardActivity]
    
    # Quick stats
    skills_count: int
    job_matches_count: int
    learning_paths_count: int
    
    # Last updated
    last_analysis_date: Optional[datetime] = None
    last_profile_update: Optional[datetime] = None
    generated_at: datetime
    
    # Analysis status
    analysis_status: Optional[str] = Field(default="completed", description="pending, in_progress, completed, error")
    needs_analysis: Optional[bool] = Field(default=False, description="True if user needs to complete profile analysis")


class UserProgressSummary(BaseModel):
    """User progress tracking summary"""
    user_id: str
    
    # Overall progress
    overall_progress: float = Field(ge=0, le=100)
    career_score_trend: List[Dict[str, Any]]  # Historical career scores
    
    # Skill progress
    skill_improvements: List[Dict[str, Any]]
    new_skills_added: int
    skills_mastered: int
    
    # Career milestones
    milestones: List[ProgressMilestone]
    milestone_completion_rate: float = Field(ge=0, le=100)
    
    # Learning progress
    learning_paths_started: int
    learning_paths_completed: int
    courses_completed: int
    
    # Job market progress
    job_compatibility_improvement: float
    interview_readiness_score: float = Field(ge=0, le=100)
    
    # Time-based metrics
    tracking_period_days: int
    generated_at: datetime


class PersonalizedContent(BaseModel):
    """Personalized dashboard content"""
    user_id: str
    
    # Personalized recommendations
    featured_jobs: List[Dict[str, Any]]
    recommended_skills: List[Dict[str, Any]]
    suggested_learning_paths: List[Dict[str, Any]]
    
    # Market insights
    market_trends: List[Dict[str, Any]]
    salary_insights: Dict[str, Any]
    industry_updates: List[Dict[str, Any]]
    
    # Networking suggestions
    networking_opportunities: List[Dict[str, Any]]
    similar_profiles: List[Dict[str, Any]]
    
    # Content preferences
    content_categories: List[str]
    personalization_score: float = Field(ge=0, le=100)
    
    generated_at: datetime


class DashboardConfiguration(BaseModel):
    """Dashboard configuration settings"""
    user_id: str
    
    # Widget preferences
    enabled_widgets: List[str]
    widget_order: List[str]
    
    # Notification preferences
    email_notifications: bool = True
    push_notifications: bool = True
    notification_frequency: str = Field(default="daily", description="daily, weekly, monthly")
    
    # Display preferences
    theme: str = Field(default="light", description="light, dark, auto")
    timezone: str = Field(default="UTC")
    date_format: str = Field(default="YYYY-MM-DD")
    
    # Privacy settings
    public_profile: bool = False
    share_progress: bool = False
    
    updated_at: datetime


class SkillRadarData(BaseModel):
    """Skill radar chart data for dashboard visualization"""
    user_id: str
    skill_categories: Dict[str, Dict[str, float]]  # category -> {skill: proficiency}
    skill_strengths: List[str]
    skill_gaps: List[str]
    improvement_areas: List[str]
    overall_scores: Dict[str, float]
    chart_config: Dict[str, Any]
    market_comparison: Optional[Dict[str, Any]] = None
    generated_at: str


class CareerProgressData(BaseModel):
    """Career progress tracking data"""
    user_id: str
    tracking_period_days: int
    overall_progress: float
    career_score_trend: List[Dict[str, Any]]
    skill_improvements: List[Dict[str, Any]]
    progress_timeline: List[Dict[str, Any]]
    progress_metrics: Dict[str, Any]
    milestones: List[Dict[str, Any]]
    milestone_completion_rate: float
    trajectory_predictions: Optional[Dict[str, Any]] = None
    milestone_tracking: Optional[Dict[str, Any]] = None
    generated_at: str


class JobRecommendationData(BaseModel):
    """Job recommendation data with personalized matching"""
    user_id: str
    target_role: Optional[str]
    inferred_target_role: Optional[str]
    preferred_cities: Optional[List[str]]
    min_match_score: float
    total_matches: int
    filtered_matches: int
    recommendations: List[Dict[str, Any]]
    recommendation_summary: Dict[str, Any]
    market_insights: Optional[Dict[str, Any]] = None
    generated_at: str


class ComprehensiveDashboardData(BaseModel):
    """Comprehensive dashboard data aggregation"""
    user_id: str
    dashboard_summary: Dict[str, Any]
    skill_radar: Optional[Dict[str, Any]] = None
    career_progress: Optional[Dict[str, Any]] = None
    job_recommendations: Optional[Dict[str, Any]] = None
    generated_at: str


class RealTimeAnalysisData(BaseModel):
    """Real-time analysis results from AI service"""
    user_id: str
    analysis_timestamp: str
    skill_assessment: Optional[Dict[str, Any]] = None
    career_recommendations: Optional[List[Dict[str, Any]]] = None
    learning_paths: Optional[List[Dict[str, Any]]] = None
    project_suggestions: Optional[List[Dict[str, Any]]] = None
    market_insights: Optional[Dict[str, Any]] = None
    is_fresh_analysis: bool