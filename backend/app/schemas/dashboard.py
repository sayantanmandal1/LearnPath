"""
Dashboard schemas for API responses
"""
from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class DashboardMetric(BaseModel):
    """Individual dashboard metric"""
    name: str
    value: Any
    change: Optional[float] = None
    change_type: Optional[str] = Field(None, description="increase, decrease, or stable")
    unit: Optional[str] = None
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
    id: str
    title: str
    description: str
    type: str  # job, learning, skill, career
    priority: str = Field(default="medium", description="low, medium, high")
    action_url: Optional[str] = None
    estimated_time: Optional[str] = None
    impact_score: Optional[float] = Field(None, ge=0, le=10)


class DashboardActivity(BaseModel):
    """Recent user activity"""
    id: str
    type: str  # profile_update, analysis_completed, skill_added, etc.
    title: str
    description: str
    timestamp: datetime
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