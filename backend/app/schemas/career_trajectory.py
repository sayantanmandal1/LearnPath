"""
Pydantic schemas for career trajectory API responses.
"""

from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple
from pydantic import BaseModel, Field


class ProgressionStep(BaseModel):
    """Individual step in career progression."""
    role: str = Field(..., description="Role title for this step")
    duration_months: int = Field(..., description="Duration of this step in months")
    description: str = Field(..., description="Description of this progression step")
    key_activities: List[str] = Field(default_factory=list, description="Key activities for this step")
    skills_to_develop: List[str] = Field(default_factory=list, description="Skills to develop in this step")
    milestones: List[str] = Field(default_factory=list, description="Key milestones to achieve")


class SalaryRange(BaseModel):
    """Salary range information."""
    min_salary: int = Field(..., description="Minimum salary")
    max_salary: int = Field(..., description="Maximum salary")
    currency: str = Field(default="USD", description="Currency code")


class AlternativeRoute(BaseModel):
    """Alternative career path route."""
    path_id: str = Field(..., description="Unique identifier for alternative path")
    approach: str = Field(..., description="Alternative approach name")
    description: str = Field(..., description="Description of alternative approach")
    progression_steps: List[ProgressionStep] = Field(..., description="Steps for alternative path")
    estimated_timeline_months: int = Field(..., description="Estimated timeline in months")
    advantages: List[str] = Field(default_factory=list, description="Advantages of this approach")
    considerations: List[str] = Field(default_factory=list, description="Important considerations")
    success_rate: float = Field(..., ge=0, le=1, description="Historical success rate")


class CareerTrajectoryResponse(BaseModel):
    """Career trajectory recommendation response."""
    trajectory_id: str = Field(..., description="Unique trajectory identifier")
    title: str = Field(..., description="Trajectory title")
    target_role: str = Field(..., description="Target role for this trajectory")
    match_score: float = Field(..., ge=0, le=1, description="Semantic similarity match score")
    confidence_score: float = Field(..., ge=0, le=1, description="Overall confidence in recommendation")
    
    # Path details
    progression_steps: List[ProgressionStep] = Field(..., description="Detailed progression steps")
    estimated_timeline_months: int = Field(..., description="Total estimated timeline in months")
    difficulty_level: str = Field(..., description="Difficulty level: easy, moderate, challenging, difficult")
    
    # Skills and requirements
    required_skills: List[str] = Field(..., description="Skills required for target role")
    skill_gaps: Dict[str, float] = Field(..., description="Missing skills with importance scores")
    transferable_skills: List[str] = Field(..., description="Existing skills that transfer well")
    
    # Market analysis
    market_demand: str = Field(..., description="Market demand level: low, moderate, high")
    salary_progression: Dict[str, SalaryRange] = Field(..., description="Salary ranges by role")
    growth_potential: float = Field(..., ge=0, le=1, description="Career growth potential score")
    
    # Alternative paths
    alternative_routes: List[AlternativeRoute] = Field(default_factory=list, description="Alternative career paths")
    lateral_opportunities: List[str] = Field(default_factory=list, description="Lateral movement opportunities")
    
    # Reasoning and guidance
    reasoning: str = Field(..., description="Detailed reasoning for recommendation")
    success_factors: List[str] = Field(..., description="Key factors for success")
    potential_challenges: List[str] = Field(..., description="Potential challenges to consider")
    
    # Metadata
    recommendation_date: datetime = Field(..., description="When recommendation was generated")
    data_sources: List[str] = Field(..., description="Data sources used for recommendation")
    
    @classmethod
    def from_domain_model(cls, trajectory) -> "CareerTrajectoryResponse":
        """Convert domain model to response schema."""
        # Convert salary progression to SalaryRange objects
        salary_progression = {}
        for role, (min_sal, max_sal) in trajectory.salary_progression.items():
            salary_progression[role] = SalaryRange(
                min_salary=min_sal,
                max_salary=max_sal,
                currency="USD"
            )
        
        # Convert progression steps
        progression_steps = []
        for step in trajectory.progression_steps:
            progression_steps.append(ProgressionStep(
                role=step['role'],
                duration_months=step['duration_months'],
                description=step['description'],
                key_activities=step.get('key_activities', []),
                skills_to_develop=step.get('skills_to_develop', []),
                milestones=step.get('milestones', [])
            ))
        
        # Convert alternative routes
        alternative_routes = []
        for alt in trajectory.alternative_routes:
            alt_steps = []
            for step in alt['progression_steps']:
                alt_steps.append(ProgressionStep(
                    role=step['role'],
                    duration_months=step['duration_months'],
                    description=step['description'],
                    key_activities=step.get('key_activities', []),
                    skills_to_develop=step.get('skills_to_develop', []),
                    milestones=step.get('milestones', [])
                ))
            
            alternative_routes.append(AlternativeRoute(
                path_id=alt['path_id'],
                approach=alt['approach'],
                description=alt['description'],
                progression_steps=alt_steps,
                estimated_timeline_months=alt['estimated_timeline_months'],
                advantages=alt.get('advantages', []),
                considerations=alt.get('considerations', []),
                success_rate=alt.get('success_rate', 0.7)
            ))
        
        return cls(
            trajectory_id=trajectory.trajectory_id,
            title=trajectory.title,
            target_role=trajectory.target_role,
            match_score=trajectory.match_score,
            confidence_score=trajectory.confidence_score,
            progression_steps=progression_steps,
            estimated_timeline_months=trajectory.estimated_timeline_months,
            difficulty_level=trajectory.difficulty_level,
            required_skills=trajectory.required_skills,
            skill_gaps=trajectory.skill_gaps,
            transferable_skills=trajectory.transferable_skills,
            market_demand=trajectory.market_demand,
            salary_progression=salary_progression,
            growth_potential=trajectory.growth_potential,
            alternative_routes=alternative_routes,
            lateral_opportunities=trajectory.lateral_opportunities,
            reasoning=trajectory.reasoning,
            success_factors=trajectory.success_factors,
            potential_challenges=trajectory.potential_challenges,
            recommendation_date=trajectory.recommendation_date,
            data_sources=trajectory.data_sources
        )


class CareerTrajectoryListResponse(BaseModel):
    """Response for list of career trajectories."""
    trajectories: List[CareerTrajectoryResponse] = Field(..., description="List of career trajectory recommendations")
    total_count: int = Field(..., description="Total number of trajectories")
    generated_at: Optional[datetime] = Field(None, description="When recommendations were generated")


class SkillGapAnalysisResponse(BaseModel):
    """Skill gap analysis response."""
    target_role: str = Field(..., description="Target role analyzed")
    missing_skills: Dict[str, float] = Field(..., description="Skills missing with importance scores")
    weak_skills: Dict[str, float] = Field(..., description="Weak skills with gap scores")
    strong_skills: List[str] = Field(..., description="Strong existing skills")
    overall_readiness: float = Field(..., ge=0, le=1, description="Overall readiness score")
    learning_time_estimate_weeks: int = Field(..., description="Estimated learning time in weeks")
    priority_skills: List[str] = Field(..., description="Priority skills to develop first")
    readiness_percentage: float = Field(..., description="Readiness as percentage")
    analysis_date: str = Field(..., description="When analysis was performed")


class JobMatchScoreResponse(BaseModel):
    """Job match score analysis response."""
    job_id: str = Field(..., description="Job posting identifier")
    job_title: str = Field(..., description="Job title")
    company: str = Field(..., description="Company name")
    match_score: float = Field(..., ge=0, le=1, description="Overall match score")
    match_percentage: float = Field(..., description="Match score as percentage")
    skill_gaps: Dict[str, float] = Field(..., description="Missing skills with importance")
    weak_skills: Dict[str, float] = Field(..., description="Weak skills with gap scores")
    strong_skills: List[str] = Field(..., description="Strong matching skills")
    overall_readiness: float = Field(..., ge=0, le=1, description="Overall readiness for role")
    readiness_percentage: float = Field(..., description="Readiness as percentage")
    analysis_date: str = Field(..., description="When analysis was performed")


class MarketInsightResponse(BaseModel):
    """Market insight response for a role."""
    role: str = Field(..., description="Role analyzed")
    demand_level: str = Field(..., description="Market demand level")
    growth_potential: float = Field(..., ge=0, le=1, description="Growth potential score")
    salary_trend: str = Field(..., description="Salary trend direction")
    job_count: int = Field(..., description="Number of recent job postings")
    average_salary: Optional[int] = Field(None, description="Average salary if available")
    analysis_date: Optional[datetime] = Field(None, description="When analysis was performed")


# Request schemas for POST endpoints

class TrajectoryRefreshRequest(BaseModel):
    """Request to refresh career trajectories."""
    n_recommendations: int = Field(5, ge=1, le=10, description="Number of recommendations")
    include_alternatives: bool = Field(True, description="Include alternative paths")
    force_refresh: bool = Field(False, description="Force refresh of cached data")


class SkillGapAnalysisRequest(BaseModel):
    """Request for skill gap analysis."""
    target_role: str = Field(..., description="Target role to analyze")
    include_learning_paths: bool = Field(False, description="Include learning path suggestions")


class JobMatchRequest(BaseModel):
    """Request for job match analysis."""
    job_id: str = Field(..., description="Job posting identifier")
    include_recommendations: bool = Field(False, description="Include improvement recommendations")


# Additional utility schemas

class SkillImportance(BaseModel):
    """Skill with importance score."""
    skill: str = Field(..., description="Skill name")
    importance: float = Field(..., ge=0, le=1, description="Importance score")
    category: Optional[str] = Field(None, description="Skill category")


class CareerPathSummary(BaseModel):
    """Summary of a career path."""
    path_name: str = Field(..., description="Career path name")
    typical_roles: List[str] = Field(..., description="Typical roles in this path")
    key_skills: List[str] = Field(..., description="Key skills for this path")
    average_timeline_months: int = Field(..., description="Average timeline to complete path")
    market_demand: str = Field(..., description="Current market demand")


class TrajectoryComparison(BaseModel):
    """Comparison between multiple trajectories."""
    trajectory_ids: List[str] = Field(..., description="Trajectory IDs being compared")
    comparison_metrics: Dict[str, Any] = Field(..., description="Comparison metrics")
    recommendation: str = Field(..., description="Recommended trajectory with reasoning")