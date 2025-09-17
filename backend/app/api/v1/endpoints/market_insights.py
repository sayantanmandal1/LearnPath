"""
Market insights API endpoints for comprehensive market analysis
"""
from typing import List, Optional, Dict, Any
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel, Field

from app.core.database import get_db
from app.services.market_insights_service import MarketInsightsService
from app.api.dependencies import get_current_user
from app.models.user import User

router = APIRouter()

# Service instance
market_insights_service = MarketInsightsService()


# Request/Response Models
class MarketInsightsRequest(BaseModel):
    """Request model for market insights"""
    role: Optional[str] = Field(None, description="Target role to analyze")
    skills: Optional[List[str]] = Field(None, description="Skills to analyze")
    location: Optional[str] = Field(None, description="Target location")
    experience_level: Optional[str] = Field(None, description="Experience level filter")
    days: int = Field(90, ge=7, le=365, description="Analysis period in days")


class MarketOverview(BaseModel):
    """Market overview data"""
    total_jobs: int = Field(..., description="Total job postings found")
    avg_salary: Optional[float] = Field(None, description="Average salary")
    salary_range: Optional[tuple] = Field(None, description="Salary range (min, max)")
    growth_rate: float = Field(..., description="Market growth rate")
    demand_score: float = Field(..., description="Demand score (0-1)")


class SkillAnalysis(BaseModel):
    """Skill analysis data"""
    skill_name: str = Field(..., description="Skill name")
    demand_count: int = Field(..., description="Number of job postings")
    growth_rate: float = Field(..., description="Growth rate")
    avg_salary: Optional[float] = Field(None, description="Average salary for this skill")
    competition_level: str = Field(..., description="Competition level")
    trend_direction: str = Field(..., description="Trend direction")


class GeographicData(BaseModel):
    """Geographic market data"""
    top_locations: List[Dict[str, Any]] = Field(..., description="Top locations by job count")
    remote_opportunities: int = Field(..., description="Number of remote opportunities")


class IndustryTrends(BaseModel):
    """Industry trends data"""
    emerging_skills: List[Dict[str, Any]] = Field(..., description="Emerging skills")
    declining_skills: List[Dict[str, Any]] = Field(..., description="Declining skills")
    hot_technologies: List[str] = Field(..., description="Hot technologies")


class MarketRecommendation(BaseModel):
    """Market recommendation"""
    type: str = Field(..., description="Recommendation type")
    message: str = Field(..., description="Recommendation message")


class MarketInsightsResponse(BaseModel):
    """Comprehensive market insights response"""
    demand_trend: str = Field(..., description="Overall demand trend")
    salary_growth: str = Field(..., description="Salary growth trend")
    top_skills: List[str] = Field(..., description="Top skills in demand")
    competition_level: str = Field(..., description="Market competition level")
    market_overview: MarketOverview = Field(..., description="Market overview data")
    skill_analysis: List[SkillAnalysis] = Field(..., description="Detailed skill analysis")
    geographic_data: GeographicData = Field(..., description="Geographic insights")
    industry_trends: IndustryTrends = Field(..., description="Industry trends")
    recommendations: List[MarketRecommendation] = Field(..., description="Market recommendations")
    analysis_date: str = Field(..., description="Analysis timestamp")
    data_freshness: str = Field(..., description="Data freshness indicator")


class SimpleMarketInsights(BaseModel):
    """Simple market insights for frontend compatibility"""
    demandTrend: str = Field(..., description="Market demand trend")
    salaryGrowth: str = Field(..., description="Salary growth trend")
    topSkills: List[str] = Field(..., description="Top skills in demand")
    competitionLevel: str = Field(..., description="Competition level")


@router.post("/comprehensive", response_model=MarketInsightsResponse)
async def get_comprehensive_market_insights(
    request: MarketInsightsRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get comprehensive market insights for role/skills combination
    
    This endpoint provides detailed market analysis including:
    - Demand trends and growth rates
    - Salary analysis and projections
    - Skill demand and competition levels
    - Geographic market distribution
    - Industry trends and emerging technologies
    - Personalized recommendations
    """
    try:
        insights = await market_insights_service.get_comprehensive_market_insights(
            db=db,
            role=request.role,
            skills=request.skills,
            location=request.location,
            experience_level=request.experience_level,
            days=request.days
        )
        
        return MarketInsightsResponse(**insights)
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate market insights: {str(e)}"
        )


@router.get("/simple", response_model=SimpleMarketInsights)
async def get_simple_market_insights(
    role: Optional[str] = Query(None, description="Target role"),
    skills: Optional[str] = Query(None, description="Comma-separated skills"),
    location: Optional[str] = Query(None, description="Target location"),
    experience_level: Optional[str] = Query(None, description="Experience level"),
    days: int = Query(90, ge=7, le=365, description="Analysis period"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get simple market insights in frontend-compatible format
    
    Returns basic market insights that match the frontend's expected format:
    - demandTrend: Market demand level
    - salaryGrowth: Salary growth trend
    - topSkills: List of top skills
    - competitionLevel: Competition level
    """
    try:
        # Parse skills
        skills_list = None
        if skills:
            skills_list = [skill.strip() for skill in skills.split(',') if skill.strip()]
        
        insights = await market_insights_service.get_comprehensive_market_insights(
            db=db,
            role=role,
            skills=skills_list,
            location=location,
            experience_level=experience_level,
            days=days
        )
        
        # Convert to simple format
        simple_insights = SimpleMarketInsights(
            demandTrend=insights['demand_trend'],
            salaryGrowth=insights['salary_growth'],
            topSkills=insights['top_skills'],
            competitionLevel=insights['competition_level']
        )
        
        return simple_insights
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate market insights: {str(e)}"
        )


@router.get("/role/{role}")
async def get_role_market_insights(
    role: str,
    location: Optional[str] = Query(None, description="Target location"),
    experience_level: Optional[str] = Query(None, description="Experience level"),
    days: int = Query(90, ge=7, le=365, description="Analysis period"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get market insights for a specific role
    
    Provides role-specific market analysis including demand trends,
    salary information, required skills, and competition levels.
    """
    try:
        insights = await market_insights_service.get_comprehensive_market_insights(
            db=db,
            role=role,
            skills=None,
            location=location,
            experience_level=experience_level,
            days=days
        )
        
        return {
            "role": role,
            "market_insights": insights,
            "analysis_date": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get role market insights: {str(e)}"
        )


@router.get("/skills/{skill}")
async def get_skill_market_insights(
    skill: str,
    location: Optional[str] = Query(None, description="Target location"),
    experience_level: Optional[str] = Query(None, description="Experience level"),
    days: int = Query(90, ge=7, le=365, description="Analysis period"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get market insights for a specific skill
    
    Provides skill-specific market analysis including demand trends,
    salary premiums, related roles, and growth projections.
    """
    try:
        insights = await market_insights_service.get_comprehensive_market_insights(
            db=db,
            role=None,
            skills=[skill],
            location=location,
            experience_level=experience_level,
            days=days
        )
        
        # Extract skill-specific data
        skill_data = None
        for skill_analysis in insights.get('skill_analysis', []):
            if skill_analysis['skill_name'].lower() == skill.lower():
                skill_data = skill_analysis
                break
        
        return {
            "skill": skill,
            "skill_data": skill_data,
            "market_insights": insights,
            "analysis_date": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get skill market insights: {str(e)}"
        )


@router.get("/trending-skills")
async def get_trending_skills(
    days: int = Query(30, ge=7, le=90, description="Analysis period"),
    limit: int = Query(20, ge=1, le=50, description="Number of skills to return"),
    location: Optional[str] = Query(None, description="Location filter"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get trending skills in the job market
    
    Returns a list of skills that are currently trending based on
    job posting frequency, growth rates, and demand patterns.
    """
    try:
        insights = await market_insights_service.get_comprehensive_market_insights(
            db=db,
            role=None,
            skills=None,
            location=location,
            experience_level=None,
            days=days
        )
        
        # Extract trending skills data
        trending_skills = []
        for skill_analysis in insights.get('skill_analysis', [])[:limit]:
            trending_skills.append({
                'skill': skill_analysis['skill_name'],
                'demand_count': skill_analysis['demand_count'],
                'growth_rate': skill_analysis['growth_rate'],
                'trend_direction': skill_analysis['trend_direction'],
                'avg_salary': skill_analysis['avg_salary']
            })
        
        return {
            "trending_skills": trending_skills,
            "analysis_period_days": days,
            "total_skills_analyzed": len(insights.get('skill_analysis', [])),
            "analysis_date": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get trending skills: {str(e)}"
        )


@router.get("/salary-analysis")
async def get_salary_analysis(
    role: Optional[str] = Query(None, description="Target role"),
    skills: Optional[str] = Query(None, description="Comma-separated skills"),
    location: Optional[str] = Query(None, description="Target location"),
    experience_level: Optional[str] = Query(None, description="Experience level"),
    days: int = Query(90, ge=30, le=365, description="Analysis period"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get detailed salary analysis for role/skills combination
    
    Provides comprehensive salary insights including:
    - Average and median salaries
    - Salary ranges and distributions
    - Growth trends and projections
    - Geographic salary variations
    - Experience level impact
    """
    try:
        # Parse skills
        skills_list = None
        if skills:
            skills_list = [skill.strip() for skill in skills.split(',') if skill.strip()]
        
        insights = await market_insights_service.get_comprehensive_market_insights(
            db=db,
            role=role,
            skills=skills_list,
            location=location,
            experience_level=experience_level,
            days=days
        )
        
        market_overview = insights.get('market_overview', {})
        
        return {
            "salary_analysis": {
                "avg_salary": market_overview.get('avg_salary'),
                "salary_range": market_overview.get('salary_range'),
                "growth_trend": insights.get('salary_growth'),
                "sample_size": len(insights.get('skill_analysis', [])),
                "confidence_level": "Medium"  # Could be calculated based on sample size
            },
            "geographic_variations": insights.get('geographic_data', {}),
            "skill_premiums": [
                {
                    "skill": skill['skill_name'],
                    "avg_salary": skill['avg_salary']
                }
                for skill in insights.get('skill_analysis', [])
                if skill.get('avg_salary')
            ],
            "analysis_date": datetime.utcnow().isoformat(),
            "analysis_period_days": days
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get salary analysis: {str(e)}"
        )


@router.get("/competition-analysis")
async def get_competition_analysis(
    role: Optional[str] = Query(None, description="Target role"),
    skills: Optional[str] = Query(None, description="Comma-separated skills"),
    location: Optional[str] = Query(None, description="Target location"),
    experience_level: Optional[str] = Query(None, description="Experience level"),
    days: int = Query(90, ge=7, le=365, description="Analysis period"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get market competition analysis
    
    Analyzes market competition levels and provides insights on:
    - Overall competition level
    - Job availability vs demand
    - Skill differentiation opportunities
    - Market positioning recommendations
    """
    try:
        # Parse skills
        skills_list = None
        if skills:
            skills_list = [skill.strip() for skill in skills.split(',') if skill.strip()]
        
        insights = await market_insights_service.get_comprehensive_market_insights(
            db=db,
            role=role,
            skills=skills_list,
            location=location,
            experience_level=experience_level,
            days=days
        )
        
        market_overview = insights.get('market_overview', {})
        
        return {
            "competition_analysis": {
                "competition_level": insights.get('competition_level'),
                "job_availability": market_overview.get('total_jobs'),
                "demand_score": market_overview.get('demand_score'),
                "market_saturation": "Low" if market_overview.get('demand_score', 0) > 0.1 else "High"
            },
            "differentiation_opportunities": [
                skill['skill_name'] for skill in insights.get('skill_analysis', [])
                if skill.get('competition_level') == 'Low'
            ][:5],
            "recommendations": insights.get('recommendations', []),
            "analysis_date": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get competition analysis: {str(e)}"
        )


@router.get("/health")
async def health_check():
    """
    Health check for market insights service
    """
    return {
        "status": "healthy",
        "service": "market_insights",
        "timestamp": datetime.utcnow().isoformat()
    }