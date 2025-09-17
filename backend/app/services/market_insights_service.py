"""
Market insights service for comprehensive market analysis and trend calculations
"""
import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from statistics import median, mean
import asyncio

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, or_, desc, text
from sqlalchemy.orm import selectinload

from app.models.job import JobPosting, JobSkill
from app.models.skill import Skill
from app.models.profile import UserProfile
from app.repositories.job import JobRepository
from app.services.market_trend_analyzer import MarketTrendAnalyzer

logger = logging.getLogger(__name__)


@dataclass
class MarketInsightData:
    """Market insight data structure"""
    demand_trend: str
    salary_growth: str
    top_skills: List[str]
    competition_level: str
    avg_salary: Optional[float]
    salary_range: Optional[Tuple[float, float]]
    job_count: int
    growth_rate: float
    demand_score: float
    market_competitiveness: str


@dataclass
class SkillDemandData:
    """Skill demand analysis data"""
    skill_name: str
    demand_count: int
    growth_rate: float
    avg_salary: Optional[float]
    competition_level: str
    trend_direction: str


@dataclass
class SalaryAnalysis:
    """Salary analysis data"""
    role: str
    location: Optional[str]
    avg_salary: float
    median_salary: float
    salary_range: Tuple[float, float]
    growth_rate: float
    sample_size: int
    confidence_level: float


class MarketInsightsService:
    """Service for generating comprehensive market insights"""
    
    def __init__(self):
        self.job_repository = JobRepository()
        self.trend_analyzer = MarketTrendAnalyzer()
        
        # Cache for market data
        self._market_cache = {}
        self._cache_ttl = 3600  # 1 hour cache
    
    async def get_comprehensive_market_insights(
        self,
        db: AsyncSession,
        role: Optional[str] = None,
        skills: Optional[List[str]] = None,
        location: Optional[str] = None,
        experience_level: Optional[str] = None,
        days: int = 90
    ) -> Dict[str, Any]:
        """
        Get comprehensive market insights for role/skills combination
        
        Args:
            db: Database session
            role: Target role
            skills: List of skills to analyze
            location: Target location
            experience_level: Experience level filter
            days: Analysis period in days
            
        Returns:
            Comprehensive market insights data
        """
        try:
            logger.info(f"Generating market insights for role: {role}, skills: {skills}")
            
            # Generate cache key
            cache_key = self._generate_cache_key(role, skills, location, experience_level, days)
            
            # Check cache
            if cache_key in self._market_cache:
                cached_data, timestamp = self._market_cache[cache_key]
                if datetime.utcnow() - timestamp < timedelta(seconds=self._cache_ttl):
                    logger.info("Returning cached market insights")
                    return cached_data
            
            # Gather market data
            market_data = await self._gather_market_data(
                db, role, skills, location, experience_level, days
            )
            
            # Analyze salary trends
            salary_analysis = await self._analyze_salary_trends(
                db, role, skills, location, experience_level, days
            )
            
            # Calculate skill demand
            skill_demand = await self._calculate_skill_demand(
                db, skills or [], days
            )
            
            # Determine competition level
            competition_level = await self._calculate_competition_level(
                db, role, skills, location, days
            )
            
            # Generate trend analysis
            trend_analysis = await self._analyze_market_trends(
                db, role, skills, days
            )
            
            # Compile comprehensive insights
            insights = {
                'demand_trend': trend_analysis.get('demand_trend', 'Medium'),
                'salary_growth': salary_analysis.get('growth_trend', '+8% YoY'),
                'top_skills': skill_demand.get('top_skills', skills or [])[:10],
                'competition_level': competition_level,
                'market_overview': {
                    'total_jobs': market_data.get('total_jobs', 0),
                    'avg_salary': salary_analysis.get('avg_salary'),
                    'salary_range': salary_analysis.get('salary_range'),
                    'growth_rate': trend_analysis.get('growth_rate', 0.0),
                    'demand_score': market_data.get('demand_score', 0.0)
                },
                'skill_analysis': skill_demand.get('skill_breakdown', []),
                'geographic_data': await self._get_geographic_insights(
                    db, role, skills, days
                ),
                'industry_trends': await self._get_industry_trends(
                    db, role, days
                ),
                'recommendations': await self._generate_market_recommendations(
                    market_data, salary_analysis, skill_demand, competition_level
                ),
                'analysis_date': datetime.utcnow().isoformat(),
                'data_freshness': 'Real-time' if days <= 30 else 'Historical'
            }
            
            # Cache the results
            self._market_cache[cache_key] = (insights, datetime.utcnow())
            
            return insights
            
        except Exception as e:
            logger.error(f"Error generating market insights: {str(e)}")
            return self._get_fallback_insights(role, skills)
    
    async def _gather_market_data(
        self,
        db: AsyncSession,
        role: Optional[str],
        skills: Optional[List[str]],
        location: Optional[str],
        experience_level: Optional[str],
        days: int
    ) -> Dict[str, Any]:
        """Gather basic market data"""
        
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        # Build base query
        query = select(JobPosting).where(
            and_(
                JobPosting.posted_date >= cutoff_date,
                JobPosting.is_active == True
            )
        )
        
        # Add filters
        if role:
            query = query.where(
                or_(
                    JobPosting.title.ilike(f"%{role}%"),
                    JobPosting.description.ilike(f"%{role}%")
                )
            )
        
        if location:
            query = query.where(JobPosting.location.ilike(f"%{location}%"))
        
        if experience_level:
            query = query.where(JobPosting.experience_level == experience_level)
        
        # Execute query
        result = await db.execute(query)
        jobs = result.scalars().all()
        
        # Filter by skills if provided
        if skills:
            filtered_jobs = []
            for job in jobs:
                if job.processed_skills:
                    job_skills = [skill.lower() for skill in job.processed_skills.keys()]
                    if any(skill.lower() in job_skills for skill in skills):
                        filtered_jobs.append(job)
            jobs = filtered_jobs
        
        # Calculate demand score
        total_jobs_query = select(func.count(JobPosting.id)).where(
            and_(
                JobPosting.posted_date >= cutoff_date,
                JobPosting.is_active == True
            )
        )
        total_result = await db.execute(total_jobs_query)
        total_jobs = total_result.scalar() or 1
        
        demand_score = len(jobs) / total_jobs if total_jobs > 0 else 0
        
        return {
            'total_jobs': len(jobs),
            'demand_score': demand_score,
            'jobs_data': jobs
        }
    
    async def _analyze_salary_trends(
        self,
        db: AsyncSession,
        role: Optional[str],
        skills: Optional[List[str]],
        location: Optional[str],
        experience_level: Optional[str],
        days: int
    ) -> Dict[str, Any]:
        """Analyze salary trends and growth"""
        
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        # Get salary data
        query = select(
            JobPosting.salary_min,
            JobPosting.salary_max,
            JobPosting.salary_period,
            JobPosting.posted_date
        ).where(
            and_(
                JobPosting.posted_date >= cutoff_date,
                JobPosting.is_active == True,
                or_(
                    JobPosting.salary_min.isnot(None),
                    JobPosting.salary_max.isnot(None)
                )
            )
        )
        
        # Add filters
        if role:
            query = query.where(
                or_(
                    JobPosting.title.ilike(f"%{role}%"),
                    JobPosting.description.ilike(f"%{role}%")
                )
            )
        
        if location:
            query = query.where(JobPosting.location.ilike(f"%{location}%"))
        
        if experience_level:
            query = query.where(JobPosting.experience_level == experience_level)
        
        result = await db.execute(query)
        salary_data = result.fetchall()
        
        if not salary_data:
            return {
                'avg_salary': None,
                'salary_range': None,
                'growth_trend': '+8% YoY'
            }
        
        # Process salary data
        salaries = []
        for row in salary_data:
            if row.salary_min and row.salary_max:
                avg_salary = (row.salary_min + row.salary_max) / 2
            else:
                avg_salary = row.salary_min or row.salary_max
            
            if avg_salary:
                # Normalize to yearly
                if row.salary_period == 'hourly':
                    avg_salary *= 2080  # 40 hours/week * 52 weeks
                elif row.salary_period == 'monthly':
                    avg_salary *= 12
                
                salaries.append(avg_salary)
        
        if not salaries:
            return {
                'avg_salary': None,
                'salary_range': None,
                'growth_trend': '+8% YoY'
            }
        
        # Calculate statistics
        avg_salary = mean(salaries)
        median_salary = median(salaries)
        min_salary = min(salaries)
        max_salary = max(salaries)
        
        # Calculate growth trend (simplified)
        # In production, this would compare with historical data
        growth_rate = self._estimate_growth_rate(role, skills, avg_salary)
        growth_trend = f"+{growth_rate:.0f}% YoY" if growth_rate > 0 else f"{growth_rate:.0f}% YoY"
        
        return {
            'avg_salary': avg_salary,
            'median_salary': median_salary,
            'salary_range': (min_salary, max_salary),
            'growth_trend': growth_trend,
            'sample_size': len(salaries)
        }
    
    async def _calculate_skill_demand(
        self,
        db: AsyncSession,
        skills: List[str],
        days: int
    ) -> Dict[str, Any]:
        """Calculate skill demand and trends"""
        
        if not skills:
            # Get top skills from recent jobs
            skills = await self._get_trending_skills(db, days, limit=20)
        
        skill_data = []
        
        for skill in skills:
            try:
                # Get skill demand data
                demand_data = await self.trend_analyzer.get_skill_market_data(db, skill)
                
                skill_info = {
                    'skill_name': skill,
                    'demand_count': demand_data.get('job_count', 0),
                    'growth_rate': demand_data.get('growth_trend', 0.0),
                    'avg_salary': demand_data.get('avg_salary'),
                    'competition_level': demand_data.get('market_competitiveness', 'medium'),
                    'trend_direction': 'growing' if demand_data.get('growth_trend', 0) > 0 else 'stable'
                }
                
                skill_data.append(skill_info)
                
            except Exception as e:
                logger.warning(f"Failed to get demand data for skill {skill}: {e}")
                continue
        
        # Sort by demand
        skill_data.sort(key=lambda x: x['demand_count'], reverse=True)
        
        return {
            'top_skills': [skill['skill_name'] for skill in skill_data[:10]],
            'skill_breakdown': skill_data,
            'total_skills_analyzed': len(skill_data)
        }
    
    async def _calculate_competition_level(
        self,
        db: AsyncSession,
        role: Optional[str],
        skills: Optional[List[str]],
        location: Optional[str],
        days: int
    ) -> str:
        """Calculate market competition level"""
        
        try:
            # Get job count for this role/skills combination
            market_data = await self._gather_market_data(
                db, role, skills, location, None, days
            )
            
            job_count = market_data['total_jobs']
            demand_score = market_data['demand_score']
            
            # Simple competition calculation based on job availability
            if job_count > 1000 and demand_score > 0.1:
                return 'Low'
            elif job_count > 500 and demand_score > 0.05:
                return 'Medium'
            elif job_count > 100:
                return 'High'
            else:
                return 'Very High'
                
        except Exception as e:
            logger.warning(f"Failed to calculate competition level: {e}")
            return 'Medium'
    
    async def _analyze_market_trends(
        self,
        db: AsyncSession,
        role: Optional[str],
        skills: Optional[List[str]],
        days: int
    ) -> Dict[str, Any]:
        """Analyze overall market trends"""
        
        try:
            # Get trend data from trend analyzer
            if skills:
                trends = await self.trend_analyzer.analyze_skill_demand_trends(
                    db, skills, days
                )
            else:
                trends = await self.trend_analyzer.analyze_skill_demand_trends(
                    db, None, days
                )
            
            if not trends:
                return {
                    'demand_trend': 'Medium',
                    'growth_rate': 0.08
                }
            
            # Calculate overall trend
            avg_growth = mean([trend.get('growth_rate_weekly', 0) for trend in trends])
            
            # Determine demand trend
            if avg_growth > 0.15:
                demand_trend = 'Very High'
            elif avg_growth > 0.08:
                demand_trend = 'High'
            elif avg_growth > 0.02:
                demand_trend = 'Medium'
            elif avg_growth > -0.05:
                demand_trend = 'Low'
            else:
                demand_trend = 'Declining'
            
            return {
                'demand_trend': demand_trend,
                'growth_rate': avg_growth,
                'trend_confidence': mean([trend.get('confidence', 0) for trend in trends])
            }
            
        except Exception as e:
            logger.warning(f"Failed to analyze market trends: {e}")
            return {
                'demand_trend': 'Medium',
                'growth_rate': 0.08
            }
    
    async def _get_geographic_insights(
        self,
        db: AsyncSession,
        role: Optional[str],
        skills: Optional[List[str]],
        days: int
    ) -> Dict[str, Any]:
        """Get geographic market insights"""
        
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            
            # Get location distribution
            query = select(
                JobPosting.location,
                func.count(JobPosting.id).label('job_count'),
                func.avg(JobPosting.salary_min).label('avg_salary_min'),
                func.avg(JobPosting.salary_max).label('avg_salary_max')
            ).where(
                and_(
                    JobPosting.posted_date >= cutoff_date,
                    JobPosting.is_active == True,
                    JobPosting.location.isnot(None)
                )
            ).group_by(JobPosting.location).order_by(desc('job_count')).limit(10)
            
            # Add role filter if provided
            if role:
                query = query.where(
                    or_(
                        JobPosting.title.ilike(f"%{role}%"),
                        JobPosting.description.ilike(f"%{role}%")
                    )
                )
            
            result = await db.execute(query)
            location_data = result.fetchall()
            
            top_locations = []
            for row in location_data:
                avg_salary = None
                if row.avg_salary_min and row.avg_salary_max:
                    avg_salary = (row.avg_salary_min + row.avg_salary_max) / 2
                
                top_locations.append({
                    'location': row.location,
                    'job_count': row.job_count,
                    'avg_salary': avg_salary
                })
            
            return {
                'top_locations': top_locations,
                'remote_opportunities': await self._get_remote_job_count(db, role, skills, days)
            }
            
        except Exception as e:
            logger.warning(f"Failed to get geographic insights: {e}")
            return {'top_locations': [], 'remote_opportunities': 0}
    
    async def _get_industry_trends(
        self,
        db: AsyncSession,
        role: Optional[str],
        days: int
    ) -> Dict[str, Any]:
        """Get industry-specific trends"""
        
        try:
            # Get emerging skills
            emerging_skills = await self.trend_analyzer.detect_emerging_skills(db, days)
            
            return {
                'emerging_skills': [
                    {
                        'skill': skill.skill_name,
                        'growth_rate': skill.growth_rate,
                        'trend_score': skill.trend_score
                    }
                    for skill in emerging_skills[:5]
                ],
                'declining_skills': [],  # Would need historical comparison
                'hot_technologies': [skill.skill_name for skill in emerging_skills[:3]]
            }
            
        except Exception as e:
            logger.warning(f"Failed to get industry trends: {e}")
            return {'emerging_skills': [], 'declining_skills': [], 'hot_technologies': []}
    
    async def _generate_market_recommendations(
        self,
        market_data: Dict[str, Any],
        salary_analysis: Dict[str, Any],
        skill_demand: Dict[str, Any],
        competition_level: str
    ) -> List[Dict[str, str]]:
        """Generate market-based recommendations"""
        
        recommendations = []
        
        # Competition-based recommendations
        if competition_level == 'Very High':
            recommendations.append({
                'type': 'skill_development',
                'message': 'Consider developing niche skills to stand out in this competitive market'
            })
        elif competition_level == 'Low':
            recommendations.append({
                'type': 'opportunity',
                'message': 'Great market opportunity with low competition - consider applying soon'
            })
        
        # Salary-based recommendations
        avg_salary = salary_analysis.get('avg_salary')
        if avg_salary and avg_salary > 100000:
            recommendations.append({
                'type': 'salary',
                'message': 'This role offers above-average compensation in the current market'
            })
        
        # Skill-based recommendations
        top_skills = skill_demand.get('top_skills', [])
        if top_skills:
            recommendations.append({
                'type': 'skills',
                'message': f'Focus on developing {", ".join(top_skills[:3])} for better market positioning'
            })
        
        return recommendations
    
    async def _get_trending_skills(
        self,
        db: AsyncSession,
        days: int,
        limit: int = 20
    ) -> List[str]:
        """Get trending skills from recent job postings"""
        
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        query = select(
            Skill.name,
            func.count(JobSkill.id).label('skill_count')
        ).select_from(
            JobSkill.__table__.join(Skill.__table__).join(JobPosting.__table__)
        ).where(
            JobPosting.posted_date >= cutoff_date
        ).group_by(
            Skill.id, Skill.name
        ).order_by(
            desc('skill_count')
        ).limit(limit)
        
        result = await db.execute(query)
        skills = result.fetchall()
        
        return [row.name for row in skills]
    
    async def _get_remote_job_count(
        self,
        db: AsyncSession,
        role: Optional[str],
        skills: Optional[List[str]],
        days: int
    ) -> int:
        """Get count of remote job opportunities"""
        
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        query = select(func.count(JobPosting.id)).where(
            and_(
                JobPosting.posted_date >= cutoff_date,
                JobPosting.is_active == True,
                or_(
                    JobPosting.remote_type == 'remote',
                    JobPosting.location.ilike('%remote%')
                )
            )
        )
        
        if role:
            query = query.where(
                or_(
                    JobPosting.title.ilike(f"%{role}%"),
                    JobPosting.description.ilike(f"%{role}%")
                )
            )
        
        result = await db.execute(query)
        return result.scalar() or 0
    
    def _estimate_growth_rate(
        self,
        role: Optional[str],
        skills: Optional[List[str]],
        avg_salary: float
    ) -> float:
        """Estimate growth rate based on role and skills"""
        
        # Simple heuristic-based growth estimation
        # In production, this would use historical data
        
        base_growth = 8.0  # Base 8% growth
        
        # Tech roles typically have higher growth
        if role and any(tech in role.lower() for tech in ['engineer', 'developer', 'data', 'ai', 'ml']):
            base_growth += 4.0
        
        # High-demand skills boost growth
        if skills:
            high_demand_skills = ['python', 'react', 'aws', 'kubernetes', 'machine learning', 'ai']
            skill_boost = sum(2.0 for skill in skills if skill.lower() in high_demand_skills)
            base_growth += min(skill_boost, 8.0)  # Cap at 8% boost
        
        # Salary level affects growth
        if avg_salary > 150000:
            base_growth += 2.0
        elif avg_salary > 100000:
            base_growth += 1.0
        
        return min(base_growth, 25.0)  # Cap at 25% growth
    
    def _generate_cache_key(
        self,
        role: Optional[str],
        skills: Optional[List[str]],
        location: Optional[str],
        experience_level: Optional[str],
        days: int
    ) -> str:
        """Generate cache key for market insights"""
        
        key_parts = [
            role or 'any_role',
            '_'.join(sorted(skills)) if skills else 'no_skills',
            location or 'any_location',
            experience_level or 'any_level',
            str(days)
        ]
        
        return '_'.join(key_parts).lower().replace(' ', '_')
    
    def _get_fallback_insights(
        self,
        role: Optional[str],
        skills: Optional[List[str]]
    ) -> Dict[str, Any]:
        """Get fallback insights when analysis fails"""
        
        return {
            'demand_trend': 'Medium',
            'salary_growth': '+8% YoY',
            'top_skills': skills[:10] if skills else ['Communication', 'Problem Solving'],
            'competition_level': 'Medium',
            'market_overview': {
                'total_jobs': 0,
                'avg_salary': None,
                'salary_range': None,
                'growth_rate': 0.08,
                'demand_score': 0.5
            },
            'skill_analysis': [],
            'geographic_data': {'top_locations': [], 'remote_opportunities': 0},
            'industry_trends': {'emerging_skills': [], 'declining_skills': [], 'hot_technologies': []},
            'recommendations': [
                {
                    'type': 'general',
                    'message': 'Market data temporarily unavailable. Please try again later.'
                }
            ],
            'analysis_date': datetime.utcnow().isoformat(),
            'data_freshness': 'Fallback'
        }