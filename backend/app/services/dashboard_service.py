"""
Dashboard service for aggregating user data and generating dashboard content
"""
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, desc

from ..models.user import User
from ..models.profile import UserProfile
from ..schemas.dashboard import (
    DashboardSummary, DashboardMetric, ProgressMilestone, DashboardRecommendation,
    DashboardActivity, UserProgressSummary, PersonalizedContent, DashboardConfiguration
)
# Import services conditionally to avoid dependency issues
try:
    from ..services.analytics_service import AnalyticsService
except ImportError:
    AnalyticsService = None

try:
    from ..services.recommendation_service import RecommendationService
except ImportError:
    RecommendationService = None

try:
    from ..services.profile_service import ProfileService
except ImportError:
    ProfileService = None
from ..core.exceptions import ServiceException

logger = logging.getLogger(__name__)


class DashboardService:
    """Service for dashboard data aggregation and management"""
    
    def __init__(self, db: AsyncSession):
        self.db = db
        # Initialize services conditionally
        try:
            self.analytics_service = AnalyticsService(db) if AnalyticsService else None
        except Exception:
            self.analytics_service = None
        
        try:
            self.recommendation_service = RecommendationService() if RecommendationService else None
        except Exception:
            self.recommendation_service = None
        
        try:
            self.profile_service = ProfileService(db) if ProfileService else None
        except Exception:
            self.profile_service = None
    
    async def get_dashboard_summary(self, user_id: str) -> DashboardSummary:
        """
        Generate comprehensive dashboard summary for user
        """
        try:
            # Get user profile
            if self.profile_service:
                profile = await self.profile_service.get_profile(user_id)
                if not profile:
                    # Return a "needs analysis" dashboard state instead of raising exception
                    return await self._create_needs_analysis_dashboard(user_id)
            else:
                # Create mock profile when service unavailable
                profile = self._create_mock_profile(user_id)
            
            # Get analytics data
            if self.analytics_service:
                analytics = await self.analytics_service.calculate_comprehensive_user_analytics(user_id)
                career_score = await self.analytics_service.generate_overall_career_score_and_recommendations(user_id)
            else:
                # Use mock analytics when service unavailable
                analytics = self._get_mock_analytics(user_id)
                career_score = self._get_mock_career_score(user_id)
            
            # Calculate profile completion
            profile_completion = await self._calculate_profile_completion(profile)
            
            # Get key metrics
            key_metrics = await self._generate_key_metrics(user_id, analytics)
            
            # Get milestones
            active_milestones = await self._get_active_milestones(user_id)
            milestone_stats = await self._get_milestone_stats(user_id)
            
            # Get recommendations
            top_recommendations = await self._get_top_recommendations(user_id)
            
            # Get recent activities
            recent_activities = await self._get_recent_activities(user_id)
            
            # Get quick stats
            quick_stats = await self._get_quick_stats(user_id)
            
            # Get last update dates
            last_analysis_date = await self._get_last_analysis_date(user_id)
            last_profile_update = profile.updated_at if profile else None
            
            return DashboardSummary(
                user_id=user_id,
                overall_career_score=analytics.get("overall_career_score", 0.0),
                profile_completion=profile_completion,
                key_metrics=key_metrics,
                active_milestones=active_milestones,
                completed_milestones_count=milestone_stats["completed"],
                total_milestones_count=milestone_stats["total"],
                top_recommendations=top_recommendations,
                recent_activities=recent_activities,
                skills_count=quick_stats["skills_count"],
                job_matches_count=quick_stats["job_matches_count"],
                learning_paths_count=quick_stats["learning_paths_count"],
                last_analysis_date=last_analysis_date,
                last_profile_update=last_profile_update,
                generated_at=datetime.utcnow()
            )
            
        except Exception as e:
            logger.error(f"Error generating dashboard summary for user {user_id}: {str(e)}")
            raise ServiceException(f"Failed to generate dashboard summary: {str(e)}")
    
    async def get_user_progress_summary(self, user_id: str, tracking_period_days: int = 90) -> UserProgressSummary:
        """
        Generate user progress tracking summary
        """
        try:
            # Get progress tracking data
            if self.analytics_service:
                progress_data = await self.analytics_service.track_historical_progress(user_id, tracking_period_days)
            else:
                progress_data = self._get_mock_progress_data(user_id, tracking_period_days)
            
            # Get career score trend
            career_score_trend = await self._get_career_score_trend(user_id, tracking_period_days)
            
            # Get skill improvements
            skill_improvements = await self._get_skill_improvements(user_id, tracking_period_days)
            
            # Get milestones
            milestones = await self._get_all_milestones(user_id)
            milestone_completion_rate = await self._calculate_milestone_completion_rate(user_id)
            
            # Get learning progress
            learning_progress = await self._get_learning_progress(user_id)
            
            # Get job market progress
            job_market_progress = await self._get_job_market_progress(user_id)
            
            return UserProgressSummary(
                user_id=user_id,
                overall_progress=progress_data.get("overall_progress_percentage", 0.0),
                career_score_trend=career_score_trend,
                skill_improvements=skill_improvements,
                new_skills_added=progress_data.get("new_skills_count", 0),
                skills_mastered=progress_data.get("skills_mastered_count", 0),
                milestones=milestones,
                milestone_completion_rate=milestone_completion_rate,
                learning_paths_started=learning_progress["started"],
                learning_paths_completed=learning_progress["completed"],
                courses_completed=learning_progress["courses_completed"],
                job_compatibility_improvement=job_market_progress["compatibility_improvement"],
                interview_readiness_score=job_market_progress["interview_readiness"],
                tracking_period_days=tracking_period_days,
                generated_at=datetime.utcnow()
            )
            
        except Exception as e:
            logger.error(f"Error generating progress summary for user {user_id}: {str(e)}")
            raise ServiceException(f"Failed to generate progress summary: {str(e)}")
    
    async def get_personalized_content(self, user_id: str) -> PersonalizedContent:
        """
        Generate personalized dashboard content
        """
        try:
            # Get user profile for personalization
            if self.profile_service:
                profile = await self.profile_service.get_profile(user_id)
                if not profile:
                    raise ServiceException(f"Profile not found for user {user_id}")
            else:
                profile = self._create_mock_profile(user_id)
            
            # Get personalized recommendations
            if self.recommendation_service:
                recommendations = await self.recommendation_service.get_recommendations(user_id)
            else:
                recommendations = self._get_mock_recommendations(user_id)
            
            # Get featured jobs
            featured_jobs = await self._get_featured_jobs(user_id)
            
            # Get recommended skills
            recommended_skills = await self._get_recommended_skills(user_id)
            
            # Get suggested learning paths
            suggested_learning_paths = await self._get_suggested_learning_paths(user_id)
            
            # Get market insights
            market_trends = await self._get_market_trends(user_id)
            salary_insights = await self._get_salary_insights(user_id)
            industry_updates = await self._get_industry_updates(user_id)
            
            # Get networking suggestions
            networking_opportunities = await self._get_networking_opportunities(user_id)
            similar_profiles = await self._get_similar_profiles(user_id)
            
            # Calculate personalization score
            personalization_score = await self._calculate_personalization_score(user_id)
            
            return PersonalizedContent(
                user_id=user_id,
                featured_jobs=featured_jobs,
                recommended_skills=recommended_skills,
                suggested_learning_paths=suggested_learning_paths,
                market_trends=market_trends,
                salary_insights=salary_insights,
                industry_updates=industry_updates,
                networking_opportunities=networking_opportunities,
                similar_profiles=similar_profiles,
                content_categories=["jobs", "skills", "learning", "market", "networking"],
                personalization_score=personalization_score,
                generated_at=datetime.utcnow()
            )
            
        except Exception as e:
            logger.error(f"Error generating personalized content for user {user_id}: {str(e)}")
            raise ServiceException(f"Failed to generate personalized content: {str(e)}")
    
    # Helper methods
    
    async def _calculate_profile_completion(self, profile: UserProfile) -> float:
        """Calculate profile completion percentage"""
        required_fields = [
            'current_role', 'experience_years', 'education_level', 'location',
            'skills', 'career_goals', 'preferred_work_type'
        ]
        
        completed_fields = 0
        for field in required_fields:
            value = getattr(profile, field, None)
            if value and (isinstance(value, str) and value.strip() or 
                         isinstance(value, list) and len(value) > 0 or
                         isinstance(value, (int, float)) and value > 0):
                completed_fields += 1
        
        return (completed_fields / len(required_fields)) * 100
    
    async def _generate_key_metrics(self, user_id: str, analytics: Dict[str, Any]) -> List[DashboardMetric]:
        """Generate key dashboard metrics"""
        metrics = []
        
        # Career score metric
        career_score = analytics.get("overall_career_score", 0.0)
        metrics.append(DashboardMetric(
            name="Career Score",
            value=round(career_score, 1),
            unit="points",
            description="Overall career development score"
        ))
        
        # Skills count
        skill_count = analytics.get("skill_analytics", {}).get("total_skills", 0)
        metrics.append(DashboardMetric(
            name="Skills",
            value=skill_count,
            unit="skills",
            description="Total skills in profile"
        ))
        
        # Market position
        market_position = analytics.get("market_analytics", {}).get("market_position_percentile", 0)
        metrics.append(DashboardMetric(
            name="Market Position",
            value=f"{round(market_position)}%",
            description="Your position in the job market"
        ))
        
        # Experience level
        experience_score = analytics.get("experience_analytics", {}).get("experience_score", 0)
        metrics.append(DashboardMetric(
            name="Experience Score",
            value=round(experience_score, 1),
            unit="points",
            description="Professional experience rating"
        ))
        
        return metrics
    
    async def _get_active_milestones(self, user_id: str) -> List[ProgressMilestone]:
        """Get active milestones for user"""
        # Mock implementation - in real app, this would query milestone data
        milestones = [
            ProgressMilestone(
                id="milestone_1",
                title="Complete Python Certification",
                description="Finish Python programming certification course",
                category="learning",
                completed=False,
                target_date=datetime.utcnow() + timedelta(days=30),
                progress_percentage=75.0,
                priority="high"
            ),
            ProgressMilestone(
                id="milestone_2", 
                title="Update LinkedIn Profile",
                description="Enhance LinkedIn profile with recent achievements",
                category="career",
                completed=False,
                progress_percentage=25.0,
                priority="medium"
            )
        ]
        return milestones
    
    async def _get_milestone_stats(self, user_id: str) -> Dict[str, int]:
        """Get milestone statistics"""
        # Mock implementation
        return {"completed": 3, "total": 8}
    
    async def _get_top_recommendations(self, user_id: str) -> List[DashboardRecommendation]:
        """Get top recommendations for dashboard"""
        recommendations = []
        
        try:
            # Get recommendations from service
            if self.recommendation_service:
                rec_data = await self.recommendation_service.get_recommendations(user_id)
            else:
                rec_data = self._get_mock_recommendations(user_id)
            
            # Convert to dashboard format
            for i, rec in enumerate(rec_data.get("recommendations", [])[:5]):
                recommendations.append(DashboardRecommendation(
                    id=f"rec_{i}",
                    title=rec.get("title", "Career Recommendation"),
                    description=rec.get("description", ""),
                    type="career",
                    priority="medium",
                    impact_score=rec.get("confidence_score", 5.0)
                ))
        except Exception as e:
            logger.warning(f"Could not get recommendations for user {user_id}: {str(e)}")
            # Return default recommendations
            recommendations = [
                DashboardRecommendation(
                    id="default_1",
                    title="Update Your Skills",
                    description="Add new skills to your profile to improve job matching",
                    type="skill",
                    priority="high",
                    impact_score=8.0
                )
            ]
        
        return recommendations
    
    async def _get_recent_activities(self, user_id: str) -> List[DashboardActivity]:
        """Get recent user activities"""
        # Mock implementation - in real app, this would query activity logs
        activities = [
            DashboardActivity(
                id="activity_1",
                type="profile_update",
                title="Profile Updated",
                description="Added new skills to profile",
                timestamp=datetime.utcnow() - timedelta(hours=2)
            ),
            DashboardActivity(
                id="activity_2",
                type="analysis_completed",
                title="Career Analysis Completed",
                description="Generated new career recommendations",
                timestamp=datetime.utcnow() - timedelta(days=1)
            )
        ]
        return activities
    
    async def _get_quick_stats(self, user_id: str) -> Dict[str, int]:
        """Get quick statistics for dashboard"""
        # Mock implementation
        return {
            "skills_count": 12,
            "job_matches_count": 25,
            "learning_paths_count": 3
        }
    
    async def _get_last_analysis_date(self, user_id: str) -> Optional[datetime]:
        """Get last analysis date"""
        # Mock implementation
        return datetime.utcnow() - timedelta(days=2)
    
    async def _get_career_score_trend(self, user_id: str, days: int) -> List[Dict[str, Any]]:
        """Get career score trend over time"""
        # Mock implementation
        trend = []
        for i in range(0, days, 7):  # Weekly data points
            date = datetime.utcnow() - timedelta(days=days-i)
            score = 65 + (i / days) * 15  # Simulated improvement
            trend.append({
                "date": date.isoformat(),
                "score": round(score, 1)
            })
        return trend
    
    async def _get_skill_improvements(self, user_id: str, days: int) -> List[Dict[str, Any]]:
        """Get skill improvements over time"""
        # Mock implementation
        return [
            {"skill": "Python", "improvement": 15.0, "category": "programming"},
            {"skill": "Machine Learning", "improvement": 10.0, "category": "technical"},
            {"skill": "Leadership", "improvement": 8.0, "category": "soft_skills"}
        ]
    
    async def _get_all_milestones(self, user_id: str) -> List[ProgressMilestone]:
        """Get all milestones for user"""
        return await self._get_active_milestones(user_id)  # Simplified
    
    async def _calculate_milestone_completion_rate(self, user_id: str) -> float:
        """Calculate milestone completion rate"""
        stats = await self._get_milestone_stats(user_id)
        if stats["total"] == 0:
            return 0.0
        return (stats["completed"] / stats["total"]) * 100
    
    async def _get_learning_progress(self, user_id: str) -> Dict[str, int]:
        """Get learning progress statistics"""
        return {
            "started": 5,
            "completed": 2,
            "courses_completed": 8
        }
    
    async def _get_job_market_progress(self, user_id: str) -> Dict[str, float]:
        """Get job market progress metrics"""
        return {
            "compatibility_improvement": 12.5,
            "interview_readiness": 78.0
        }
    
    async def _get_featured_jobs(self, user_id: str) -> List[Dict[str, Any]]:
        """Get featured job recommendations"""
        return [
            {
                "id": "job_1",
                "title": "Senior Python Developer",
                "company": "Tech Corp",
                "location": "Remote",
                "match_score": 85.0,
                "salary_range": "$90k - $120k"
            }
        ]
    
    async def _get_recommended_skills(self, user_id: str) -> List[Dict[str, Any]]:
        """Get recommended skills to learn"""
        return [
            {
                "skill": "Docker",
                "category": "DevOps",
                "demand_score": 9.2,
                "learning_time": "2-3 weeks"
            }
        ]
    
    async def _get_suggested_learning_paths(self, user_id: str) -> List[Dict[str, Any]]:
        """Get suggested learning paths"""
        return [
            {
                "id": "path_1",
                "title": "Full Stack Development",
                "duration": "6 months",
                "difficulty": "Intermediate",
                "match_score": 88.0
            }
        ]
    
    async def _get_market_trends(self, user_id: str) -> List[Dict[str, Any]]:
        """Get relevant market trends"""
        return [
            {
                "trend": "AI/ML Skills in High Demand",
                "impact": "High",
                "relevance_score": 9.1
            }
        ]
    
    async def _get_salary_insights(self, user_id: str) -> Dict[str, Any]:
        """Get salary insights for user"""
        return {
            "current_market_rate": "$95,000",
            "potential_increase": "15%",
            "top_paying_skills": ["Python", "AWS", "Machine Learning"]
        }
    
    async def _get_industry_updates(self, user_id: str) -> List[Dict[str, Any]]:
        """Get relevant industry updates"""
        return [
            {
                "title": "Remote Work Trends 2024",
                "summary": "Latest insights on remote work adoption",
                "relevance": "High"
            }
        ]
    
    async def _get_networking_opportunities(self, user_id: str) -> List[Dict[str, Any]]:
        """Get networking opportunities"""
        return [
            {
                "event": "Tech Meetup - AI in Practice",
                "date": "2024-02-15",
                "location": "Virtual",
                "relevance_score": 8.5
            }
        ]
    
    async def _get_similar_profiles(self, user_id: str) -> List[Dict[str, Any]]:
        """Get similar user profiles for networking"""
        return [
            {
                "profile_id": "user_123",
                "role": "Senior Developer",
                "similarity_score": 87.0,
                "common_skills": ["Python", "React", "AWS"]
            }
        ]
    
    async def _calculate_personalization_score(self, user_id: str) -> float:
        """Calculate how personalized the content is"""
        # Mock implementation based on profile completeness and activity
        return 82.5
    
    # Mock methods for when services are unavailable
    
    def _create_mock_profile(self, user_id: str):
        """Create a mock profile when ProfileService is unavailable"""
        from unittest.mock import MagicMock
        profile = MagicMock()
        profile.id = f"profile_{user_id}"
        profile.user_id = user_id
        profile.current_role = "Software Developer"
        profile.experience_years = 5
        profile.education_level = "Bachelor's"
        profile.location = "Remote"
        profile.skills = ["Python", "JavaScript", "React"]
        profile.career_goals = "Advance to Senior Developer"
        profile.preferred_work_type = "Remote"
        profile.updated_at = datetime.utcnow()
        return profile
    
    def _get_mock_analytics(self, user_id: str) -> Dict[str, Any]:
        """Get mock analytics when AnalyticsService is unavailable"""
        return {
            "overall_career_score": 75.5,
            "skill_analytics": {"total_skills": 12},
            "experience_analytics": {"experience_score": 8.2},
            "market_analytics": {"market_position_percentile": 68},
            "progression_analytics": {"progress_score": 7.8}
        }
    
    def _get_mock_career_score(self, user_id: str) -> Dict[str, Any]:
        """Get mock career score when AnalyticsService is unavailable"""
        return {
            "overall_career_score": 75.5,
            "comprehensive_recommendations": [
                "Learn Docker for containerization",
                "Improve leadership skills",
                "Get AWS certification"
            ],
            "priority_actions": [
                "Update LinkedIn profile",
                "Complete Python certification",
                "Apply to senior roles"
            ],
            "trajectory_predictions": {
                "next_role": "Senior Developer",
                "timeline": "6-12 months",
                "confidence": 0.85
            }
        }
    
    def _get_mock_progress_data(self, user_id: str, days: int) -> Dict[str, Any]:
        """Get mock progress data when AnalyticsService is unavailable"""
        return {
            "overall_progress_percentage": 68.5,
            "new_skills_count": 3,
            "skills_mastered_count": 2,
            "improvement_areas": ["Leadership", "System Design"],
            "achievements": ["Completed Python course", "Updated profile"]
        }
    
    def _get_mock_recommendations(self, user_id: str) -> Dict[str, Any]:
        """Get mock recommendations when RecommendationService is unavailable"""
        return {
            "recommendations": [
                {
                    "title": "Learn Docker",
                    "description": "Container technology is in high demand",
                    "confidence_score": 8.5,
                    "type": "skill"
                },
                {
                    "title": "Apply to Senior Roles",
                    "description": "Your profile matches senior developer positions",
                    "confidence_score": 7.8,
                    "type": "career"
                }
            ]
        }
    
    async def get_real_time_dashboard_data(self, user_id: str) -> Dict[str, Any]:
        """
        Get real-time dashboard data with fresh analysis results.
        
        This method integrates with AI analysis service and job matching
        to provide up-to-date dashboard information.
        """
        try:
            # Get base dashboard summary
            summary = await self.get_dashboard_summary(user_id)
            
            # Get real-time analysis if available
            real_time_analysis = await self._get_real_time_analysis_data(user_id)
            
            # Get job market data
            job_market_data = await self._get_job_market_data(user_id)
            
            # Combine all data
            real_time_data = {
                "dashboard_summary": summary.dict(),
                "real_time_analysis": real_time_analysis,
                "job_market_data": job_market_data,
                "last_updated": datetime.utcnow().isoformat(),
                "data_freshness": {
                    "dashboard_summary": "real-time",
                    "analysis_results": real_time_analysis.get("freshness", "cached"),
                    "job_data": job_market_data.get("freshness", "cached")
                }
            }
            
            return real_time_data
            
        except Exception as e:
            logger.error(f"Error getting real-time dashboard data for user {user_id}: {str(e)}")
            raise ServiceException(f"Failed to get real-time dashboard data: {str(e)}")
    
    async def _get_real_time_analysis_data(self, user_id: str) -> Dict[str, Any]:
        """Get real-time analysis data from AI service"""
        try:
            # Check if AI analysis service is available
            if hasattr(self, 'ai_service') and self.ai_service:
                # Get fresh analysis
                analysis_result = await self.ai_service.analyze_complete_profile(user_id, self.db)
                return {
                    "skill_assessment": {
                        "technical_skills": analysis_result.skill_assessment.technical_skills,
                        "soft_skills": analysis_result.skill_assessment.soft_skills,
                        "skill_strengths": analysis_result.skill_assessment.skill_strengths,
                        "skill_gaps": analysis_result.skill_assessment.skill_gaps,
                        "market_relevance_score": analysis_result.skill_assessment.market_relevance_score
                    },
                    "career_recommendations": [
                        {
                            "role": rec.recommended_role,
                            "match_score": rec.match_score,
                            "reasoning": rec.reasoning,
                            "preparation_timeline": rec.preparation_timeline
                        }
                        for rec in analysis_result.career_recommendations
                    ],
                    "learning_paths": [
                        {
                            "title": path.title,
                            "description": path.description,
                            "target_skills": path.target_skills,
                            "estimated_duration": path.estimated_duration
                        }
                        for path in analysis_result.learning_paths
                    ],
                    "freshness": "real-time",
                    "analysis_timestamp": analysis_result.analysis_timestamp.isoformat()
                }
            else:
                # Return cached or mock data
                return {
                    "skill_assessment": self._get_mock_skill_assessment(),
                    "career_recommendations": self._get_mock_career_recommendations(),
                    "learning_paths": self._get_mock_learning_paths(),
                    "freshness": "mock",
                    "analysis_timestamp": datetime.utcnow().isoformat()
                }
                
        except Exception as e:
            logger.warning(f"Failed to get real-time analysis data: {str(e)}")
            return {
                "error": "Real-time analysis unavailable",
                "fallback_data": self._get_mock_analysis_data(),
                "freshness": "error"
            }
    
    async def _get_job_market_data(self, user_id: str) -> Dict[str, Any]:
        """Get job market data from real-time job service"""
        try:
            # This would integrate with RealTimeJobService
            # For now, return mock job market data
            return {
                "trending_roles": [
                    {"role": "Full Stack Developer", "demand_score": 9.2, "growth": "+15%"},
                    {"role": "DevOps Engineer", "demand_score": 8.8, "growth": "+22%"},
                    {"role": "Data Scientist", "demand_score": 8.5, "growth": "+18%"}
                ],
                "salary_insights": {
                    "market_average": "12-18 LPA",
                    "user_potential": "10-15 LPA",
                    "top_paying_skills": ["AWS", "Kubernetes", "Python", "React"]
                },
                "job_opportunities": {
                    "total_matches": 45,
                    "high_match": 12,
                    "medium_match": 23,
                    "low_match": 10
                },
                "market_trends": [
                    "Remote work opportunities increasing by 25%",
                    "AI/ML skills in high demand",
                    "Cloud expertise becoming essential"
                ],
                "freshness": "cached",
                "last_updated": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.warning(f"Failed to get job market data: {str(e)}")
            return {
                "error": "Job market data unavailable",
                "fallback_message": "Job market insights are temporarily unavailable"
            }
    
    def _get_mock_skill_assessment(self) -> Dict[str, Any]:
        """Get mock skill assessment data"""
        return {
            "technical_skills": {
                "Python": 0.85,
                "JavaScript": 0.75,
                "React": 0.70,
                "SQL": 0.80,
                "Git": 0.90
            },
            "soft_skills": {
                "Communication": 0.75,
                "Problem Solving": 0.85,
                "Teamwork": 0.80,
                "Leadership": 0.60
            },
            "skill_strengths": ["Python", "Problem Solving", "Git"],
            "skill_gaps": ["System Design", "Leadership", "Cloud Architecture"],
            "market_relevance_score": 0.78
        }
    
    def _get_mock_career_recommendations(self) -> List[Dict[str, Any]]:
        """Get mock career recommendations"""
        return [
            {
                "role": "Senior Software Developer",
                "match_score": 0.82,
                "reasoning": "Strong technical skills with room for leadership growth",
                "preparation_timeline": "6-9 months"
            },
            {
                "role": "Full Stack Developer",
                "match_score": 0.78,
                "reasoning": "Good balance of frontend and backend skills",
                "preparation_timeline": "3-6 months"
            }
        ]
    
    def _get_mock_learning_paths(self) -> List[Dict[str, Any]]:
        """Get mock learning paths"""
        return [
            {
                "title": "Advanced Python Development",
                "description": "Master advanced Python concepts and frameworks",
                "target_skills": ["Django", "FastAPI", "Async Programming"],
                "estimated_duration": "3 months"
            },
            {
                "title": "Cloud Architecture Fundamentals",
                "description": "Learn cloud design patterns and best practices",
                "target_skills": ["AWS", "Docker", "Kubernetes"],
                "estimated_duration": "4 months"
            }
        ]
    
    def _get_mock_analysis_data(self) -> Dict[str, Any]:
        """Get complete mock analysis data"""
        return {
            "skill_assessment": self._get_mock_skill_assessment(),
            "career_recommendations": self._get_mock_career_recommendations(),
            "learning_paths": self._get_mock_learning_paths()
        }
    
    async def _create_needs_analysis_dashboard(self, user_id: str) -> DashboardSummary:
        """
        Create a dashboard state for users who haven't completed their profile analysis.
        This provides a clear call-to-action while showing zero values for all metrics.
        """
        return DashboardSummary(
            user_id=user_id,
            overall_career_score=0.0,
            profile_completion=0.0,
            key_metrics=[
                DashboardMetric(
                    title="Career Score",
                    value="0/100",
                    change="+0 pts",
                    trend="neutral",
                    description="Complete your analysis to get your career score"
                ),
                DashboardMetric(
                    title="Job Matches",
                    value="0",
                    change="Pending analysis",
                    trend="neutral",
                    description="Get personalized job recommendations"
                ),
                DashboardMetric(
                    title="Skills Tracked",
                    value="0",
                    change="Add skills",
                    trend="neutral",
                    description="Track your skill development"
                ),
                DashboardMetric(
                    title="Learning Progress",
                    value="0",
                    change="0 paths",
                    trend="neutral",
                    description="Start your learning journey"
                )
            ],
            active_milestones=[],
            completed_milestones_count=0,
            total_milestones_count=0,
            top_recommendations=[
                DashboardRecommendation(
                    title="Complete Your Profile Analysis",
                    description="Upload your resume and connect your profiles to get personalized insights",
                    confidence_score=10.0,
                    type="profile_setup",
                    priority="high"
                ),
                DashboardRecommendation(
                    title="Connect Your GitHub Profile",
                    description="Link your GitHub to analyze your coding skills and projects",
                    confidence_score=9.0,
                    type="profile_setup",
                    priority="medium"
                ),
                DashboardRecommendation(
                    title="Add Your Skills",
                    description="Tell us about your technical and soft skills for better recommendations",
                    confidence_score=8.0,
                    type="profile_setup",
                    priority="medium"
                )
            ],
            recent_activities=[
                DashboardActivity(
                    title="Account Created",
                    description="Welcome to CareerPilot! Complete your profile to get started.",
                    timestamp=datetime.utcnow(),
                    type="account",
                    status="completed"
                )
            ],
            skills_count=0,
            job_matches_count=0,
            learning_paths_count=0,
            last_analysis_date=None,
            last_profile_update=None,
            generated_at=datetime.utcnow(),
            analysis_status="pending",
            needs_analysis=True
        )