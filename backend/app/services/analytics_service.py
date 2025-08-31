"""
Analytics service for generating comprehensive career analytics and reports
"""
import json
import uuid
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
import asyncio
import logging
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, or_

from ..models.user import User
from ..models.profile import UserProfile
from ..models.skill import Skill, UserSkill
from ..models.job import JobPosting, JobSkill
from ..schemas.analytics import (
    SkillRadarChart, CareerRoadmapVisualization, CareerRoadmapNode, CareerRoadmapEdge,
    SkillGapAnalysis, SkillGapReport, JobCompatibilityScore, JobCompatibilityReport,
    ProgressTrackingEntry, HistoricalProgressReport, CareerAnalysisReport,
    AnalyticsRequest, ChartConfiguration, VisualizationResponse, ChartType
)
from ..core.exceptions import AnalyticsError, DataNotFoundError

logger = logging.getLogger(__name__)


class AnalyticsService:
    """Service for generating comprehensive career analytics and visualizations"""
    
    def __init__(self, db: AsyncSession):
        self.db = db
        self.chart_cache = {}
        self.report_cache = {}
    
    async def generate_skill_radar_chart(self, user_id: str, target_role: Optional[str] = None) -> SkillRadarChart:
        """Generate skill radar chart with professional visualization data"""
        try:
            # Get user profile and skills
            user_profile = await self._get_user_profile(user_id)
            user_skills = await self._get_user_skills(user_id)
            
            if not user_skills:
                raise DataNotFoundError("No skills found for user")
            
            # Define skill categories for radar chart
            categories = [
                "Programming Languages", "Frameworks & Libraries", "Databases", 
                "Cloud & DevOps", "Soft Skills", "Tools & Technologies"
            ]
            
            # Calculate user scores for each category
            user_scores = []
            market_averages = []
            target_scores = []
            
            for category in categories:
                category_skills = await self._get_skills_by_category(category.lower().replace(" & ", "_").replace(" ", "_"))
                user_category_score = await self._calculate_category_score(user_skills, category_skills)
                market_avg = await self._get_market_average_score(category_skills)
                
                user_scores.append(user_category_score)
                market_averages.append(market_avg)
                
                # Calculate target scores if target role is provided
                if target_role:
                    target_score = await self._calculate_target_score(category_skills, target_role)
                    target_scores.append(target_score)
            
            return SkillRadarChart(
                user_id=user_id,
                categories=categories,
                user_scores=user_scores,
                market_average=market_averages,
                target_scores=target_scores if target_role else None,
                max_score=100.0
            )
            
        except Exception as e:
            logger.error(f"Error generating skill radar chart for user {user_id}: {str(e)}")
            raise AnalyticsError(f"Failed to generate skill radar chart: {str(e)}")
    
    async def generate_career_roadmap(self, user_id: str, target_role: str) -> CareerRoadmapVisualization:
        """Generate interactive career roadmap visualization"""
        try:
            user_profile = await self._get_user_profile(user_id)
            current_role = user_profile.current_role or "Current Position"
            
            # Create roadmap nodes
            nodes = []
            edges = []
            
            # Current position node
            current_node = CareerRoadmapNode(
                id="current",
                title=current_role,
                description="Your current position",
                position={"x": 0, "y": 0},
                node_type="current",
                timeline_months=0,
                completion_status="completed"
            )
            nodes.append(current_node)
            
            # Generate milestone nodes based on skill gaps
            skill_gaps = await self.analyze_skill_gaps(user_id, target_role)
            milestones = await self._generate_career_milestones(skill_gaps.skill_gaps, target_role)
            
            for i, milestone in enumerate(milestones):
                milestone_node = CareerRoadmapNode(
                    id=f"milestone_{i}",
                    title=milestone["title"],
                    description=milestone["description"],
                    position={"x": (i + 1) * 200, "y": 0},
                    node_type="milestone",
                    timeline_months=milestone["timeline_months"],
                    required_skills=milestone["required_skills"],
                    completion_status="not_started"
                )
                nodes.append(milestone_node)
                
                # Create edge from previous node
                prev_node_id = "current" if i == 0 else f"milestone_{i-1}"
                edge = CareerRoadmapEdge(
                    id=f"edge_{i}",
                    source_id=prev_node_id,
                    target_id=f"milestone_{i}",
                    edge_type="direct",
                    difficulty=milestone["difficulty"],
                    estimated_duration_months=milestone["timeline_months"],
                    required_actions=milestone["required_actions"]
                )
                edges.append(edge)
            
            # Target role node
            target_node = CareerRoadmapNode(
                id="target",
                title=target_role,
                description="Your target career goal",
                position={"x": (len(milestones) + 1) * 200, "y": 0},
                node_type="target",
                timeline_months=sum(m["timeline_months"] for m in milestones),
                completion_status="not_started"
            )
            nodes.append(target_node)
            
            # Final edge to target
            final_edge = CareerRoadmapEdge(
                id="final_edge",
                source_id=f"milestone_{len(milestones)-1}" if milestones else "current",
                target_id="target",
                edge_type="direct",
                difficulty=0.7,
                estimated_duration_months=6
            )
            edges.append(final_edge)
            
            # Add alternative paths
            alternative_roles = await self._get_alternative_career_paths(user_id, target_role)
            for i, alt_role in enumerate(alternative_roles[:2]):  # Limit to 2 alternatives
                alt_node = CareerRoadmapNode(
                    id=f"alternative_{i}",
                    title=alt_role["title"],
                    description=alt_role["description"],
                    position={"x": len(milestones) * 100, "y": (i + 1) * 150},
                    node_type="alternative",
                    timeline_months=alt_role["timeline_months"]
                )
                nodes.append(alt_node)
                
                # Edge from appropriate milestone
                alt_edge = CareerRoadmapEdge(
                    id=f"alt_edge_{i}",
                    source_id="current",
                    target_id=f"alternative_{i}",
                    edge_type="alternative",
                    difficulty=alt_role["difficulty"]
                )
                edges.append(alt_edge)
            
            return CareerRoadmapVisualization(
                user_id=user_id,
                nodes=nodes,
                edges=edges,
                metadata={
                    "target_role": target_role,
                    "total_timeline_months": sum(m["timeline_months"] for m in milestones),
                    "difficulty_level": "intermediate"
                }
            )
            
        except Exception as e:
            logger.error(f"Error generating career roadmap for user {user_id}: {str(e)}")
            raise AnalyticsError(f"Failed to generate career roadmap: {str(e)}")
    
    async def analyze_skill_gaps(self, user_id: str, target_role: str) -> SkillGapReport:
        """Analyze skill gaps with progress tracking capabilities"""
        try:
            user_skills = await self._get_user_skills(user_id)
            target_skills = await self._get_target_role_skills(target_role)
            
            skill_gaps = []
            strengths = []
            total_learning_hours = 0
            priority_skills = []
            
            # Analyze each target skill
            for target_skill in target_skills:
                skill_name = target_skill["name"]
                target_level = target_skill["required_level"]
                
                # Find user's current level for this skill
                current_level = 0.0
                for user_skill in user_skills:
                    if user_skill.skill.name.lower() == skill_name.lower():
                        current_level = user_skill.confidence_score * 100
                        break
                
                if current_level >= target_level * 0.8:  # 80% threshold for strength
                    strengths.append(skill_name)
                else:
                    gap_size = target_level - current_level
                    priority = self._calculate_skill_priority(gap_size, target_skill.get("importance", "medium"))
                    learning_hours = self._estimate_learning_hours(gap_size, skill_name)
                    
                    gap_analysis = SkillGapAnalysis(
                        skill_name=skill_name,
                        current_level=current_level,
                        target_level=target_level,
                        gap_size=gap_size,
                        priority=priority,
                        estimated_learning_hours=learning_hours,
                        recommended_resources=await self._get_learning_resources(skill_name),
                        market_demand=target_skill.get("market_demand", 0.5),
                        salary_impact=target_skill.get("salary_impact", 0.0)
                    )
                    skill_gaps.append(gap_analysis)
                    total_learning_hours += learning_hours
                    
                    if priority == "high":
                        priority_skills.append(skill_name)
            
            # Calculate overall match score
            total_skills = len(target_skills)
            matched_skills = len(strengths)
            overall_match_score = (matched_skills / total_skills) * 100 if total_skills > 0 else 0
            
            return SkillGapReport(
                user_id=user_id,
                target_role=target_role,
                overall_match_score=overall_match_score,
                skill_gaps=skill_gaps,
                strengths=strengths,
                total_learning_hours=total_learning_hours,
                priority_skills=priority_skills
            )
            
        except Exception as e:
            logger.error(f"Error analyzing skill gaps for user {user_id}: {str(e)}")
            raise AnalyticsError(f"Failed to analyze skill gaps: {str(e)}")
    
    async def generate_job_compatibility_scores(
        self, 
        user_id: str, 
        job_filters: Optional[Dict[str, Any]] = None,
        limit: int = 20
    ) -> JobCompatibilityReport:
        """Generate job compatibility scoring with requirement overlay"""
        try:
            user_skills = await self._get_user_skills(user_id)
            user_profile = await self._get_user_profile(user_id)
            
            # Get relevant job postings
            jobs = await self._get_filtered_jobs(job_filters, limit)
            
            job_matches = []
            for job in jobs:
                compatibility_score = await self._calculate_job_compatibility(
                    user_skills, user_profile, job
                )
                job_matches.append(compatibility_score)
            
            # Sort by overall score
            job_matches.sort(key=lambda x: x.overall_score, reverse=True)
            
            return JobCompatibilityReport(
                user_id=user_id,
                job_matches=job_matches,
                filters_applied=job_filters or {},
                total_jobs_analyzed=len(jobs)
            )
            
        except Exception as e:
            logger.error(f"Error generating job compatibility scores for user {user_id}: {str(e)}")
            raise AnalyticsError(f"Failed to generate job compatibility scores: {str(e)}")
    
    async def track_historical_progress(
        self, 
        user_id: str, 
        tracking_period_days: int = 90
    ) -> HistoricalProgressReport:
        """Track historical progress and improvement trends"""
        try:
            # Get historical skill data
            cutoff_date = datetime.utcnow() - timedelta(days=tracking_period_days)
            
            # This would typically query historical skill snapshots
            # For now, we'll simulate progress tracking
            skill_improvements = await self._calculate_skill_improvements(user_id, tracking_period_days)
            
            overall_improvement = sum(entry.improvement for entry in skill_improvements) / len(skill_improvements) if skill_improvements else 0
            
            milestones_achieved = await self._get_achieved_milestones(user_id, cutoff_date)
            
            trend_analysis = await self._analyze_progress_trends(skill_improvements)
            
            return HistoricalProgressReport(
                user_id=user_id,
                tracking_period_days=tracking_period_days,
                skill_improvements=skill_improvements,
                overall_improvement_score=overall_improvement,
                milestones_achieved=milestones_achieved,
                trend_analysis=trend_analysis
            )
            
        except Exception as e:
            logger.error(f"Error tracking historical progress for user {user_id}: {str(e)}")
            raise AnalyticsError(f"Failed to track historical progress: {str(e)}")
    
    async def generate_comprehensive_report(self, request: AnalyticsRequest) -> CareerAnalysisReport:
        """Generate comprehensive career analysis report"""
        try:
            # Generate all components
            tasks = []
            
            if "skill_radar" in request.analysis_types or "full_report" in request.analysis_types:
                tasks.append(self.generate_skill_radar_chart(request.user_id, request.target_role))
            
            if "career_roadmap" in request.analysis_types or "full_report" in request.analysis_types:
                if request.target_role:
                    tasks.append(self.generate_career_roadmap(request.user_id, request.target_role))
            
            if "skill_gaps" in request.analysis_types or "full_report" in request.analysis_types:
                if request.target_role:
                    tasks.append(self.analyze_skill_gaps(request.user_id, request.target_role))
            
            if "job_compatibility" in request.analysis_types or "full_report" in request.analysis_types:
                if request.include_job_matches:
                    tasks.append(self.generate_job_compatibility_scores(
                        request.user_id, request.job_search_filters
                    ))
            
            if "progress_tracking" in request.analysis_types or "full_report" in request.analysis_types:
                if request.include_progress_tracking:
                    tasks.append(self.track_historical_progress(
                        request.user_id, request.tracking_period_days
                    ))
            
            # Execute all tasks concurrently
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Extract results
            skill_radar_chart = None
            career_roadmap = None
            skill_gap_report = None
            job_compatibility_report = None
            progress_report = None
            
            result_index = 0
            if "skill_radar" in request.analysis_types or "full_report" in request.analysis_types:
                skill_radar_chart = results[result_index] if not isinstance(results[result_index], Exception) else None
                result_index += 1
            
            if ("career_roadmap" in request.analysis_types or "full_report" in request.analysis_types) and request.target_role:
                career_roadmap = results[result_index] if not isinstance(results[result_index], Exception) else None
                result_index += 1
            
            if ("skill_gaps" in request.analysis_types or "full_report" in request.analysis_types) and request.target_role:
                skill_gap_report = results[result_index] if not isinstance(results[result_index], Exception) else None
                result_index += 1
            
            if ("job_compatibility" in request.analysis_types or "full_report" in request.analysis_types) and request.include_job_matches:
                job_compatibility_report = results[result_index] if not isinstance(results[result_index], Exception) else None
                result_index += 1
            
            if ("progress_tracking" in request.analysis_types or "full_report" in request.analysis_types) and request.include_progress_tracking:
                progress_report = results[result_index] if not isinstance(results[result_index], Exception) else None
            
            # Get profile summary
            profile_summary = await self._get_profile_summary(request.user_id)
            
            # Generate recommendations and next steps
            recommendations = await self._generate_recommendations(
                skill_gap_report, job_compatibility_report, progress_report
            )
            next_steps = await self._generate_next_steps(skill_gap_report, career_roadmap)
            
            return CareerAnalysisReport(
                user_id=request.user_id,
                profile_summary=profile_summary,
                skill_radar_chart=skill_radar_chart,
                career_roadmap=career_roadmap,
                skill_gap_report=skill_gap_report,
                job_compatibility_report=job_compatibility_report,
                progress_report=progress_report,
                recommendations=recommendations,
                next_steps=next_steps
            )
            
        except Exception as e:
            logger.error(f"Error generating comprehensive report for user {request.user_id}: {str(e)}")
            raise AnalyticsError(f"Failed to generate comprehensive report: {str(e)}")
    
    # Helper methods
    async def _get_user_profile(self, user_id: str) -> UserProfile:
        """Get user profile"""
        result = await self.db.execute(
            select(UserProfile).where(UserProfile.user_id == user_id)
        )
        profile = result.scalar_one_or_none()
        if not profile:
            raise DataNotFoundError(f"Profile not found for user {user_id}")
        return profile
    
    async def _get_user_skills(self, user_id: str) -> List[UserSkill]:
        """Get user skills with skill details"""
        result = await self.db.execute(
            select(UserSkill).join(Skill).where(UserSkill.user_id == user_id)
        )
        return result.scalars().all()
    
    async def _get_skills_by_category(self, category: str) -> List[Skill]:
        """Get skills by category"""
        result = await self.db.execute(
            select(Skill).where(Skill.category == category)
        )
        return result.scalars().all()
    
    async def _calculate_category_score(self, user_skills: List[UserSkill], category_skills: List[Skill]) -> float:
        """Calculate user's score for a skill category"""
        if not category_skills:
            return 0.0
        
        category_skill_ids = {skill.id for skill in category_skills}
        user_category_skills = [
            skill for skill in user_skills 
            if skill.skill_id in category_skill_ids
        ]
        
        if not user_category_skills:
            return 0.0
        
        total_score = sum(skill.confidence_score * 100 for skill in user_category_skills)
        return total_score / len(user_category_skills)
    
    async def _get_market_average_score(self, category_skills: List[Skill]) -> float:
        """Get market average score for skill category"""
        # This would typically query market data
        # For now, return a simulated average
        return 65.0  # Simulated market average
    
    async def _calculate_target_score(self, category_skills: List[Skill], target_role: str) -> float:
        """Calculate target score for role"""
        # This would analyze job requirements for the target role
        # For now, return a simulated target score
        return 80.0  # Simulated target score
    
    async def _generate_career_milestones(self, skill_gaps: List[SkillGapAnalysis], target_role: str) -> List[Dict[str, Any]]:
        """Generate career milestones based on skill gaps"""
        milestones = []
        
        # Group skills by priority and create milestones
        high_priority_skills = [gap.skill_name for gap in skill_gaps if gap.priority == "high"]
        medium_priority_skills = [gap.skill_name for gap in skill_gaps if gap.priority == "medium"]
        
        if high_priority_skills:
            milestones.append({
                "title": "Core Skills Development",
                "description": f"Master essential skills: {', '.join(high_priority_skills[:3])}",
                "timeline_months": 6,
                "required_skills": high_priority_skills[:3],
                "difficulty": 0.8,
                "required_actions": [
                    "Complete online courses",
                    "Build practice projects",
                    "Seek mentorship"
                ]
            })
        
        if medium_priority_skills:
            milestones.append({
                "title": "Advanced Skills Enhancement",
                "description": f"Develop advanced capabilities: {', '.join(medium_priority_skills[:3])}",
                "timeline_months": 4,
                "required_skills": medium_priority_skills[:3],
                "difficulty": 0.6,
                "required_actions": [
                    "Take specialized courses",
                    "Contribute to open source",
                    "Network with professionals"
                ]
            })
        
        return milestones
    
    async def _get_alternative_career_paths(self, user_id: str, target_role: str) -> List[Dict[str, Any]]:
        """Get alternative career paths"""
        # This would analyze similar roles and career transitions
        # For now, return simulated alternatives
        return [
            {
                "title": "Senior Developer",
                "description": "Technical leadership role",
                "timeline_months": 12,
                "difficulty": 0.6
            },
            {
                "title": "Product Manager",
                "description": "Product strategy and management",
                "timeline_months": 18,
                "difficulty": 0.7
            }
        ]
    
    async def _get_target_role_skills(self, target_role: str) -> List[Dict[str, Any]]:
        """Get required skills for target role"""
        # This would query job market data for the role
        # For now, return simulated requirements
        return [
            {"name": "Python", "required_level": 85, "importance": "high", "market_demand": 0.9},
            {"name": "React", "required_level": 80, "importance": "high", "market_demand": 0.8},
            {"name": "AWS", "required_level": 70, "importance": "medium", "market_demand": 0.7},
            {"name": "Docker", "required_level": 65, "importance": "medium", "market_demand": 0.6}
        ]
    
    def _calculate_skill_priority(self, gap_size: float, importance: str) -> str:
        """Calculate skill priority based on gap size and importance"""
        if importance == "high" and gap_size > 50:
            return "high"
        elif importance == "high" or gap_size > 70:
            return "high"
        elif gap_size > 40:
            return "medium"
        else:
            return "low"
    
    def _estimate_learning_hours(self, gap_size: float, skill_name: str) -> int:
        """Estimate learning hours needed to close skill gap"""
        # Base hours per skill level point
        base_hours_per_point = 2
        
        # Skill complexity multiplier
        complexity_multipliers = {
            "python": 1.2, "javascript": 1.1, "react": 1.3, "aws": 1.5,
            "machine learning": 2.0, "data science": 1.8
        }
        
        multiplier = complexity_multipliers.get(skill_name.lower(), 1.0)
        return int(gap_size * base_hours_per_point * multiplier)
    
    async def _get_learning_resources(self, skill_name: str) -> List[Dict[str, Any]]:
        """Get recommended learning resources for skill"""
        # This would query learning resource database
        # For now, return simulated resources
        return [
            {
                "title": f"Complete {skill_name} Course",
                "type": "course",
                "provider": "Coursera",
                "rating": 4.5,
                "duration_hours": 40,
                "cost": 49.99
            },
            {
                "title": f"{skill_name} Documentation",
                "type": "documentation",
                "provider": "Official",
                "rating": 4.8,
                "duration_hours": 10,
                "cost": 0
            }
        ]
    
    async def _get_filtered_jobs(self, filters: Optional[Dict[str, Any]], limit: int) -> List[JobPosting]:
        """Get filtered job postings"""
        query = select(JobPosting).where(JobPosting.is_active == True)
        
        if filters:
            if "location" in filters:
                query = query.where(JobPosting.location.ilike(f"%{filters['location']}%"))
            if "experience_level" in filters:
                query = query.where(JobPosting.experience_level == filters["experience_level"])
            if "remote_type" in filters:
                query = query.where(JobPosting.remote_type == filters["remote_type"])
        
        query = query.limit(limit)
        result = await self.db.execute(query)
        return result.scalars().all()
    
    async def _calculate_job_compatibility(
        self, 
        user_skills: List[UserSkill], 
        user_profile: UserProfile, 
        job: JobPosting
    ) -> JobCompatibilityScore:
        """Calculate job compatibility score"""
        # Get job required skills
        result = await self.db.execute(
            select(JobSkill).join(Skill).where(JobSkill.job_posting_id == job.id)
        )
        job_skills = result.scalars().all()
        
        # Calculate skill match
        user_skill_names = {skill.skill.name.lower() for skill in user_skills}
        job_skill_names = {skill.skill.name.lower() for skill in job_skills}
        
        matched_skills = list(user_skill_names.intersection(job_skill_names))
        missing_skills = list(job_skill_names - user_skill_names)
        
        skill_match_score = (len(matched_skills) / len(job_skill_names)) * 100 if job_skill_names else 0
        
        # Calculate experience match
        required_experience = self._parse_experience_level(job.experience_level)
        user_experience = user_profile.experience_years or 0
        experience_match_score = min(100, (user_experience / required_experience) * 100) if required_experience > 0 else 100
        
        # Calculate overall score
        overall_score = (skill_match_score * 0.6 + experience_match_score * 0.4)
        
        # Determine recommendation
        if overall_score >= 80:
            recommendation = "apply"
        elif overall_score >= 60:
            recommendation = "consider"
        else:
            recommendation = "improve_first"
        
        return JobCompatibilityScore(
            job_id=job.id,
            job_title=job.title,
            company=job.company,
            overall_score=overall_score,
            skill_match_score=skill_match_score,
            experience_match_score=experience_match_score,
            matched_skills=matched_skills,
            missing_skills=missing_skills,
            recommendation=recommendation
        )
    
    def _parse_experience_level(self, experience_level: Optional[str]) -> int:
        """Parse experience level to years"""
        if not experience_level:
            return 0
        
        level_mapping = {
            "entry": 1,
            "mid": 3,
            "senior": 5,
            "lead": 7,
            "executive": 10
        }
        
        return level_mapping.get(experience_level.lower(), 0)
    
    async def _calculate_skill_improvements(self, user_id: str, tracking_period_days: int) -> List[ProgressTrackingEntry]:
        """Calculate skill improvements over tracking period"""
        # This would typically query historical skill snapshots
        # For now, simulate progress tracking with sample data
        current_skills = await self._get_user_skills(user_id)
        
        improvements = []
        for skill in current_skills[:5]:  # Limit to top 5 skills for demo
            # Simulate historical data
            current_score = skill.confidence_score * 100
            previous_score = max(0, current_score - (10 + (hash(skill.skill.name) % 20)))  # Simulate improvement
            improvement = current_score - previous_score
            
            if improvement > 0:
                improvements.append(ProgressTrackingEntry(
                    user_id=user_id,
                    skill_name=skill.skill.name,
                    previous_score=previous_score,
                    current_score=current_score,
                    improvement=improvement,
                    tracking_period_days=tracking_period_days,
                    evidence=f"Improved through practice and learning",
                    milestone_achieved=f"Reached {current_score:.0f}% proficiency" if current_score >= 80 else None
                ))
        
        return improvements
    
    async def _get_achieved_milestones(self, user_id: str, cutoff_date: datetime) -> List[str]:
        """Get milestones achieved since cutoff date"""
        # This would typically query milestone tracking data
        # For now, return simulated milestones
        return [
            "Completed Python Advanced Course",
            "Built first React application",
            "Contributed to open source project"
        ]
    
    async def _analyze_progress_trends(self, skill_improvements: List[ProgressTrackingEntry]) -> Dict[str, Any]:
        """Analyze progress trends from skill improvements"""
        if not skill_improvements:
            return {"trend": "stable", "velocity": 0.0}
        
        total_improvement = sum(entry.improvement for entry in skill_improvements)
        avg_improvement = total_improvement / len(skill_improvements)
        
        # Determine trend
        if avg_improvement > 10:
            trend = "accelerating"
        elif avg_improvement > 5:
            trend = "improving"
        elif avg_improvement > 0:
            trend = "stable"
        else:
            trend = "declining"
        
        return {
            "trend": trend,
            "velocity": avg_improvement,
            "total_skills_improved": len(skill_improvements),
            "best_performing_skill": max(skill_improvements, key=lambda x: x.improvement).skill_name if skill_improvements else None
        }
    
    async def _get_profile_summary(self, user_id: str) -> Dict[str, Any]:
        """Get profile summary for reports"""
        try:
            profile = await self._get_user_profile(user_id)
            user_skills = await self._get_user_skills(user_id)
            
            return {
                "name": "User",  # UserProfile doesn't have name field
                "current_role": profile.current_role or "Professional",
                "experience_years": profile.experience_years or 0,
                "location": profile.location or "Not specified",
                "dream_job": profile.dream_job or "Not specified",
                "skills": [skill.skill.name for skill in user_skills],
                "skill_count": len(user_skills),
                "profile_completeness": self._calculate_profile_completeness(profile)
            }
        except Exception as e:
            logger.error(f"Error getting profile summary: {str(e)}")
            return {
                "name": "User",
                "current_role": "Professional",
                "experience_years": 0,
                "location": "Not specified",
                "dream_job": "Not specified",
                "skills": [],
                "skill_count": 0,
                "profile_completeness": 0.0
            }
    
    def _calculate_profile_completeness(self, profile: UserProfile) -> float:
        """Calculate profile completeness percentage"""
        fields = [
            profile.current_role,
            profile.location,
            profile.dream_job,
            profile.experience_years,
            profile.github_username,
            profile.linkedin_url
        ]
        
        completed_fields = sum(1 for field in fields if field)
        return (completed_fields / len(fields)) * 100
    
    async def _generate_recommendations(
        self, 
        skill_gap_report: Optional[SkillGapReport],
        job_compatibility_report: Optional[JobCompatibilityReport],
        progress_report: Optional[HistoricalProgressReport]
    ) -> List[str]:
        """Generate personalized recommendations"""
        recommendations = []
        
        if skill_gap_report:
            # Skill-based recommendations
            high_priority_skills = [gap.skill_name for gap in skill_gap_report.skill_gaps if gap.priority == "high"]
            if high_priority_skills:
                recommendations.append(f"Focus on developing high-priority skills: {', '.join(high_priority_skills[:3])}")
            
            if skill_gap_report.total_learning_hours > 0:
                recommendations.append(f"Dedicate approximately {skill_gap_report.total_learning_hours} hours to close skill gaps")
        
        if job_compatibility_report:
            # Job market recommendations
            top_matches = [job for job in job_compatibility_report.job_matches if job.overall_score >= 80]
            if top_matches:
                recommendations.append(f"Consider applying to {len(top_matches)} high-compatibility positions")
            
            improve_jobs = [job for job in job_compatibility_report.job_matches if job.recommendation == "improve_first"]
            if improve_jobs:
                common_missing_skills = self._find_common_missing_skills(improve_jobs)
                if common_missing_skills:
                    recommendations.append(f"Develop {', '.join(common_missing_skills[:2])} to unlock more opportunities")
        
        if progress_report:
            # Progress-based recommendations
            if progress_report.overall_improvement_score > 10:
                recommendations.append("Excellent progress! Continue your current learning approach")
            elif progress_report.overall_improvement_score > 0:
                recommendations.append("Good progress. Consider increasing learning intensity for faster growth")
            else:
                recommendations.append("Focus on consistent skill development to accelerate career growth")
        
        return recommendations
    
    async def _generate_next_steps(
        self, 
        skill_gap_report: Optional[SkillGapReport],
        career_roadmap: Optional[CareerRoadmapVisualization]
    ) -> List[str]:
        """Generate actionable next steps"""
        next_steps = []
        
        if skill_gap_report:
            # Immediate skill development steps
            priority_skills = skill_gap_report.priority_skills[:2]  # Top 2 priority skills
            for skill in priority_skills:
                next_steps.append(f"Start learning {skill} through online courses or tutorials")
            
            if skill_gap_report.skill_gaps:
                # Find skills with learning resources
                skills_with_resources = [gap for gap in skill_gap_report.skill_gaps if gap.recommended_resources]
                if skills_with_resources:
                    skill = skills_with_resources[0]
                    next_steps.append(f"Enroll in recommended course for {skill.skill_name}")
        
        if career_roadmap:
            # Career progression steps
            milestone_nodes = [node for node in career_roadmap.nodes if node.node_type == "milestone"]
            if milestone_nodes:
                first_milestone = milestone_nodes[0]
                next_steps.append(f"Work towards: {first_milestone.title}")
                
                if first_milestone.required_skills:
                    next_steps.append(f"Focus on developing: {', '.join(first_milestone.required_skills[:2])}")
        
        # General next steps
        next_steps.extend([
            "Update your LinkedIn profile with new skills",
            "Build a portfolio project showcasing your abilities",
            "Network with professionals in your target field"
        ])
        
        return next_steps[:5]  # Limit to 5 next steps
    
    def _find_common_missing_skills(self, jobs: List[JobCompatibilityScore]) -> List[str]:
        """Find commonly missing skills across job matches"""
        skill_counts = {}
        
        for job in jobs:
            for skill in job.missing_skills:
                skill_counts[skill] = skill_counts.get(skill, 0) + 1
        
        # Sort by frequency and return top skills
        sorted_skills = sorted(skill_counts.items(), key=lambda x: x[1], reverse=True)
        return [skill for skill, count in sorted_skills if count >= 2]  # Skills missing in 2+ jobs