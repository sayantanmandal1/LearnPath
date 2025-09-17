"""
Recommendation service that integrates ML recommendation engine with backend data.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import json
import asyncio
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
import numpy as np

from app.core.database import get_db
from app.models.user import User
from app.models.profile import UserProfile
from app.models.job import JobPosting, JobSkill
from app.models.skill import Skill, UserSkill
from app.repositories.profile import ProfileRepository
from app.repositories.job import JobRepository
from app.repositories.skill import SkillRepository
from app.core.exceptions import ServiceException

# Import ML components
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'machinelearningmodel'))

try:
    from machinelearningmodel.recommendation_engine import (
        HybridRecommendationEngine,
        CareerRecommendation,
        LearningPath,
        SkillGapAnalysis,
        SkillGapAnalyzer,
        RecommendationExplainer
    )
except ImportError:
    # Fallback classes for when ML module is not available
    class HybridRecommendationEngine:
        def __init__(self):
            pass
        def fit(self, *args, **kwargs):
            pass
        def recommend_careers(self, *args, **kwargs):
            return []
        def recommend_learning_paths(self, *args, **kwargs):
            return []
    
    class CareerRecommendation:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
    
    class LearningPath:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
    
    class SkillGapAnalysis:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
    
    class SkillGapAnalyzer:
        def __init__(self):
            pass
        def analyze_skill_gaps(self, *args, **kwargs):
            return SkillGapAnalysis(
                target_role="",
                missing_skills={},
                weak_skills={},
                strong_skills=[],
                overall_readiness=0.5,
                learning_time_estimate=4,
                priority_skills=[]
            )
    
    class RecommendationExplainer:
        def __init__(self):
            pass


logger = logging.getLogger(__name__)


class RecommendationService:
    """Service for generating career and learning path recommendations."""
    
    def __init__(self):
        self.recommendation_engine = HybridRecommendationEngine()
        self.skill_gap_analyzer = SkillGapAnalyzer()
        self.explainer = RecommendationExplainer()
        self.model_trained = False
        self.last_training_time = None
        self.training_interval = timedelta(hours=24)  # Retrain every 24 hours
        self.training_interval = timedelta(hours=24)  # Retrain daily
        
    async def initialize_and_train_models(self, db: AsyncSession):
        """Initialize and train recommendation models with current data."""
        try:
            logger.info("Initializing recommendation models...")
            
            # Check if retraining is needed
            if (self.model_trained and self.last_training_time and 
                datetime.utcnow() - self.last_training_time < self.training_interval):
                logger.info("Models are up to date, skipping training")
                return
            
            # Fetch training data
            user_data = await self._fetch_user_data(db)
            job_data = await self._fetch_job_data(db)
            user_item_matrix, user_ids = await self._build_user_item_matrix(db)
            
            if len(user_data) == 0 or len(job_data) == 0:
                logger.warning("Insufficient data for training recommendation models")
                return
            
            # Train the hybrid recommendation engine
            await asyncio.get_event_loop().run_in_executor(
                None, 
                self.recommendation_engine.fit,
                user_item_matrix, user_ids, job_data, user_data
            )
            
            self.model_trained = True
            self.last_training_time = datetime.utcnow()
            logger.info("Recommendation models trained successfully")
            
        except Exception as e:
            logger.error(f"Error training recommendation models: {str(e)}")
            raise ServiceException(f"Failed to initialize recommendation models: {str(e)}")
    
    async def get_career_recommendations(self, user_id: str, db: AsyncSession, 
                                       n_recommendations: int = 5) -> List[Dict[str, Any]]:
        """
        Get career recommendations for a user.
        
        Args:
            user_id: User identifier
            db: Database session
            n_recommendations: Number of recommendations to return
            
        Returns:
            List of career recommendations with detailed analysis
        """
        try:
            # Ensure models are trained
            await self.initialize_and_train_models(db)
            
            if not self.model_trained:
                raise ServiceException("Recommendation models not available")
            
            # Fetch user profile
            profile_repo = ProfileRepository(db)
            user_profile = await profile_repo.get_by_user_id(user_id)
            
            if not user_profile:
                raise ServiceException("User profile not found")
            
            # Get user skills
            user_skills = user_profile.skills or {}
            
            # Prepare user profile data for ML engine
            user_profile_data = {
                'id': user_id,
                'current_role': user_profile.current_role or '',
                'dream_job': user_profile.dream_job or '',
                'skills': list(user_skills.keys()),
                'experience_years': user_profile.experience_years or 0
            }
            
            # Generate recommendations using ML engine
            ml_recommendations = await asyncio.get_event_loop().run_in_executor(
                None,
                self.recommendation_engine.recommend_careers,
                user_id, user_profile_data, n_recommendations
            )
            
            # Enrich recommendations with database data
            enriched_recommendations = []
            job_repo = JobRepository(db)
            
            for ml_rec in ml_recommendations:
                # Fetch actual job data (in practice, ml_rec would contain job_id)
                # For now, we'll create synthetic recommendations based on user profile
                job_recommendation = await self._create_job_recommendation(
                    user_profile, user_skills, ml_rec, db
                )
                enriched_recommendations.append(job_recommendation)
            
            return enriched_recommendations
            
        except Exception as e:
            logger.error(f"Error generating career recommendations for user {user_id}: {str(e)}")
            raise ServiceException(f"Failed to generate career recommendations: {str(e)}")
    
    async def get_learning_path_recommendations(self, user_id: str, target_role: str, 
                                              db: AsyncSession) -> List[Dict[str, Any]]:
        """
        Get learning path recommendations for a user targeting a specific role.
        
        Args:
            user_id: User identifier
            target_role: Target job role
            db: Database session
            
        Returns:
            List of learning path recommendations
        """
        try:
            # Fetch user profile and skills
            profile_repo = ProfileRepository(db)
            user_profile = await profile_repo.get_by_user_id(user_id)
            
            if not user_profile:
                raise ServiceException("User profile not found")
            
            user_skills = user_profile.skills or {}
            
            # Get target role skill requirements
            target_skills = await self._get_target_role_skills(target_role, db)
            
            if not target_skills:
                # Use default skills for common roles
                target_skills = self._get_default_role_skills(target_role)
            
            # Generate learning paths using ML engine
            learning_paths = await asyncio.get_event_loop().run_in_executor(
                None,
                self.recommendation_engine.recommend_learning_paths,
                user_skills, target_role, target_skills
            )
            
            # Enrich learning paths with actual resources
            enriched_paths = []
            for path in learning_paths:
                enriched_path = await self._enrich_learning_path(path, db)
                enriched_paths.append(enriched_path)
            
            return enriched_paths
            
        except Exception as e:
            logger.error(f"Error generating learning paths for user {user_id}: {str(e)}")
            raise ServiceException(f"Failed to generate learning path recommendations: {str(e)}")
    
    async def analyze_skill_gaps(self, user_id: str, target_role: str, 
                               db: AsyncSession) -> Dict[str, Any]:
        """
        Analyze skill gaps between user profile and target role.
        
        Args:
            user_id: User identifier
            target_role: Target job role
            db: Database session
            
        Returns:
            Detailed skill gap analysis
        """
        try:
            # Fetch user profile
            profile_repo = ProfileRepository(db)
            user_profile = await profile_repo.get_by_user_id(user_id)
            
            if not user_profile:
                raise ServiceException("User profile not found")
            
            user_skills = user_profile.skills or {}
            
            # Get target role requirements
            target_skills = await self._get_target_role_skills(target_role, db)
            if not target_skills:
                target_skills = self._get_default_role_skills(target_role)
            
            # Perform skill gap analysis
            gap_analysis = await asyncio.get_event_loop().run_in_executor(
                None,
                self.skill_gap_analyzer.analyze_skill_gaps,
                user_skills, target_skills, target_role
            )
            
            # Convert to dictionary format
            return {
                'target_role': gap_analysis.target_role,
                'missing_skills': gap_analysis.missing_skills,
                'weak_skills': gap_analysis.weak_skills,
                'strong_skills': gap_analysis.strong_skills,
                'overall_readiness': gap_analysis.overall_readiness,
                'learning_time_estimate_weeks': gap_analysis.learning_time_estimate,
                'priority_skills': gap_analysis.priority_skills,
                'readiness_percentage': round(gap_analysis.overall_readiness * 100, 1),
                'analysis_date': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error analyzing skill gaps for user {user_id}: {str(e)}")
            raise ServiceException(f"Failed to analyze skill gaps: {str(e)}")
    
    async def get_job_match_score(self, user_id: str, job_id: str, 
                                db: AsyncSession) -> Dict[str, Any]:
        """
        Calculate match score between user and specific job.
        
        Args:
            user_id: User identifier
            job_id: Job posting identifier
            db: Database session
            
        Returns:
            Job match analysis with score and details
        """
        try:
            # Fetch user profile and job posting
            profile_repo = ProfileRepository(db)
            job_repo = JobRepository(db)
            
            user_profile = await profile_repo.get_by_user_id(user_id)
            job_posting = await job_repo.get_by_id(job_id)
            
            if not user_profile or not job_posting:
                raise ServiceException("User profile or job posting not found")
            
            user_skills = user_profile.skills or {}
            job_skills = job_posting.processed_skills or {}
            
            # Calculate match score using content-based filtering
            if self.model_trained:
                match_score = await asyncio.get_event_loop().run_in_executor(
                    None,
                    self.recommendation_engine.content_based_engine.calculate_job_similarity,
                    user_id, job_id
                )
            else:
                # Fallback to simple skill overlap calculation
                match_score = self._calculate_simple_match_score(user_skills, job_skills)
            
            # Perform detailed analysis
            gap_analysis = await asyncio.get_event_loop().run_in_executor(
                None,
                self.skill_gap_analyzer.analyze_skill_gaps,
                user_skills, job_skills, job_posting.title
            )
            
            return {
                'job_id': job_id,
                'job_title': job_posting.title,
                'company': job_posting.company,
                'match_score': round(match_score, 3),
                'match_percentage': round(match_score * 100, 1),
                'skill_gaps': gap_analysis.missing_skills,
                'weak_skills': gap_analysis.weak_skills,
                'strong_skills': gap_analysis.strong_skills,
                'overall_readiness': gap_analysis.overall_readiness,
                'readiness_percentage': round(gap_analysis.overall_readiness * 100, 1),
                'analysis_date': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error calculating job match score: {str(e)}")
            raise ServiceException(f"Failed to calculate job match score: {str(e)}")
    
    async def _fetch_user_data(self, db: AsyncSession) -> List[Dict[str, Any]]:
        """Fetch user data for training."""
        try:
            query = select(UserProfile).where(UserProfile.skills.isnot(None))
            result = await db.execute(query)
            profiles = result.scalars().all()
            
            user_data = []
            for profile in profiles:
                user_data.append({
                    'id': profile.user_id,
                    'current_role': profile.current_role or '',
                    'dream_job': profile.dream_job or '',
                    'skills': list((profile.skills or {}).keys()),
                    'experience_years': profile.experience_years or 0
                })
            
            return user_data
            
        except Exception as e:
            logger.error(f"Error fetching user data: {str(e)}")
            return []
    
    async def _fetch_job_data(self, db: AsyncSession) -> List[Dict[str, Any]]:
        """Fetch job data for training."""
        try:
            query = select(JobPosting).where(
                JobPosting.is_active == True,
                JobPosting.processed_skills.isnot(None)
            ).limit(1000)  # Limit for performance
            
            result = await db.execute(query)
            jobs = result.scalars().all()
            
            job_data = []
            for job in jobs:
                job_data.append({
                    'id': job.id,
                    'title': job.title,
                    'description': job.description,
                    'skills': list((job.processed_skills or {}).keys()),
                    'company': job.company,
                    'location': job.location,
                    'experience_level': job.experience_level
                })
            
            return job_data
            
        except Exception as e:
            logger.error(f"Error fetching job data: {str(e)}")
            return []
    
    async def _build_user_item_matrix(self, db: AsyncSession) -> Tuple[np.ndarray, List[str]]:
        """Build user-item interaction matrix for collaborative filtering."""
        try:
            # For now, create a synthetic matrix based on user skills and job requirements
            # In practice, this would be based on actual user interactions (applications, views, etc.)
            
            user_data = await self._fetch_user_data(db)
            job_data = await self._fetch_job_data(db)
            
            if not user_data or not job_data:
                return np.array([]), []
            
            user_ids = [user['id'] for user in user_data]
            job_ids = [job['id'] for job in job_data]
            
            # Create synthetic interaction matrix based on skill overlap
            matrix = np.zeros((len(user_ids), len(job_ids)))
            
            for i, user in enumerate(user_data):
                user_skills = set(user['skills'])
                for j, job in enumerate(job_data):
                    job_skills = set(job['skills'])
                    if user_skills and job_skills:
                        overlap = len(user_skills.intersection(job_skills))
                        total = len(user_skills.union(job_skills))
                        similarity = overlap / total if total > 0 else 0
                        # Convert to rating scale (1-5)
                        matrix[i, j] = similarity * 4 + 1
            
            return matrix, user_ids
            
        except Exception as e:
            logger.error(f"Error building user-item matrix: {str(e)}")
            return np.array([]), []
    
    async def _create_job_recommendation(self, user_profile: UserProfile, 
                                       user_skills: Dict[str, float],
                                       ml_rec: CareerRecommendation, 
                                       db: AsyncSession) -> Dict[str, Any]:
        """Create enriched job recommendation."""
        # In practice, this would fetch actual job data
        # For now, create synthetic recommendations based on user profile
        
        dream_job = user_profile.dream_job or "Software Engineer"
        experience_level = "Senior" if (user_profile.experience_years or 0) > 5 else "Mid-level"
        
        return {
            'job_title': f"{experience_level} {dream_job}",
            'company': "Tech Company",  # Placeholder
            'match_score': ml_rec.match_score,
            'match_percentage': round(ml_rec.match_score * 100, 1),
            'required_skills': ml_rec.required_skills,
            'skill_gaps': ml_rec.skill_gaps,
            'salary_range': {
                'min': ml_rec.salary_range[0],
                'max': ml_rec.salary_range[1],
                'currency': 'USD'
            },
            'growth_potential': ml_rec.growth_potential,
            'market_demand': ml_rec.market_demand,
            'confidence_score': ml_rec.confidence_score,
            'reasoning': ml_rec.reasoning,
            'alternative_paths': ml_rec.alternative_paths,
            'location': user_profile.location or "Remote",
            'employment_type': "Full-time",
            'recommendation_date': datetime.utcnow().isoformat()
        }
    
    async def _get_target_role_skills(self, target_role: str, db: AsyncSession) -> Dict[str, float]:
        """Get skill requirements for target role from job postings."""
        try:
            # Query job postings for similar roles
            query = select(JobPosting).where(
                JobPosting.title.ilike(f"%{target_role}%"),
                JobPosting.processed_skills.isnot(None),
                JobPosting.is_active == True
            ).limit(50)
            
            result = await db.execute(query)
            jobs = result.scalars().all()
            
            if not jobs:
                return {}
            
            # Aggregate skill requirements
            skill_counts = {}
            total_jobs = len(jobs)
            
            for job in jobs:
                job_skills = job.processed_skills or {}
                for skill, importance in job_skills.items():
                    if skill not in skill_counts:
                        skill_counts[skill] = 0
                    skill_counts[skill] += importance
            
            # Normalize by frequency
            target_skills = {}
            for skill, total_importance in skill_counts.items():
                frequency = total_importance / total_jobs
                target_skills[skill] = min(frequency, 1.0)  # Cap at 1.0
            
            return target_skills
            
        except Exception as e:
            logger.error(f"Error fetching target role skills: {str(e)}")
            return {}
    
    def _get_default_role_skills(self, target_role: str) -> Dict[str, float]:
        """Get default skill requirements for common roles."""
        role_skills = {
            'software engineer': {
                'python': 0.8, 'javascript': 0.7, 'sql': 0.6, 'git': 0.9,
                'algorithms': 0.7, 'system design': 0.6, 'testing': 0.5
            },
            'data scientist': {
                'python': 0.9, 'sql': 0.8, 'machine learning': 0.9, 'statistics': 0.8,
                'pandas': 0.7, 'numpy': 0.7, 'scikit-learn': 0.6, 'jupyter': 0.5
            },
            'frontend developer': {
                'javascript': 0.9, 'html': 0.8, 'css': 0.8, 'react': 0.7,
                'typescript': 0.6, 'webpack': 0.5, 'responsive design': 0.7
            },
            'backend developer': {
                'python': 0.8, 'java': 0.7, 'sql': 0.8, 'api design': 0.7,
                'microservices': 0.6, 'docker': 0.6, 'databases': 0.8
            },
            'devops engineer': {
                'docker': 0.9, 'kubernetes': 0.8, 'aws': 0.7, 'linux': 0.8,
                'terraform': 0.6, 'jenkins': 0.6, 'monitoring': 0.7
            }
        }
        
        role_lower = target_role.lower()
        for role, skills in role_skills.items():
            if role in role_lower:
                return skills
        
        # Default skills for unknown roles
        return {
            'communication': 0.8, 'problem solving': 0.8, 'teamwork': 0.7,
            'project management': 0.6, 'analytical thinking': 0.7
        }
    
    async def _enrich_learning_path(self, path: LearningPath, db: AsyncSession) -> Dict[str, Any]:
        """Enrich learning path with actual resources."""
        # In practice, this would fetch actual learning resources from database
        # For now, create synthetic resources
        
        resources = [
            {
                'title': f"Complete {path.target_skills[0]} Course",
                'type': 'course',
                'provider': 'Coursera',
                'url': f"https://coursera.org/{path.target_skills[0].lower()}",
                'rating': 4.5,
                'duration_hours': 40,
                'cost': 49.99,
                'prerequisites': []
            },
            {
                'title': f"{path.target_skills[0]} Documentation",
                'type': 'documentation',
                'provider': 'Official',
                'url': f"https://docs.{path.target_skills[0].lower()}.org",
                'rating': 4.8,
                'duration_hours': 10,
                'cost': 0,
                'prerequisites': []
            }
        ]
        
        milestones = [
            {
                'title': f"Complete {path.target_skills[0]} Basics",
                'description': f"Learn fundamental concepts of {path.target_skills[0]}",
                'estimated_weeks': 2,
                'completion_criteria': ['Complete course modules 1-3', 'Pass quiz with 80%+']
            },
            {
                'title': f"Build {path.target_skills[0]} Project",
                'description': f"Create a practical project using {path.target_skills[0]}",
                'estimated_weeks': 3,
                'completion_criteria': ['Deploy working application', 'Document code and process']
            }
        ]
        
        return {
            'path_id': path.path_id,
            'title': path.title,
            'target_skills': path.target_skills,
            'estimated_duration_weeks': path.estimated_duration_weeks,
            'difficulty_level': path.difficulty_level,
            'priority_score': path.priority_score,
            'reasoning': path.reasoning,
            'resources': resources,
            'milestones': milestones,
            'created_date': datetime.utcnow().isoformat()
        }
    
    def _calculate_simple_match_score(self, user_skills: Dict[str, float], 
                                    job_skills: Dict[str, float]) -> float:
        """Calculate simple match score based on skill overlap."""
        if not user_skills or not job_skills:
            return 0.0
        
        user_skill_set = set(user_skills.keys())
        job_skill_set = set(job_skills.keys())
        
        intersection = user_skill_set.intersection(job_skill_set)
        union = user_skill_set.union(job_skill_set)
        
        if not union:
            return 0.0
        
        # Jaccard similarity with confidence weighting
        overlap_score = 0.0
        for skill in intersection:
            user_confidence = user_skills.get(skill, 0)
            job_importance = job_skills.get(skill, 0)
            overlap_score += min(user_confidence, job_importance)
        
        total_importance = sum(job_skills.values())
        if total_importance > 0:
            return overlap_score / total_importance
        
        return len(intersection) / len(union)    

    async def get_filtered_career_recommendations(self, user_id: str, db: AsyncSession,
                                                target_role: Optional[str] = None,
                                                filters: Dict[str, Any] = None,
                                                n_recommendations: int = 10,
                                                include_alternatives: bool = True) -> List[Dict[str, Any]]:
        """
        Get career recommendations with advanced filtering.
        
        Args:
            user_id: User identifier
            db: Database session
            target_role: Optional target role filter
            filters: Dictionary of filters (location, experience_level, salary_min, etc.)
            n_recommendations: Number of recommendations
            include_alternatives: Include alternative career paths
            
        Returns:
            Filtered list of career recommendations
        """
        try:
            # Get base recommendations
            recommendations = await self.get_career_recommendations(
                user_id, db, n_recommendations * 2  # Get more to filter from
            )
            
            if not filters:
                return recommendations[:n_recommendations]
            
            # Apply filters
            filtered_recommendations = []
            
            for rec in recommendations:
                # Location filter
                if filters.get('location') and filters['location'].lower() not in rec.get('location', '').lower():
                    continue
                
                # Experience level filter
                if filters.get('experience_level') and filters['experience_level'].lower() not in rec.get('job_title', '').lower():
                    continue
                
                # Salary filters
                salary_range = rec.get('salary_range', {})
                if filters.get('salary_min') and salary_range.get('min', 0) < filters['salary_min']:
                    continue
                if filters.get('salary_max') and salary_range.get('max', float('inf')) > filters['salary_max']:
                    continue
                
                # Skills filter
                if filters.get('skills'):
                    required_skills = set(rec.get('required_skills', []))
                    filter_skills = set(filters['skills'])
                    if not filter_skills.intersection(required_skills):
                        continue
                
                filtered_recommendations.append(rec)
                
                if len(filtered_recommendations) >= n_recommendations:
                    break
            
            return filtered_recommendations
            
        except Exception as e:
            logger.error(f"Error getting filtered career recommendations: {e}")
            raise ServiceException(f"Failed to get filtered recommendations: {str(e)}")
    
    async def get_advanced_job_matches(self, user_profile: UserProfile,
                                     filters: Dict[str, Any] = None,
                                     match_threshold: float = 0.6,
                                     include_skill_gaps: bool = True,
                                     db: AsyncSession = None) -> List[Dict[str, Any]]:
        """
        Get advanced job matches with filtering and analysis.
        
        Args:
            user_profile: User profile object
            filters: Dictionary of filters
            match_threshold: Minimum match score
            include_skill_gaps: Include skill gap analysis
            db: Database session
            
        Returns:
            List of job matches with analysis
        """
        try:
            job_repo = JobRepository()
            
            # Build query with filters
            query_filters = {}
            if filters:
                if filters.get('location'):
                    query_filters['location'] = filters['location']
                if filters.get('experience_level'):
                    query_filters['experience_level'] = filters['experience_level']
                if filters.get('remote_type'):
                    query_filters['remote_type'] = filters['remote_type']
                if filters.get('min_salary'):
                    query_filters['min_salary'] = filters['min_salary']
                if filters.get('max_salary'):
                    query_filters['max_salary'] = filters['max_salary']
            
            # Get job postings
            jobs = await job_repo.search_jobs(
                db=db,
                **query_filters,
                limit=100  # Get more jobs to analyze
            )
            
            job_matches = []
            user_skills = user_profile.skills or {}
            
            for job in jobs:
                # Calculate match score
                job_skills = job.processed_skills or {}
                match_score = self._calculate_simple_match_score(user_skills, job_skills)
                
                if match_score < match_threshold:
                    continue
                
                match_data = {
                    'job_id': job.id,
                    'job_title': job.title,
                    'company': job.company,
                    'location': job.location or 'Not specified',
                    'remote_type': job.remote_type,
                    'employment_type': job.employment_type,
                    'experience_level': job.experience_level,
                    'match_score': round(match_score, 3),
                    'match_percentage': round(match_score * 100, 1),
                    'salary_min': job.salary_min,
                    'salary_max': job.salary_max,
                    'salary_currency': job.salary_currency or 'USD',
                    'posted_date': job.posted_date.isoformat() if job.posted_date else None,
                    'source_url': job.source_url,
                    'description': job.description[:500] + '...' if len(job.description) > 500 else job.description,
                    'required_skills': list(job_skills.keys()) if job_skills else []
                }
                
                # Add skill gap analysis if requested
                if include_skill_gaps:
                    gap_analysis = await asyncio.get_event_loop().run_in_executor(
                        None,
                        self.skill_gap_analyzer.analyze_skill_gaps,
                        user_skills, job_skills, job.title
                    )
                    
                    match_data.update({
                        'skill_gaps': gap_analysis.missing_skills,
                        'weak_skills': gap_analysis.weak_skills,
                        'strong_skills': gap_analysis.strong_skills,
                        'overall_readiness': gap_analysis.overall_readiness,
                        'readiness_percentage': round(gap_analysis.overall_readiness * 100, 1),
                        'priority_skills': gap_analysis.priority_skills
                    })
                
                job_matches.append(match_data)
            
            # Sort by match score
            job_matches.sort(key=lambda x: x['match_score'], reverse=True)
            
            return job_matches
            
        except Exception as e:
            logger.error(f"Error getting advanced job matches: {e}")
            raise ServiceException(f"Failed to get job matches: {str(e)}")
    
    async def get_job_recommendations_with_ml(self, user_id: str, db: AsyncSession,
                                            filters: Dict[str, Any] = None,
                                            n_recommendations: int = 10) -> List[Dict[str, Any]]:
        """
        Get job recommendations using ML algorithms with enhanced matching.
        
        Args:
            user_id: User identifier
            db: Database session
            filters: Optional filters for job search
            n_recommendations: Number of recommendations to return
            
        Returns:
            List of job recommendations with detailed analysis
        """
        try:
            # Ensure models are trained
            await self.initialize_and_train_models(db)
            
            # Get user profile
            profile_repo = ProfileRepository()
            user_profile = await profile_repo.get_by_user_id(db, user_id)
            
            if not user_profile:
                raise ServiceException("User profile not found")
            
            # Get job matches using advanced matching
            job_matches = await self.get_advanced_job_matches(
                user_profile=user_profile,
                filters=filters,
                match_threshold=0.3,  # Lower threshold to get more candidates
                include_skill_gaps=True,
                db=db
            )
            
            # If we have ML models trained, enhance with collaborative filtering
            if self.model_trained:
                try:
                    # Get collaborative filtering recommendations
                    ml_recommendations = await asyncio.get_event_loop().run_in_executor(
                        None,
                        self.recommendation_engine.recommend_careers,
                        user_id, {
                            'id': user_id,
                            'current_role': user_profile.current_role or '',
                            'dream_job': user_profile.dream_job or '',
                            'skills': list((user_profile.skills or {}).keys()),
                            'experience_years': user_profile.experience_years or 0
                        }, n_recommendations * 2
                    )
                    
                    # Combine content-based and collaborative filtering results
                    job_matches = self._combine_recommendation_results(job_matches, ml_recommendations)
                    
                except Exception as ml_error:
                    logger.warning(f"ML recommendation failed, using content-based only: {ml_error}")
            
            # Apply additional ranking based on user preferences
            job_matches = self._rank_job_recommendations(job_matches, user_profile)
            
            # Return top N recommendations
            return job_matches[:n_recommendations]
            
        except Exception as e:
            logger.error(f"Error getting job recommendations with ML: {e}")
            raise ServiceException(f"Failed to get job recommendations: {str(e)}")
    
    def _combine_recommendation_results(self, content_based: List[Dict], 
                                      ml_based: List) -> List[Dict]:
        """Combine content-based and ML-based recommendations."""
        # For now, prioritize content-based results since they have actual job data
        # In a full implementation, this would intelligently merge both approaches
        return content_based
    
    def _rank_job_recommendations(self, job_matches: List[Dict], 
                                user_profile: UserProfile) -> List[Dict]:
        """Apply additional ranking based on user preferences."""
        try:
            # Add preference-based scoring
            for match in job_matches:
                preference_score = 0.0
                
                # Boost score for dream job matches
                if user_profile.dream_job:
                    dream_job_lower = user_profile.dream_job.lower()
                    job_title_lower = match['job_title'].lower()
                    if dream_job_lower in job_title_lower or job_title_lower in dream_job_lower:
                        preference_score += 0.2
                
                # Boost score for location preferences
                if user_profile.location and match.get('location'):
                    if user_profile.location.lower() in match['location'].lower():
                        preference_score += 0.1
                
                # Boost score for remote work preference
                if user_profile.remote_work_preference and match.get('remote_type') == 'remote':
                    preference_score += 0.15
                
                # Apply preference boost to match score
                original_score = match['match_score']
                boosted_score = min(1.0, original_score + preference_score)
                match['match_score'] = round(boosted_score, 3)
                match['match_percentage'] = round(boosted_score * 100, 1)
                match['preference_boost'] = round(preference_score, 3)
            
            # Re-sort by updated match scores
            job_matches.sort(key=lambda x: x['match_score'], reverse=True)
            
            return job_matches
            
        except Exception as e:
            logger.warning(f"Error ranking job recommendations: {e}")
            return job_matches