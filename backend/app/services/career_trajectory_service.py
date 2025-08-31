"""
Career trajectory recommendation service for personalized career path planning.

This service implements:
1. Career path matching using semantic similarity between profiles and jobs
2. Career progression modeling based on historical data
3. Dream job optimization and path planning algorithms
4. Alternative career route discovery and lateral movement suggestions
5. Market demand integration for career recommendations
6. Confidence scores and detailed reasoning for each recommendation
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import asyncio
import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, or_
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer

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
ml_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'machinelearningmodel')
if ml_path not in sys.path:
    sys.path.append(ml_path)

try:
    from recommendation_engine import (
        ContentBasedFilteringEngine,
        SkillGapAnalyzer,
        RecommendationExplainer
    )
except ImportError:
    # Fallback to creating minimal implementations
    import logging
    logging.getLogger(__name__).warning("ML recommendation engine not available, using fallback implementations")
    
    class ContentBasedFilteringEngine:
        def __init__(self):
            pass
        
        def calculate_job_similarity(self, user_id: str, job_id: str) -> float:
            return 0.5  # Default similarity
    
    class SkillGapAnalyzer:
        def __init__(self):
            pass
        
        def analyze_skill_gaps(self, user_skills, target_skills, target_role):
            from types import SimpleNamespace
            return SimpleNamespace(
                target_role=target_role,
                missing_skills={},
                weak_skills={},
                strong_skills=list(user_skills.keys()),
                overall_readiness=0.7,
                learning_time_estimate=16,
                priority_skills=list(target_skills.keys())[:3] if target_skills else []
            )
    
    class RecommendationExplainer:
        def __init__(self):
            pass


logger = logging.getLogger(__name__)


@dataclass
class CareerTrajectoryRecommendation:
    """Career trajectory recommendation with detailed path analysis."""
    trajectory_id: str
    title: str
    target_role: str
    match_score: float
    confidence_score: float
    
    # Path details
    progression_steps: List[Dict[str, Any]]
    estimated_timeline_months: int
    difficulty_level: str
    
    # Skills and requirements
    required_skills: List[str]
    skill_gaps: Dict[str, float]
    transferable_skills: List[str]
    
    # Market analysis
    market_demand: str
    salary_progression: Dict[str, Tuple[int, int]]
    growth_potential: float
    
    # Alternative paths
    alternative_routes: List[Dict[str, Any]]
    lateral_opportunities: List[str]
    
    # Reasoning and explanation
    reasoning: str
    success_factors: List[str]
    potential_challenges: List[str]
    
    # Metadata
    recommendation_date: datetime
    data_sources: List[str]


@dataclass
class CareerProgressionModel:
    """Career progression model for a specific career path."""
    career_path: str
    typical_progression: List[Dict[str, Any]]
    skill_evolution: Dict[str, List[str]]
    timeline_ranges: Dict[str, Tuple[int, int]]  # role -> (min_months, max_months)
    success_metrics: Dict[str, Any]
    market_trends: Dict[str, Any]


class CareerTrajectoryService:
    """Service for generating personalized career trajectory recommendations."""
    
    def __init__(self):
        self.embedding_model = None
        self.content_engine = ContentBasedFilteringEngine()
        self.skill_gap_analyzer = SkillGapAnalyzer()
        self.explainer = RecommendationExplainer()
        self.career_progression_models = {}
        self.market_demand_cache = {}
        self.cache_expiry = timedelta(hours=6)
        
    async def _load_embedding_model(self):
        """Load sentence transformer model for semantic similarity."""
        if self.embedding_model is None:
            logger.info("Loading embedding model for career trajectory analysis")
            self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    
    async def get_career_trajectory_recommendations(
        self, 
        user_id: str, 
        db: AsyncSession,
        n_recommendations: int = 5,
        include_alternatives: bool = True
    ) -> List[CareerTrajectoryRecommendation]:
        """
        Generate comprehensive career trajectory recommendations for a user.
        
        Args:
            user_id: User identifier
            db: Database session
            n_recommendations: Number of main trajectories to recommend
            include_alternatives: Whether to include alternative paths
            
        Returns:
            List of career trajectory recommendations
        """
        try:
            logger.info(f"Generating career trajectory recommendations for user {user_id}")
            
            # Load embedding model
            await self._load_embedding_model()
            
            # Fetch user profile and data
            profile_repo = ProfileRepository()
            user_profile = await profile_repo.get_by_user_id(db, user_id)
            
            if not user_profile:
                raise ServiceException("User profile not found")
            
            # Extract user data
            user_skills = user_profile.skills or {}
            dream_job = user_profile.dream_job
            current_role = user_profile.current_role
            experience_years = user_profile.experience_years or 0
            
            # Generate career trajectory recommendations
            trajectories = []
            
            # 1. Dream job optimization path
            if dream_job:
                dream_trajectory = await self._generate_dream_job_trajectory(
                    user_profile, user_skills, dream_job, db
                )
                if dream_trajectory:
                    trajectories.append(dream_trajectory)
            
            # 2. Natural progression paths from current role
            if current_role:
                progression_trajectories = await self._generate_progression_trajectories(
                    user_profile, user_skills, current_role, db, n_recommendations - 1
                )
                trajectories.extend(progression_trajectories)
            
            # 3. Lateral movement opportunities
            lateral_trajectories = await self._generate_lateral_trajectories(
                user_profile, user_skills, db, max(1, n_recommendations - len(trajectories))
            )
            trajectories.extend(lateral_trajectories)
            
            # 4. Market-driven opportunities
            if len(trajectories) < n_recommendations:
                market_trajectories = await self._generate_market_driven_trajectories(
                    user_profile, user_skills, db, n_recommendations - len(trajectories)
                )
                trajectories.extend(market_trajectories)
            
            # Sort by confidence score and return top N
            trajectories.sort(key=lambda x: x.confidence_score, reverse=True)
            final_trajectories = trajectories[:n_recommendations]
            
            # Add alternative routes if requested
            if include_alternatives:
                for trajectory in final_trajectories:
                    trajectory.alternative_routes = await self._find_alternative_routes(
                        trajectory, user_skills, db
                    )
            
            logger.info(f"Generated {len(final_trajectories)} career trajectory recommendations")
            return final_trajectories
            
        except Exception as e:
            logger.error(f"Error generating career trajectories for user {user_id}: {str(e)}")
            raise ServiceException(f"Failed to generate career trajectory recommendations: {str(e)}")
    
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
            profile_repo = ProfileRepository()
            user_profile = await profile_repo.get_by_user_id(db, user_id)
            
            if not user_profile:
                raise ServiceException("User profile not found")
            
            user_skills = user_profile.skills or {}
            
            # Get target role requirements
            target_skills = await self._get_role_skill_requirements(target_role, db)
            
            # Perform skill gap analysis
            gap_analysis = self.skill_gap_analyzer.analyze_skill_gaps(
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
            profile_repo = ProfileRepository()
            job_repo = JobRepository()
            
            user_profile = await profile_repo.get_by_user_id(db, user_id)
            job_posting = await job_repo.get_by_id(db, job_id)
            
            if not user_profile or not job_posting:
                raise ServiceException("User profile or job posting not found")
            
            user_skills = user_profile.skills or {}
            job_skills = job_posting.processed_skills or {}
            
            # Calculate match score using semantic similarity
            await self._load_embedding_model()
            
            user_text = self._create_user_text(user_profile)
            job_text = f"{job_posting.title} {job_posting.description} {' '.join(job_skills.keys())}"
            
            user_embedding = self.embedding_model.encode([user_text])
            job_embedding = self.embedding_model.encode([job_text])
            similarity = cosine_similarity(user_embedding, job_embedding)[0][0]
            
            # Perform detailed analysis
            gap_analysis = self.skill_gap_analyzer.analyze_skill_gaps(
                user_skills, job_skills, job_posting.title
            )
            
            return {
                'job_id': job_id,
                'job_title': job_posting.title,
                'company': job_posting.company,
                'match_score': round(similarity, 3),
                'match_percentage': round(similarity * 100, 1),
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

    async def _generate_dream_job_trajectory(
        self, 
        user_profile: UserProfile, 
        user_skills: Dict[str, float],
        dream_job: str, 
        db: AsyncSession
    ) -> Optional[CareerTrajectoryRecommendation]:
        """Generate trajectory towards user's dream job."""
        try:
            # Find jobs matching dream job description
            job_repo = JobRepository()
            matching_jobs = await job_repo.search_jobs(
                db, title=dream_job, limit=20
            )
            
            if not matching_jobs:
                # Create synthetic dream job requirements
                target_skills = self._get_default_role_skills(dream_job)
            else:
                # Aggregate skills from matching jobs
                target_skills = await self._aggregate_job_skills(matching_jobs, db)
            
            # Analyze skill gaps
            gap_analysis = self.skill_gap_analyzer.analyze_skill_gaps(
                user_skills, target_skills, dream_job
            )
            
            # Calculate semantic similarity
            user_text = self._create_user_text(user_profile)
            dream_job_text = f"{dream_job} {' '.join(target_skills.keys())}"
            
            user_embedding = self.embedding_model.encode([user_text])
            job_embedding = self.embedding_model.encode([dream_job_text])
            similarity = cosine_similarity(user_embedding, job_embedding)[0][0]
            
            # Generate progression steps
            progression_steps = await self._plan_progression_steps(
                user_profile.current_role or "Entry Level",
                dream_job,
                gap_analysis,
                db
            )
            
            # Calculate timeline
            timeline_months = self._estimate_trajectory_timeline(
                progression_steps, gap_analysis.learning_time_estimate
            )
            
            # Get market data
            market_data = await self._get_market_demand_data(dream_job, db)
            
            # Generate salary progression
            salary_progression = await self._calculate_salary_progression(
                progression_steps, db
            )
            
            # Create recommendation
            trajectory = CareerTrajectoryRecommendation(
                trajectory_id=f"dream_{user_profile.user_id}_{hash(dream_job)}",
                title=f"Path to {dream_job}",
                target_role=dream_job,
                match_score=similarity,
                confidence_score=self._calculate_confidence_score(
                    similarity, gap_analysis.overall_readiness, market_data
                ),
                progression_steps=progression_steps,
                estimated_timeline_months=timeline_months,
                difficulty_level=self._assess_difficulty_level(gap_analysis),
                required_skills=list(target_skills.keys()),
                skill_gaps=gap_analysis.missing_skills,
                transferable_skills=gap_analysis.strong_skills,
                market_demand=market_data.get('demand_level', 'moderate'),
                salary_progression=salary_progression,
                growth_potential=market_data.get('growth_potential', 0.7),
                alternative_routes=[],  # Will be populated later
                lateral_opportunities=[],
                reasoning=self._generate_trajectory_reasoning(
                    dream_job, gap_analysis, market_data, "dream_job"
                ),
                success_factors=self._identify_success_factors(gap_analysis, market_data),
                potential_challenges=self._identify_challenges(gap_analysis, market_data),
                recommendation_date=datetime.utcnow(),
                data_sources=["job_postings", "market_analysis", "skill_taxonomy"]
            )
            
            return trajectory
            
        except Exception as e:
            logger.error(f"Error generating dream job trajectory: {str(e)}")
            return None
    
    async def _generate_progression_trajectories(
        self,
        user_profile: UserProfile,
        user_skills: Dict[str, float],
        current_role: str,
        db: AsyncSession,
        n_trajectories: int = 3
    ) -> List[CareerTrajectoryRecommendation]:
        """Generate natural career progression trajectories."""
        trajectories = []
        
        try:
            # Get career progression model for current role
            progression_model = await self._get_career_progression_model(current_role, db)
            
            if not progression_model:
                return trajectories
            
            # Generate trajectories for each typical next step
            for next_step in progression_model.typical_progression[:n_trajectories]:
                target_role = next_step['role']
                
                # Get target role requirements
                target_skills = await self._get_role_skill_requirements(target_role, db)
                
                # Analyze skill gaps
                gap_analysis = self.skill_gap_analyzer.analyze_skill_gaps(
                    user_skills, target_skills, target_role
                )
                
                # Calculate semantic similarity
                user_text = self._create_user_text(user_profile)
                target_text = f"{target_role} {' '.join(target_skills.keys())}"
                
                user_embedding = self.embedding_model.encode([user_text])
                target_embedding = self.embedding_model.encode([target_text])
                similarity = cosine_similarity(user_embedding, target_embedding)[0][0]
                
                # Generate progression steps
                progression_steps = [
                    {
                        'role': current_role,
                        'duration_months': 0,
                        'description': 'Current position',
                        'key_activities': ['Strengthen current skills', 'Build foundation']
                    },
                    {
                        'role': target_role,
                        'duration_months': next_step.get('typical_duration_months', 24),
                        'description': f'Transition to {target_role}',
                        'key_activities': next_step.get('key_activities', [])
                    }
                ]
                
                # Get market data
                market_data = await self._get_market_demand_data(target_role, db)
                
                # Create trajectory
                trajectory = CareerTrajectoryRecommendation(
                    trajectory_id=f"progression_{user_profile.user_id}_{hash(target_role)}",
                    title=f"Natural Progression to {target_role}",
                    target_role=target_role,
                    match_score=similarity,
                    confidence_score=self._calculate_confidence_score(
                        similarity, gap_analysis.overall_readiness, market_data
                    ),
                    progression_steps=progression_steps,
                    estimated_timeline_months=next_step.get('typical_duration_months', 24),
                    difficulty_level=self._assess_difficulty_level(gap_analysis),
                    required_skills=list(target_skills.keys()),
                    skill_gaps=gap_analysis.missing_skills,
                    transferable_skills=gap_analysis.strong_skills,
                    market_demand=market_data.get('demand_level', 'moderate'),
                    salary_progression=await self._calculate_salary_progression(
                        progression_steps, db
                    ),
                    growth_potential=market_data.get('growth_potential', 0.7),
                    alternative_routes=[],
                    lateral_opportunities=[],
                    reasoning=self._generate_trajectory_reasoning(
                        target_role, gap_analysis, market_data, "natural_progression"
                    ),
                    success_factors=self._identify_success_factors(gap_analysis, market_data),
                    potential_challenges=self._identify_challenges(gap_analysis, market_data),
                    recommendation_date=datetime.utcnow(),
                    data_sources=["career_progression_models", "job_postings", "market_analysis"]
                )
                
                trajectories.append(trajectory)
                
        except Exception as e:
            logger.error(f"Error generating progression trajectories: {str(e)}")
        
        return trajectories
    
    async def _generate_lateral_trajectories(
        self,
        user_profile: UserProfile,
        user_skills: Dict[str, float],
        db: AsyncSession,
        n_trajectories: int = 2
    ) -> List[CareerTrajectoryRecommendation]:
        """Generate lateral movement opportunities."""
        trajectories = []
        
        try:
            # Find roles with similar skill requirements
            similar_roles = await self._find_similar_roles(user_skills, db, n_trajectories * 2)
            
            for role_data in similar_roles[:n_trajectories]:
                target_role = role_data['role']
                target_skills = role_data['skills']
                similarity_score = role_data['similarity']
                
                # Analyze skill gaps
                gap_analysis = self.skill_gap_analyzer.analyze_skill_gaps(
                    user_skills, target_skills, target_role
                )
                
                # Generate lateral movement steps
                progression_steps = [
                    {
                        'role': user_profile.current_role or 'Current Role',
                        'duration_months': 0,
                        'description': 'Current position',
                        'key_activities': ['Leverage transferable skills']
                    },
                    {
                        'role': f"Transition Period",
                        'duration_months': 6,
                        'description': 'Skill bridging and networking',
                        'key_activities': [
                            'Build missing technical skills',
                            'Network in target industry',
                            'Complete relevant projects'
                        ]
                    },
                    {
                        'role': target_role,
                        'duration_months': 12,
                        'description': f'Lateral move to {target_role}',
                        'key_activities': ['Apply transferable experience', 'Adapt to new domain']
                    }
                ]
                
                # Get market data
                market_data = await self._get_market_demand_data(target_role, db)
                
                # Create trajectory
                trajectory = CareerTrajectoryRecommendation(
                    trajectory_id=f"lateral_{user_profile.user_id}_{hash(target_role)}",
                    title=f"Lateral Move to {target_role}",
                    target_role=target_role,
                    match_score=similarity_score,
                    confidence_score=self._calculate_confidence_score(
                        similarity_score, gap_analysis.overall_readiness, market_data
                    ),
                    progression_steps=progression_steps,
                    estimated_timeline_months=18,  # Typical lateral move timeline
                    difficulty_level="moderate",  # Lateral moves are typically moderate difficulty
                    required_skills=list(target_skills.keys()),
                    skill_gaps=gap_analysis.missing_skills,
                    transferable_skills=gap_analysis.strong_skills,
                    market_demand=market_data.get('demand_level', 'moderate'),
                    salary_progression=await self._calculate_salary_progression(
                        progression_steps, db
                    ),
                    growth_potential=market_data.get('growth_potential', 0.6),
                    alternative_routes=[],
                    lateral_opportunities=[],
                    reasoning=self._generate_trajectory_reasoning(
                        target_role, gap_analysis, market_data, "lateral_move"
                    ),
                    success_factors=self._identify_success_factors(gap_analysis, market_data),
                    potential_challenges=self._identify_challenges(gap_analysis, market_data),
                    recommendation_date=datetime.utcnow(),
                    data_sources=["skill_similarity", "job_postings", "market_analysis"]
                )
                
                trajectories.append(trajectory)
                
        except Exception as e:
            logger.error(f"Error generating lateral trajectories: {str(e)}")
        
        return trajectories    

    async def _generate_market_driven_trajectories(
        self,
        user_profile: UserProfile,
        user_skills: Dict[str, float],
        db: AsyncSession,
        n_trajectories: int = 2
    ) -> List[CareerTrajectoryRecommendation]:
        """Generate trajectories based on high-demand market opportunities."""
        trajectories = []
        
        try:
            # Get high-demand roles
            high_demand_roles = await self._get_high_demand_roles(db, n_trajectories * 2)
            
            for role_data in high_demand_roles[:n_trajectories]:
                target_role = role_data['role']
                target_skills = role_data['skills']
                market_score = role_data['demand_score']
                
                # Analyze skill gaps
                gap_analysis = self.skill_gap_analyzer.analyze_skill_gaps(
                    user_skills, target_skills, target_role
                )
                
                # Calculate semantic similarity
                user_text = self._create_user_text(user_profile)
                target_text = f"{target_role} {' '.join(target_skills.keys())}"
                
                user_embedding = self.embedding_model.encode([user_text])
                target_embedding = self.embedding_model.encode([target_text])
                similarity = cosine_similarity(user_embedding, target_embedding)[0][0]
                
                # Generate market-driven progression steps
                progression_steps = await self._plan_market_driven_steps(
                    user_profile, target_role, gap_analysis, db
                )
                
                # Calculate timeline based on skill gaps
                timeline_months = max(12, gap_analysis.learning_time_estimate * 4)  # Convert weeks to months
                
                # Get detailed market data
                market_data = await self._get_market_demand_data(target_role, db)
                
                # Create trajectory
                trajectory = CareerTrajectoryRecommendation(
                    trajectory_id=f"market_{user_profile.user_id}_{hash(target_role)}",
                    title=f"Market Opportunity: {target_role}",
                    target_role=target_role,
                    match_score=similarity,
                    confidence_score=self._calculate_confidence_score(
                        similarity, gap_analysis.overall_readiness, market_data, market_weight=0.4
                    ),
                    progression_steps=progression_steps,
                    estimated_timeline_months=timeline_months,
                    difficulty_level=self._assess_difficulty_level(gap_analysis),
                    required_skills=list(target_skills.keys()),
                    skill_gaps=gap_analysis.missing_skills,
                    transferable_skills=gap_analysis.strong_skills,
                    market_demand=market_data.get('demand_level', 'high'),
                    salary_progression=await self._calculate_salary_progression(
                        progression_steps, db
                    ),
                    growth_potential=market_data.get('growth_potential', 0.8),
                    alternative_routes=[],
                    lateral_opportunities=[],
                    reasoning=self._generate_trajectory_reasoning(
                        target_role, gap_analysis, market_data, "market_driven"
                    ),
                    success_factors=self._identify_success_factors(gap_analysis, market_data),
                    potential_challenges=self._identify_challenges(gap_analysis, market_data),
                    recommendation_date=datetime.utcnow(),
                    data_sources=["market_trends", "job_demand_analysis", "skill_taxonomy"]
                )
                
                trajectories.append(trajectory)
                
        except Exception as e:
            logger.error(f"Error generating market-driven trajectories: {str(e)}")
        
        return trajectories
    
    async def _find_alternative_routes(
        self,
        main_trajectory: CareerTrajectoryRecommendation,
        user_skills: Dict[str, float],
        db: AsyncSession
    ) -> List[Dict[str, Any]]:
        """Find alternative routes to the same target role."""
        alternatives = []
        
        try:
            target_role = main_trajectory.target_role
            
            # Find different career paths to the same role
            alternative_paths = await self._discover_alternative_paths(target_role, db)
            
            for path in alternative_paths[:3]:  # Limit to top 3 alternatives
                # Calculate different approach
                alt_steps = await self._plan_alternative_progression(
                    user_skills, target_role, path, db
                )
                
                # Estimate timeline for alternative
                alt_timeline = self._estimate_alternative_timeline(alt_steps)
                
                alternative = {
                    'path_id': f"alt_{hash(path['approach'])}",
                    'approach': path['approach'],
                    'description': path['description'],
                    'progression_steps': alt_steps,
                    'estimated_timeline_months': alt_timeline,
                    'advantages': path.get('advantages', []),
                    'considerations': path.get('considerations', []),
                    'success_rate': path.get('success_rate', 0.7)
                }
                
                alternatives.append(alternative)
                
        except Exception as e:
            logger.error(f"Error finding alternative routes: {str(e)}")
        
        return alternatives
    
    # Helper methods for data processing and analysis
    
    def _create_user_text(self, user_profile: UserProfile) -> str:
        """Create text representation of user profile for embedding."""
        parts = []
        
        if user_profile.current_role:
            parts.append(user_profile.current_role)
        
        if user_profile.dream_job:
            parts.append(user_profile.dream_job)
        
        if user_profile.skills:
            parts.extend(list(user_profile.skills.keys()))
        
        experience = user_profile.experience_years or 0
        if experience > 0:
            parts.append(f"{experience} years experience")
        
        return " ".join(parts)
    
    async def _aggregate_job_skills(
        self, 
        jobs: List[JobPosting], 
        db: AsyncSession
    ) -> Dict[str, float]:
        """Aggregate skills from multiple job postings."""
        skill_counts = {}
        total_jobs = len(jobs)
        
        for job in jobs:
            job_skills = job.processed_skills or {}
            for skill, importance in job_skills.items():
                if skill not in skill_counts:
                    skill_counts[skill] = 0
                skill_counts[skill] += importance
        
        # Normalize by frequency
        aggregated_skills = {}
        for skill, total_importance in skill_counts.items():
            frequency = total_importance / total_jobs
            aggregated_skills[skill] = min(frequency, 1.0)
        
        return aggregated_skills
    
    def _get_default_role_skills(self, role: str) -> Dict[str, float]:
        """Get default skill requirements for common roles."""
        role_skills = {
            'software engineer': {
                'python': 0.8, 'javascript': 0.7, 'sql': 0.6, 'git': 0.9,
                'algorithms': 0.7, 'system design': 0.6, 'testing': 0.5
            },
            'data scientist': {
                'python': 0.9, 'sql': 0.8, 'machine learning': 0.9, 'statistics': 0.8,
                'pandas': 0.7, 'numpy': 0.7, 'scikit-learn': 0.6
            },
            'product manager': {
                'product strategy': 0.9, 'user research': 0.7, 'data analysis': 0.6,
                'project management': 0.8, 'communication': 0.9, 'market analysis': 0.6
            },
            'frontend developer': {
                'javascript': 0.9, 'html': 0.8, 'css': 0.8, 'react': 0.7,
                'typescript': 0.6, 'responsive design': 0.7
            },
            'backend developer': {
                'python': 0.8, 'java': 0.7, 'sql': 0.8, 'api design': 0.7,
                'microservices': 0.6, 'databases': 0.8
            },
            'devops engineer': {
                'docker': 0.9, 'kubernetes': 0.8, 'aws': 0.7, 'linux': 0.8,
                'terraform': 0.6, 'monitoring': 0.7
            },
            'data engineer': {
                'python': 0.8, 'sql': 0.9, 'spark': 0.7, 'kafka': 0.6,
                'airflow': 0.6, 'data modeling': 0.8
            },
            'machine learning engineer': {
                'python': 0.9, 'machine learning': 0.9, 'tensorflow': 0.7,
                'pytorch': 0.7, 'mlops': 0.6, 'docker': 0.6
            }
        }
        
        role_lower = role.lower()
        for role_key, skills in role_skills.items():
            if role_key in role_lower:
                return skills
        
        # Default skills for unknown roles
        return {
            'communication': 0.8, 'problem solving': 0.8, 'teamwork': 0.7,
            'project management': 0.6, 'analytical thinking': 0.7
        }
    
    async def _plan_progression_steps(
        self,
        current_role: str,
        target_role: str,
        gap_analysis,
        db: AsyncSession
    ) -> List[Dict[str, Any]]:
        """Plan detailed progression steps for career trajectory."""
        steps = []
        
        # Current position
        steps.append({
            'role': current_role,
            'duration_months': 0,
            'description': 'Current position - foundation building',
            'key_activities': [
                'Strengthen existing skills',
                'Identify skill gaps',
                'Build professional network'
            ],
            'skills_to_develop': [],
            'milestones': ['Complete current projects', 'Seek feedback']
        })
        
        # Intermediate steps based on skill gaps
        if gap_analysis.learning_time_estimate > 24:  # More than 2 years of learning
            # Add intermediate role
            intermediate_role = self._suggest_intermediate_role(current_role, target_role)
            steps.append({
                'role': intermediate_role,
                'duration_months': 18,
                'description': f'Intermediate step towards {target_role}',
                'key_activities': [
                    'Develop core technical skills',
                    'Gain relevant experience',
                    'Build domain expertise'
                ],
                'skills_to_develop': gap_analysis.priority_skills[:3],
                'milestones': ['Complete certification', 'Lead project', 'Mentor others']
            })
        
        # Skill development phase
        steps.append({
            'role': f"{target_role} Preparation",
            'duration_months': max(6, gap_analysis.learning_time_estimate // 4),
            'description': 'Intensive skill development and preparation',
            'key_activities': [
                'Complete targeted learning paths',
                'Build portfolio projects',
                'Network with industry professionals'
            ],
            'skills_to_develop': gap_analysis.priority_skills,
            'milestones': [
                'Complete key certifications',
                'Deploy portfolio projects',
                'Attend industry events'
            ]
        })
        
        # Target role
        steps.append({
            'role': target_role,
            'duration_months': 0,
            'description': f'Transition to {target_role}',
            'key_activities': [
                'Apply for target positions',
                'Leverage network connections',
                'Demonstrate acquired skills'
            ],
            'skills_to_develop': [],
            'milestones': ['Secure target role', 'Successfully onboard']
        })
        
        return steps
    
    def _suggest_intermediate_role(self, current_role: str, target_role: str) -> str:
        """Suggest an intermediate role between current and target."""
        # Simple logic - in practice, this would use career progression data
        role_progressions = {
            ('junior developer', 'senior developer'): 'developer',
            ('developer', 'tech lead'): 'senior developer',
            ('analyst', 'data scientist'): 'senior analyst',
            ('coordinator', 'manager'): 'senior coordinator',
        }
        
        current_lower = current_role.lower()
        target_lower = target_role.lower()
        
        for (curr, targ), intermediate in role_progressions.items():
            if curr in current_lower and targ in target_lower:
                return intermediate
        
        # Default intermediate role
        if 'senior' not in current_lower and 'senior' in target_lower:
            return f"Senior {current_role}"
        
        return f"{target_role} Associate"    

    def _estimate_trajectory_timeline(
        self, 
        progression_steps: List[Dict[str, Any]], 
        learning_weeks: int
    ) -> int:
        """Estimate total timeline for career trajectory in months."""
        total_months = 0
        
        for step in progression_steps:
            total_months += step.get('duration_months', 0)
        
        # Add learning time (convert weeks to months)
        learning_months = max(6, learning_weeks // 4)
        total_months += learning_months
        
        return total_months
    
    def _assess_difficulty_level(self, gap_analysis) -> str:
        """Assess difficulty level of career trajectory."""
        readiness = gap_analysis.overall_readiness
        learning_time = gap_analysis.learning_time_estimate
        
        if readiness > 0.8 and learning_time < 12:
            return "easy"
        elif readiness > 0.6 and learning_time < 24:
            return "moderate"
        elif readiness > 0.4 and learning_time < 48:
            return "challenging"
        else:
            return "difficult"
    
    async def _get_market_demand_data(self, role: str, db: AsyncSession) -> Dict[str, Any]:
        """Get market demand data for a role."""
        # Check cache first
        cache_key = f"market_demand_{role.lower().replace(' ', '_')}"
        if cache_key in self.market_demand_cache:
            cached_data, timestamp = self.market_demand_cache[cache_key]
            if datetime.utcnow() - timestamp < self.cache_expiry:
                return cached_data
        
        try:
            # Query recent job postings for this role
            job_repo = JobRepository()
            recent_jobs = await job_repo.search_jobs(
                db, title=role, limit=100
            )
            
            if not recent_jobs:
                # Default market data
                market_data = {
                    'demand_level': 'moderate',
                    'growth_potential': 0.6,
                    'salary_trend': 'stable',
                    'job_count': 0
                }
            else:
                # Analyze market data
                job_count = len(recent_jobs)
                
                # Calculate demand level based on job count
                if job_count > 50:
                    demand_level = 'high'
                elif job_count > 20:
                    demand_level = 'moderate'
                else:
                    demand_level = 'low'
                
                # Calculate growth potential (simplified)
                growth_potential = min(0.9, 0.5 + (job_count / 100))
                
                market_data = {
                    'demand_level': demand_level,
                    'growth_potential': growth_potential,
                    'salary_trend': 'growing' if job_count > 30 else 'stable',
                    'job_count': job_count,
                    'average_salary': self._calculate_average_salary(recent_jobs)
                }
            
            # Cache the result
            self.market_demand_cache[cache_key] = (market_data, datetime.utcnow())
            return market_data
            
        except Exception as e:
            logger.error(f"Error getting market demand data: {str(e)}")
            return {
                'demand_level': 'moderate',
                'growth_potential': 0.6,
                'salary_trend': 'stable',
                'job_count': 0
            }
    
    def _calculate_average_salary(self, jobs: List[JobPosting]) -> Optional[int]:
        """Calculate average salary from job postings."""
        salaries = []
        
        for job in jobs:
            if job.salary_min and job.salary_max:
                avg_salary = (job.salary_min + job.salary_max) / 2
                salaries.append(avg_salary)
        
        return int(np.mean(salaries)) if salaries else None
    
    async def _calculate_salary_progression(
        self, 
        progression_steps: List[Dict[str, Any]], 
        db: AsyncSession
    ) -> Dict[str, Tuple[int, int]]:
        """Calculate salary progression for each step."""
        salary_progression = {}
        
        # Default salary ranges by role level
        default_salaries = {
            'entry': (45000, 65000),
            'junior': (55000, 75000),
            'mid': (70000, 95000),
            'senior': (90000, 130000),
            'lead': (120000, 160000),
            'principal': (150000, 200000),
            'manager': (110000, 150000),
            'director': (160000, 220000)
        }
        
        for step in progression_steps:
            role = step['role']
            
            # Try to get actual salary data
            try:
                job_repo = JobRepository()
                similar_jobs = await job_repo.search_jobs(
                    db, title=role, limit=20
                )
                
                if similar_jobs:
                    salaries = []
                    for job in similar_jobs:
                        if job.salary_min and job.salary_max:
                            salaries.append((job.salary_min, job.salary_max))
                    
                    if salaries:
                        min_salaries = [s[0] for s in salaries]
                        max_salaries = [s[1] for s in salaries]
                        salary_range = (
                            int(np.mean(min_salaries)),
                            int(np.mean(max_salaries))
                        )
                        salary_progression[role] = salary_range
                        continue
            except Exception:
                pass
            
            # Use default salary based on role level
            role_lower = role.lower()
            for level, salary_range in default_salaries.items():
                if level in role_lower:
                    salary_progression[role] = salary_range
                    break
            else:
                # Default salary
                salary_progression[role] = (60000, 85000)
        
        return salary_progression
    
    def _calculate_confidence_score(
        self, 
        similarity: float, 
        readiness: float, 
        market_data: Dict[str, Any],
        market_weight: float = 0.2
    ) -> float:
        """Calculate confidence score for trajectory recommendation."""
        # Base score from similarity and readiness
        base_score = (similarity * 0.4) + (readiness * 0.4)
        
        # Market demand factor
        demand_scores = {'low': 0.3, 'moderate': 0.6, 'high': 0.9}
        market_score = demand_scores.get(market_data.get('demand_level', 'moderate'), 0.6)
        
        # Growth potential factor
        growth_score = market_data.get('growth_potential', 0.6)
        
        # Combined confidence score
        confidence = base_score + (market_score * market_weight) + (growth_score * 0.2)
        
        return min(1.0, confidence)
    
    def _generate_trajectory_reasoning(
        self, 
        target_role: str, 
        gap_analysis, 
        market_data: Dict[str, Any],
        trajectory_type: str
    ) -> str:
        """Generate reasoning for trajectory recommendation."""
        reasoning_parts = []
        
        # Trajectory type specific reasoning
        if trajectory_type == "dream_job":
            reasoning_parts.append(f"This trajectory aligns with your stated goal of becoming a {target_role}.")
        elif trajectory_type == "natural_progression":
            reasoning_parts.append(f"This represents a natural career progression from your current role.")
        elif trajectory_type == "lateral_move":
            reasoning_parts.append(f"This lateral move leverages your transferable skills while opening new opportunities.")
        elif trajectory_type == "market_driven":
            reasoning_parts.append(f"This trajectory targets a high-demand role with strong market opportunities.")
        
        # Skill analysis
        if gap_analysis.overall_readiness > 0.7:
            reasoning_parts.append(f"You already possess {len(gap_analysis.strong_skills)} key skills required for this role.")
        else:
            reasoning_parts.append(f"While you'll need to develop {len(gap_analysis.missing_skills)} new skills, your existing expertise provides a solid foundation.")
        
        # Market analysis
        demand_level = market_data.get('demand_level', 'moderate')
        if demand_level == 'high':
            reasoning_parts.append("The job market shows strong demand for this role, improving your chances of success.")
        elif demand_level == 'low':
            reasoning_parts.append("While market demand is currently lower, this could represent a strategic long-term opportunity.")
        
        # Timeline consideration
        if gap_analysis.learning_time_estimate < 24:
            reasoning_parts.append("The relatively short learning timeline makes this an achievable near-term goal.")
        else:
            reasoning_parts.append("This is a longer-term trajectory that will require sustained effort and commitment.")
        
        return " ".join(reasoning_parts)
    
    def _identify_success_factors(
        self, 
        gap_analysis, 
        market_data: Dict[str, Any]
    ) -> List[str]:
        """Identify key success factors for the trajectory."""
        factors = []
        
        # Skill-based factors
        if gap_analysis.priority_skills:
            factors.append(f"Master {gap_analysis.priority_skills[0]} as the highest priority skill")
        
        if len(gap_analysis.strong_skills) > 3:
            factors.append("Leverage your strong existing skill foundation")
        
        # Market-based factors
        if market_data.get('demand_level') == 'high':
            factors.append("Take advantage of high market demand")
        
        # General success factors
        factors.extend([
            "Build a strong professional network in the target field",
            "Create a portfolio demonstrating relevant skills",
            "Seek mentorship from professionals in the target role",
            "Stay updated with industry trends and technologies"
        ])
        
        return factors[:5]  # Limit to top 5 factors
    
    def _identify_challenges(
        self, 
        gap_analysis, 
        market_data: Dict[str, Any]
    ) -> List[str]:
        """Identify potential challenges for the trajectory."""
        challenges = []
        
        # Skill gap challenges
        if len(gap_analysis.missing_skills) > 5:
            challenges.append("Significant skill development required across multiple areas")
        
        if gap_analysis.learning_time_estimate > 48:
            challenges.append("Long learning timeline requiring sustained commitment")
        
        # Market challenges
        if market_data.get('demand_level') == 'low':
            challenges.append("Limited job opportunities in current market")
        
        # Readiness challenges
        if gap_analysis.overall_readiness < 0.5:
            challenges.append("Low current readiness requiring substantial preparation")
        
        # General challenges
        challenges.extend([
            "Competition from other candidates with similar goals",
            "Potential need for additional education or certifications",
            "Balancing skill development with current work responsibilities"
        ])
        
        return challenges[:4]  # Limit to top 4 challenges
    
    # Additional helper methods for advanced features
    
    async def _get_career_progression_model(
        self, 
        current_role: str, 
        db: AsyncSession
    ) -> Optional[CareerProgressionModel]:
        """Get career progression model for a role."""
        # This would typically load from a database or ML model
        # For now, return a simplified model
        
        progression_models = {
            'software engineer': {
                'typical_progression': [
                    {
                        'role': 'Senior Software Engineer',
                        'typical_duration_months': 24,
                        'key_activities': ['Lead technical projects', 'Mentor junior developers']
                    },
                    {
                        'role': 'Tech Lead',
                        'typical_duration_months': 36,
                        'key_activities': ['Architect solutions', 'Guide technical decisions']
                    },
                    {
                        'role': 'Engineering Manager',
                        'typical_duration_months': 48,
                        'key_activities': ['Manage engineering teams', 'Strategic planning']
                    }
                ]
            },
            'data analyst': {
                'typical_progression': [
                    {
                        'role': 'Senior Data Analyst',
                        'typical_duration_months': 18,
                        'key_activities': ['Lead analysis projects', 'Develop insights']
                    },
                    {
                        'role': 'Data Scientist',
                        'typical_duration_months': 30,
                        'key_activities': ['Build ML models', 'Advanced analytics']
                    },
                    {
                        'role': 'Principal Data Scientist',
                        'typical_duration_months': 42,
                        'key_activities': ['Research and innovation', 'Technical leadership']
                    }
                ]
            }
        }
        
        role_lower = current_role.lower()
        for role_key, model_data in progression_models.items():
            if role_key in role_lower:
                return CareerProgressionModel(
                    career_path=role_key,
                    typical_progression=model_data['typical_progression'],
                    skill_evolution={},
                    timeline_ranges={},
                    success_metrics={},
                    market_trends={}
                )
        
        return None
    
    async def _get_role_skill_requirements(
        self, 
        role: str, 
        db: AsyncSession
    ) -> Dict[str, float]:
        """Get skill requirements for a specific role."""
        try:
            job_repo = JobRepository()
            role_jobs = await job_repo.search_jobs(db, title=role, limit=50)
            
            if role_jobs:
                return await self._aggregate_job_skills(role_jobs, db)
            else:
                return self._get_default_role_skills(role)
                
        except Exception as e:
            logger.error(f"Error getting role skill requirements: {str(e)}")
            return self._get_default_role_skills(role)
    
    async def _find_similar_roles(
        self, 
        user_skills: Dict[str, float], 
        db: AsyncSession, 
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Find roles with similar skill requirements."""
        similar_roles = []
        
        try:
            # Get all unique job titles from recent postings
            job_repo = JobRepository()
            recent_jobs = await job_repo.get_recent_jobs(db, days=30, limit=500)
            
            # Group jobs by title and calculate skill similarity
            role_skills = {}
            for job in recent_jobs:
                title = job.title
                if title not in role_skills:
                    role_skills[title] = {}
                
                job_skills = job.processed_skills or {}
                for skill, importance in job_skills.items():
                    if skill not in role_skills[title]:
                        role_skills[title][skill] = 0
                    role_skills[title][skill] += importance
            
            # Calculate similarity scores
            user_skill_set = set(user_skills.keys())
            
            for role, skills in role_skills.items():
                role_skill_set = set(skills.keys())
                
                # Calculate Jaccard similarity
                intersection = user_skill_set.intersection(role_skill_set)
                union = user_skill_set.union(role_skill_set)
                
                if union:
                    similarity = len(intersection) / len(union)
                    
                    # Weight by skill confidence
                    weighted_similarity = 0
                    for skill in intersection:
                        user_conf = user_skills.get(skill, 0)
                        role_imp = skills.get(skill, 0)
                        weighted_similarity += min(user_conf, role_imp)
                    
                    if len(intersection) > 0:
                        weighted_similarity /= len(intersection)
                    
                    similar_roles.append({
                        'role': role,
                        'skills': skills,
                        'similarity': (similarity + weighted_similarity) / 2,
                        'skill_overlap': len(intersection)
                    })
            
            # Sort by similarity and return top results
            similar_roles.sort(key=lambda x: x['similarity'], reverse=True)
            return similar_roles[:limit]
            
        except Exception as e:
            logger.error(f"Error finding similar roles: {str(e)}")
            return []
    
    async def _get_high_demand_roles(
        self, 
        db: AsyncSession, 
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get roles with high market demand."""
        try:
            # Query job postings to find high-demand roles
            job_repo = JobRepository()
            recent_jobs = await job_repo.get_recent_jobs(db, days=30, limit=1000)
            
            # Count job postings by role
            role_counts = {}
            role_skills = {}
            
            for job in recent_jobs:
                title = job.title
                if title not in role_counts:
                    role_counts[title] = 0
                    role_skills[title] = {}
                
                role_counts[title] += 1
                
                # Aggregate skills
                job_skills = job.processed_skills or {}
                for skill, importance in job_skills.items():
                    if skill not in role_skills[title]:
                        role_skills[title][skill] = 0
                    role_skills[title][skill] += importance
            
            # Calculate demand scores
            high_demand_roles = []
            max_count = max(role_counts.values()) if role_counts else 1
            
            for role, count in role_counts.items():
                demand_score = count / max_count
                
                if demand_score > 0.1:  # Only include roles with reasonable demand
                    high_demand_roles.append({
                        'role': role,
                        'skills': role_skills[role],
                        'demand_score': demand_score,
                        'job_count': count
                    })
            
            # Sort by demand score
            high_demand_roles.sort(key=lambda x: x['demand_score'], reverse=True)
            return high_demand_roles[:limit]
            
        except Exception as e:
            logger.error(f"Error getting high-demand roles: {str(e)}")
            return []
    
    async def _plan_market_driven_steps(
        self,
        user_profile: UserProfile,
        target_role: str,
        gap_analysis,
        db: AsyncSession
    ) -> List[Dict[str, Any]]:
        """Plan progression steps for market-driven trajectory."""
        steps = []
        
        # Current position
        steps.append({
            'role': user_profile.current_role or 'Current Position',
            'duration_months': 0,
            'description': 'Current position - market research and preparation',
            'key_activities': [
                'Research market trends and opportunities',
                'Identify key players in target industry',
                'Assess current skill relevance'
            ],
            'skills_to_develop': [],
            'milestones': ['Complete market analysis', 'Network with industry professionals']
        })
        
        # Skill development phase
        priority_skills = gap_analysis.priority_skills[:3]
        steps.append({
            'role': 'Skill Development Phase',
            'duration_months': max(6, gap_analysis.learning_time_estimate // 4),
            'description': 'Intensive development of high-demand skills',
            'key_activities': [
                'Complete targeted training programs',
                'Build projects showcasing new skills',
                'Obtain relevant certifications'
            ],
            'skills_to_develop': priority_skills,
            'milestones': [
                f'Master {priority_skills[0] if priority_skills else "key skills"}',
                'Complete certification program',
                'Build portfolio project'
            ]
        })
        
        # Market entry phase
        steps.append({
            'role': f'{target_role} Candidate',
            'duration_months': 6,
            'description': 'Active job search and market entry',
            'key_activities': [
                'Apply to target positions',
                'Leverage professional network',
                'Participate in industry events'
            ],
            'skills_to_develop': [],
            'milestones': [
                'Submit applications to target companies',
                'Complete technical interviews',
                'Negotiate job offer'
            ]
        })
        
        # Target role
        steps.append({
            'role': target_role,
            'duration_months': 0,
            'description': f'Successfully transition to {target_role}',
            'key_activities': [
                'Onboard effectively',
                'Apply learned skills',
                'Continue professional development'
            ],
            'skills_to_develop': [],
            'milestones': ['Complete onboarding', 'Deliver first project']
        })
        
        return steps
    
    async def _discover_alternative_paths(
        self, 
        target_role: str, 
        db: AsyncSession
    ) -> List[Dict[str, Any]]:
        """Discover alternative career paths to the same target role."""
        alternatives = []
        
        # Common alternative approaches
        alternative_approaches = [
            {
                'approach': 'Bootcamp Route',
                'description': 'Intensive bootcamp or accelerated program',
                'advantages': ['Fast-track learning', 'Practical focus', 'Job placement support'],
                'considerations': ['Intensive time commitment', 'May lack depth in some areas'],
                'success_rate': 0.7
            },
            {
                'approach': 'Freelance/Contract Path',
                'description': 'Build experience through freelance work',
                'advantages': ['Flexible learning', 'Real-world experience', 'Portfolio building'],
                'considerations': ['Income uncertainty', 'Self-directed learning required'],
                'success_rate': 0.6
            },
            {
                'approach': 'Internal Transition',
                'description': 'Transition within current company',
                'advantages': ['Known environment', 'Existing relationships', 'Lower risk'],
                'considerations': ['Limited by company opportunities', 'May require patience'],
                'success_rate': 0.8
            },
            {
                'approach': 'Academic Route',
                'description': 'Formal education or advanced degree',
                'advantages': ['Comprehensive knowledge', 'Credibility', 'Network building'],
                'considerations': ['Time and cost intensive', 'May be theoretical'],
                'success_rate': 0.75
            }
        ]
        
        return alternative_approaches
    
    async def _plan_alternative_progression(
        self,
        user_skills: Dict[str, float],
        target_role: str,
        alternative_path: Dict[str, Any],
        db: AsyncSession
    ) -> List[Dict[str, Any]]:
        """Plan progression steps for alternative path."""
        approach = alternative_path['approach']
        
        if approach == 'Bootcamp Route':
            return [
                {
                    'role': 'Bootcamp Student',
                    'duration_months': 6,
                    'description': 'Intensive skill development program',
                    'key_activities': ['Complete bootcamp curriculum', 'Build portfolio projects']
                },
                {
                    'role': f'Junior {target_role}',
                    'duration_months': 12,
                    'description': 'Entry-level position with growth focus',
                    'key_activities': ['Apply bootcamp skills', 'Continue learning on the job']
                },
                {
                    'role': target_role,
                    'duration_months': 0,
                    'description': 'Target role achievement',
                    'key_activities': ['Leverage experience', 'Take on advanced responsibilities']
                }
            ]
        elif approach == 'Internal Transition':
            return [
                {
                    'role': 'Internal Skill Building',
                    'duration_months': 9,
                    'description': 'Develop skills while in current role',
                    'key_activities': ['Take on relevant projects', 'Shadow target role professionals']
                },
                {
                    'role': 'Internal Transfer',
                    'duration_months': 3,
                    'description': 'Formal transition within company',
                    'key_activities': ['Apply for internal position', 'Complete transition training']
                },
                {
                    'role': target_role,
                    'duration_months': 0,
                    'description': 'Target role in current company',
                    'key_activities': ['Apply existing company knowledge', 'Excel in new role']
                }
            ]
        else:
            # Generic alternative progression
            return [
                {
                    'role': 'Preparation Phase',
                    'duration_months': 8,
                    'description': f'Prepare for {target_role} via {approach.lower()}',
                    'key_activities': ['Follow alternative path', 'Build relevant experience']
                },
                {
                    'role': target_role,
                    'duration_months': 0,
                    'description': 'Target role achievement',
                    'key_activities': ['Apply learned skills', 'Demonstrate competency']
                }
            ]
    
    def _estimate_alternative_timeline(self, alt_steps: List[Dict[str, Any]]) -> int:
        """Estimate timeline for alternative progression."""
        total_months = 0
        for step in alt_steps:
            total_months += step.get('duration_months', 0)
        return total_months