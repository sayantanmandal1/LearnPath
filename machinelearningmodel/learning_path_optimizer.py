"""
Learning Path Optimizer for the AI Career Recommender System.

This module implements advanced ML algorithms for:
- Skill gap identification and prioritization
- Learning timeline estimation
- Resource quality scoring
- Personalized learning path optimization
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import logging
import json
from datetime import datetime, timedelta


logger = logging.getLogger(__name__)


@dataclass
class SkillGapAnalysis:
    """Result of skill gap analysis."""
    skill_name: str
    current_level: float
    target_level: float
    gap_size: float
    priority_score: float
    estimated_hours: int
    difficulty_level: str
    prerequisites: List[str]
    market_demand: float


@dataclass
class LearningTimelineEstimate:
    """Learning timeline estimation result."""
    total_hours: int
    total_weeks: int
    milestones: List[Dict[str, Any]]
    confidence_score: float
    factors_considered: List[str]


class LearningPathOptimizer:
    """Advanced ML-based learning path optimization."""
    
    def __init__(self):
        self.skill_embeddings = {}
        self.timeline_model = None
        self.quality_model = None
        self.scaler = StandardScaler()
        self._initialize_models()
        
    def _initialize_models(self):
        """Initialize ML models for optimization."""
        try:
            # Initialize timeline estimation model
            self.timeline_model = RandomForestRegressor(
                n_estimators=100,
                random_state=42,
                max_depth=10
            )
            
            # Initialize resource quality scoring model
            self.quality_model = RandomForestRegressor(
                n_estimators=50,
                random_state=42,
                max_depth=8
            )
            
            # Load pre-trained models if available
            self._load_pretrained_models()
            
            logger.info("Learning path optimizer models initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
            raise

    def analyze_skill_gaps(
        self, 
        current_skills: Dict[str, float], 
        target_skills: List[str],
        target_role: Optional[str] = None,
        market_data: Optional[Dict[str, Any]] = None
    ) -> List[SkillGapAnalysis]:
        """
        Perform advanced skill gap analysis with ML-based prioritization.
        
        Implements requirement 4.1 - skill gap identification and prioritization.
        """
        try:
            logger.info(f"Analyzing skill gaps for {len(target_skills)} target skills")
            
            skill_gaps = []
            
            # Get role-specific skill requirements
            role_requirements = self._get_role_skill_requirements(target_role)
            
            # Combine target skills with role requirements
            all_target_skills = list(set(target_skills + list(role_requirements.keys())))
            
            for skill in all_target_skills:
                current_level = current_skills.get(skill, 0.0)
                target_level = role_requirements.get(skill, 0.8)  # Default target level
                
                if current_level < target_level:
                    gap_size = target_level - current_level
                    
                    # Calculate priority using multiple factors
                    priority_score = self._calculate_skill_priority(
                        skill, gap_size, target_role, market_data
                    )
                    
                    # Estimate learning time
                    estimated_hours = self._estimate_skill_learning_time(
                        skill, gap_size, current_level
                    )
                    
                    # Get skill metadata
                    skill_meta = self._get_skill_metadata(skill)
                    
                    gap_analysis = SkillGapAnalysis(
                        skill_name=skill,
                        current_level=current_level,
                        target_level=target_level,
                        gap_size=gap_size,
                        priority_score=priority_score,
                        estimated_hours=estimated_hours,
                        difficulty_level=skill_meta.get('difficulty', 'intermediate'),
                        prerequisites=skill_meta.get('prerequisites', []),
                        market_demand=self._get_market_demand_score(skill, market_data)
                    )
                    
                    skill_gaps.append(gap_analysis)
            
            # Sort by priority score (highest first)
            skill_gaps.sort(key=lambda x: x.priority_score, reverse=True)
            
            logger.info(f"Identified {len(skill_gaps)} skill gaps")
            return skill_gaps
            
        except Exception as e:
            logger.error(f"Error analyzing skill gaps: {e}")
            raise

    def estimate_learning_timeline(
        self,
        skill_gaps: List[SkillGapAnalysis],
        user_profile: Dict[str, Any],
        time_commitment_hours_per_week: int = 10
    ) -> LearningTimelineEstimate:
        """
        Estimate learning timeline using ML models and user-specific factors.
        
        Implements requirement 4.5 - timeline estimation and milestone tracking.
        """
        try:
            logger.info(f"Estimating timeline for {len(skill_gaps)} skills")
            
            # Extract features for timeline prediction
            features = self._extract_timeline_features(skill_gaps, user_profile)
            
            # Predict base learning time
            if self.timeline_model and hasattr(self.timeline_model, 'predict'):
                # Use trained model if available
                base_hours = self.timeline_model.predict([features])[0]
            else:
                # Fallback to heuristic calculation
                base_hours = sum(gap.estimated_hours for gap in skill_gaps)
            
            # Apply user-specific adjustments
            adjusted_hours = self._apply_user_adjustments(
                base_hours, user_profile, skill_gaps
            )
            
            # Calculate weeks based on time commitment
            total_weeks = max(1, int(adjusted_hours / time_commitment_hours_per_week))
            
            # Create milestone timeline
            milestones = self._create_milestone_timeline(
                skill_gaps, total_weeks, time_commitment_hours_per_week
            )
            
            # Calculate confidence score
            confidence_score = self._calculate_timeline_confidence(
                skill_gaps, user_profile, adjusted_hours
            )
            
            # Identify factors considered
            factors_considered = [
                "skill_difficulty",
                "user_experience_level",
                "prerequisite_knowledge",
                "time_commitment",
                "learning_style_preferences"
            ]
            
            timeline_estimate = LearningTimelineEstimate(
                total_hours=int(adjusted_hours),
                total_weeks=total_weeks,
                milestones=milestones,
                confidence_score=confidence_score,
                factors_considered=factors_considered
            )
            
            logger.info(f"Timeline estimated: {total_weeks} weeks, {int(adjusted_hours)} hours")
            return timeline_estimate
            
        except Exception as e:
            logger.error(f"Error estimating learning timeline: {e}")
            raise

    def score_resource_quality(
        self,
        resource_data: Dict[str, Any],
        user_preferences: Dict[str, Any],
        skill_context: str
    ) -> float:
        """
        Score learning resource quality using ML models.
        
        Implements requirement 4.6 - resource quality scoring and filtering.
        """
        try:
            # Extract features for quality scoring
            features = self._extract_resource_features(
                resource_data, user_preferences, skill_context
            )
            
            # Predict quality score
            if self.quality_model and hasattr(self.quality_model, 'predict'):
                quality_score = self.quality_model.predict([features])[0]
            else:
                # Fallback to heuristic scoring
                quality_score = self._heuristic_quality_score(resource_data)
            
            # Normalize to 0-1 range
            quality_score = max(0.0, min(1.0, quality_score))
            
            return quality_score
            
        except Exception as e:
            logger.error(f"Error scoring resource quality: {e}")
            return 0.5  # Default neutral score

    def optimize_learning_sequence(
        self,
        skill_gaps: List[SkillGapAnalysis],
        user_constraints: Dict[str, Any]
    ) -> List[SkillGapAnalysis]:
        """
        Optimize the sequence of skills to learn based on dependencies and efficiency.
        
        Implements requirement 4.4 - learning path sequencing.
        """
        try:
            logger.info("Optimizing learning sequence")
            
            # Build skill dependency graph
            dependency_graph = self._build_skill_dependency_graph(skill_gaps)
            
            # Perform topological sort with priority weighting
            optimized_sequence = self._topological_sort_with_priority(
                dependency_graph, skill_gaps, user_constraints
            )
            
            logger.info(f"Optimized sequence for {len(optimized_sequence)} skills")
            return optimized_sequence
            
        except Exception as e:
            logger.error(f"Error optimizing learning sequence: {e}")
            return skill_gaps  # Return original order as fallback

    # Private helper methods
    
    def _get_role_skill_requirements(self, target_role: Optional[str]) -> Dict[str, float]:
        """Get skill requirements for a specific role."""
        if not target_role:
            return {}
        
        # This would typically query a database of job requirements
        role_requirements = {
            "software_engineer": {
                "python": 0.8, "javascript": 0.7, "git": 0.9, "sql": 0.6,
                "algorithms": 0.8, "system_design": 0.7, "testing": 0.6
            },
            "data_scientist": {
                "python": 0.9, "machine_learning": 0.9, "statistics": 0.8,
                "sql": 0.8, "data_visualization": 0.7, "pandas": 0.8
            },
            "frontend_developer": {
                "javascript": 0.9, "react": 0.8, "html": 0.8, "css": 0.8,
                "typescript": 0.7, "responsive_design": 0.7
            },
            "backend_developer": {
                "python": 0.8, "java": 0.7, "sql": 0.8, "api_design": 0.8,
                "microservices": 0.7, "docker": 0.6
            }
        }
        
        role_key = target_role.lower().replace(" ", "_")
        return role_requirements.get(role_key, {})

    def _calculate_skill_priority(
        self,
        skill: str,
        gap_size: float,
        target_role: Optional[str],
        market_data: Optional[Dict[str, Any]]
    ) -> float:
        """Calculate priority score for a skill gap."""
        priority = gap_size  # Base priority on gap size
        
        # Role criticality factor
        if target_role:
            role_requirements = self._get_role_skill_requirements(target_role)
            if skill in role_requirements:
                priority *= 1.5  # Boost priority for role-critical skills
        
        # Market demand factor
        market_demand = self._get_market_demand_score(skill, market_data)
        priority *= (1 + market_demand)
        
        # Skill difficulty factor (easier skills get slight priority boost)
        difficulty_boost = {
            'beginner': 1.2,
            'intermediate': 1.0,
            'advanced': 0.9,
            'expert': 0.8
        }
        skill_meta = self._get_skill_metadata(skill)
        difficulty = skill_meta.get('difficulty', 'intermediate')
        priority *= difficulty_boost.get(difficulty, 1.0)
        
        # Prerequisite factor (skills with fewer prerequisites get priority)
        prerequisites = skill_meta.get('prerequisites', [])
        if len(prerequisites) == 0:
            priority *= 1.3  # Boost for foundational skills
        elif len(prerequisites) > 3:
            priority *= 0.9  # Slight penalty for complex skills
        
        return min(priority, 1.0)  # Cap at 1.0

    def _estimate_skill_learning_time(
        self,
        skill: str,
        gap_size: float,
        current_level: float
    ) -> int:
        """Estimate learning time for a specific skill."""
        # Base learning hours for different skills
        base_hours = {
            "python": 80, "javascript": 70, "react": 60, "java": 90,
            "machine_learning": 120, "data_science": 100, "sql": 50,
            "docker": 40, "kubernetes": 60, "aws": 80, "git": 30,
            "html": 40, "css": 50, "typescript": 45, "node.js": 55
        }
        
        skill_hours = base_hours.get(skill.lower(), 60)  # Default 60 hours
        
        # Adjust based on gap size
        estimated_hours = int(skill_hours * gap_size)
        
        # Adjust based on current level (higher current level = faster learning)
        if current_level > 0.3:
            estimated_hours = int(estimated_hours * 0.8)  # 20% reduction
        elif current_level > 0.5:
            estimated_hours = int(estimated_hours * 0.6)  # 40% reduction
        
        return max(10, estimated_hours)  # Minimum 10 hours

    def _get_skill_metadata(self, skill: str) -> Dict[str, Any]:
        """Get metadata for a skill including difficulty and prerequisites."""
        skill_metadata = {
            "python": {
                "difficulty": "beginner",
                "category": "programming",
                "prerequisites": []
            },
            "javascript": {
                "difficulty": "beginner",
                "category": "programming",
                "prerequisites": []
            },
            "react": {
                "difficulty": "intermediate",
                "category": "frontend",
                "prerequisites": ["javascript", "html", "css"]
            },
            "machine_learning": {
                "difficulty": "advanced",
                "category": "ai",
                "prerequisites": ["python", "statistics", "linear_algebra"]
            },
            "docker": {
                "difficulty": "intermediate",
                "category": "devops",
                "prerequisites": ["linux", "command_line"]
            },
            "kubernetes": {
                "difficulty": "advanced",
                "category": "devops",
                "prerequisites": ["docker", "networking", "yaml"]
            }
        }
        
        return skill_metadata.get(skill.lower(), {
            "difficulty": "intermediate",
            "category": "general",
            "prerequisites": []
        })

    def _get_market_demand_score(
        self,
        skill: str,
        market_data: Optional[Dict[str, Any]]
    ) -> float:
        """Get market demand score for a skill."""
        if market_data and skill in market_data:
            return market_data[skill].get('demand_score', 0.5)
        
        # Default demand scores based on industry trends
        demand_scores = {
            "python": 0.9, "javascript": 0.9, "react": 0.8, "java": 0.8,
            "machine_learning": 0.85, "data_science": 0.8, "sql": 0.8,
            "docker": 0.7, "kubernetes": 0.75, "aws": 0.8, "git": 0.7,
            "typescript": 0.75, "node.js": 0.7, "vue.js": 0.6
        }
        
        return demand_scores.get(skill.lower(), 0.5)

    def _extract_timeline_features(
        self,
        skill_gaps: List[SkillGapAnalysis],
        user_profile: Dict[str, Any]
    ) -> List[float]:
        """Extract features for timeline prediction model."""
        features = []
        
        # Skill-related features
        features.append(len(skill_gaps))  # Number of skills
        features.append(np.mean([gap.gap_size for gap in skill_gaps]))  # Average gap size
        features.append(np.mean([gap.priority_score for gap in skill_gaps]))  # Average priority
        
        # Difficulty distribution
        difficulty_counts = {"beginner": 0, "intermediate": 0, "advanced": 0}
        for gap in skill_gaps:
            difficulty_counts[gap.difficulty_level] += 1
        
        total_skills = len(skill_gaps) or 1
        features.extend([
            difficulty_counts["beginner"] / total_skills,
            difficulty_counts["intermediate"] / total_skills,
            difficulty_counts["advanced"] / total_skills
        ])
        
        # User profile features
        features.append(user_profile.get('experience_years', 0) / 10.0)  # Normalized experience
        features.append(user_profile.get('learning_speed_factor', 1.0))  # Learning speed
        features.append(len(user_profile.get('current_skills', {})) / 20.0)  # Skill breadth
        
        return features

    def _apply_user_adjustments(
        self,
        base_hours: float,
        user_profile: Dict[str, Any],
        skill_gaps: List[SkillGapAnalysis]
    ) -> float:
        """Apply user-specific adjustments to learning time estimate."""
        adjusted_hours = base_hours
        
        # Experience level adjustment
        experience_years = user_profile.get('experience_years', 0)
        if experience_years > 5:
            adjusted_hours *= 0.8  # Experienced learners are faster
        elif experience_years < 2:
            adjusted_hours *= 1.2  # Beginners need more time
        
        # Learning style adjustment
        learning_style = user_profile.get('learning_style', 'mixed')
        style_factors = {
            'visual': 1.0,
            'hands_on': 0.9,  # Hands-on learners are often faster
            'theoretical': 1.1,  # Theoretical learners might need more time
            'mixed': 1.0
        }
        adjusted_hours *= style_factors.get(learning_style, 1.0)
        
        # Prerequisite knowledge adjustment
        current_skills = set(user_profile.get('current_skills', {}).keys())
        total_prerequisites = set()
        for gap in skill_gaps:
            total_prerequisites.update(gap.prerequisites)
        
        prerequisite_coverage = len(current_skills & total_prerequisites) / max(1, len(total_prerequisites))
        if prerequisite_coverage > 0.7:
            adjusted_hours *= 0.85  # Good prerequisite coverage reduces time
        elif prerequisite_coverage < 0.3:
            adjusted_hours *= 1.15  # Poor prerequisite coverage increases time
        
        return adjusted_hours

    def _create_milestone_timeline(
        self,
        skill_gaps: List[SkillGapAnalysis],
        total_weeks: int,
        hours_per_week: int
    ) -> List[Dict[str, Any]]:
        """Create milestone timeline for learning path."""
        milestones = []
        
        # Group skills by difficulty for milestone creation
        skill_groups = {"beginner": [], "intermediate": [], "advanced": []}
        for gap in skill_gaps:
            skill_groups[gap.difficulty_level].append(gap)
        
        current_week = 0
        milestone_id = 1
        
        # Create milestones for each difficulty level
        for difficulty in ["beginner", "intermediate", "advanced"]:
            skills_in_group = skill_groups[difficulty]
            if not skills_in_group:
                continue
            
            # Calculate weeks needed for this group
            group_hours = sum(gap.estimated_hours for gap in skills_in_group)
            group_weeks = max(1, group_hours // hours_per_week)
            
            milestone = {
                "id": milestone_id,
                "title": f"{difficulty.title()} Skills Milestone",
                "description": f"Complete {difficulty} level skills",
                "skills": [gap.skill_name for gap in skills_in_group],
                "start_week": current_week + 1,
                "end_week": current_week + group_weeks,
                "estimated_hours": group_hours,
                "difficulty": difficulty
            }
            
            milestones.append(milestone)
            current_week += group_weeks
            milestone_id += 1
        
        return milestones

    def _calculate_timeline_confidence(
        self,
        skill_gaps: List[SkillGapAnalysis],
        user_profile: Dict[str, Any],
        estimated_hours: float
    ) -> float:
        """Calculate confidence score for timeline estimate."""
        confidence_factors = []
        
        # Data quality factor
        if len(skill_gaps) > 0:
            avg_priority = np.mean([gap.priority_score for gap in skill_gaps])
            confidence_factors.append(avg_priority)
        
        # User profile completeness
        profile_completeness = len(user_profile) / 10.0  # Assume 10 key fields
        confidence_factors.append(min(1.0, profile_completeness))
        
        # Skill complexity factor
        complexity_scores = {"beginner": 0.9, "intermediate": 0.7, "advanced": 0.5}
        if skill_gaps:
            avg_complexity = np.mean([
                complexity_scores.get(gap.difficulty_level, 0.7) 
                for gap in skill_gaps
            ])
            confidence_factors.append(avg_complexity)
        
        # Time estimate reasonableness (not too high or too low)
        if 20 <= estimated_hours <= 500:  # Reasonable range
            confidence_factors.append(0.8)
        else:
            confidence_factors.append(0.6)
        
        return np.mean(confidence_factors) if confidence_factors else 0.7

    def _extract_resource_features(
        self,
        resource_data: Dict[str, Any],
        user_preferences: Dict[str, Any],
        skill_context: str
    ) -> List[float]:
        """Extract features for resource quality scoring."""
        features = []
        
        # Resource intrinsic features
        features.append(resource_data.get('rating', 3.0) / 5.0)  # Normalized rating
        features.append(resource_data.get('duration_hours', 50) / 100.0)  # Normalized duration
        features.append(1.0 if resource_data.get('certificate_available') else 0.0)
        features.append(1.0 if resource_data.get('hands_on_projects') else 0.0)
        
        # Provider reputation
        provider_scores = {
            'coursera': 0.9, 'edx': 0.9, 'udemy': 0.8, 'freecodecamp': 0.8,
            'pluralsight': 0.85, 'udacity': 0.85, 'codecademy': 0.75
        }
        provider = resource_data.get('provider', '').lower()
        features.append(provider_scores.get(provider, 0.7))
        
        # Cost factor (free resources get slight boost)
        cost = resource_data.get('cost', 0)
        if cost == 0:
            features.append(0.8)  # Free resource bonus
        elif cost < 50:
            features.append(0.7)  # Affordable
        else:
            features.append(0.6)  # Expensive
        
        # User preference alignment
        preferred_providers = user_preferences.get('preferred_providers', [])
        if not preferred_providers or provider in [p.lower() for p in preferred_providers]:
            features.append(1.0)
        else:
            features.append(0.5)
        
        # Skill relevance (simplified)
        skills_taught = resource_data.get('skills_taught', [])
        if skill_context.lower() in [s.lower() for s in skills_taught]:
            features.append(1.0)
        else:
            features.append(0.7)
        
        return features

    def _heuristic_quality_score(self, resource_data: Dict[str, Any]) -> float:
        """Fallback heuristic quality scoring."""
        score = 0.5  # Base score
        
        # Rating contribution (40%)
        if 'rating' in resource_data:
            score += (resource_data['rating'] / 5.0) * 0.4
        
        # Provider contribution (30%)
        provider_scores = {
            'coursera': 0.3, 'edx': 0.3, 'udemy': 0.25, 'freecodecamp': 0.25,
            'pluralsight': 0.28, 'udacity': 0.28, 'codecademy': 0.22
        }
        provider = resource_data.get('provider', '').lower()
        score += provider_scores.get(provider, 0.2)
        
        # Features contribution (30%)
        if resource_data.get('certificate_available'):
            score += 0.1
        if resource_data.get('hands_on_projects'):
            score += 0.1
        if resource_data.get('cost', 0) == 0:  # Free resource
            score += 0.1
        
        return min(1.0, score)

    def _build_skill_dependency_graph(
        self,
        skill_gaps: List[SkillGapAnalysis]
    ) -> Dict[str, List[str]]:
        """Build skill dependency graph."""
        graph = {}
        skill_names = {gap.skill_name for gap in skill_gaps}
        
        for gap in skill_gaps:
            # Only include prerequisites that are also in our skill list
            prerequisites = [p for p in gap.prerequisites if p in skill_names]
            graph[gap.skill_name] = prerequisites
        
        return graph

    def _topological_sort_with_priority(
        self,
        dependency_graph: Dict[str, List[str]],
        skill_gaps: List[SkillGapAnalysis],
        user_constraints: Dict[str, Any]
    ) -> List[SkillGapAnalysis]:
        """Perform topological sort with priority weighting."""
        # Create skill lookup
        skill_lookup = {gap.skill_name: gap for gap in skill_gaps}
        
        # Calculate in-degrees
        in_degree = {skill: 0 for skill in dependency_graph}
        for skill, deps in dependency_graph.items():
            for dep in deps:
                if dep in in_degree:
                    in_degree[skill] += 1
        
        # Priority queue (skills with no dependencies and high priority first)
        available = []
        for skill, degree in in_degree.items():
            if degree == 0:
                gap = skill_lookup[skill]
                available.append((gap.priority_score, skill, gap))
        
        available.sort(reverse=True)  # Sort by priority (highest first)
        
        result = []
        
        while available:
            # Take highest priority available skill
            _, skill_name, skill_gap = available.pop(0)
            result.append(skill_gap)
            
            # Update dependencies
            for other_skill in dependency_graph:
                if skill_name in dependency_graph[other_skill]:
                    in_degree[other_skill] -= 1
                    if in_degree[other_skill] == 0:
                        other_gap = skill_lookup[other_skill]
                        available.append((other_gap.priority_score, other_skill, other_gap))
            
            # Re-sort available skills by priority
            available.sort(reverse=True)
        
        return result

    def _load_pretrained_models(self):
        """Load pre-trained models if available."""
        try:
            # This would load actual trained models from disk
            # For now, we'll use synthetic training data to initialize models
            self._train_with_synthetic_data()
            
        except Exception as e:
            logger.warning(f"Could not load pre-trained models: {e}")

    def _train_with_synthetic_data(self):
        """Train models with synthetic data for demonstration."""
        try:
            # Generate synthetic training data for timeline model
            n_samples = 1000
            X_timeline = np.random.rand(n_samples, 9)  # 9 features
            y_timeline = (
                X_timeline[:, 0] * 50 +  # Number of skills
                X_timeline[:, 1] * 100 +  # Average gap size
                X_timeline[:, 6] * 30 +   # Experience factor
                np.random.normal(0, 10, n_samples)  # Noise
            )
            
            self.timeline_model.fit(X_timeline, y_timeline)
            
            # Generate synthetic training data for quality model
            X_quality = np.random.rand(n_samples, 8)  # 8 features
            y_quality = (
                X_quality[:, 0] * 0.4 +  # Rating factor
                X_quality[:, 4] * 0.3 +  # Provider factor
                X_quality[:, 2] * 0.1 +  # Certificate factor
                X_quality[:, 3] * 0.1 +  # Projects factor
                np.random.normal(0, 0.1, n_samples)  # Noise
            )
            y_quality = np.clip(y_quality, 0, 1)  # Clip to valid range
            
            self.quality_model.fit(X_quality, y_quality)
            
            logger.info("Models trained with synthetic data")
            
        except Exception as e:
            logger.error(f"Error training with synthetic data: {e}")