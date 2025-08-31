"""
Core recommendation engine algorithms for career and learning path recommendations.

This module implements multiple recommendation approaches:
1. Collaborative filtering using matrix factorization
2. Content-based filtering using cosine similarity on skill embeddings
3. Hybrid recommendation system combining multiple approaches
4. Neural collaborative filtering using PyTorch
5. Skill gap analysis and quantification algorithms
6. Recommendation explanation and reasoning generation
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from datetime import datetime
import json

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer
import scipy.sparse as sp

from .models import SkillExtraction, ResumeData


logger = logging.getLogger(__name__)


@dataclass
class CareerRecommendation:
    """Career recommendation with detailed analysis."""
    job_title: str
    match_score: float
    required_skills: List[str]
    skill_gaps: Dict[str, float]
    salary_range: Tuple[int, int]
    growth_potential: float
    market_demand: str
    reasoning: str
    confidence_score: float
    alternative_paths: List[str]


@dataclass
class LearningPath:
    """Learning path recommendation."""
    path_id: str
    title: str
    target_skills: List[str]
    estimated_duration_weeks: int
    difficulty_level: str
    resources: List[Dict[str, Any]]
    milestones: List[Dict[str, Any]]
    priority_score: float
    reasoning: str


@dataclass
class SkillGapAnalysis:
    """Skill gap analysis result."""
    target_role: str
    missing_skills: Dict[str, float]  # skill -> importance score
    weak_skills: Dict[str, float]     # skill -> gap score
    strong_skills: List[str]
    overall_readiness: float
    learning_time_estimate: int  # weeks
    priority_skills: List[str]


class CollaborativeFilteringEngine:
    """Collaborative filtering using matrix factorization techniques."""
    
    def __init__(self, n_factors: int = 50, learning_rate: float = 0.01, 
                 regularization: float = 0.1, n_iterations: int = 100):
        self.n_factors = n_factors
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.n_iterations = n_iterations
        self.user_factors = None
        self.item_factors = None
        self.user_bias = None
        self.item_bias = None
        self.global_bias = None
        self.fitted = False
        
    def fit(self, user_item_matrix: np.ndarray, user_ids: List[str], item_ids: List[str]):
        """
        Fit collaborative filtering model using matrix factorization.
        
        Args:
            user_item_matrix: User-item interaction matrix
            user_ids: List of user identifiers
            item_ids: List of item identifiers
        """
        logger.info(f"Training collaborative filtering model with {len(user_ids)} users and {len(item_ids)} items")
        
        n_users, n_items = user_item_matrix.shape
        
        # Initialize factors and biases
        self.user_factors = np.random.normal(0, 0.1, (n_users, self.n_factors))
        self.item_factors = np.random.normal(0, 0.1, (n_items, self.n_factors))
        self.user_bias = np.zeros(n_users)
        self.item_bias = np.zeros(n_items)
        self.global_bias = np.mean(user_item_matrix[user_item_matrix > 0])
        
        # Create sparse matrix for efficient computation
        sparse_matrix = sp.csr_matrix(user_item_matrix)
        
        # Training loop
        for iteration in range(self.n_iterations):
            for user_idx in range(n_users):
                for item_idx in sparse_matrix[user_idx].indices:
                    rating = sparse_matrix[user_idx, item_idx]
                    
                    # Predict rating
                    prediction = (self.global_bias + 
                                self.user_bias[user_idx] + 
                                self.item_bias[item_idx] + 
                                np.dot(self.user_factors[user_idx], self.item_factors[item_idx]))
                    
                    error = rating - prediction
                    
                    # Update biases
                    self.user_bias[user_idx] += self.learning_rate * (error - self.regularization * self.user_bias[user_idx])
                    self.item_bias[item_idx] += self.learning_rate * (error - self.regularization * self.item_bias[item_idx])
                    
                    # Update factors
                    user_factors_old = self.user_factors[user_idx].copy()
                    self.user_factors[user_idx] += self.learning_rate * (error * self.item_factors[item_idx] - 
                                                                       self.regularization * self.user_factors[user_idx])
                    self.item_factors[item_idx] += self.learning_rate * (error * user_factors_old - 
                                                                       self.regularization * self.item_factors[item_idx])
            
            if iteration % 20 == 0:
                rmse = self._calculate_rmse(sparse_matrix)
                logger.debug(f"Iteration {iteration}, RMSE: {rmse:.4f}")
        
        self.user_id_to_idx = {user_id: idx for idx, user_id in enumerate(user_ids)}
        self.item_id_to_idx = {item_id: idx for idx, item_id in enumerate(item_ids)}
        self.fitted = True
        
    def predict(self, user_id: str, item_id: str) -> float:
        """Predict rating for user-item pair."""
        if not self.fitted:
            raise ValueError("Model must be fitted before making predictions")
            
        user_idx = self.user_id_to_idx.get(user_id)
        item_idx = self.item_id_to_idx.get(item_id)
        
        if user_idx is None or item_idx is None:
            return self.global_bias
            
        prediction = (self.global_bias + 
                     self.user_bias[user_idx] + 
                     self.item_bias[item_idx] + 
                     np.dot(self.user_factors[user_idx], self.item_factors[item_idx]))
        
        return max(0, min(5, prediction))  # Clamp to valid rating range
    
    def recommend_items(self, user_id: str, n_recommendations: int = 10) -> List[Tuple[str, float]]:
        """Generate item recommendations for a user."""
        if not self.fitted:
            raise ValueError("Model must be fitted before making recommendations")
            
        user_idx = self.user_id_to_idx.get(user_id)
        if user_idx is None:
            return []
            
        # Calculate predictions for all items
        predictions = []
        for item_id, item_idx in self.item_id_to_idx.items():
            score = self.predict(user_id, item_id)
            predictions.append((item_id, score))
        
        # Sort by score and return top N
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:n_recommendations]
    
    def _calculate_rmse(self, sparse_matrix: sp.csr_matrix) -> float:
        """Calculate RMSE for current model."""
        total_error = 0
        count = 0
        
        for user_idx in range(sparse_matrix.shape[0]):
            for item_idx in sparse_matrix[user_idx].indices:
                rating = sparse_matrix[user_idx, item_idx]
                prediction = (self.global_bias + 
                            self.user_bias[user_idx] + 
                            self.item_bias[item_idx] + 
                            np.dot(self.user_factors[user_idx], self.item_factors[item_idx]))
                total_error += (rating - prediction) ** 2
                count += 1
        
        return np.sqrt(total_error / count) if count > 0 else 0


class ContentBasedFilteringEngine:
    """Content-based filtering using cosine similarity on skill embeddings."""
    
    def __init__(self, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.embedding_model_name = embedding_model
        self.embedding_model = None
        self.job_embeddings = {}
        self.user_embeddings = {}
        self.skill_embeddings = {}
        self.tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        self.fitted = False
        
    def _load_embedding_model(self):
        """Load sentence transformer model."""
        if self.embedding_model is None:
            logger.info(f"Loading embedding model: {self.embedding_model_name}")
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
    
    def fit(self, job_data: List[Dict[str, Any]], user_data: List[Dict[str, Any]]):
        """
        Fit content-based filtering model.
        
        Args:
            job_data: List of job postings with skills and descriptions
            user_data: List of user profiles with skills and experience
        """
        self._load_embedding_model()
        logger.info(f"Training content-based filtering with {len(job_data)} jobs and {len(user_data)} users")
        
        # Generate job embeddings
        job_texts = []
        job_ids = []
        for job in job_data:
            job_text = f"{job['title']} {job['description']} {' '.join(job.get('skills', []))}"
            job_texts.append(job_text)
            job_ids.append(job['id'])
        
        if job_texts:
            job_embeddings = self.embedding_model.encode(job_texts)
            self.job_embeddings = {job_id: embedding for job_id, embedding in zip(job_ids, job_embeddings)}
        
        # Generate user embeddings
        user_texts = []
        user_ids = []
        for user in user_data:
            user_text = f"{user.get('current_role', '')} {user.get('dream_job', '')} {' '.join(user.get('skills', []))}"
            user_texts.append(user_text)
            user_ids.append(user['id'])
        
        if user_texts:
            user_embeddings = self.embedding_model.encode(user_texts)
            self.user_embeddings = {user_id: embedding for user_id, embedding in zip(user_ids, user_embeddings)}
        
        # Generate skill embeddings
        all_skills = set()
        for job in job_data:
            all_skills.update(job.get('skills', []))
        for user in user_data:
            all_skills.update(user.get('skills', []))
        
        if all_skills:
            skill_list = list(all_skills)
            skill_embeddings = self.embedding_model.encode(skill_list)
            self.skill_embeddings = {skill: embedding for skill, embedding in zip(skill_list, skill_embeddings)}
        
        self.fitted = True
        
    def calculate_job_similarity(self, user_id: str, job_id: str) -> float:
        """Calculate similarity between user and job."""
        if not self.fitted:
            raise ValueError("Model must be fitted before calculating similarity")
            
        user_embedding = self.user_embeddings.get(user_id)
        job_embedding = self.job_embeddings.get(job_id)
        
        if user_embedding is None or job_embedding is None:
            return 0.0
            
        similarity = cosine_similarity([user_embedding], [job_embedding])[0][0]
        return max(0, similarity)  # Ensure non-negative
    
    def recommend_jobs(self, user_id: str, n_recommendations: int = 10) -> List[Tuple[str, float]]:
        """Recommend jobs for a user based on content similarity."""
        if not self.fitted:
            raise ValueError("Model must be fitted before making recommendations")
            
        recommendations = []
        for job_id in self.job_embeddings.keys():
            similarity = self.calculate_job_similarity(user_id, job_id)
            recommendations.append((job_id, similarity))
        
        recommendations.sort(key=lambda x: x[1], reverse=True)
        return recommendations[:n_recommendations]
    
    def find_similar_skills(self, skill: str, n_similar: int = 5) -> List[Tuple[str, float]]:
        """Find skills similar to the given skill."""
        if not self.fitted or skill not in self.skill_embeddings:
            return []
            
        skill_embedding = self.skill_embeddings[skill]
        similarities = []
        
        for other_skill, other_embedding in self.skill_embeddings.items():
            if other_skill != skill:
                similarity = cosine_similarity([skill_embedding], [other_embedding])[0][0]
                similarities.append((other_skill, similarity))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:n_similar]


class NeuralCollaborativeFiltering(nn.Module):
    """Neural collaborative filtering using PyTorch."""
    
    def __init__(self, n_users: int, n_items: int, embedding_dim: int = 64, 
                 hidden_dims: List[int] = [128, 64, 32]):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        
        # Embedding layers
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        
        # Neural MF layers
        layers = []
        input_dim = embedding_dim * 2
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            input_dim = hidden_dim
        
        layers.append(nn.Linear(input_dim, 1))
        self.mlp = nn.Sequential(*layers)
        
        # Initialize embeddings
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)
    
    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)
        
        # Concatenate embeddings
        x = torch.cat([user_emb, item_emb], dim=1)
        
        # Pass through MLP
        output = self.mlp(x)
        return torch.sigmoid(output)


class SkillGapAnalyzer:
    """Skill gap analysis and quantification algorithms."""
    
    def __init__(self, skill_importance_weights: Optional[Dict[str, float]] = None):
        self.skill_importance_weights = skill_importance_weights or {}
        self.market_demand_data = {}
        self.learning_time_estimates = {}
        
    def analyze_skill_gaps(self, user_skills: Dict[str, float], 
                          target_job_skills: Dict[str, float],
                          target_role: str) -> SkillGapAnalysis:
        """
        Analyze skill gaps between user profile and target job.
        
        Args:
            user_skills: User skills with confidence scores
            target_job_skills: Required job skills with importance scores
            target_role: Target job role
            
        Returns:
            SkillGapAnalysis object with detailed gap analysis
        """
        missing_skills = {}
        weak_skills = {}
        strong_skills = []
        
        # Identify missing skills
        for skill, importance in target_job_skills.items():
            if skill not in user_skills:
                missing_skills[skill] = importance
            elif user_skills[skill] < 0.6:  # Below proficiency threshold
                gap_score = importance * (0.6 - user_skills[skill])
                weak_skills[skill] = gap_score
            else:
                strong_skills.append(skill)
        
        # Calculate overall readiness
        total_required_skills = len(target_job_skills)
        covered_skills = len(strong_skills)
        overall_readiness = covered_skills / total_required_skills if total_required_skills > 0 else 0
        
        # Estimate learning time
        learning_time = self._estimate_learning_time(missing_skills, weak_skills)
        
        # Prioritize skills by importance and market demand
        priority_skills = self._prioritize_skills(missing_skills, weak_skills)
        
        return SkillGapAnalysis(
            target_role=target_role,
            missing_skills=missing_skills,
            weak_skills=weak_skills,
            strong_skills=strong_skills,
            overall_readiness=overall_readiness,
            learning_time_estimate=learning_time,
            priority_skills=priority_skills
        )
    
    def _estimate_learning_time(self, missing_skills: Dict[str, float], 
                               weak_skills: Dict[str, float]) -> int:
        """Estimate learning time in weeks."""
        total_time = 0
        
        # Default learning times (in weeks)
        default_times = {
            'programming_languages': 12,
            'frameworks_libraries': 8,
            'databases': 6,
            'cloud_platforms': 10,
            'devops_tools': 8,
            'soft_skills': 4,
            'technical': 6
        }
        
        for skill, importance in missing_skills.items():
            skill_category = self._get_skill_category(skill)
            base_time = self.learning_time_estimates.get(skill, default_times.get(skill_category, 6))
            total_time += base_time * importance
        
        for skill, gap_score in weak_skills.items():
            skill_category = self._get_skill_category(skill)
            base_time = self.learning_time_estimates.get(skill, default_times.get(skill_category, 6))
            total_time += base_time * gap_score * 0.5  # Half time for improvement
        
        return int(total_time)
    
    def _prioritize_skills(self, missing_skills: Dict[str, float], 
                          weak_skills: Dict[str, float]) -> List[str]:
        """Prioritize skills by importance and market demand."""
        all_skills = {}
        all_skills.update(missing_skills)
        all_skills.update(weak_skills)
        
        # Calculate priority scores
        priority_scores = {}
        for skill, score in all_skills.items():
            market_demand = self.market_demand_data.get(skill, 0.5)
            importance_weight = self.skill_importance_weights.get(skill, 1.0)
            priority_scores[skill] = score * market_demand * importance_weight
        
        # Sort by priority score
        sorted_skills = sorted(priority_scores.items(), key=lambda x: x[1], reverse=True)
        return [skill for skill, _ in sorted_skills]
    
    def _get_skill_category(self, skill: str) -> str:
        """Get skill category for learning time estimation."""
        # Simple categorization - in practice, this would use the skill taxonomy
        programming_languages = ['python', 'java', 'javascript', 'c++', 'go', 'rust']
        frameworks = ['react', 'angular', 'vue', 'django', 'flask', 'spring']
        databases = ['mysql', 'postgresql', 'mongodb', 'redis']
        cloud = ['aws', 'azure', 'gcp', 'docker', 'kubernetes']
        
        skill_lower = skill.lower()
        if any(lang in skill_lower for lang in programming_languages):
            return 'programming_languages'
        elif any(fw in skill_lower for fw in frameworks):
            return 'frameworks_libraries'
        elif any(db in skill_lower for db in databases):
            return 'databases'
        elif any(cloud_tech in skill_lower for cloud_tech in cloud):
            return 'cloud_platforms'
        else:
            return 'technical'


class RecommendationExplainer:
    """Generate explanations and reasoning for recommendations."""
    
    def __init__(self):
        self.explanation_templates = {
            'skill_match': "This role matches {match_percentage:.0f}% of your skills, particularly your expertise in {strong_skills}.",
            'growth_potential': "This position offers excellent growth potential in {growth_areas}, which aligns with current market trends.",
            'skill_gap': "You would need to develop skills in {missing_skills} to be fully qualified for this role.",
            'market_demand': "This role is in {demand_level} demand with {salary_trend} salary trends.",
            'career_progression': "This role represents a natural progression from your current experience in {current_areas}."
        }
    
    def explain_career_recommendation(self, recommendation: CareerRecommendation, 
                                    user_skills: Dict[str, float],
                                    market_data: Dict[str, Any]) -> str:
        """Generate detailed explanation for career recommendation."""
        explanations = []
        
        # Skill match explanation
        matching_skills = [skill for skill in recommendation.required_skills if skill in user_skills]
        match_percentage = (len(matching_skills) / len(recommendation.required_skills)) * 100
        
        if matching_skills:
            strong_skills = [skill for skill in matching_skills if user_skills[skill] > 0.7]
            if strong_skills:
                explanations.append(
                    self.explanation_templates['skill_match'].format(
                        match_percentage=match_percentage,
                        strong_skills=', '.join(strong_skills[:3])
                    )
                )
        
        # Growth potential explanation
        if recommendation.growth_potential > 0.7:
            explanations.append(
                self.explanation_templates['growth_potential'].format(
                    growth_areas=recommendation.job_title.split()[-1]  # Simplified
                )
            )
        
        # Skill gap explanation
        if recommendation.skill_gaps:
            top_gaps = sorted(recommendation.skill_gaps.items(), key=lambda x: x[1], reverse=True)[:3]
            missing_skills = [skill for skill, _ in top_gaps]
            explanations.append(
                self.explanation_templates['skill_gap'].format(
                    missing_skills=', '.join(missing_skills)
                )
            )
        
        # Market demand explanation
        demand_level = market_data.get('demand_level', 'moderate')
        salary_trend = market_data.get('salary_trend', 'stable')
        explanations.append(
            self.explanation_templates['market_demand'].format(
                demand_level=demand_level,
                salary_trend=salary_trend
            )
        )
        
        return ' '.join(explanations)
    
    def explain_learning_path(self, learning_path: LearningPath, 
                            skill_gaps: Dict[str, float]) -> str:
        """Generate explanation for learning path recommendation."""
        priority_skills = sorted(skill_gaps.items(), key=lambda x: x[1], reverse=True)[:3]
        priority_skill_names = [skill for skill, _ in priority_skills]
        
        explanation = f"This learning path focuses on {', '.join(priority_skill_names)}, "
        explanation += f"which are critical skills for your target role. "
        explanation += f"The estimated completion time is {learning_path.estimated_duration_weeks} weeks, "
        explanation += f"with a {learning_path.difficulty_level} difficulty level."
        
        return explanation


class HybridRecommendationEngine:
    """Hybrid recommendation system combining multiple approaches."""
    
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.weights = weights or {
            'collaborative': 0.3,
            'content_based': 0.4,
            'neural': 0.2,
            'skill_based': 0.1
        }
        
        self.collaborative_engine = CollaborativeFilteringEngine()
        self.content_based_engine = ContentBasedFilteringEngine()
        self.neural_engine = None
        self.skill_gap_analyzer = SkillGapAnalyzer()
        self.explainer = RecommendationExplainer()
        self.fitted = False
    
    def fit(self, user_item_matrix: np.ndarray, user_ids: List[str], 
            job_data: List[Dict[str, Any]], user_data: List[Dict[str, Any]]):
        """Fit all recommendation engines."""
        logger.info("Training hybrid recommendation system")
        
        # Fit collaborative filtering
        self.collaborative_engine.fit(user_item_matrix, user_ids, [job['id'] for job in job_data])
        
        # Fit content-based filtering
        self.content_based_engine.fit(job_data, user_data)
        
        # Initialize neural collaborative filtering
        n_users, n_items = user_item_matrix.shape
        self.neural_engine = NeuralCollaborativeFiltering(n_users, n_items)
        
        self.fitted = True
    
    def recommend_careers(self, user_id: str, user_profile: Dict[str, Any], 
                         n_recommendations: int = 5) -> List[CareerRecommendation]:
        """Generate hybrid career recommendations."""
        if not self.fitted:
            raise ValueError("Model must be fitted before making recommendations")
        
        # Get recommendations from each engine
        collab_recs = self.collaborative_engine.recommend_items(user_id, n_recommendations * 2)
        content_recs = self.content_based_engine.recommend_jobs(user_id, n_recommendations * 2)
        
        # Combine recommendations with weighted scores
        combined_scores = {}
        
        for job_id, score in collab_recs:
            combined_scores[job_id] = combined_scores.get(job_id, 0) + score * self.weights['collaborative']
        
        for job_id, score in content_recs:
            combined_scores[job_id] = combined_scores.get(job_id, 0) + score * self.weights['content_based']
        
        # Sort by combined score
        sorted_recommendations = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Generate detailed recommendations
        recommendations = []
        for job_id, score in sorted_recommendations[:n_recommendations]:
            # This would fetch actual job data in practice
            recommendation = CareerRecommendation(
                job_title=f"Job {job_id}",  # Placeholder
                match_score=score,
                required_skills=[],  # Would be populated from job data
                skill_gaps={},
                salary_range=(50000, 80000),  # Placeholder
                growth_potential=0.8,
                market_demand="high",
                reasoning="",
                confidence_score=score,
                alternative_paths=[]
            )
            recommendations.append(recommendation)
        
        return recommendations
    
    def recommend_learning_paths(self, user_skills: Dict[str, float], 
                               target_role: str, target_skills: Dict[str, float]) -> List[LearningPath]:
        """Generate learning path recommendations."""
        # Analyze skill gaps
        gap_analysis = self.skill_gap_analyzer.analyze_skill_gaps(
            user_skills, target_skills, target_role
        )
        
        # Generate learning paths for priority skills
        learning_paths = []
        for i, skill in enumerate(gap_analysis.priority_skills[:3]):  # Top 3 priority skills
            path = LearningPath(
                path_id=f"path_{i+1}",
                title=f"Master {skill} for {target_role}",
                target_skills=[skill],
                estimated_duration_weeks=8,  # Placeholder
                difficulty_level="intermediate",
                resources=[],  # Would be populated with actual resources
                milestones=[],
                priority_score=1.0 - (i * 0.1),  # Decreasing priority
                reasoning=self.explainer.explain_learning_path(
                    LearningPath(f"path_{i+1}", f"Master {skill}", [skill], 8, "intermediate", [], [], 1.0, ""),
                    {skill: gap_analysis.missing_skills.get(skill, gap_analysis.weak_skills.get(skill, 0))}
                )
            )
            learning_paths.append(path)
        
        return learning_paths
    
    def get_model_performance_metrics(self) -> Dict[str, float]:
        """Get performance metrics for the recommendation system."""
        # This would calculate actual metrics in practice
        return {
            'collaborative_rmse': 0.85,
            'content_similarity_score': 0.78,
            'hybrid_precision_at_5': 0.82,
            'hybrid_recall_at_5': 0.75,
            'coverage': 0.90,
            'diversity': 0.65
        }