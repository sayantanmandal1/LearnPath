"""
Tests for the recommendation engine algorithms.
"""

import pytest
import numpy as np
import torch
from unittest.mock import Mock, patch
from typing import Dict, List

from machinelearningmodel.recommendation_engine import (
    CollaborativeFilteringEngine,
    ContentBasedFilteringEngine,
    NeuralCollaborativeFiltering,
    SkillGapAnalyzer,
    RecommendationExplainer,
    HybridRecommendationEngine,
    CareerRecommendation,
    LearningPath,
    SkillGapAnalysis
)


class TestCollaborativeFilteringEngine:
    """Test collaborative filtering engine."""
    
    def test_initialization(self):
        """Test engine initialization."""
        engine = CollaborativeFilteringEngine(n_factors=10, learning_rate=0.05)
        assert engine.n_factors == 10
        assert engine.learning_rate == 0.05
        assert not engine.fitted
    
    def test_fit_and_predict(self):
        """Test model fitting and prediction."""
        # Create sample user-item matrix
        user_item_matrix = np.array([
            [5, 3, 0, 1],
            [4, 0, 0, 1],
            [1, 1, 0, 5],
            [1, 0, 0, 4],
            [0, 1, 5, 4]
        ], dtype=float)
        
        user_ids = ['user1', 'user2', 'user3', 'user4', 'user5']
        item_ids = ['item1', 'item2', 'item3', 'item4']
        
        engine = CollaborativeFilteringEngine(n_factors=5, n_iterations=10)
        engine.fit(user_item_matrix, user_ids, item_ids)
        
        assert engine.fitted
        assert engine.user_factors.shape == (5, 5)
        assert engine.item_factors.shape == (4, 5)
        
        # Test prediction
        prediction = engine.predict('user1', 'item3')
        assert isinstance(prediction, float)
        assert 0 <= prediction <= 5
    
    def test_recommend_items(self):
        """Test item recommendation."""
        user_item_matrix = np.array([
            [5, 3, 0, 1],
            [4, 0, 0, 1],
            [1, 1, 0, 5]
        ], dtype=float)
        
        user_ids = ['user1', 'user2', 'user3']
        item_ids = ['item1', 'item2', 'item3', 'item4']
        
        engine = CollaborativeFilteringEngine(n_factors=3, n_iterations=5)
        engine.fit(user_item_matrix, user_ids, item_ids)
        
        recommendations = engine.recommend_items('user1', n_recommendations=2)
        assert len(recommendations) == 2
        assert all(isinstance(item_id, str) and isinstance(score, float) for item_id, score in recommendations)
    
    def test_predict_unknown_user_item(self):
        """Test prediction for unknown user/item."""
        user_item_matrix = np.array([[5, 3], [4, 0]], dtype=float)
        user_ids = ['user1', 'user2']
        item_ids = ['item1', 'item2']
        
        engine = CollaborativeFilteringEngine(n_factors=2, n_iterations=5)
        engine.fit(user_item_matrix, user_ids, item_ids)
        
        # Test unknown user
        prediction = engine.predict('unknown_user', 'item1')
        assert prediction == engine.global_bias
        
        # Test unknown item
        prediction = engine.predict('user1', 'unknown_item')
        assert prediction == engine.global_bias


class TestContentBasedFilteringEngine:
    """Test content-based filtering engine."""
    
    @patch('machinelearningmodel.recommendation_engine.SentenceTransformer')
    def test_initialization(self, mock_transformer):
        """Test engine initialization."""
        engine = ContentBasedFilteringEngine()
        assert engine.embedding_model_name == "sentence-transformers/all-MiniLM-L6-v2"
        assert not engine.fitted
    
    @patch('machinelearningmodel.recommendation_engine.SentenceTransformer')
    def test_fit(self, mock_transformer):
        """Test model fitting."""
        # Mock the sentence transformer
        mock_model = Mock()
        mock_model.encode.return_value = np.random.rand(3, 384)  # Mock embeddings
        mock_transformer.return_value = mock_model
        
        job_data = [
            {'id': 'job1', 'title': 'Software Engineer', 'description': 'Python development', 'skills': ['python', 'django']},
            {'id': 'job2', 'title': 'Data Scientist', 'description': 'Machine learning', 'skills': ['python', 'sklearn']},
            {'id': 'job3', 'title': 'Frontend Developer', 'description': 'React development', 'skills': ['javascript', 'react']}
        ]
        
        user_data = [
            {'id': 'user1', 'current_role': 'Junior Developer', 'dream_job': 'Senior Engineer', 'skills': ['python', 'flask']},
            {'id': 'user2', 'current_role': 'Analyst', 'dream_job': 'Data Scientist', 'skills': ['python', 'pandas']},
            {'id': 'user3', 'current_role': 'Designer', 'dream_job': 'Full Stack', 'skills': ['html', 'css']}
        ]
        
        engine = ContentBasedFilteringEngine()
        engine.fit(job_data, user_data)
        
        assert engine.fitted
        assert len(engine.job_embeddings) == 3
        assert len(engine.user_embeddings) == 3
        assert len(engine.skill_embeddings) > 0
    
    @patch('machinelearningmodel.recommendation_engine.SentenceTransformer')
    def test_calculate_job_similarity(self, mock_transformer):
        """Test job similarity calculation."""
        mock_model = Mock()
        mock_model.encode.return_value = np.random.rand(2, 384)
        mock_transformer.return_value = mock_model
        
        job_data = [{'id': 'job1', 'title': 'Engineer', 'description': 'Python', 'skills': ['python']}]
        user_data = [{'id': 'user1', 'current_role': 'Developer', 'dream_job': 'Engineer', 'skills': ['python']}]
        
        engine = ContentBasedFilteringEngine()
        engine.fit(job_data, user_data)
        
        similarity = engine.calculate_job_similarity('user1', 'job1')
        assert isinstance(similarity, float)
        assert 0 <= similarity <= 1.01  # Allow for small floating point errors
    
    @patch('machinelearningmodel.recommendation_engine.SentenceTransformer')
    def test_recommend_jobs(self, mock_transformer):
        """Test job recommendation."""
        mock_model = Mock()
        mock_model.encode.return_value = np.random.rand(3, 384)
        mock_transformer.return_value = mock_model
        
        job_data = [
            {'id': 'job1', 'title': 'Engineer', 'description': 'Python', 'skills': ['python']},
            {'id': 'job2', 'title': 'Analyst', 'description': 'Data', 'skills': ['sql']},
            {'id': 'job3', 'title': 'Designer', 'description': 'UI', 'skills': ['figma']}
        ]
        user_data = [{'id': 'user1', 'current_role': 'Developer', 'dream_job': 'Engineer', 'skills': ['python']}]
        
        engine = ContentBasedFilteringEngine()
        engine.fit(job_data, user_data)
        
        recommendations = engine.recommend_jobs('user1', n_recommendations=2)
        assert len(recommendations) == 2
        assert all(isinstance(job_id, str) and isinstance(score, float) for job_id, score in recommendations)


class TestNeuralCollaborativeFiltering:
    """Test neural collaborative filtering."""
    
    def test_initialization(self):
        """Test model initialization."""
        model = NeuralCollaborativeFiltering(n_users=100, n_items=50, embedding_dim=32)
        assert model.n_users == 100
        assert model.n_items == 50
        assert model.embedding_dim == 32
    
    def test_forward_pass(self):
        """Test forward pass."""
        model = NeuralCollaborativeFiltering(n_users=10, n_items=5, embedding_dim=8)
        
        user_ids = torch.tensor([0, 1, 2])
        item_ids = torch.tensor([0, 1, 2])
        
        output = model(user_ids, item_ids)
        assert output.shape == (3, 1)
        assert torch.all(output >= 0) and torch.all(output <= 1)  # Sigmoid output


class TestSkillGapAnalyzer:
    """Test skill gap analyzer."""
    
    def test_initialization(self):
        """Test analyzer initialization."""
        analyzer = SkillGapAnalyzer()
        assert isinstance(analyzer.skill_importance_weights, dict)
    
    def test_analyze_skill_gaps(self):
        """Test skill gap analysis."""
        analyzer = SkillGapAnalyzer()
        
        user_skills = {
            'python': 0.8,
            'sql': 0.6,
            'javascript': 0.4
        }
        
        target_job_skills = {
            'python': 0.9,
            'sql': 0.7,
            'javascript': 0.8,
            'react': 0.6,
            'aws': 0.5
        }
        
        analysis = analyzer.analyze_skill_gaps(user_skills, target_job_skills, 'Full Stack Developer')
        
        assert isinstance(analysis, SkillGapAnalysis)
        assert analysis.target_role == 'Full Stack Developer'
        assert 'react' in analysis.missing_skills
        assert 'aws' in analysis.missing_skills
        assert 'javascript' in analysis.weak_skills
        assert 'python' in analysis.strong_skills
        assert 0 <= analysis.overall_readiness <= 1
        assert analysis.learning_time_estimate > 0
    
    def test_prioritize_skills(self):
        """Test skill prioritization."""
        analyzer = SkillGapAnalyzer()
        analyzer.market_demand_data = {'react': 0.9, 'aws': 0.8, 'javascript': 0.7}
        
        missing_skills = {'react': 0.8, 'aws': 0.6}
        weak_skills = {'javascript': 0.4}
        
        priority_skills = analyzer._prioritize_skills(missing_skills, weak_skills)
        assert isinstance(priority_skills, list)
        assert len(priority_skills) == 3
        assert 'react' in priority_skills  # Should be high priority due to high market demand


class TestRecommendationExplainer:
    """Test recommendation explainer."""
    
    def test_initialization(self):
        """Test explainer initialization."""
        explainer = RecommendationExplainer()
        assert isinstance(explainer.explanation_templates, dict)
    
    def test_explain_career_recommendation(self):
        """Test career recommendation explanation."""
        explainer = RecommendationExplainer()
        
        recommendation = CareerRecommendation(
            job_title='Senior Python Developer',
            match_score=0.85,
            required_skills=['python', 'django', 'postgresql'],
            skill_gaps={'kubernetes': 0.3},
            salary_range=(80000, 120000),
            growth_potential=0.8,
            market_demand='high',
            reasoning='',
            confidence_score=0.85,
            alternative_paths=[]
        )
        
        user_skills = {'python': 0.9, 'django': 0.7, 'sql': 0.6}
        market_data = {'demand_level': 'high', 'salary_trend': 'increasing'}
        
        explanation = explainer.explain_career_recommendation(recommendation, user_skills, market_data)
        assert isinstance(explanation, str)
        assert len(explanation) > 0
        assert 'python' in explanation.lower()
    
    def test_explain_learning_path(self):
        """Test learning path explanation."""
        explainer = RecommendationExplainer()
        
        learning_path = LearningPath(
            path_id='path1',
            title='Master React Development',
            target_skills=['react'],
            estimated_duration_weeks=8,
            difficulty_level='intermediate',
            resources=[],
            milestones=[],
            priority_score=0.9,
            reasoning=''
        )
        
        skill_gaps = {'react': 0.8, 'redux': 0.6}
        
        explanation = explainer.explain_learning_path(learning_path, skill_gaps)
        assert isinstance(explanation, str)
        assert len(explanation) > 0
        assert 'react' in explanation.lower()
        assert '8 weeks' in explanation


class TestHybridRecommendationEngine:
    """Test hybrid recommendation engine."""
    
    def test_initialization(self):
        """Test engine initialization."""
        engine = HybridRecommendationEngine()
        assert isinstance(engine.weights, dict)
        assert 'collaborative' in engine.weights
        assert 'content_based' in engine.weights
        assert not engine.fitted
    
    @patch('machinelearningmodel.recommendation_engine.SentenceTransformer')
    def test_fit(self, mock_transformer):
        """Test hybrid engine fitting."""
        mock_model = Mock()
        mock_model.encode.return_value = np.random.rand(2, 384)
        mock_transformer.return_value = mock_model
        
        user_item_matrix = np.array([[5, 3], [4, 0]], dtype=float)
        user_ids = ['user1', 'user2']
        
        job_data = [
            {'id': 'job1', 'title': 'Engineer', 'description': 'Python', 'skills': ['python']},
            {'id': 'job2', 'title': 'Analyst', 'description': 'Data', 'skills': ['sql']}
        ]
        
        user_data = [
            {'id': 'user1', 'current_role': 'Developer', 'skills': ['python']},
            {'id': 'user2', 'current_role': 'Analyst', 'skills': ['sql']}
        ]
        
        engine = HybridRecommendationEngine()
        engine.fit(user_item_matrix, user_ids, job_data, user_data)
        
        assert engine.fitted
        assert engine.collaborative_engine.fitted
        assert engine.content_based_engine.fitted
    
    def test_get_model_performance_metrics(self):
        """Test performance metrics retrieval."""
        engine = HybridRecommendationEngine()
        metrics = engine.get_model_performance_metrics()
        
        assert isinstance(metrics, dict)
        assert 'collaborative_rmse' in metrics
        assert 'content_similarity_score' in metrics
        assert 'hybrid_precision_at_5' in metrics
        assert all(isinstance(value, float) for value in metrics.values())


@pytest.fixture
def sample_user_item_matrix():
    """Sample user-item interaction matrix."""
    return np.array([
        [5, 3, 0, 1, 0],
        [4, 0, 0, 1, 2],
        [1, 1, 0, 5, 0],
        [1, 0, 0, 4, 3],
        [0, 1, 5, 4, 0]
    ], dtype=float)


@pytest.fixture
def sample_job_data():
    """Sample job data."""
    return [
        {
            'id': 'job1',
            'title': 'Senior Python Developer',
            'description': 'Develop web applications using Python and Django',
            'skills': ['python', 'django', 'postgresql', 'redis']
        },
        {
            'id': 'job2',
            'title': 'Data Scientist',
            'description': 'Build machine learning models and analyze data',
            'skills': ['python', 'sklearn', 'pandas', 'tensorflow']
        },
        {
            'id': 'job3',
            'title': 'Frontend Developer',
            'description': 'Create responsive web interfaces using React',
            'skills': ['javascript', 'react', 'css', 'html']
        }
    ]


@pytest.fixture
def sample_user_data():
    """Sample user data."""
    return [
        {
            'id': 'user1',
            'current_role': 'Junior Python Developer',
            'dream_job': 'Senior Python Developer',
            'skills': ['python', 'flask', 'mysql']
        },
        {
            'id': 'user2',
            'current_role': 'Data Analyst',
            'dream_job': 'Data Scientist',
            'skills': ['python', 'pandas', 'sql']
        },
        {
            'id': 'user3',
            'current_role': 'Web Designer',
            'dream_job': 'Frontend Developer',
            'skills': ['html', 'css', 'photoshop']
        }
    ]


class TestIntegration:
    """Integration tests for the recommendation system."""
    
    @patch('machinelearningmodel.recommendation_engine.SentenceTransformer')
    def test_end_to_end_recommendation_flow(self, mock_transformer, sample_user_item_matrix, 
                                          sample_job_data, sample_user_data):
        """Test complete recommendation flow."""
        # Mock sentence transformer
        mock_model = Mock()
        mock_model.encode.return_value = np.random.rand(len(sample_job_data), 384)
        mock_transformer.return_value = mock_model
        
        # Initialize and fit hybrid engine
        engine = HybridRecommendationEngine()
        user_ids = ['user1', 'user2', 'user3', 'user4', 'user5']
        
        engine.fit(sample_user_item_matrix, user_ids, sample_job_data, sample_user_data)
        
        # Test career recommendations
        user_profile = sample_user_data[0]
        recommendations = engine.recommend_careers('user1', user_profile, n_recommendations=2)
        
        assert len(recommendations) == 2
        assert all(isinstance(rec, CareerRecommendation) for rec in recommendations)
        
        # Test learning path recommendations
        user_skills = {'python': 0.7, 'flask': 0.6}
        target_skills = {'python': 0.9, 'django': 0.8, 'postgresql': 0.7}
        
        learning_paths = engine.recommend_learning_paths(user_skills, 'Senior Python Developer', target_skills)
        
        assert len(learning_paths) > 0
        assert all(isinstance(path, LearningPath) for path in learning_paths)
    
    def test_skill_gap_analysis_integration(self):
        """Test skill gap analysis integration."""
        analyzer = SkillGapAnalyzer()
        
        user_skills = {
            'python': 0.8,
            'sql': 0.6,
            'html': 0.7
        }
        
        target_job_skills = {
            'python': 0.9,
            'django': 0.8,
            'postgresql': 0.7,
            'redis': 0.5,
            'docker': 0.6
        }
        
        analysis = analyzer.analyze_skill_gaps(user_skills, target_job_skills, 'Full Stack Developer')
        
        # Verify analysis results
        assert analysis.target_role == 'Full Stack Developer'
        assert len(analysis.missing_skills) > 0
        assert len(analysis.priority_skills) > 0
        assert analysis.learning_time_estimate > 0
        assert 0 <= analysis.overall_readiness <= 1
        
        # Verify that missing skills are correctly identified
        expected_missing = {'django', 'postgresql', 'redis', 'docker'}
        actual_missing = set(analysis.missing_skills.keys())
        assert expected_missing.issubset(actual_missing) or len(actual_missing.intersection(expected_missing)) > 0