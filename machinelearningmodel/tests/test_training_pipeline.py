"""
Tests for the ML training pipeline components.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from machinelearningmodel.training.data_preparation import (
    DataPreprocessor, FeatureEngineer, UserItemMatrixBuilder,
    TrainingDatasetBuilder, create_synthetic_training_data
)
from machinelearningmodel.training.model_trainer import (
    CollaborativeFilteringTrainer, TrainingConfig
)
from machinelearningmodel.training.model_evaluator import (
    ClassificationEvaluator, RecommendationEvaluator, EvaluationMetrics
)
from machinelearningmodel.training.ab_testing import (
    ABTestFramework, ExperimentConfig, ExperimentVariant,
    TrafficSplitMethod, create_model_comparison_experiment
)
from machinelearningmodel.training.continuous_learning import (
    ContinuousLearningEngine, OnlineRecommendationModel,
    UserFeedback, FeedbackType, create_feedback_simulation
)
from machinelearningmodel.training.model_versioning import (
    ModelRegistry, ModelVersion, ModelStatus, create_model_version_from_training
)
from machinelearningmodel.training.training_pipeline import (
    MLTrainingPipeline, PipelineConfig
)


class TestDataPreparation:
    """Test data preparation components."""
    
    def test_data_preprocessor_initialization(self):
        """Test DataPreprocessor initialization."""
        config = {'remove_personal_info': True}
        preprocessor = DataPreprocessor(config)
        
        assert preprocessor.config == config
        assert not preprocessor.fitted
    
    def test_clean_text_data(self):
        """Test text cleaning functionality."""
        preprocessor = DataPreprocessor()
        
        texts = [
            "Contact me at john@example.com or 555-123-4567",
            "I live at 123 Main Street",
            "I know javascript and python"
        ]
        
        cleaned = preprocessor.clean_text_data(texts)
        
        assert "[EMAIL]" in cleaned[0]
        assert "[PHONE]" in cleaned[0]
        assert "[ADDRESS]" in cleaned[1]
        assert "JavaScript" in cleaned[2]
        assert "Python" in cleaned[2]
    
    def test_extract_numerical_features(self):
        """Test numerical feature extraction."""
        preprocessor = DataPreprocessor()
        
        data = [
            {
                'experience_years': 5,
                'skills': ['Python', 'SQL'],
                'education_level': 'bachelor',
                'github_stats': {'public_repos': 10, 'followers': 5},
                'leetcode_stats': {'problems_solved': 100, 'contest_rating': 1500}
            }
        ]
        
        features = preprocessor.extract_numerical_features(data)
        
        assert features.shape[0] == 1
        assert features.shape[1] > 5  # Should have multiple features
        assert features[0, 0] == 5  # experience_years
    
    def test_user_item_matrix_builder(self):
        """Test user-item matrix building."""
        builder = UserItemMatrixBuilder(min_interactions=1)
        
        interactions = [
            {'user_id': 'user1', 'item_id': 'item1', 'rating': 5.0},
            {'user_id': 'user1', 'item_id': 'item2', 'rating': 4.0},
            {'user_id': 'user2', 'item_id': 'item1', 'rating': 3.0}
        ]
        
        matrix = builder.build_matrix(interactions)
        
        assert matrix.matrix.shape == (2, 2)  # 2 users, 2 items
        assert len(matrix.user_ids) == 2
        assert len(matrix.item_ids) == 2
        assert matrix.metadata['n_interactions'] == 3
    
    def test_create_synthetic_training_data(self):
        """Test synthetic data generation."""
        data = create_synthetic_training_data(n_users=10, n_items=5, n_interactions=20)
        
        assert len(data['users']) == 10
        assert len(data['items']) == 5
        assert len(data['interactions']) == 20
        assert 'skill_taxonomy' in data


class TestModelTraining:
    """Test model training components."""
    
    def test_training_config(self):
        """Test training configuration."""
        config = TrainingConfig(
            model_type='test_model',
            hyperparameters={'param1': 1},
            training_params={'epochs': 10},
            validation_params={}
        )
        
        assert config.model_type == 'test_model'
        assert config.hyperparameters['param1'] == 1
        assert config.early_stopping is True
    
    @patch('machinelearningmodel.training.model_trainer.mlflow')
    def test_collaborative_filtering_trainer(self, mock_mlflow):
        """Test collaborative filtering trainer."""
        config = TrainingConfig(
            model_type='collaborative_filtering',
            hyperparameters={
                'n_factors': 10,
                'learning_rate': 0.1,
                'regularization': 0.01,
                'n_iterations': 5
            },
            training_params={},
            validation_params={}
        )
        
        trainer = CollaborativeFilteringTrainer(config)
        
        # Create simple user-item matrix
        from machinelearningmodel.training.data_preparation import UserItemMatrix
        import scipy.sparse as sp
        
        matrix = sp.csr_matrix(np.array([[5, 0, 3], [4, 0, 0], [0, 2, 4]]))
        user_item_matrix = UserItemMatrix(
            matrix=matrix,
            user_ids=['user1', 'user2', 'user3'],
            item_ids=['item1', 'item2', 'item3'],
            user_id_to_idx={'user1': 0, 'user2': 1, 'user3': 2},
            item_id_to_idx={'item1': 0, 'item2': 1, 'item3': 2},
            metadata={'n_users': 3, 'n_items': 3}
        )
        
        # Mock MLflow context manager
        mock_mlflow.start_run.return_value.__enter__ = Mock()
        mock_mlflow.start_run.return_value.__exit__ = Mock()
        
        result = trainer.train(user_item_matrix)
        
        assert result.model_type == 'collaborative_filtering'
        assert 'rmse' in result.training_metrics
        assert result.training_time > 0


class TestModelEvaluation:
    """Test model evaluation components."""
    
    def test_classification_evaluator(self):
        """Test classification evaluation."""
        evaluator = ClassificationEvaluator()
        
        y_true = np.array([0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0, 1])
        
        metrics = evaluator.evaluate(y_true, y_pred)
        
        assert metrics.accuracy is not None
        assert metrics.precision is not None
        assert metrics.recall is not None
        assert metrics.f1_score is not None
        assert 0 <= metrics.accuracy <= 1
    
    def test_recommendation_evaluator(self):
        """Test recommendation evaluation."""
        evaluator = RecommendationEvaluator(k_values=[1, 3, 5])
        
        recommendations = {
            'user1': [('item1', 0.9), ('item2', 0.8), ('item3', 0.7)],
            'user2': [('item2', 0.95), ('item1', 0.85), ('item4', 0.75)]
        }
        
        ground_truth = {
            'user1': ['item1', 'item3'],
            'user2': ['item2']
        }
        
        metrics = evaluator.evaluate_recommendations(recommendations, ground_truth)
        
        assert 1 in metrics.precision_at_k
        assert 3 in metrics.precision_at_k
        assert 5 in metrics.precision_at_k
        assert metrics.map_score >= 0
        assert metrics.coverage >= 0


class TestABTesting:
    """Test A/B testing components."""
    
    def test_ab_test_framework_initialization(self):
        """Test A/B test framework initialization."""
        framework = ABTestFramework()
        
        assert framework.traffic_splitter is not None
        assert framework.experiment_tracker is not None
        assert framework.statistical_analyzer is not None
    
    def test_create_experiment(self):
        """Test experiment creation."""
        framework = ABTestFramework()
        
        variants = [
            ExperimentVariant(
                variant_id='control',
                name='Control',
                description='Control variant',
                model_config={'model': 'baseline'},
                traffic_allocation=0.5,
                is_control=True
            ),
            ExperimentVariant(
                variant_id='treatment',
                name='Treatment',
                description='Treatment variant',
                model_config={'model': 'new'},
                traffic_allocation=0.5,
                is_control=False
            )
        ]
        
        config = ExperimentConfig(
            experiment_id='test_exp',
            name='Test Experiment',
            description='Test experiment',
            variants=variants,
            primary_metric='accuracy',
            secondary_metrics=['precision'],
            minimum_sample_size=100,
            significance_level=0.05,
            power=0.8,
            traffic_split_method=TrafficSplitMethod.HASH_BASED,
            start_date=datetime.now()
        )
        
        experiment_id = framework.create_experiment(config)
        assert experiment_id == 'test_exp'
        assert 'test_exp' in framework.active_experiments
    
    def test_user_assignment(self):
        """Test user assignment to variants."""
        framework = ABTestFramework()
        
        # Create experiment first
        variants = [
            ExperimentVariant('control', 'Control', 'Control', {}, 0.5, True),
            ExperimentVariant('treatment', 'Treatment', 'Treatment', {}, 0.5, False)
        ]
        
        config = ExperimentConfig(
            experiment_id='test_exp',
            name='Test',
            description='Test',
            variants=variants,
            primary_metric='accuracy',
            secondary_metrics=[],
            minimum_sample_size=100,
            significance_level=0.05,
            power=0.8,
            traffic_split_method=TrafficSplitMethod.HASH_BASED,
            start_date=datetime.now()
        )
        
        framework.create_experiment(config)
        framework.start_experiment('test_exp')
        
        # Test user assignment
        variant = framework.assign_user_to_experiment('user123', 'test_exp')
        assert variant in ['control', 'treatment']
        
        # Same user should get same variant
        variant2 = framework.assign_user_to_experiment('user123', 'test_exp')
        assert variant == variant2


class TestContinuousLearning:
    """Test continuous learning components."""
    
    def test_online_recommendation_model(self):
        """Test online recommendation model."""
        model = OnlineRecommendationModel('test_model')
        
        assert model.model_id == 'test_model'
        assert not model.is_fitted
        
        # Test partial fit
        X = np.random.randn(10, 5)
        y = np.random.randn(10)
        
        success = model.partial_fit(X, y)
        assert success
        assert model.is_fitted
        assert model.update_count == 10
        
        # Test prediction
        predictions = model.predict(X)
        assert len(predictions) == 10
    
    def test_feedback_processing(self):
        """Test feedback processing."""
        engine = ContinuousLearningEngine()
        
        feedback = UserFeedback(
            user_id='user1',
            item_id='item1',
            feedback_type=FeedbackType.EXPLICIT,
            feedback_value=4.0,
            context={'device': 'mobile'},
            timestamp=datetime.now()
        )
        
        engine.add_feedback(feedback)
        
        # Check feedback was processed
        recent_feedback = engine.feedback_processor.get_recent_feedback(hours=1)
        assert len(recent_feedback) == 1
        assert recent_feedback[0].user_id == 'user1'
    
    def test_create_feedback_simulation(self):
        """Test feedback simulation creation."""
        feedback_list = create_feedback_simulation(n_users=5, n_items=3, n_feedback=10)
        
        assert len(feedback_list) == 10
        assert all(isinstance(fb, UserFeedback) for fb in feedback_list)
        assert all(fb.user_id.startswith('user_') for fb in feedback_list)


class TestModelVersioning:
    """Test model versioning components."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.registry = ModelRegistry(self.temp_dir)
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def test_model_registry_initialization(self):
        """Test model registry initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            registry = ModelRegistry(temp_dir)
            
            assert registry.registry_path.exists()
            assert isinstance(registry.models, dict)
    
    def test_model_version_creation(self):
        """Test model version creation."""
        from machinelearningmodel.training.model_trainer import TrainingResult
        
        training_result = TrainingResult(
            model_id='test_model',
            model_type='test',
            training_metrics={'accuracy': 0.8},
            validation_metrics={'accuracy': 0.75},
            test_metrics={'accuracy': 0.77},
            best_epoch=10,
            training_time=100.0,
            hyperparameters={'param1': 1},
            model_path='/tmp/model.pkl',
            metadata={}
        )
        
        eval_metrics = EvaluationMetrics(accuracy=0.77)
        
        model_version = create_model_version_from_training(
            training_result, eval_metrics, 'test_model'
        )
        
        assert model_version.model_id == 'test_model'
        assert model_version.model_type == 'test_model'
        assert model_version.status == ModelStatus.TRAINING
        assert model_version.training_result == training_result


class TestTrainingPipeline:
    """Test complete training pipeline."""
    
    def test_pipeline_config(self):
        """Test pipeline configuration."""
        config = PipelineConfig(
            data_sources={'test': 'data'},
            preprocessing_config={},
            feature_engineering_config={},
            models_to_train=['test_model'],
            hyperparameter_optimization=False,
            optimization_trials=10,
            evaluation_metrics=['accuracy'],
            cross_validation_folds=5,
            test_size=0.2,
            enable_ab_testing=False,
            ab_test_duration_days=7,
            significance_level=0.05,
            enable_continuous_learning=False,
            learning_mode='online',
            feedback_window_hours=24,
            enable_auto_deployment=False,
            deployment_strategy='blue_green',
            auto_deploy_to_staging=False,
            auto_deploy_to_production=False,
            output_directory='./test_output',
            max_workers=1,
            enable_gpu=False
        )
        
        assert config.models_to_train == ['test_model']
        assert config.test_size == 0.2
        assert not config.enable_ab_testing
    
    def test_pipeline_initialization(self):
        """Test pipeline initialization."""
        config = PipelineConfig(
            data_sources={},
            preprocessing_config={},
            feature_engineering_config={},
            models_to_train=[],
            hyperparameter_optimization=False,
            optimization_trials=10,
            evaluation_metrics=[],
            cross_validation_folds=5,
            test_size=0.2,
            enable_ab_testing=False,
            ab_test_duration_days=7,
            significance_level=0.05,
            enable_continuous_learning=False,
            learning_mode='online',
            feedback_window_hours=24,
            enable_auto_deployment=False,
            deployment_strategy='blue_green',
            auto_deploy_to_staging=False,
            auto_deploy_to_production=False,
            output_directory='./test_output',
            max_workers=1,
            enable_gpu=False
        )
        
        with tempfile.TemporaryDirectory() as temp_dir:
            config.output_directory = temp_dir
            pipeline = MLTrainingPipeline(config)
            
            assert pipeline.config == config
            assert pipeline.output_dir.exists()
            assert pipeline.data_preprocessor is not None
            assert pipeline.pipeline_state['status'] == 'initialized'
    
    def test_pipeline_status(self):
        """Test pipeline status tracking."""
        config = PipelineConfig(
            data_sources={},
            preprocessing_config={},
            feature_engineering_config={},
            models_to_train=[],
            hyperparameter_optimization=False,
            optimization_trials=10,
            evaluation_metrics=[],
            cross_validation_folds=5,
            test_size=0.2,
            enable_ab_testing=False,
            ab_test_duration_days=7,
            significance_level=0.05,
            enable_continuous_learning=False,
            learning_mode='online',
            feedback_window_hours=24,
            enable_auto_deployment=False,
            deployment_strategy='blue_green',
            auto_deploy_to_staging=False,
            auto_deploy_to_production=False,
            output_directory='./test_output',
            max_workers=1,
            enable_gpu=False
        )
        
        with tempfile.TemporaryDirectory() as temp_dir:
            config.output_directory = temp_dir
            pipeline = MLTrainingPipeline(config)
            
            status = pipeline.get_pipeline_status()
            
            assert status['status'] == 'initialized'
            assert status['current_step'] is None
            assert status['duration_seconds'] == 0


@pytest.fixture
def sample_training_data():
    """Fixture providing sample training data."""
    return create_synthetic_training_data(n_users=10, n_items=5, n_interactions=20)


@pytest.fixture
def temp_directory():
    """Fixture providing temporary directory."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


class TestIntegration:
    """Integration tests for the complete pipeline."""
    
    @pytest.mark.asyncio
    async def test_minimal_pipeline_run(self, temp_directory):
        """Test minimal pipeline run."""
        config = PipelineConfig(
            data_sources={'synthetic': 'demo'},
            preprocessing_config={},
            feature_engineering_config={},
            models_to_train=['collaborative_filtering'],
            hyperparameter_optimization=False,
            optimization_trials=5,
            evaluation_metrics=['accuracy'],
            cross_validation_folds=3,
            test_size=0.2,
            enable_ab_testing=False,
            ab_test_duration_days=1,
            significance_level=0.05,
            enable_continuous_learning=False,
            learning_mode='online',
            feedback_window_hours=24,
            enable_auto_deployment=False,
            deployment_strategy='blue_green',
            auto_deploy_to_staging=False,
            auto_deploy_to_production=False,
            output_directory=temp_directory,
            max_workers=1,
            enable_gpu=False
        )
        
        pipeline = MLTrainingPipeline(config)
        
        # Test data preparation step
        datasets = await pipeline._prepare_data()
        
        assert 'user_item_matrix' in datasets
        assert 'processed_data' in datasets
        
        # Test training step
        training_results = await pipeline._train_models(datasets)
        
        assert 'collaborative_filtering' in training_results
        
        # Check pipeline status
        status = pipeline.get_pipeline_status()
        assert status['status'] == 'running'


if __name__ == '__main__':
    pytest.main([__file__])