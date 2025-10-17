"""
Complete ML training pipeline orchestrator.

This module orchestrates the entire ML training pipeline including data preparation,
model training, evaluation, A/B testing, continuous learning, and deployment.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
import json
import os
from pathlib import Path
from dataclasses import dataclass, asdict
import asyncio
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor

from .data_preparation import (
    DataPreprocessor, FeatureEngineer, UserItemMatrixBuilder, 
    TrainingDatasetBuilder, create_synthetic_training_data
)
from .model_trainer import (
    ModelTrainingPipeline, CollaborativeFilteringTrainer,
    NeuralCollaborativeFilteringTrainer, SkillClassificationTrainer,
    HyperparameterOptimizer, TrainingConfig
)
from .model_evaluator import (
    ValidationFramework, ClassificationEvaluator, RegressionEvaluator,
    RecommendationEvaluator, ModelComparator
)
from .ab_testing import (
    ABTestFramework, create_model_comparison_experiment,
    ExperimentConfig, ExperimentVariant
)
from .continuous_learning import (
    ContinuousLearningEngine, OnlineRecommendationModel,
    OnlineClassificationModel, create_feedback_simulation
)
from .model_versioning import (
    ModelRegistry, ModelDeploymentManager, AutomatedDeploymentPipeline,
    create_model_version_from_training, ModelStatus, DeploymentStrategy
)


logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for the complete training pipeline."""
    # Data preparation
    data_sources: Dict[str, str]
    preprocessing_config: Dict[str, Any]
    feature_engineering_config: Dict[str, Any]
    
    # Model training
    models_to_train: List[str]
    hyperparameter_optimization: bool
    optimization_trials: int
    
    # Evaluation
    evaluation_metrics: List[str]
    cross_validation_folds: int
    test_size: float
    
    # A/B testing
    enable_ab_testing: bool
    ab_test_duration_days: int
    significance_level: float
    
    # Continuous learning
    enable_continuous_learning: bool
    learning_mode: str
    feedback_window_hours: int
    
    # Model versioning and deployment
    enable_auto_deployment: bool
    deployment_strategy: str
    auto_deploy_to_staging: bool
    auto_deploy_to_production: bool
    
    # Infrastructure
    output_directory: str
    max_workers: int
    enable_gpu: bool


class MLTrainingPipeline:
    """Complete ML training pipeline orchestrator."""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.output_dir = Path(config.output_directory)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.data_preprocessor = DataPreprocessor(config.preprocessing_config)
        self.feature_engineer = FeatureEngineer()
        self.matrix_builder = UserItemMatrixBuilder()
        self.dataset_builder = TrainingDatasetBuilder(test_size=config.test_size)
        
        # Training components
        self.training_pipeline = ModelTrainingPipeline()
        self.validation_framework = ValidationFramework()
        
        # A/B testing
        if config.enable_ab_testing:
            self.ab_test_framework = ABTestFramework()
        
        # Continuous learning
        if config.enable_continuous_learning:
            self.continuous_learning_engine = ContinuousLearningEngine(
                learning_mode=config.learning_mode,
                update_frequency_minutes=60
            )
        
        # Model versioning and deployment
        self.model_registry = ModelRegistry(str(self.output_dir / "model_registry"))
        self.deployment_manager = ModelDeploymentManager(self.model_registry)
        
        if config.enable_auto_deployment:
            self.deployment_pipeline = AutomatedDeploymentPipeline(
                self.model_registry, self.deployment_manager
            )
        
        # Execution state
        self.pipeline_state = {
            'status': 'initialized',
            'current_step': None,
            'start_time': None,
            'end_time': None,
            'results': {}
        }
    
    async def run_complete_pipeline(self) -> Dict[str, Any]:
        """Run the complete ML training pipeline."""
        logger.info("Starting complete ML training pipeline")
        
        self.pipeline_state['status'] = 'running'
        self.pipeline_state['start_time'] = datetime.now()
        
        try:
            # Step 1: Data preparation
            logger.info("Step 1: Data preparation")
            self.pipeline_state['current_step'] = 'data_preparation'
            datasets = await self._prepare_data()
            self.pipeline_state['results']['datasets'] = datasets
            
            # Step 2: Model training
            logger.info("Step 2: Model training")
            self.pipeline_state['current_step'] = 'model_training'
            training_results = await self._train_models(datasets)
            self.pipeline_state['results']['training_results'] = training_results
            
            # Step 3: Model evaluation
            logger.info("Step 3: Model evaluation")
            self.pipeline_state['current_step'] = 'model_evaluation'
            evaluation_results = await self._evaluate_models(training_results, datasets)
            self.pipeline_state['results']['evaluation_results'] = evaluation_results
            
            # Step 4: Model versioning
            logger.info("Step 4: Model versioning")
            self.pipeline_state['current_step'] = 'model_versioning'
            versioning_results = await self._version_models(training_results, evaluation_results)
            self.pipeline_state['results']['versioning_results'] = versioning_results
            
            # Step 5: A/B testing (if enabled)
            if self.config.enable_ab_testing:
                logger.info("Step 5: A/B testing")
                self.pipeline_state['current_step'] = 'ab_testing'
                ab_test_results = await self._setup_ab_tests(versioning_results)
                self.pipeline_state['results']['ab_test_results'] = ab_test_results
            
            # Step 6: Continuous learning setup (if enabled)
            if self.config.enable_continuous_learning:
                logger.info("Step 6: Continuous learning setup")
                self.pipeline_state['current_step'] = 'continuous_learning'
                continuous_learning_results = await self._setup_continuous_learning(versioning_results)
                self.pipeline_state['results']['continuous_learning_results'] = continuous_learning_results
            
            # Step 7: Deployment (if enabled)
            if self.config.enable_auto_deployment:
                logger.info("Step 7: Model deployment")
                self.pipeline_state['current_step'] = 'deployment'
                deployment_results = await self._deploy_models(versioning_results)
                self.pipeline_state['results']['deployment_results'] = deployment_results
            
            self.pipeline_state['status'] = 'completed'
            self.pipeline_state['end_time'] = datetime.now()
            
            # Generate final report
            final_report = self._generate_pipeline_report()
            
            logger.info("ML training pipeline completed successfully")
            return final_report
            
        except Exception as e:
            self.pipeline_state['status'] = 'failed'
            self.pipeline_state['end_time'] = datetime.now()
            self.pipeline_state['error'] = str(e)
            
            logger.error(f"ML training pipeline failed: {e}")
            raise
    
    async def _prepare_data(self) -> Dict[str, Any]:
        """Prepare training datasets."""
        logger.info("Preparing training data")
        
        # For demonstration, create synthetic data
        # In practice, this would load from actual data sources
        synthetic_data = create_synthetic_training_data(
            n_users=1000, n_items=500, n_interactions=10000
        )
        
        # Preprocess data
        processed_data = self.data_preprocessor.fit_transform({
            'numerical_data': synthetic_data['users'],
            'text_data': [item['description'] for item in synthetic_data['items']],
            'skill_data': synthetic_data['users'],
            'skill_taxonomy': synthetic_data['skill_taxonomy'],
            'categorical_data': {
                'education_level': [user['education_level'] for user in synthetic_data['users']]
            }
        })
        
        # Build user-item matrix
        user_item_matrix = self.matrix_builder.build_matrix(
            synthetic_data['interactions'], implicit_feedback=False
        )
        
        # Create training datasets
        datasets = {}
        
        # Recommendation dataset
        if 'recommendation' in self.config.models_to_train:
            recommendation_dataset = self.dataset_builder.build_recommendation_dataset(
                user_item_matrix,
                user_features=processed_data.get('numerical'),
                item_features=processed_data.get('text')
            )
            datasets['recommendation'] = recommendation_dataset
        
        # Skill classification dataset
        if 'skill_classification' in self.config.models_to_train:
            skill_data = []
            for user in synthetic_data['users']:
                skill_data.append({
                    'text': f"Experience with {', '.join(user['skills'])}",
                    'skills': user['skills']
                })
            
            skill_dataset = self.dataset_builder.build_skill_classification_dataset(skill_data)
            datasets['skill_classification'] = skill_dataset
        
        # Save datasets
        for name, dataset in datasets.items():
            dataset_path = self.output_dir / f"{name}_dataset.pkl"
            self.dataset_builder.save_dataset(dataset, str(dataset_path))
        
        datasets['user_item_matrix'] = user_item_matrix
        datasets['processed_data'] = processed_data
        
        return datasets
    
    async def _train_models(self, datasets: Dict[str, Any]) -> Dict[str, Any]:
        """Train all configured models."""
        logger.info("Training models")
        
        training_results = {}
        
        # Use thread pool for parallel training
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = {}
            
            # Submit training jobs
            for model_type in self.config.models_to_train:
                if model_type == 'collaborative_filtering' and 'user_item_matrix' in datasets:
                    future = executor.submit(self._train_collaborative_filtering, datasets['user_item_matrix'])
                    futures[model_type] = future
                
                elif model_type == 'neural_collaborative_filtering' and 'recommendation' in datasets:
                    future = executor.submit(self._train_neural_cf, datasets['recommendation'])
                    futures[model_type] = future
                
                elif model_type == 'skill_classification' and 'skill_classification' in datasets:
                    future = executor.submit(self._train_skill_classification, datasets['skill_classification'])
                    futures[model_type] = future
            
            # Collect results
            for model_type, future in futures.items():
                try:
                    result = future.result(timeout=3600)  # 1 hour timeout
                    training_results[model_type] = result
                    logger.info(f"Completed training {model_type}")
                except Exception as e:
                    logger.error(f"Failed to train {model_type}: {e}")
                    training_results[model_type] = {'error': str(e)}
        
        return training_results
    
    def _train_collaborative_filtering(self, user_item_matrix):
        """Train collaborative filtering model."""
        config = TrainingConfig(
            model_type='collaborative_filtering',
            hyperparameters={
                'n_factors': 50,
                'learning_rate': 0.01,
                'regularization': 0.1,
                'n_iterations': 100
            },
            training_params={},
            validation_params={},
            run_name='collaborative_filtering_training'
        )
        
        trainer = CollaborativeFilteringTrainer(config)
        return trainer.train(user_item_matrix)
    
    def _train_neural_cf(self, dataset):
        """Train neural collaborative filtering model."""
        config = TrainingConfig(
            model_type='neural_collaborative_filtering',
            hyperparameters={
                'embedding_dim': 64,
                'hidden_dims': [128, 64, 32]
            },
            training_params={
                'learning_rate': 0.001,
                'batch_size': 256,
                'epochs': 50  # Reduced for demo
            },
            validation_params={},
            run_name='neural_cf_training'
        )
        
        trainer = NeuralCollaborativeFilteringTrainer(config)
        return trainer.train(dataset)
    
    def _train_skill_classification(self, dataset):
        """Train skill classification model."""
        config = TrainingConfig(
            model_type='random_forest',
            hyperparameters={
                'n_estimators': 100,
                'max_depth': 10,
                'random_state': 42
            },
            training_params={},
            validation_params={},
            run_name='skill_classification_training'
        )
        
        trainer = SkillClassificationTrainer(config)
        return trainer.train(dataset)
    
    async def _evaluate_models(self, training_results: Dict[str, Any], 
                              datasets: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate all trained models."""
        logger.info("Evaluating models")
        
        evaluation_results = {}
        
        for model_type, training_result in training_results.items():
            if 'error' in training_result:
                continue
            
            try:
                # Load the trained model for evaluation
                # This is simplified - in practice, you'd load the actual model
                evaluation_results[model_type] = {
                    'training_metrics': training_result.training_metrics,
                    'validation_metrics': training_result.validation_metrics,
                    'test_metrics': training_result.test_metrics,
                    'model_path': training_result.model_path
                }
                
                logger.info(f"Completed evaluation for {model_type}")
                
            except Exception as e:
                logger.error(f"Failed to evaluate {model_type}: {e}")
                evaluation_results[model_type] = {'error': str(e)}
        
        return evaluation_results
    
    async def _version_models(self, training_results: Dict[str, Any], 
                             evaluation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Version all trained models."""
        logger.info("Versioning models")
        
        versioning_results = {}
        
        for model_type in training_results.keys():
            if 'error' in training_results[model_type]:
                continue
            
            try:
                training_result = training_results[model_type]
                
                # Create evaluation metrics object (simplified)
                from .model_evaluator import EvaluationMetrics
                eval_metrics = EvaluationMetrics(
                    accuracy=training_result.test_metrics.get('accuracy', 0.8),
                    precision=training_result.test_metrics.get('precision', 0.75),
                    recall=training_result.test_metrics.get('recall', 0.7),
                    f1_score=training_result.test_metrics.get('f1_score', 0.72)
                )
                
                # Create model version
                model_version = create_model_version_from_training(
                    training_result, eval_metrics, model_type
                )
                
                # Register with model registry
                version_id = self.model_registry.register_model(model_version)
                
                # Update status to validation
                model_id, version_str = version_id.split(':')
                self.model_registry.update_model_status(model_id, version_str, ModelStatus.VALIDATION)
                
                versioning_results[model_type] = {
                    'version_id': version_id,
                    'model_path': model_version.model_path,
                    'status': ModelStatus.VALIDATION.value
                }
                
                logger.info(f"Versioned model {model_type} as {version_id}")
                
            except Exception as e:
                logger.error(f"Failed to version {model_type}: {e}")
                versioning_results[model_type] = {'error': str(e)}
        
        return versioning_results
    
    async def _setup_ab_tests(self, versioning_results: Dict[str, Any]) -> Dict[str, Any]:
        """Set up A/B tests for model comparison."""
        logger.info("Setting up A/B tests")
        
        ab_test_results = {}
        
        # Group models by type for comparison
        model_groups = {}
        for model_type, result in versioning_results.items():
            if 'error' in result:
                continue
            
            base_type = model_type.split('_')[0]  # e.g., 'collaborative' from 'collaborative_filtering'
            if base_type not in model_groups:
                model_groups[base_type] = []
            model_groups[base_type].append((model_type, result))
        
        # Create A/B tests for each group with multiple models
        for group_name, models in model_groups.items():
            if len(models) < 2:
                continue  # Need at least 2 models for A/B test
            
            try:
                # Create model configs for A/B test
                model_configs = {}
                for model_type, result in models:
                    model_configs[model_type] = {
                        'name': model_type.replace('_', ' ').title(),
                        'description': f'{model_type} model',
                        'version_id': result['version_id']
                    }
                
                # Create experiment
                experiment_config = create_model_comparison_experiment(
                    model_configs, f"{group_name.title()} Model Comparison"
                )
                
                # Create experiment in framework
                experiment_id = self.ab_test_framework.create_experiment(experiment_config)
                
                ab_test_results[group_name] = {
                    'experiment_id': experiment_id,
                    'models': list(model_configs.keys()),
                    'status': 'created'
                }
                
                logger.info(f"Created A/B test {experiment_id} for {group_name} models")
                
            except Exception as e:
                logger.error(f"Failed to create A/B test for {group_name}: {e}")
                ab_test_results[group_name] = {'error': str(e)}
        
        return ab_test_results
    
    async def _setup_continuous_learning(self, versioning_results: Dict[str, Any]) -> Dict[str, Any]:
        """Set up continuous learning for models."""
        logger.info("Setting up continuous learning")
        
        continuous_learning_results = {}
        
        for model_type, result in versioning_results.items():
            if 'error' in result:
                continue
            
            try:
                # Create online learning model
                if 'classification' in model_type:
                    online_model = OnlineClassificationModel(
                        model_id=result['version_id'],
                        learning_rate=0.01,
                        classes=[0, 1]  # Binary classification
                    )
                else:
                    online_model = OnlineRecommendationModel(
                        model_id=result['version_id'],
                        learning_rate=0.01
                    )
                
                # Register with continuous learning engine
                self.continuous_learning_engine.register_model(online_model)
                
                continuous_learning_results[model_type] = {
                    'model_id': result['version_id'],
                    'status': 'registered'
                }
                
                logger.info(f"Registered {model_type} for continuous learning")
                
            except Exception as e:
                logger.error(f"Failed to setup continuous learning for {model_type}: {e}")
                continuous_learning_results[model_type] = {'error': str(e)}
        
        # Start continuous learning engine
        if continuous_learning_results:
            self.continuous_learning_engine.start_continuous_learning()
            
            # Simulate some feedback for demonstration
            feedback_data = create_feedback_simulation(n_users=50, n_items=25, n_feedback=100)
            for feedback in feedback_data[:10]:  # Add first 10 feedback items
                self.continuous_learning_engine.add_feedback(feedback)
        
        return continuous_learning_results
    
    async def _deploy_models(self, versioning_results: Dict[str, Any]) -> Dict[str, Any]:
        """Deploy models to staging/production."""
        logger.info("Deploying models")
        
        deployment_results = {}
        
        for model_type, result in versioning_results.items():
            if 'error' in result:
                continue
            
            try:
                version_id = result['version_id']
                model_id, version_str = version_id.split(':')
                
                # Trigger automated deployment pipeline
                pipeline_id = self.deployment_pipeline.trigger_pipeline(model_id, version_str)
                
                deployment_results[model_type] = {
                    'version_id': version_id,
                    'pipeline_id': pipeline_id,
                    'status': 'deployed_to_staging'
                }
                
                logger.info(f"Deployed {model_type} to staging via pipeline {pipeline_id}")
                
            except Exception as e:
                logger.error(f"Failed to deploy {model_type}: {e}")
                deployment_results[model_type] = {'error': str(e)}
        
        return deployment_results
    
    def _generate_pipeline_report(self) -> Dict[str, Any]:
        """Generate comprehensive pipeline report."""
        duration = (self.pipeline_state['end_time'] - self.pipeline_state['start_time']).total_seconds()
        
        report = {
            'pipeline_summary': {
                'status': self.pipeline_state['status'],
                'duration_seconds': duration,
                'start_time': self.pipeline_state['start_time'].isoformat(),
                'end_time': self.pipeline_state['end_time'].isoformat(),
                'models_trained': len(self.config.models_to_train),
                'steps_completed': len([k for k in self.pipeline_state['results'].keys()])
            },
            'configuration': asdict(self.config),
            'results': self.pipeline_state['results'],
            'recommendations': self._generate_recommendations()
        }
        
        # Save report
        report_path = self.output_dir / f"pipeline_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Pipeline report saved to {report_path}")
        
        return report
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on pipeline results."""
        recommendations = []
        
        # Check training results
        training_results = self.pipeline_state['results'].get('training_results', {})
        failed_models = [model for model, result in training_results.items() if 'error' in result]
        
        if failed_models:
            recommendations.append(f"Review failed models: {', '.join(failed_models)}")
        
        # Check evaluation results
        evaluation_results = self.pipeline_state['results'].get('evaluation_results', {})
        low_performance_models = []
        
        for model, result in evaluation_results.items():
            if 'error' not in result:
                accuracy = result.get('test_metrics', {}).get('accuracy', 0)
                if accuracy < 0.8:
                    low_performance_models.append(model)
        
        if low_performance_models:
            recommendations.append(f"Consider retraining models with low performance: {', '.join(low_performance_models)}")
        
        # Check A/B test setup
        if self.config.enable_ab_testing:
            ab_results = self.pipeline_state['results'].get('ab_test_results', {})
            if ab_results:
                recommendations.append("Monitor A/B test results and implement winning variants")
        
        # Check continuous learning
        if self.config.enable_continuous_learning:
            recommendations.append("Monitor continuous learning performance and adjust learning rates as needed")
        
        # Check deployment
        if self.config.enable_auto_deployment:
            recommendations.append("Monitor deployed models and set up alerting for performance degradation")
        
        return recommendations
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current pipeline status."""
        return {
            'status': self.pipeline_state['status'],
            'current_step': self.pipeline_state['current_step'],
            'start_time': self.pipeline_state['start_time'],
            'duration_seconds': (datetime.now() - self.pipeline_state['start_time']).total_seconds() if self.pipeline_state['start_time'] else 0,
            'completed_steps': list(self.pipeline_state['results'].keys())
        }


async def run_demo_pipeline():
    """Run a demonstration of the complete ML training pipeline."""
    # Create pipeline configuration
    config = PipelineConfig(
        data_sources={'synthetic': 'demo'},
        preprocessing_config={'remove_personal_info': True},
        feature_engineering_config={'create_interactions': True},
        models_to_train=['collaborative_filtering', 'skill_classification'],
        hyperparameter_optimization=False,
        optimization_trials=10,
        evaluation_metrics=['accuracy', 'precision', 'recall', 'f1_score'],
        cross_validation_folds=5,
        test_size=0.2,
        enable_ab_testing=True,
        ab_test_duration_days=7,
        significance_level=0.05,
        enable_continuous_learning=True,
        learning_mode='hybrid',
        feedback_window_hours=24,
        enable_auto_deployment=True,
        deployment_strategy='blue_green',
        auto_deploy_to_staging=True,
        auto_deploy_to_production=False,
        output_directory='./ml_pipeline_output',
        max_workers=2,
        enable_gpu=False
    )
    
    # Create and run pipeline
    pipeline = MLTrainingPipeline(config)
    
    try:
        results = await pipeline.run_complete_pipeline()
        print("Pipeline completed successfully!")
        print(f"Results summary: {results['pipeline_summary']}")
        return results
    except Exception as e:
        print(f"Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    # Run the demo pipeline
    import asyncio
    asyncio.run(run_demo_pipeline())