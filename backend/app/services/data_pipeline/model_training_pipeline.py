"""
Model Training and Retraining Pipeline
Handles automated model training with performance monitoring and A/B testing.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import json
import os
from pathlib import Path

from app.core.logging import get_logger
from app.services.data_pipeline.pipeline_monitor import get_pipeline_monitor
from machinelearningmodel.training.training_pipeline import TrainingPipeline
from machinelearningmodel.training.model_evaluator import ModelEvaluator
from machinelearningmodel.training.ab_testing import ABTestingFramework
from machinelearningmodel.training.model_versioning import ModelVersionManager
from app.repositories.profile import ProfileRepository
from app.repositories.job import JobRepository
from app.core.database import get_db_session

logger = get_logger(__name__)


class ModelTrainingPipeline:
    """
    Pipeline for automated model training and retraining with performance monitoring
    """
    
    def __init__(self):
        self.training_pipeline = TrainingPipeline()
        self.model_evaluator = ModelEvaluator()
        self.ab_testing = ABTestingFramework()
        self.version_manager = ModelVersionManager()
        self.monitor = None
        
    async def execute(self, metadata: Dict[str, Any] = None):
        """Execute the model training pipeline"""
        execution_id = metadata.get('execution_id', f"model_training_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}")
        
        self.monitor = await get_pipeline_monitor()
        
        try:
            logger.info(f"Starting model training pipeline: {execution_id}")
            
            # Initialize metrics
            await self.monitor.update_job_metrics(
                execution_id,
                status="running",
                records_processed=0,
                records_failed=0
            )
            
            # Get training parameters
            training_mode = metadata.get('training_mode', 'incremental')  # 'full' or 'incremental'
            models_to_train = metadata.get('models', ['recommendation', 'skill_extraction', 'career_prediction'])
            enable_ab_testing = metadata.get('enable_ab_testing', True)
            
            training_results = {
                'models_trained': [],
                'models_failed': [],
                'performance_metrics': {},
                'ab_test_results': {},
                'deployment_status': {}
            }
            
            total_processed = 0
            total_failed = 0
            
            # Step 1: Prepare training data
            logger.info("Preparing training data")
            training_data = await self._prepare_training_data(training_mode)
            
            if not training_data or not training_data.get('has_sufficient_data'):
                raise ValueError("Insufficient training data available")
            
            # Step 2: Train each model
            for model_name in models_to_train:
                try:
                    logger.info(f"Training model: {model_name}")
                    
                    # Train the model
                    training_result = await self._train_model(
                        model_name, 
                        training_data, 
                        training_mode
                    )
                    
                    if training_result['success']:
                        training_results['models_trained'].append(model_name)
                        training_results['performance_metrics'][model_name] = training_result['metrics']
                        total_processed += 1
                        
                        # Evaluate model performance
                        evaluation_result = await self._evaluate_model(model_name, training_result['model_path'])
                        training_results['performance_metrics'][model_name].update(evaluation_result)
                        
                        # Set up A/B testing if enabled
                        if enable_ab_testing:
                            ab_test_result = await self._setup_ab_test(model_name, training_result['model_path'])
                            training_results['ab_test_results'][model_name] = ab_test_result
                        else:
                            # Deploy directly if A/B testing is disabled
                            deployment_result = await self._deploy_model(model_name, training_result['model_path'])
                            training_results['deployment_status'][model_name] = deployment_result
                    else:
                        training_results['models_failed'].append({
                            'model_name': model_name,
                            'error': training_result.get('error', 'Unknown error')
                        })
                        total_failed += 1
                        
                except Exception as e:
                    logger.error(f"Failed to train model {model_name}: {e}")
                    training_results['models_failed'].append({
                        'model_name': model_name,
                        'error': str(e)
                    })
                    total_failed += 1
                
                # Update metrics after each model
                await self.monitor.update_job_metrics(
                    execution_id,
                    records_processed=total_processed,
                    records_failed=total_failed
                )
            
            # Step 3: Generate training summary and reports
            training_summary = await self._generate_training_summary(training_results, training_data)
            
            # Calculate data quality score based on training success
            success_rate = total_processed / (total_processed + total_failed) if (total_processed + total_failed) > 0 else 0.0
            
            # Store training summary
            await self._store_training_summary(execution_id, training_summary)
            
            # Update final metrics
            await self.monitor.update_job_metrics(
                execution_id,
                status="completed",
                records_processed=total_processed,
                records_failed=total_failed,
                data_quality_score=success_rate
            )
            
            logger.info(f"Model training pipeline completed: {total_processed} models trained successfully")
            
        except Exception as e:
            logger.error(f"Model training pipeline failed: {e}")
            
            await self.monitor.update_job_metrics(
                execution_id,
                status="failed",
                error_count=1
            )
            raise
    
    async def _prepare_training_data(self, training_mode: str) -> Dict[str, Any]:
        """Prepare training data for model training"""
        try:
            logger.info("Preparing training data")
            
            async with get_db_session() as db:
                profile_repo = ProfileRepository(db)
                job_repo = JobRepository(db)
                
                # Get training data based on mode
                if training_mode == 'full':
                    # Full retraining - use all available data
                    profiles = await profile_repo.get_all_profiles()
                    jobs = await job_repo.get_all_jobs()
                    cutoff_date = None
                else:
                    # Incremental training - use recent data
                    cutoff_date = datetime.utcnow() - timedelta(days=30)
                    profiles = await profile_repo.get_profiles_updated_since(cutoff_date)
                    jobs = await job_repo.get_jobs_created_since(cutoff_date)
                
                # Check data sufficiency
                min_profiles = 100
                min_jobs = 500
                has_sufficient_data = len(profiles) >= min_profiles and len(jobs) >= min_jobs
                
                if not has_sufficient_data:
                    logger.warning(f"Insufficient training data: {len(profiles)} profiles, {len(jobs)} jobs")
                
                # Prepare data for different model types
                training_data = {
                    'has_sufficient_data': has_sufficient_data,
                    'training_mode': training_mode,
                    'cutoff_date': cutoff_date.isoformat() if cutoff_date else None,
                    'data_stats': {
                        'total_profiles': len(profiles),
                        'total_jobs': len(jobs),
                        'profiles_with_skills': len([p for p in profiles if p.skills]),
                        'jobs_with_requirements': len([j for j in jobs if j.required_skills])
                    },
                    'recommendation_data': await self._prepare_recommendation_data(profiles, jobs),
                    'skill_extraction_data': await self._prepare_skill_extraction_data(profiles, jobs),
                    'career_prediction_data': await self._prepare_career_prediction_data(profiles, jobs)
                }
                
                return training_data
                
        except Exception as e:
            logger.error(f"Failed to prepare training data: {e}")
            return {'has_sufficient_data': False, 'error': str(e)}
    
    async def _prepare_recommendation_data(self, profiles: List[Any], jobs: List[Any]) -> Dict[str, Any]:
        """Prepare data for recommendation model training"""
        try:
            # Create user-item interaction matrix
            user_job_interactions = []
            user_features = []
            job_features = []
            
            for profile in profiles:
                if not profile.skills:
                    continue
                
                user_features.append({
                    'user_id': profile.user_id,
                    'skills': profile.skills,
                    'experience_level': getattr(profile, 'experience_level', 'mid'),
                    'career_interests': getattr(profile, 'career_interests', [])
                })
                
                # Create implicit interactions based on skill matches
                for job in jobs:
                    if not job.required_skills:
                        continue
                    
                    # Calculate skill match score
                    profile_skills = set(profile.skills.keys()) if profile.skills else set()
                    job_skills = set(job.required_skills)
                    
                    if profile_skills and job_skills:
                        match_score = len(profile_skills.intersection(job_skills)) / len(job_skills)
                        
                        if match_score > 0.3:  # Threshold for implicit positive interaction
                            user_job_interactions.append({
                                'user_id': profile.user_id,
                                'job_id': job.id,
                                'interaction_score': match_score,
                                'timestamp': datetime.utcnow().isoformat()
                            })
            
            # Prepare job features
            for job in jobs:
                if job.required_skills:
                    job_features.append({
                        'job_id': job.id,
                        'title': job.title,
                        'required_skills': job.required_skills,
                        'experience_level': getattr(job, 'experience_level', 'mid'),
                        'salary_range': getattr(job, 'salary_range', None),
                        'location': getattr(job, 'location', 'remote')
                    })
            
            return {
                'user_job_interactions': user_job_interactions,
                'user_features': user_features,
                'job_features': job_features,
                'interaction_count': len(user_job_interactions)
            }
            
        except Exception as e:
            logger.error(f"Failed to prepare recommendation data: {e}")
            return {}
    
    async def _prepare_skill_extraction_data(self, profiles: List[Any], jobs: List[Any]) -> Dict[str, Any]:
        """Prepare data for skill extraction model training"""
        try:
            training_examples = []
            
            # Use job descriptions as training text with known skills as labels
            for job in jobs:
                if job.description and job.required_skills:
                    training_examples.append({
                        'text': job.description,
                        'skills': job.required_skills,
                        'source': 'job_description'
                    })
            
            # Use profile data if available
            for profile in profiles:
                if hasattr(profile, 'resume_text') and profile.resume_text and profile.skills:
                    training_examples.append({
                        'text': profile.resume_text,
                        'skills': list(profile.skills.keys()),
                        'source': 'resume'
                    })
            
            return {
                'training_examples': training_examples,
                'example_count': len(training_examples),
                'unique_skills': len(set(
                    skill for example in training_examples 
                    for skill in example['skills']
                ))
            }
            
        except Exception as e:
            logger.error(f"Failed to prepare skill extraction data: {e}")
            return {}
    
    async def _prepare_career_prediction_data(self, profiles: List[Any], jobs: List[Any]) -> Dict[str, Any]:
        """Prepare data for career prediction model training"""
        try:
            career_transitions = []
            
            # This would typically require historical career data
            # For now, we'll create synthetic transitions based on skill similarity
            
            for profile in profiles:
                if not profile.skills:
                    continue
                
                current_role = getattr(profile, 'current_role', 'unknown')
                profile_skills = set(profile.skills.keys()) if profile.skills else set()
                
                # Find potential career transitions based on skill overlap
                for job in jobs:
                    if not job.required_skills:
                        continue
                    
                    job_skills = set(job.required_skills)
                    skill_overlap = len(profile_skills.intersection(job_skills))
                    skill_gap = len(job_skills - profile_skills)
                    
                    if skill_overlap > 0:
                        career_transitions.append({
                            'current_role': current_role,
                            'target_role': job.title,
                            'current_skills': list(profile_skills),
                            'required_skills': job.required_skills,
                            'skill_overlap': skill_overlap,
                            'skill_gap': skill_gap,
                            'transition_difficulty': skill_gap / len(job_skills) if job_skills else 1.0
                        })
            
            return {
                'career_transitions': career_transitions,
                'transition_count': len(career_transitions),
                'unique_roles': len(set(
                    t['current_role'] for t in career_transitions
                ).union(set(
                    t['target_role'] for t in career_transitions
                )))
            }
            
        except Exception as e:
            logger.error(f"Failed to prepare career prediction data: {e}")
            return {}
    
    async def _train_model(self, model_name: str, training_data: Dict[str, Any], training_mode: str) -> Dict[str, Any]:
        """Train a specific model"""
        try:
            logger.info(f"Training {model_name} model")
            
            # Get model-specific training data
            model_data = training_data.get(f'{model_name}_data', {})
            
            if not model_data:
                return {
                    'success': False,
                    'error': f'No training data available for {model_name}'
                }
            
            # Train the model using the training pipeline
            training_result = await self.training_pipeline.train_model(
                model_name=model_name,
                training_data=model_data,
                training_mode=training_mode
            )
            
            if training_result['success']:
                # Save model version
                model_version = await self.version_manager.save_model_version(
                    model_name=model_name,
                    model_path=training_result['model_path'],
                    metrics=training_result['metrics'],
                    training_data_info=model_data
                )
                
                training_result['model_version'] = model_version
                
                logger.info(f"Successfully trained {model_name} model (version: {model_version})")
            
            return training_result
            
        except Exception as e:
            logger.error(f"Failed to train {model_name} model: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def _evaluate_model(self, model_name: str, model_path: str) -> Dict[str, Any]:
        """Evaluate a trained model"""
        try:
            logger.info(f"Evaluating {model_name} model")
            
            evaluation_result = await self.model_evaluator.evaluate_model(
                model_name=model_name,
                model_path=model_path
            )
            
            return evaluation_result
            
        except Exception as e:
            logger.error(f"Failed to evaluate {model_name} model: {e}")
            return {'evaluation_error': str(e)}
    
    async def _setup_ab_test(self, model_name: str, model_path: str) -> Dict[str, Any]:
        """Set up A/B testing for a new model"""
        try:
            logger.info(f"Setting up A/B test for {model_name} model")
            
            ab_test_result = await self.ab_testing.create_ab_test(
                model_name=model_name,
                new_model_path=model_path,
                traffic_split=0.1  # Start with 10% traffic to new model
            )
            
            return ab_test_result
            
        except Exception as e:
            logger.error(f"Failed to setup A/B test for {model_name}: {e}")
            return {'ab_test_error': str(e)}
    
    async def _deploy_model(self, model_name: str, model_path: str) -> Dict[str, Any]:
        """Deploy a model to production"""
        try:
            logger.info(f"Deploying {model_name} model")
            
            deployment_result = await self.version_manager.deploy_model(
                model_name=model_name,
                model_path=model_path
            )
            
            return deployment_result
            
        except Exception as e:
            logger.error(f"Failed to deploy {model_name} model: {e}")
            return {'deployment_error': str(e)}
    
    async def _generate_training_summary(self, training_results: Dict[str, Any], training_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive training summary"""
        try:
            summary = {
                'timestamp': datetime.utcnow().isoformat(),
                'training_data_stats': training_data.get('data_stats', {}),
                'models_trained': len(training_results['models_trained']),
                'models_failed': len(training_results['models_failed']),
                'success_rate': len(training_results['models_trained']) / (
                    len(training_results['models_trained']) + len(training_results['models_failed'])
                ) if (len(training_results['models_trained']) + len(training_results['models_failed'])) > 0 else 0.0,
                'performance_summary': {},
                'recommendations': []
            }
            
            # Summarize performance metrics
            for model_name, metrics in training_results['performance_metrics'].items():
                summary['performance_summary'][model_name] = {
                    'accuracy': metrics.get('accuracy', 0.0),
                    'precision': metrics.get('precision', 0.0),
                    'recall': metrics.get('recall', 0.0),
                    'f1_score': metrics.get('f1_score', 0.0),
                    'training_time': metrics.get('training_time', 0.0)
                }
            
            # Generate recommendations
            if summary['success_rate'] < 0.8:
                summary['recommendations'].append(
                    "Consider increasing training data quality or quantity"
                )
            
            if any(metrics.get('accuracy', 0) < 0.7 for metrics in training_results['performance_metrics'].values()):
                summary['recommendations'].append(
                    "Some models have low accuracy - consider hyperparameter tuning"
                )
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to generate training summary: {e}")
            return {}
    
    async def _store_training_summary(self, execution_id: str, summary: Dict[str, Any]):
        """Store training summary for reporting"""
        try:
            from app.core.redis import get_redis_client
            
            redis_client = await get_redis_client()
            
            # Store detailed summary
            summary_key = f"model_training_summary:{execution_id}"
            await redis_client.set(
                summary_key,
                json.dumps(summary),
                ex=86400 * 90  # Keep for 90 days
            )
            
            # Update monthly stats
            month = datetime.utcnow().strftime('%Y-%m')
            monthly_stats_key = f"model_training_monthly:{month}"
            
            await redis_client.hincrby(monthly_stats_key, 'total_trainings', 1)
            await redis_client.hincrby(monthly_stats_key, 'models_trained', summary.get('models_trained', 0))
            await redis_client.hincrby(monthly_stats_key, 'models_failed', summary.get('models_failed', 0))
            await redis_client.expire(monthly_stats_key, 86400 * 365)  # Keep for 1 year
            
        except Exception as e:
            logger.error(f"Failed to store training summary: {e}")
    
    async def get_training_stats(self, months: int = 6) -> Dict[str, Any]:
        """Get model training statistics"""
        try:
            from app.core.redis import get_redis_client
            
            redis_client = await get_redis_client()
            stats = {
                'monthly_stats': {},
                'total_trainings': 0,
                'total_models_trained': 0,
                'total_models_failed': 0,
                'success_rate': 0.0
            }
            
            # Get monthly stats
            for i in range(months):
                date = (datetime.utcnow() - timedelta(days=i*30)).strftime('%Y-%m')
                monthly_key = f"model_training_monthly:{date}"
                
                monthly_data = await redis_client.hgetall(monthly_key)
                if monthly_data:
                    trainings = int(monthly_data.get('total_trainings', 0))
                    trained = int(monthly_data.get('models_trained', 0))
                    failed = int(monthly_data.get('models_failed', 0))
                    
                    stats['monthly_stats'][date] = {
                        'total_trainings': trainings,
                        'models_trained': trained,
                        'models_failed': failed,
                        'success_rate': trained / (trained + failed) if (trained + failed) > 0 else 0.0
                    }
                    
                    stats['total_trainings'] += trainings
                    stats['total_models_trained'] += trained
                    stats['total_models_failed'] += failed
            
            # Calculate overall success rate
            total_attempts = stats['total_models_trained'] + stats['total_models_failed']
            if total_attempts > 0:
                stats['success_rate'] = stats['total_models_trained'] / total_attempts
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get training stats: {e}")
            return {}
    
    async def trigger_model_retraining(self, model_name: str, force: bool = False) -> Dict[str, Any]:
        """Trigger retraining for a specific model"""
        try:
            # Check if retraining is needed (unless forced)
            if not force:
                current_performance = await self.model_evaluator.get_current_performance(model_name)
                if current_performance.get('accuracy', 0) > 0.8:
                    return {
                        'triggered': False,
                        'reason': 'Model performance is still acceptable',
                        'current_accuracy': current_performance.get('accuracy', 0)
                    }
            
            # Trigger retraining
            metadata = {
                'execution_id': f"retrain_{model_name}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                'training_mode': 'incremental',
                'models': [model_name],
                'enable_ab_testing': True
            }
            
            # Execute training pipeline
            await self.execute(metadata)
            
            return {
                'triggered': True,
                'model_name': model_name,
                'execution_id': metadata['execution_id']
            }
            
        except Exception as e:
            logger.error(f"Failed to trigger retraining for {model_name}: {e}")
            return {
                'triggered': False,
                'error': str(e)
            }