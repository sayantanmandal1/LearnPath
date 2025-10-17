"""
Model versioning and deployment automation system.

This module provides model versioning, deployment pipelines, rollback capabilities,
and automated model lifecycle management for production ML systems.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from datetime import datetime, timedelta
import json
import pickle
import hashlib
import shutil
import os
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum
import uuid
import git
import mlflow
import mlflow.pytorch
import mlflow.sklearn
from packaging import version
import boto3
from botocore.exceptions import ClientError

from .model_trainer import TrainingResult
from .model_evaluator import EvaluationMetrics
from .ab_testing import ExperimentAnalysis


logger = logging.getLogger(__name__)


class ModelStatus(str, Enum):
    """Model deployment status."""
    TRAINING = "training"
    VALIDATION = "validation"
    STAGING = "staging"
    PRODUCTION = "production"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"
    FAILED = "failed"


class DeploymentStrategy(str, Enum):
    """Model deployment strategies."""
    BLUE_GREEN = "blue_green"
    CANARY = "canary"
    ROLLING = "rolling"
    SHADOW = "shadow"
    A_B_TEST = "a_b_test"


@dataclass
class ModelVersion:
    """Model version information."""
    model_id: str
    version: str
    model_type: str
    training_result: TrainingResult
    evaluation_metrics: EvaluationMetrics
    model_path: str
    metadata: Dict[str, Any]
    created_at: datetime
    created_by: str
    status: ModelStatus
    tags: List[str]
    parent_version: Optional[str] = None
    deployment_config: Optional[Dict[str, Any]] = None


@dataclass
class DeploymentConfig:
    """Deployment configuration."""
    strategy: DeploymentStrategy
    target_environment: str
    traffic_percentage: float
    health_check_config: Dict[str, Any]
    rollback_config: Dict[str, Any]
    monitoring_config: Dict[str, Any]
    resource_requirements: Dict[str, Any]
    auto_scaling_config: Optional[Dict[str, Any]] = None


@dataclass
class DeploymentRecord:
    """Record of model deployment."""
    deployment_id: str
    model_version: str
    environment: str
    strategy: DeploymentStrategy
    status: str
    deployed_at: datetime
    deployed_by: str
    traffic_percentage: float
    health_status: str
    performance_metrics: Dict[str, float]
    rollback_version: Optional[str] = None


class ModelRegistry:
    """Central registry for model versions and metadata."""
    
    def __init__(self, registry_path: str = "./model_registry"):
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize registry database (JSON file for simplicity)
        self.registry_file = self.registry_path / "registry.json"
        self.models = self._load_registry()
        
        # Initialize MLflow tracking
        mlflow.set_tracking_uri(str(self.registry_path / "mlruns"))
        
    def _load_registry(self) -> Dict[str, Dict[str, ModelVersion]]:
        """Load model registry from disk."""
        if self.registry_file.exists():
            try:
                with open(self.registry_file, 'r') as f:
                    data = json.load(f)
                
                # Convert back to ModelVersion objects
                models = {}
                for model_id, versions in data.items():
                    models[model_id] = {}
                    for version_str, version_data in versions.items():
                        # Convert datetime strings back to datetime objects
                        version_data['created_at'] = datetime.fromisoformat(version_data['created_at'])
                        if version_data['training_result']:
                            # Reconstruct TrainingResult object
                            training_result = TrainingResult(**version_data['training_result'])
                            version_data['training_result'] = training_result
                        
                        models[model_id][version_str] = ModelVersion(**version_data)
                
                return models
            except Exception as e:
                logger.error(f"Error loading registry: {e}")
                return {}
        else:
            return {}
    
    def _save_registry(self):
        """Save model registry to disk."""
        try:
            # Convert ModelVersion objects to serializable format
            data = {}
            for model_id, versions in self.models.items():
                data[model_id] = {}
                for version_str, model_version in versions.items():
                    version_dict = asdict(model_version)
                    # Convert datetime to string
                    version_dict['created_at'] = model_version.created_at.isoformat()
                    # Convert TrainingResult to dict
                    if model_version.training_result:
                        version_dict['training_result'] = asdict(model_version.training_result)
                    data[model_id][version_str] = version_dict
            
            with open(self.registry_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
                
        except Exception as e:
            logger.error(f"Error saving registry: {e}")
    
    def register_model(self, model_version: ModelVersion) -> str:
        """Register a new model version."""
        model_id = model_version.model_id
        version_str = model_version.version
        
        # Initialize model entry if it doesn't exist
        if model_id not in self.models:
            self.models[model_id] = {}
        
        # Check if version already exists
        if version_str in self.models[model_id]:
            raise ValueError(f"Model version {model_id}:{version_str} already exists")
        
        # Store model files
        model_dir = self.registry_path / model_id / version_str
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy model files to registry
        if os.path.exists(model_version.model_path):
            target_path = model_dir / "model.pkl"
            shutil.copy2(model_version.model_path, target_path)
            model_version.model_path = str(target_path)
        
        # Register with MLflow
        with mlflow.start_run(run_name=f"{model_id}_{version_str}"):
            mlflow.log_params(model_version.training_result.hyperparameters)
            mlflow.log_metrics(model_version.training_result.training_metrics)
            
            # Log model artifact
            if os.path.exists(model_version.model_path):
                mlflow.log_artifact(model_version.model_path, "model")
        
        # Add to registry
        self.models[model_id][version_str] = model_version
        self._save_registry()
        
        logger.info(f"Registered model {model_id}:{version_str}")
        return f"{model_id}:{version_str}"
    
    def get_model_version(self, model_id: str, version: str) -> Optional[ModelVersion]:
        """Get specific model version."""
        return self.models.get(model_id, {}).get(version)
    
    def get_latest_version(self, model_id: str, status: Optional[ModelStatus] = None) -> Optional[ModelVersion]:
        """Get latest version of a model, optionally filtered by status."""
        if model_id not in self.models:
            return None
        
        versions = self.models[model_id]
        if status:
            versions = {v: mv for v, mv in versions.items() if mv.status == status}
        
        if not versions:
            return None
        
        # Sort by version number (assuming semantic versioning)
        sorted_versions = sorted(versions.keys(), key=lambda x: version.parse(x), reverse=True)
        return versions[sorted_versions[0]]
    
    def list_model_versions(self, model_id: str) -> List[ModelVersion]:
        """List all versions of a model."""
        if model_id not in self.models:
            return []
        
        return list(self.models[model_id].values())
    
    def update_model_status(self, model_id: str, version_str: str, status: ModelStatus):
        """Update model status."""
        if model_id in self.models and version_str in self.models[model_id]:
            self.models[model_id][version_str].status = status
            self._save_registry()
            logger.info(f"Updated {model_id}:{version_str} status to {status.value}")
        else:
            raise ValueError(f"Model version {model_id}:{version_str} not found")
    
    def add_model_tags(self, model_id: str, version_str: str, tags: List[str]):
        """Add tags to a model version."""
        if model_id in self.models and version_str in self.models[model_id]:
            existing_tags = set(self.models[model_id][version_str].tags)
            existing_tags.update(tags)
            self.models[model_id][version_str].tags = list(existing_tags)
            self._save_registry()
        else:
            raise ValueError(f"Model version {model_id}:{version_str} not found")
    
    def search_models(self, query: Dict[str, Any]) -> List[ModelVersion]:
        """Search models by criteria."""
        results = []
        
        for model_id, versions in self.models.items():
            for version_str, model_version in versions.items():
                match = True
                
                # Check each query criterion
                for key, value in query.items():
                    if key == 'model_id' and model_id != value:
                        match = False
                        break
                    elif key == 'model_type' and model_version.model_type != value:
                        match = False
                        break
                    elif key == 'status' and model_version.status != value:
                        match = False
                        break
                    elif key == 'tags' and not set(value).issubset(set(model_version.tags)):
                        match = False
                        break
                    elif key == 'created_after' and model_version.created_at < value:
                        match = False
                        break
                    elif key == 'created_before' and model_version.created_at > value:
                        match = False
                        break
                
                if match:
                    results.append(model_version)
        
        return results
    
    def delete_model_version(self, model_id: str, version_str: str):
        """Delete a model version."""
        if model_id in self.models and version_str in self.models[model_id]:
            # Remove model files
            model_dir = self.registry_path / model_id / version_str
            if model_dir.exists():
                shutil.rmtree(model_dir)
            
            # Remove from registry
            del self.models[model_id][version_str]
            
            # Remove model entry if no versions left
            if not self.models[model_id]:
                del self.models[model_id]
            
            self._save_registry()
            logger.info(f"Deleted model version {model_id}:{version_str}")
        else:
            raise ValueError(f"Model version {model_id}:{version_str} not found")


class ModelDeploymentManager:
    """Manage model deployments and lifecycle."""
    
    def __init__(self, registry: ModelRegistry, deployment_config_path: str = "./deployment_configs"):
        self.registry = registry
        self.deployment_config_path = Path(deployment_config_path)
        self.deployment_config_path.mkdir(parents=True, exist_ok=True)
        
        # Deployment history
        self.deployment_history = self._load_deployment_history()
        
        # Environment configurations
        self.environments = {
            'staging': {'url': 'http://staging-api.example.com', 'replicas': 1},
            'production': {'url': 'http://api.example.com', 'replicas': 3}
        }
    
    def _load_deployment_history(self) -> List[DeploymentRecord]:
        """Load deployment history from disk."""
        history_file = self.deployment_config_path / "deployment_history.json"
        
        if history_file.exists():
            try:
                with open(history_file, 'r') as f:
                    data = json.load(f)
                
                history = []
                for record_data in data:
                    record_data['deployed_at'] = datetime.fromisoformat(record_data['deployed_at'])
                    history.append(DeploymentRecord(**record_data))
                
                return history
            except Exception as e:
                logger.error(f"Error loading deployment history: {e}")
                return []
        else:
            return []
    
    def _save_deployment_history(self):
        """Save deployment history to disk."""
        history_file = self.deployment_config_path / "deployment_history.json"
        
        try:
            data = []
            for record in self.deployment_history:
                record_dict = asdict(record)
                record_dict['deployed_at'] = record.deployed_at.isoformat()
                data.append(record_dict)
            
            with open(history_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
                
        except Exception as e:
            logger.error(f"Error saving deployment history: {e}")
    
    def deploy_model(self, model_id: str, version_str: str, 
                    environment: str, config: DeploymentConfig) -> str:
        """Deploy a model version to an environment."""
        # Get model version
        model_version = self.registry.get_model_version(model_id, version_str)
        if not model_version:
            raise ValueError(f"Model version {model_id}:{version_str} not found")
        
        # Validate model is ready for deployment
        if model_version.status not in [ModelStatus.VALIDATION, ModelStatus.STAGING]:
            raise ValueError(f"Model status {model_version.status} not suitable for deployment")
        
        # Generate deployment ID
        deployment_id = str(uuid.uuid4())
        
        # Execute deployment based on strategy
        success = self._execute_deployment(model_version, environment, config, deployment_id)
        
        if success:
            # Update model status
            if environment == 'production':
                self.registry.update_model_status(model_id, version_str, ModelStatus.PRODUCTION)
            elif environment == 'staging':
                self.registry.update_model_status(model_id, version_str, ModelStatus.STAGING)
            
            # Record deployment
            deployment_record = DeploymentRecord(
                deployment_id=deployment_id,
                model_version=f"{model_id}:{version_str}",
                environment=environment,
                strategy=config.strategy,
                status='deployed',
                deployed_at=datetime.now(),
                deployed_by='system',  # Would be actual user in practice
                traffic_percentage=config.traffic_percentage,
                health_status='healthy',
                performance_metrics={}
            )
            
            self.deployment_history.append(deployment_record)
            self._save_deployment_history()
            
            logger.info(f"Successfully deployed {model_id}:{version_str} to {environment}")
            return deployment_id
        else:
            raise RuntimeError(f"Failed to deploy {model_id}:{version_str} to {environment}")
    
    def _execute_deployment(self, model_version: ModelVersion, environment: str, 
                          config: DeploymentConfig, deployment_id: str) -> bool:
        """Execute the actual deployment."""
        logger.info(f"Executing {config.strategy.value} deployment to {environment}")
        
        try:
            if config.strategy == DeploymentStrategy.BLUE_GREEN:
                return self._blue_green_deployment(model_version, environment, config)
            elif config.strategy == DeploymentStrategy.CANARY:
                return self._canary_deployment(model_version, environment, config)
            elif config.strategy == DeploymentStrategy.ROLLING:
                return self._rolling_deployment(model_version, environment, config)
            elif config.strategy == DeploymentStrategy.SHADOW:
                return self._shadow_deployment(model_version, environment, config)
            else:
                logger.error(f"Unsupported deployment strategy: {config.strategy}")
                return False
                
        except Exception as e:
            logger.error(f"Deployment failed: {e}")
            return False
    
    def _blue_green_deployment(self, model_version: ModelVersion, environment: str, 
                              config: DeploymentConfig) -> bool:
        """Execute blue-green deployment."""
        # In a real implementation, this would:
        # 1. Deploy to green environment
        # 2. Run health checks
        # 3. Switch traffic from blue to green
        # 4. Keep blue as rollback option
        
        logger.info("Executing blue-green deployment")
        
        # Simulate deployment steps
        steps = [
            "Preparing green environment",
            "Deploying model to green environment",
            "Running health checks",
            "Switching traffic to green environment",
            "Monitoring deployment"
        ]
        
        for step in steps:
            logger.info(f"Blue-green deployment: {step}")
            # Simulate work
            import time
            time.sleep(1)
        
        return True
    
    def _canary_deployment(self, model_version: ModelVersion, environment: str, 
                          config: DeploymentConfig) -> bool:
        """Execute canary deployment."""
        logger.info("Executing canary deployment")
        
        # Gradually increase traffic to new version
        traffic_steps = [5, 10, 25, 50, 100]  # Percentage of traffic
        
        for traffic_pct in traffic_steps:
            logger.info(f"Canary deployment: Routing {traffic_pct}% traffic to new version")
            
            # In practice, you would:
            # 1. Update load balancer configuration
            # 2. Monitor metrics
            # 3. Check for errors or performance degradation
            # 4. Rollback if issues detected
            
            import time
            time.sleep(2)  # Simulate monitoring period
            
            # Simulate health check
            if not self._health_check(model_version, environment):
                logger.error("Health check failed during canary deployment")
                return False
        
        return True
    
    def _rolling_deployment(self, model_version: ModelVersion, environment: str, 
                           config: DeploymentConfig) -> bool:
        """Execute rolling deployment."""
        logger.info("Executing rolling deployment")
        
        # Update instances one by one
        replicas = self.environments[environment]['replicas']
        
        for i in range(replicas):
            logger.info(f"Rolling deployment: Updating replica {i+1}/{replicas}")
            
            # In practice, you would:
            # 1. Stop one instance
            # 2. Deploy new version
            # 3. Start instance
            # 4. Wait for health check
            # 5. Move to next instance
            
            import time
            time.sleep(1)
        
        return True
    
    def _shadow_deployment(self, model_version: ModelVersion, environment: str, 
                          config: DeploymentConfig) -> bool:
        """Execute shadow deployment."""
        logger.info("Executing shadow deployment")
        
        # Deploy alongside existing version without serving traffic
        # Used for testing and comparison
        
        steps = [
            "Deploying shadow version",
            "Configuring traffic mirroring",
            "Starting shadow monitoring",
            "Collecting comparison metrics"
        ]
        
        for step in steps:
            logger.info(f"Shadow deployment: {step}")
            import time
            time.sleep(1)
        
        return True
    
    def _health_check(self, model_version: ModelVersion, environment: str) -> bool:
        """Perform health check on deployed model."""
        # In practice, this would make actual HTTP requests to the deployed model
        # and check response times, error rates, etc.
        
        logger.info("Performing health check")
        
        # Simulate health check
        import random
        return random.random() > 0.1  # 90% success rate
    
    def rollback_deployment(self, deployment_id: str) -> bool:
        """Rollback a deployment to previous version."""
        # Find deployment record
        deployment_record = None
        for record in self.deployment_history:
            if record.deployment_id == deployment_id:
                deployment_record = record
                break
        
        if not deployment_record:
            raise ValueError(f"Deployment {deployment_id} not found")
        
        logger.info(f"Rolling back deployment {deployment_id}")
        
        # Find previous deployment in same environment
        previous_deployments = [
            r for r in self.deployment_history
            if (r.environment == deployment_record.environment and 
                r.deployed_at < deployment_record.deployed_at and
                r.status == 'deployed')
        ]
        
        if not previous_deployments:
            logger.error("No previous deployment found for rollback")
            return False
        
        # Get most recent previous deployment
        previous_deployment = max(previous_deployments, key=lambda x: x.deployed_at)
        
        # Execute rollback
        logger.info(f"Rolling back to {previous_deployment.model_version}")
        
        # Update deployment status
        deployment_record.status = 'rolled_back'
        deployment_record.rollback_version = previous_deployment.model_version
        
        self._save_deployment_history()
        
        return True
    
    def get_deployment_status(self, deployment_id: str) -> Optional[DeploymentRecord]:
        """Get status of a deployment."""
        for record in self.deployment_history:
            if record.deployment_id == deployment_id:
                return record
        return None
    
    def get_active_deployments(self, environment: Optional[str] = None) -> List[DeploymentRecord]:
        """Get currently active deployments."""
        active_deployments = [
            record for record in self.deployment_history
            if record.status == 'deployed'
        ]
        
        if environment:
            active_deployments = [
                record for record in active_deployments
                if record.environment == environment
            ]
        
        return active_deployments
    
    def monitor_deployments(self) -> Dict[str, Any]:
        """Monitor all active deployments."""
        active_deployments = self.get_active_deployments()
        
        monitoring_results = {
            'total_active_deployments': len(active_deployments),
            'deployments_by_environment': {},
            'health_status': {},
            'performance_metrics': {}
        }
        
        # Group by environment
        for deployment in active_deployments:
            env = deployment.environment
            if env not in monitoring_results['deployments_by_environment']:
                monitoring_results['deployments_by_environment'][env] = 0
            monitoring_results['deployments_by_environment'][env] += 1
            
            # Simulate health check
            health_status = 'healthy' if self._health_check(None, env) else 'unhealthy'
            monitoring_results['health_status'][deployment.deployment_id] = health_status
            
            # Simulate performance metrics
            monitoring_results['performance_metrics'][deployment.deployment_id] = {
                'response_time_ms': np.random.normal(100, 20),
                'error_rate': np.random.uniform(0, 0.05),
                'throughput_rps': np.random.normal(1000, 100)
            }
        
        return monitoring_results


class AutomatedDeploymentPipeline:
    """Automated deployment pipeline with CI/CD integration."""
    
    def __init__(self, registry: ModelRegistry, deployment_manager: ModelDeploymentManager):
        self.registry = registry
        self.deployment_manager = deployment_manager
        self.pipeline_config = self._load_pipeline_config()
        
    def _load_pipeline_config(self) -> Dict[str, Any]:
        """Load pipeline configuration."""
        return {
            'auto_deploy_to_staging': True,
            'auto_deploy_to_production': False,
            'required_tests': ['unit_tests', 'integration_tests', 'performance_tests'],
            'approval_required_for_production': True,
            'rollback_on_failure': True,
            'monitoring_duration_minutes': 60
        }
    
    def trigger_pipeline(self, model_id: str, version_str: str) -> str:
        """Trigger automated deployment pipeline."""
        logger.info(f"Triggering deployment pipeline for {model_id}:{version_str}")
        
        pipeline_id = str(uuid.uuid4())
        
        try:
            # Step 1: Validate model
            if not self._validate_model(model_id, version_str):
                raise RuntimeError("Model validation failed")
            
            # Step 2: Run tests
            if not self._run_tests(model_id, version_str):
                raise RuntimeError("Tests failed")
            
            # Step 3: Deploy to staging
            if self.pipeline_config['auto_deploy_to_staging']:
                staging_config = DeploymentConfig(
                    strategy=DeploymentStrategy.BLUE_GREEN,
                    target_environment='staging',
                    traffic_percentage=100.0,
                    health_check_config={'timeout': 30, 'retries': 3},
                    rollback_config={'auto_rollback': True, 'threshold': 0.95},
                    monitoring_config={'duration_minutes': 30},
                    resource_requirements={'cpu': '500m', 'memory': '1Gi'}
                )
                
                staging_deployment_id = self.deployment_manager.deploy_model(
                    model_id, version_str, 'staging', staging_config
                )
                
                # Monitor staging deployment
                if not self._monitor_deployment(staging_deployment_id, 30):
                    raise RuntimeError("Staging deployment monitoring failed")
            
            # Step 4: Deploy to production (if configured)
            if self.pipeline_config['auto_deploy_to_production']:
                if self.pipeline_config['approval_required_for_production']:
                    logger.info("Production deployment requires manual approval")
                    return pipeline_id
                
                production_config = DeploymentConfig(
                    strategy=DeploymentStrategy.CANARY,
                    target_environment='production',
                    traffic_percentage=100.0,
                    health_check_config={'timeout': 60, 'retries': 5},
                    rollback_config={'auto_rollback': True, 'threshold': 0.99},
                    monitoring_config={'duration_minutes': 60},
                    resource_requirements={'cpu': '1000m', 'memory': '2Gi'}
                )
                
                production_deployment_id = self.deployment_manager.deploy_model(
                    model_id, version_str, 'production', production_config
                )
                
                # Monitor production deployment
                if not self._monitor_deployment(production_deployment_id, 60):
                    if self.pipeline_config['rollback_on_failure']:
                        self.deployment_manager.rollback_deployment(production_deployment_id)
                    raise RuntimeError("Production deployment monitoring failed")
            
            logger.info(f"Deployment pipeline {pipeline_id} completed successfully")
            return pipeline_id
            
        except Exception as e:
            logger.error(f"Deployment pipeline {pipeline_id} failed: {e}")
            raise
    
    def _validate_model(self, model_id: str, version_str: str) -> bool:
        """Validate model before deployment."""
        model_version = self.registry.get_model_version(model_id, version_str)
        if not model_version:
            return False
        
        # Check if model file exists
        if not os.path.exists(model_version.model_path):
            logger.error(f"Model file not found: {model_version.model_path}")
            return False
        
        # Check evaluation metrics meet minimum thresholds
        metrics = model_version.evaluation_metrics
        if hasattr(metrics, 'accuracy') and metrics.accuracy and metrics.accuracy < 0.8:
            logger.error(f"Model accuracy {metrics.accuracy} below threshold 0.8")
            return False
        
        logger.info("Model validation passed")
        return True
    
    def _run_tests(self, model_id: str, version_str: str) -> bool:
        """Run required tests."""
        logger.info("Running automated tests")
        
        for test_type in self.pipeline_config['required_tests']:
            logger.info(f"Running {test_type}")
            
            # Simulate test execution
            import time
            time.sleep(2)
            
            # Simulate test result (90% pass rate)
            import random
            if random.random() < 0.1:
                logger.error(f"{test_type} failed")
                return False
        
        logger.info("All tests passed")
        return True
    
    def _monitor_deployment(self, deployment_id: str, duration_minutes: int) -> bool:
        """Monitor deployment for specified duration."""
        logger.info(f"Monitoring deployment {deployment_id} for {duration_minutes} minutes")
        
        # Simulate monitoring
        import time
        time.sleep(5)  # Simulate monitoring period
        
        # Check deployment health
        deployment_record = self.deployment_manager.get_deployment_status(deployment_id)
        if not deployment_record:
            return False
        
        # Simulate monitoring results (95% success rate)
        import random
        success = random.random() > 0.05
        
        if success:
            logger.info(f"Deployment {deployment_id} monitoring successful")
        else:
            logger.error(f"Deployment {deployment_id} monitoring failed")
        
        return success
    
    def approve_production_deployment(self, model_id: str, version_str: str) -> str:
        """Manually approve production deployment."""
        logger.info(f"Approving production deployment for {model_id}:{version_str}")
        
        production_config = DeploymentConfig(
            strategy=DeploymentStrategy.CANARY,
            target_environment='production',
            traffic_percentage=100.0,
            health_check_config={'timeout': 60, 'retries': 5},
            rollback_config={'auto_rollback': True, 'threshold': 0.99},
            monitoring_config={'duration_minutes': 60},
            resource_requirements={'cpu': '1000m', 'memory': '2Gi'}
        )
        
        return self.deployment_manager.deploy_model(
            model_id, version_str, 'production', production_config
        )


def create_model_version_from_training(training_result: TrainingResult, 
                                     evaluation_metrics: EvaluationMetrics,
                                     model_type: str = "recommendation",
                                     created_by: str = "system") -> ModelVersion:
    """Create ModelVersion from training results."""
    # Generate version number (simplified - in practice, use semantic versioning)
    version_str = f"1.0.{int(datetime.now().timestamp())}"
    
    return ModelVersion(
        model_id=training_result.model_id,
        version=version_str,
        model_type=model_type,
        training_result=training_result,
        evaluation_metrics=evaluation_metrics,
        model_path=training_result.model_path,
        metadata={
            'training_time': training_result.training_time,
            'best_epoch': training_result.best_epoch,
            'hyperparameters': training_result.hyperparameters
        },
        created_at=datetime.now(),
        created_by=created_by,
        status=ModelStatus.TRAINING,
        tags=['automated', 'v1']
    )


def create_model_version_from_training(training_result: TrainingResult, 
                                     evaluation_metrics: EvaluationMetrics,
                                     model_type: str) -> ModelVersion:
    """Create a ModelVersion from training and evaluation results."""
    import uuid
    
    # Generate version string (simplified semantic versioning)
    version = "1.0.0"
    
    # Create model version
    model_version = ModelVersion(
        model_id=training_result.model_id,
        version=version,
        model_type=model_type,
        training_result=training_result,
        evaluation_metrics=evaluation_metrics,
        model_path=training_result.model_path,
        metadata={
            'training_time': training_result.training_time,
            'best_epoch': training_result.best_epoch,
            'hyperparameters': training_result.hyperparameters
        },
        created_at=datetime.now(),
        created_by='system',
        status=ModelStatus.TRAINING,
        tags=['automated', 'v1'],
        parent_version=None,
        deployment_config=None
    )
    
    return model_version