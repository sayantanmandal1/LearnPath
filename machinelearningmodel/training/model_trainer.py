"""
Model training scripts for recommendation algorithms and NLP models.

This module provides training infrastructure for various ML models including
collaborative filtering, neural networks, and skill classification models.
"""

import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from datetime import datetime
import json
import pickle
from pathlib import Path
from dataclasses import dataclass, asdict
try:
    import mlflow
    import mlflow.pytorch
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    mlflow = None
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.multioutput import MultiOutputClassifier
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    optuna = None

try:
    from .data_preparation import TrainingDataset, UserItemMatrix
    from ..recommendation_engine import (
        CollaborativeFilteringEngine, 
        ContentBasedFilteringEngine,
        NeuralCollaborativeFiltering
    )
except ImportError:
    # Fallback for when running as standalone
    pass


logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for model training."""
    model_type: str
    hyperparameters: Dict[str, Any]
    training_params: Dict[str, Any]
    validation_params: Dict[str, Any]
    early_stopping: bool = True
    patience: int = 10
    save_best_model: bool = True
    experiment_name: str = "ml_training"
    run_name: Optional[str] = None


@dataclass
class TrainingResult:
    """Results from model training."""
    model_id: str
    model_type: str
    training_metrics: Dict[str, float]
    validation_metrics: Dict[str, float]
    test_metrics: Dict[str, float]
    best_epoch: int
    training_time: float
    hyperparameters: Dict[str, Any]
    model_path: str
    metadata: Dict[str, Any]


class BaseModelTrainer:
    """Base class for model trainers."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.model = None
        self.training_history = []
        self.best_model_state = None
        self.best_metric = float('-inf')
        
    def train(self, dataset: TrainingDataset) -> TrainingResult:
        """Train the model."""
        raise NotImplementedError("Subclasses must implement train method")
    
    def evaluate(self, X: Union[np.ndarray, torch.Tensor], 
                y: Union[np.ndarray, torch.Tensor]) -> Dict[str, float]:
        """Evaluate the model."""
        raise NotImplementedError("Subclasses must implement evaluate method")
    
    def save_model(self, filepath: str):
        """Save the trained model."""
        raise NotImplementedError("Subclasses must implement save_model method")
    
    def load_model(self, filepath: str):
        """Load a trained model."""
        raise NotImplementedError("Subclasses must implement load_model method")


class CollaborativeFilteringTrainer(BaseModelTrainer):
    """Trainer for collaborative filtering models."""
    
    def __init__(self, config: TrainingConfig):
        super().__init__(config)
        self.model = CollaborativeFilteringEngine(**config.hyperparameters)
    
    def train(self, user_item_matrix: UserItemMatrix) -> TrainingResult:
        """Train collaborative filtering model."""
        logger.info(f"Training collaborative filtering model")
        
        start_time = datetime.now()
        
        # Start MLflow run (if available)
        if MLFLOW_AVAILABLE:
            mlflow_context = mlflow.start_run(run_name=self.config.run_name)
            mlflow_context.__enter__()
            # Log hyperparameters
            mlflow.log_params(self.config.hyperparameters)
        else:
            mlflow_context = None
            
            # Train the model
            self.model.fit(
                user_item_matrix.matrix.toarray(),
                user_item_matrix.user_ids,
                user_item_matrix.item_ids
            )
            
            # Calculate training metrics
            training_metrics = self._calculate_cf_metrics(user_item_matrix, 'train')
            
            # Log metrics
            for metric, value in training_metrics.items():
                mlflow.log_metric(f"train_{metric}", value)
            
            training_time = (datetime.now() - start_time).total_seconds()
            
            # Save model
            model_path = f"models/collaborative_filtering_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
            Path(model_path).parent.mkdir(parents=True, exist_ok=True)
            self.save_model(model_path)
            
            # Log model
            mlflow.sklearn.log_model(self.model, "model")
            
            result = TrainingResult(
                model_id=f"cf_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                model_type="collaborative_filtering",
                training_metrics=training_metrics,
                validation_metrics={},  # CF doesn't have separate validation
                test_metrics={},
                best_epoch=self.model.n_iterations,
                training_time=training_time,
                hyperparameters=self.config.hyperparameters,
                model_path=model_path,
                metadata=user_item_matrix.metadata
            )
            
            return result
    
    def _calculate_cf_metrics(self, user_item_matrix: UserItemMatrix, split: str) -> Dict[str, float]:
        """Calculate metrics for collaborative filtering."""
        # Calculate RMSE on known interactions
        total_error = 0
        count = 0
        
        coo_matrix = user_item_matrix.matrix.tocoo()
        for user_idx, item_idx, rating in zip(coo_matrix.row, coo_matrix.col, coo_matrix.data):
            user_id = user_item_matrix.user_ids[user_idx]
            item_id = user_item_matrix.item_ids[item_idx]
            
            predicted_rating = self.model.predict(user_id, item_id)
            total_error += (rating - predicted_rating) ** 2
            count += 1
        
        rmse = np.sqrt(total_error / count) if count > 0 else 0
        
        return {
            'rmse': rmse,
            'n_predictions': count
        }
    
    def evaluate(self, user_item_matrix: UserItemMatrix) -> Dict[str, float]:
        """Evaluate collaborative filtering model."""
        return self._calculate_cf_metrics(user_item_matrix, 'test')
    
    def save_model(self, filepath: str):
        """Save collaborative filtering model."""
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)
    
    def load_model(self, filepath: str):
        """Load collaborative filtering model."""
        with open(filepath, 'rb') as f:
            self.model = pickle.load(f)


class NeuralCollaborativeFilteringTrainer(BaseModelTrainer):
    """Trainer for neural collaborative filtering models."""
    
    def __init__(self, config: TrainingConfig):
        super().__init__(config)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def train(self, dataset: TrainingDataset) -> TrainingResult:
        """Train neural collaborative filtering model."""
        logger.info(f"Training neural collaborative filtering model on {self.device}")
        
        start_time = datetime.now()
        
        # Extract dimensions from metadata
        n_users = dataset.metadata['n_users']
        n_items = dataset.metadata['n_items']
        
        # Initialize model
        self.model = NeuralCollaborativeFiltering(
            n_users=n_users,
            n_items=n_items,
            **self.config.hyperparameters
        ).to(self.device)
        
        # Prepare data loaders
        train_loader = self._create_dataloader(dataset.X_train, dataset.y_train, 
                                             self.config.training_params.get('batch_size', 256))
        val_loader = self._create_dataloader(dataset.X_val, dataset.y_val, 
                                           self.config.training_params.get('batch_size', 256))
        
        # Initialize optimizer and loss function
        optimizer = optim.Adam(self.model.parameters(), 
                             lr=self.config.training_params.get('learning_rate', 0.001))
        criterion = nn.MSELoss()
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        
        with mlflow.start_run(run_name=self.config.run_name):
            # Log hyperparameters
            mlflow.log_params(self.config.hyperparameters)
            mlflow.log_params(self.config.training_params)
            
            for epoch in range(self.config.training_params.get('epochs', 100)):
                # Training phase
                train_loss = self._train_epoch(train_loader, optimizer, criterion)
                
                # Validation phase
                val_loss = self._validate_epoch(val_loader, criterion)
                
                # Log metrics
                mlflow.log_metric("train_loss", train_loss, step=epoch)
                mlflow.log_metric("val_loss", val_loss, step=epoch)
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    if self.config.save_best_model:
                        self.best_model_state = self.model.state_dict().copy()
                else:
                    patience_counter += 1
                
                if self.config.early_stopping and patience_counter >= self.config.patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
                
                # Log progress
                if epoch % 10 == 0:
                    logger.info(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
            
            # Load best model
            if self.best_model_state is not None:
                self.model.load_state_dict(self.best_model_state)
            
            # Calculate final metrics
            training_metrics = {'loss': train_loss}
            validation_metrics = {'loss': best_val_loss}
            
            # Test evaluation
            test_loader = self._create_dataloader(dataset.X_test, dataset.y_test, 
                                                self.config.training_params.get('batch_size', 256))
            test_metrics = {'loss': self._validate_epoch(test_loader, criterion)}
            
            training_time = (datetime.now() - start_time).total_seconds()
            
            # Save model
            model_path = f"models/neural_cf_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
            Path(model_path).parent.mkdir(parents=True, exist_ok=True)
            self.save_model(model_path)
            
            # Log model
            mlflow.pytorch.log_model(self.model, "model")
            
            result = TrainingResult(
                model_id=f"ncf_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                model_type="neural_collaborative_filtering",
                training_metrics=training_metrics,
                validation_metrics=validation_metrics,
                test_metrics=test_metrics,
                best_epoch=epoch - patience_counter,
                training_time=training_time,
                hyperparameters=self.config.hyperparameters,
                model_path=model_path,
                metadata=dataset.metadata
            )
            
            return result
    
    def _create_dataloader(self, X: np.ndarray, y: np.ndarray, batch_size: int) -> DataLoader:
        """Create PyTorch DataLoader."""
        # Assuming X contains [user_indices, item_indices] in first two columns
        user_ids = torch.LongTensor(X[:, 0].astype(int))
        item_ids = torch.LongTensor(X[:, 1].astype(int))
        ratings = torch.FloatTensor(y)
        
        dataset = TensorDataset(user_ids, item_ids, ratings)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    def _train_epoch(self, dataloader: DataLoader, optimizer: optim.Optimizer, 
                    criterion: nn.Module) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        
        for user_ids, item_ids, ratings in dataloader:
            user_ids = user_ids.to(self.device)
            item_ids = item_ids.to(self.device)
            ratings = ratings.to(self.device)
            
            optimizer.zero_grad()
            predictions = self.model(user_ids, item_ids).squeeze()
            loss = criterion(predictions, ratings)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    def _validate_epoch(self, dataloader: DataLoader, criterion: nn.Module) -> float:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for user_ids, item_ids, ratings in dataloader:
                user_ids = user_ids.to(self.device)
                item_ids = item_ids.to(self.device)
                ratings = ratings.to(self.device)
                
                predictions = self.model(user_ids, item_ids).squeeze()
                loss = criterion(predictions, ratings)
                total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    def evaluate(self, X: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
        """Evaluate neural collaborative filtering model."""
        dataloader = self._create_dataloader(X.numpy(), y.numpy(), 256)
        criterion = nn.MSELoss()
        loss = self._validate_epoch(dataloader, criterion)
        
        return {'loss': loss, 'rmse': np.sqrt(loss)}
    
    def save_model(self, filepath: str):
        """Save neural collaborative filtering model."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_config': self.config.hyperparameters
        }, filepath)
    
    def load_model(self, filepath: str):
        """Load neural collaborative filtering model."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # Recreate model with saved config
        model_config = checkpoint['model_config']
        self.model = NeuralCollaborativeFiltering(**model_config).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])


class SkillClassificationTrainer(BaseModelTrainer):
    """Trainer for skill classification models."""
    
    def __init__(self, config: TrainingConfig):
        super().__init__(config)
        
    def train(self, dataset: TrainingDataset) -> TrainingResult:
        """Train skill classification model."""
        logger.info(f"Training skill classification model")
        
        start_time = datetime.now()
        
        # Initialize model based on configuration
        if self.config.model_type == 'random_forest':
            base_model = RandomForestClassifier(**self.config.hyperparameters)
            self.model = MultiOutputClassifier(base_model)
        elif self.config.model_type == 'logistic_regression':
            base_model = LogisticRegression(**self.config.hyperparameters)
            self.model = MultiOutputClassifier(base_model)
        else:
            raise ValueError(f"Unsupported model type: {self.config.model_type}")
        
        with mlflow.start_run(run_name=self.config.run_name):
            # Log hyperparameters
            mlflow.log_params(self.config.hyperparameters)
            
            # Train the model
            self.model.fit(dataset.X_train, dataset.y_train)
            
            # Calculate metrics
            training_metrics = self._calculate_classification_metrics(
                dataset.X_train, dataset.y_train, 'train'
            )
            validation_metrics = self._calculate_classification_metrics(
                dataset.X_val, dataset.y_val, 'validation'
            )
            test_metrics = self._calculate_classification_metrics(
                dataset.X_test, dataset.y_test, 'test'
            )
            
            # Log metrics
            for metric, value in training_metrics.items():
                mlflow.log_metric(f"train_{metric}", value)
            for metric, value in validation_metrics.items():
                mlflow.log_metric(f"val_{metric}", value)
            for metric, value in test_metrics.items():
                mlflow.log_metric(f"test_{metric}", value)
            
            training_time = (datetime.now() - start_time).total_seconds()
            
            # Save model
            model_path = f"models/skill_classifier_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
            Path(model_path).parent.mkdir(parents=True, exist_ok=True)
            self.save_model(model_path)
            
            # Log model
            mlflow.sklearn.log_model(self.model, "model")
            
            result = TrainingResult(
                model_id=f"skill_clf_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                model_type=self.config.model_type,
                training_metrics=training_metrics,
                validation_metrics=validation_metrics,
                test_metrics=test_metrics,
                best_epoch=1,  # Single epoch for sklearn models
                training_time=training_time,
                hyperparameters=self.config.hyperparameters,
                model_path=model_path,
                metadata=dataset.metadata
            )
            
            return result
    
    def _calculate_classification_metrics(self, X, y, split: str) -> Dict[str, float]:
        """Calculate classification metrics."""
        y_pred = self.model.predict(X)
        
        # Calculate metrics for multi-label classification
        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred, average='macro', zero_division=0)
        recall = recall_score(y, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y, y_pred, average='macro', zero_division=0)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
    
    def evaluate(self, X, y) -> Dict[str, float]:
        """Evaluate skill classification model."""
        return self._calculate_classification_metrics(X, y, 'evaluation')
    
    def save_model(self, filepath: str):
        """Save skill classification model."""
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)
    
    def load_model(self, filepath: str):
        """Load skill classification model."""
        with open(filepath, 'rb') as f:
            self.model = pickle.load(f)


class HyperparameterOptimizer:
    """Hyperparameter optimization using Optuna."""
    
    def __init__(self, model_trainer_class: type, dataset: TrainingDataset, 
                 n_trials: int = 100, direction: str = 'minimize'):
        self.model_trainer_class = model_trainer_class
        self.dataset = dataset
        self.n_trials = n_trials
        self.direction = direction
        self.study = None
        
    def optimize_collaborative_filtering(self) -> Dict[str, Any]:
        """Optimize hyperparameters for collaborative filtering."""
        def objective(trial):
            # Define hyperparameter search space
            hyperparameters = {
                'n_factors': trial.suggest_int('n_factors', 10, 200),
                'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1, log=True),
                'regularization': trial.suggest_float('regularization', 0.001, 0.1, log=True),
                'n_iterations': trial.suggest_int('n_iterations', 50, 500)
            }
            
            config = TrainingConfig(
                model_type='collaborative_filtering',
                hyperparameters=hyperparameters,
                training_params={},
                validation_params={}
            )
            
            trainer = self.model_trainer_class(config)
            result = trainer.train(self.dataset)
            
            # Return validation metric to optimize
            return result.validation_metrics.get('rmse', result.training_metrics.get('rmse', float('inf')))
        
        self.study = optuna.create_study(direction=self.direction)
        self.study.optimize(objective, n_trials=self.n_trials)
        
        return {
            'best_params': self.study.best_params,
            'best_value': self.study.best_value,
            'n_trials': len(self.study.trials)
        }
    
    def optimize_neural_cf(self) -> Dict[str, Any]:
        """Optimize hyperparameters for neural collaborative filtering."""
        def objective(trial):
            # Define hyperparameter search space
            hyperparameters = {
                'embedding_dim': trial.suggest_int('embedding_dim', 32, 256),
                'hidden_dims': [
                    trial.suggest_int('hidden_dim_1', 64, 512),
                    trial.suggest_int('hidden_dim_2', 32, 256),
                    trial.suggest_int('hidden_dim_3', 16, 128)
                ]
            }
            
            training_params = {
                'learning_rate': trial.suggest_float('learning_rate', 0.0001, 0.01, log=True),
                'batch_size': trial.suggest_categorical('batch_size', [128, 256, 512]),
                'epochs': 50  # Fixed for optimization
            }
            
            config = TrainingConfig(
                model_type='neural_collaborative_filtering',
                hyperparameters=hyperparameters,
                training_params=training_params,
                validation_params={}
            )
            
            trainer = self.model_trainer_class(config)
            result = trainer.train(self.dataset)
            
            return result.validation_metrics.get('loss', float('inf'))
        
        self.study = optuna.create_study(direction=self.direction)
        self.study.optimize(objective, n_trials=self.n_trials)
        
        return {
            'best_params': self.study.best_params,
            'best_value': self.study.best_value,
            'n_trials': len(self.study.trials)
        }
    
    def optimize_skill_classification(self) -> Dict[str, Any]:
        """Optimize hyperparameters for skill classification."""
        def objective(trial):
            model_type = trial.suggest_categorical('model_type', ['random_forest', 'logistic_regression'])
            
            if model_type == 'random_forest':
                hyperparameters = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                    'max_depth': trial.suggest_int('max_depth', 3, 20),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                    'random_state': 42
                }
            else:  # logistic_regression
                hyperparameters = {
                    'C': trial.suggest_float('C', 0.001, 100, log=True),
                    'max_iter': trial.suggest_int('max_iter', 100, 1000),
                    'random_state': 42
                }
            
            config = TrainingConfig(
                model_type=model_type,
                hyperparameters=hyperparameters,
                training_params={},
                validation_params={}
            )
            
            trainer = self.model_trainer_class(config)
            result = trainer.train(self.dataset)
            
            # Return negative F1 score for minimization
            return -result.validation_metrics.get('f1_score', 0)
        
        self.study = optuna.create_study(direction='minimize')
        self.study.optimize(objective, n_trials=self.n_trials)
        
        return {
            'best_params': self.study.best_params,
            'best_value': -self.study.best_value,  # Convert back to positive
            'n_trials': len(self.study.trials)
        }


class ModelTrainingPipeline:
    """Complete model training pipeline."""
    
    def __init__(self, experiment_name: str = "ml_training"):
        self.experiment_name = experiment_name
        mlflow.set_experiment(experiment_name)
        
    def train_all_models(self, datasets: Dict[str, Any]) -> Dict[str, TrainingResult]:
        """Train all models in the pipeline."""
        results = {}
        
        # Train collaborative filtering
        if 'user_item_matrix' in datasets:
            logger.info("Training collaborative filtering model")
            cf_config = TrainingConfig(
                model_type='collaborative_filtering',
                hyperparameters={
                    'n_factors': 50,
                    'learning_rate': 0.01,
                    'regularization': 0.1,
                    'n_iterations': 100
                },
                training_params={},
                validation_params={},
                run_name='collaborative_filtering'
            )
            
            cf_trainer = CollaborativeFilteringTrainer(cf_config)
            results['collaborative_filtering'] = cf_trainer.train(datasets['user_item_matrix'])
        
        # Train neural collaborative filtering
        if 'recommendation_dataset' in datasets:
            logger.info("Training neural collaborative filtering model")
            ncf_config = TrainingConfig(
                model_type='neural_collaborative_filtering',
                hyperparameters={
                    'embedding_dim': 64,
                    'hidden_dims': [128, 64, 32]
                },
                training_params={
                    'learning_rate': 0.001,
                    'batch_size': 256,
                    'epochs': 100
                },
                validation_params={},
                run_name='neural_collaborative_filtering'
            )
            
            ncf_trainer = NeuralCollaborativeFilteringTrainer(ncf_config)
            results['neural_collaborative_filtering'] = ncf_trainer.train(datasets['recommendation_dataset'])
        
        # Train skill classification
        if 'skill_classification_dataset' in datasets:
            logger.info("Training skill classification model")
            skill_config = TrainingConfig(
                model_type='random_forest',
                hyperparameters={
                    'n_estimators': 100,
                    'max_depth': 10,
                    'random_state': 42
                },
                training_params={},
                validation_params={},
                run_name='skill_classification'
            )
            
            skill_trainer = SkillClassificationTrainer(skill_config)
            results['skill_classification'] = skill_trainer.train(datasets['skill_classification_dataset'])
        
        return results
    
    def optimize_hyperparameters(self, datasets: Dict[str, Any], 
                                n_trials: int = 50) -> Dict[str, Dict[str, Any]]:
        """Optimize hyperparameters for all models."""
        optimization_results = {}
        
        # Optimize collaborative filtering
        if 'user_item_matrix' in datasets:
            logger.info("Optimizing collaborative filtering hyperparameters")
            optimizer = HyperparameterOptimizer(
                CollaborativeFilteringTrainer, 
                datasets['user_item_matrix'], 
                n_trials=n_trials
            )
            optimization_results['collaborative_filtering'] = optimizer.optimize_collaborative_filtering()
        
        # Optimize neural collaborative filtering
        if 'recommendation_dataset' in datasets:
            logger.info("Optimizing neural collaborative filtering hyperparameters")
            optimizer = HyperparameterOptimizer(
                NeuralCollaborativeFilteringTrainer, 
                datasets['recommendation_dataset'], 
                n_trials=n_trials
            )
            optimization_results['neural_collaborative_filtering'] = optimizer.optimize_neural_cf()
        
        # Optimize skill classification
        if 'skill_classification_dataset' in datasets:
            logger.info("Optimizing skill classification hyperparameters")
            optimizer = HyperparameterOptimizer(
                SkillClassificationTrainer, 
                datasets['skill_classification_dataset'], 
                n_trials=n_trials
            )
            optimization_results['skill_classification'] = optimizer.optimize_skill_classification()
        
        return optimization_results