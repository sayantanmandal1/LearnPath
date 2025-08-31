"""
Continuous learning system for model improvement from user feedback.

This module implements online learning, feedback processing, model updating,
and adaptive recommendation systems that improve over time.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from datetime import datetime, timedelta
import json
import pickle
from dataclasses import dataclass, asdict
from collections import deque, defaultdict
from enum import Enum
import threading
import time
from queue import Queue, Empty
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.base import BaseEstimator
from sklearn.linear_model import SGDClassifier, SGDRegressor
from sklearn.preprocessing import StandardScaler
import joblib

from .model_trainer import TrainingResult
from .model_evaluator import EvaluationMetrics
from ..recommendation_engine import HybridRecommendationEngine


logger = logging.getLogger(__name__)


class FeedbackType(str, Enum):
    """Types of user feedback."""
    EXPLICIT = "explicit"  # Direct ratings, thumbs up/down
    IMPLICIT = "implicit"  # Clicks, time spent, conversions
    NEGATIVE = "negative"  # Complaints, corrections
    CONTEXTUAL = "contextual"  # Context-aware feedback


class LearningMode(str, Enum):
    """Learning modes for continuous learning."""
    ONLINE = "online"  # Real-time learning
    BATCH = "batch"  # Periodic batch updates
    HYBRID = "hybrid"  # Combination of online and batch


@dataclass
class UserFeedback:
    """User feedback data structure."""
    user_id: str
    item_id: str
    feedback_type: FeedbackType
    feedback_value: float  # Rating, click (1/0), time spent, etc.
    context: Dict[str, Any]  # Additional context information
    timestamp: datetime
    session_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class LearningUpdate:
    """Learning update information."""
    model_id: str
    update_type: str
    samples_processed: int
    performance_change: Dict[str, float]
    timestamp: datetime
    metadata: Dict[str, Any]


@dataclass
class ModelPerformanceMetrics:
    """Model performance tracking."""
    model_id: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    user_satisfaction: float
    engagement_rate: float
    conversion_rate: float
    timestamp: datetime
    sample_size: int


class FeedbackProcessor:
    """Process and validate user feedback."""
    
    def __init__(self, feedback_window_hours: int = 24):
        self.feedback_window_hours = feedback_window_hours
        self.feedback_buffer = deque(maxlen=10000)
        self.user_feedback_history = defaultdict(list)
        
    def process_feedback(self, feedback: UserFeedback) -> bool:
        """Process and validate user feedback."""
        # Validate feedback
        if not self._validate_feedback(feedback):
            logger.warning(f"Invalid feedback from user {feedback.user_id}")
            return False
        
        # Add to buffer
        self.feedback_buffer.append(feedback)
        
        # Update user history
        self.user_feedback_history[feedback.user_id].append(feedback)
        
        # Clean old feedback
        self._clean_old_feedback()
        
        return True
    
    def _validate_feedback(self, feedback: UserFeedback) -> bool:
        """Validate feedback data."""
        # Check required fields
        if not all([feedback.user_id, feedback.item_id, feedback.feedback_type]):
            return False
        
        # Check feedback value range
        if feedback.feedback_type == FeedbackType.EXPLICIT:
            if not 1 <= feedback.feedback_value <= 5:  # Assuming 1-5 rating scale
                return False
        elif feedback.feedback_type == FeedbackType.IMPLICIT:
            if feedback.feedback_value < 0:  # Non-negative values
                return False
        
        # Check timestamp
        if feedback.timestamp > datetime.now():
            return False
        
        return True
    
    def _clean_old_feedback(self):
        """Remove old feedback outside the window."""
        cutoff_time = datetime.now() - timedelta(hours=self.feedback_window_hours)
        
        # Clean buffer
        while self.feedback_buffer and self.feedback_buffer[0].timestamp < cutoff_time:
            self.feedback_buffer.popleft()
        
        # Clean user history
        for user_id in list(self.user_feedback_history.keys()):
            user_feedback = self.user_feedback_history[user_id]
            self.user_feedback_history[user_id] = [
                fb for fb in user_feedback if fb.timestamp >= cutoff_time
            ]
            
            # Remove empty entries
            if not self.user_feedback_history[user_id]:
                del self.user_feedback_history[user_id]
    
    def get_recent_feedback(self, hours: int = 1) -> List[UserFeedback]:
        """Get feedback from the last N hours."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [fb for fb in self.feedback_buffer if fb.timestamp >= cutoff_time]
    
    def get_user_feedback_summary(self, user_id: str) -> Dict[str, Any]:
        """Get summary of user's feedback patterns."""
        user_feedback = self.user_feedback_history.get(user_id, [])
        
        if not user_feedback:
            return {}
        
        # Calculate statistics
        explicit_ratings = [fb.feedback_value for fb in user_feedback 
                          if fb.feedback_type == FeedbackType.EXPLICIT]
        implicit_values = [fb.feedback_value for fb in user_feedback 
                         if fb.feedback_type == FeedbackType.IMPLICIT]
        
        summary = {
            'total_feedback_count': len(user_feedback),
            'explicit_count': len(explicit_ratings),
            'implicit_count': len(implicit_values),
            'avg_explicit_rating': np.mean(explicit_ratings) if explicit_ratings else 0,
            'avg_implicit_value': np.mean(implicit_values) if implicit_values else 0,
            'feedback_frequency': len(user_feedback) / max(1, 
                (datetime.now() - min(fb.timestamp for fb in user_feedback)).days),
            'last_feedback_time': max(fb.timestamp for fb in user_feedback)
        }
        
        return summary


class OnlineLearningModel:
    """Base class for online learning models."""
    
    def __init__(self, model_id: str, learning_rate: float = 0.01):
        self.model_id = model_id
        self.learning_rate = learning_rate
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.update_count = 0
        
    def partial_fit(self, X: np.ndarray, y: np.ndarray) -> bool:
        """Update model with new data."""
        raise NotImplementedError("Subclasses must implement partial_fit")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        raise NotImplementedError("Subclasses must implement predict")
    
    def get_model_state(self) -> Dict[str, Any]:
        """Get current model state."""
        return {
            'model_id': self.model_id,
            'is_fitted': self.is_fitted,
            'update_count': self.update_count,
            'learning_rate': self.learning_rate
        }


class OnlineRecommendationModel(OnlineLearningModel):
    """Online learning recommendation model using SGD."""
    
    def __init__(self, model_id: str, learning_rate: float = 0.01, 
                 loss: str = 'squared_loss'):
        super().__init__(model_id, learning_rate)
        self.loss = loss
        self.model = SGDRegressor(
            learning_rate='constant',
            eta0=learning_rate,
            loss=loss,
            random_state=42
        )
        
    def partial_fit(self, X: np.ndarray, y: np.ndarray) -> bool:
        """Update model with new feedback data."""
        try:
            if not self.is_fitted:
                # First fit - need to establish feature space
                X_scaled = self.scaler.fit_transform(X)
                self.model.partial_fit(X_scaled, y)
                self.is_fitted = True
            else:
                # Incremental update
                X_scaled = self.scaler.transform(X)
                self.model.partial_fit(X_scaled, y)
            
            self.update_count += len(X)
            return True
            
        except Exception as e:
            logger.error(f"Error updating model {self.model_id}: {e}")
            return False
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            return np.zeros(len(X))
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities (for classification models)."""
        if hasattr(self.model, 'predict_proba'):
            X_scaled = self.scaler.transform(X)
            return self.model.predict_proba(X_scaled)
        else:
            # For regression, return predictions as probabilities
            predictions = self.predict(X)
            return np.column_stack([1 - predictions, predictions])


class OnlineClassificationModel(OnlineLearningModel):
    """Online learning classification model."""
    
    def __init__(self, model_id: str, learning_rate: float = 0.01, 
                 loss: str = 'log', classes: Optional[List] = None):
        super().__init__(model_id, learning_rate)
        self.loss = loss
        self.classes = classes
        self.model = SGDClassifier(
            learning_rate='constant',
            eta0=learning_rate,
            loss=loss,
            random_state=42
        )
        
    def partial_fit(self, X: np.ndarray, y: np.ndarray) -> bool:
        """Update model with new data."""
        try:
            if not self.is_fitted:
                X_scaled = self.scaler.fit_transform(X)
                self.model.partial_fit(X_scaled, y, classes=self.classes)
                self.is_fitted = True
            else:
                X_scaled = self.scaler.transform(X)
                self.model.partial_fit(X_scaled, y)
            
            self.update_count += len(X)
            return True
            
        except Exception as e:
            logger.error(f"Error updating model {self.model_id}: {e}")
            return False
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            return np.zeros(len(X))
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities."""
        if not self.is_fitted:
            return np.ones((len(X), 2)) * 0.5  # Uniform probabilities
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)


class ContinuousLearningEngine:
    """Main engine for continuous learning from user feedback."""
    
    def __init__(self, learning_mode: LearningMode = LearningMode.HYBRID,
                 batch_size: int = 100, update_frequency_minutes: int = 60):
        self.learning_mode = learning_mode
        self.batch_size = batch_size
        self.update_frequency_minutes = update_frequency_minutes
        
        # Components
        self.feedback_processor = FeedbackProcessor()
        self.models = {}
        self.performance_tracker = {}
        
        # Threading for continuous processing
        self.feedback_queue = Queue()
        self.is_running = False
        self.learning_thread = None
        
        # Batch processing
        self.pending_updates = defaultdict(list)
        self.last_batch_update = datetime.now()
        
    def register_model(self, model: OnlineLearningModel):
        """Register a model for continuous learning."""
        self.models[model.model_id] = model
        self.performance_tracker[model.model_id] = []
        logger.info(f"Registered model {model.model_id} for continuous learning")
    
    def start_continuous_learning(self):
        """Start continuous learning process."""
        if self.is_running:
            logger.warning("Continuous learning is already running")
            return
        
        self.is_running = True
        self.learning_thread = threading.Thread(target=self._learning_loop, daemon=True)
        self.learning_thread.start()
        logger.info("Started continuous learning engine")
    
    def stop_continuous_learning(self):
        """Stop continuous learning process."""
        self.is_running = False
        if self.learning_thread:
            self.learning_thread.join(timeout=5)
        logger.info("Stopped continuous learning engine")
    
    def add_feedback(self, feedback: UserFeedback):
        """Add user feedback for processing."""
        if self.feedback_processor.process_feedback(feedback):
            self.feedback_queue.put(feedback)
    
    def _learning_loop(self):
        """Main learning loop running in separate thread."""
        while self.is_running:
            try:
                # Process feedback from queue
                self._process_feedback_batch()
                
                # Perform batch updates if needed
                if self.learning_mode in [LearningMode.BATCH, LearningMode.HYBRID]:
                    self._perform_batch_updates()
                
                # Update performance metrics
                self._update_performance_metrics()
                
                # Sleep before next iteration
                time.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in learning loop: {e}")
                time.sleep(10)  # Wait before retrying
    
    def _process_feedback_batch(self):
        """Process a batch of feedback from the queue."""
        feedback_batch = []
        
        # Collect feedback from queue
        try:
            while len(feedback_batch) < self.batch_size:
                feedback = self.feedback_queue.get(timeout=1)
                feedback_batch.append(feedback)
        except Empty:
            pass  # No more feedback in queue
        
        if not feedback_batch:
            return
        
        logger.info(f"Processing {len(feedback_batch)} feedback items")
        
        # Group feedback by model (item type)
        model_feedback = defaultdict(list)
        for feedback in feedback_batch:
            # Determine which model this feedback applies to
            model_id = self._get_model_for_item(feedback.item_id)
            if model_id:
                model_feedback[model_id].append(feedback)
        
        # Process feedback for each model
        for model_id, feedback_list in model_feedback.items():
            if self.learning_mode == LearningMode.ONLINE:
                self._update_model_online(model_id, feedback_list)
            else:
                # Add to pending updates for batch processing
                self.pending_updates[model_id].extend(feedback_list)
    
    def _update_model_online(self, model_id: str, feedback_list: List[UserFeedback]):
        """Update model with online learning."""
        if model_id not in self.models:
            return
        
        model = self.models[model_id]
        
        # Convert feedback to training data
        X, y = self._feedback_to_training_data(feedback_list)
        
        if len(X) == 0:
            return
        
        # Update model
        success = model.partial_fit(X, y)
        
        if success:
            logger.info(f"Updated model {model_id} with {len(X)} samples online")
            
            # Record update
            update = LearningUpdate(
                model_id=model_id,
                update_type='online',
                samples_processed=len(X),
                performance_change={},  # Would calculate actual performance change
                timestamp=datetime.now(),
                metadata={'feedback_types': [fb.feedback_type.value for fb in feedback_list]}
            )
    
    def _perform_batch_updates(self):
        """Perform batch updates for accumulated feedback."""
        current_time = datetime.now()
        time_since_last_update = (current_time - self.last_batch_update).total_seconds() / 60
        
        if time_since_last_update < self.update_frequency_minutes:
            return
        
        logger.info("Performing batch updates")
        
        for model_id, feedback_list in self.pending_updates.items():
            if not feedback_list:
                continue
            
            if model_id not in self.models:
                continue
            
            model = self.models[model_id]
            
            # Convert feedback to training data
            X, y = self._feedback_to_training_data(feedback_list)
            
            if len(X) == 0:
                continue
            
            # Update model
            success = model.partial_fit(X, y)
            
            if success:
                logger.info(f"Updated model {model_id} with {len(X)} samples in batch")
                
                # Clear processed feedback
                self.pending_updates[model_id] = []
        
        self.last_batch_update = current_time
    
    def _feedback_to_training_data(self, feedback_list: List[UserFeedback]) -> Tuple[np.ndarray, np.ndarray]:
        """Convert feedback to training data format."""
        if not feedback_list:
            return np.array([]), np.array([])
        
        # Extract features and targets
        features = []
        targets = []
        
        for feedback in feedback_list:
            # Create feature vector from user, item, and context
            feature_vector = self._create_feature_vector(feedback)
            target_value = self._process_feedback_value(feedback)
            
            if feature_vector is not None and target_value is not None:
                features.append(feature_vector)
                targets.append(target_value)
        
        if not features:
            return np.array([]), np.array([])
        
        return np.array(features), np.array(targets)
    
    def _create_feature_vector(self, feedback: UserFeedback) -> Optional[np.ndarray]:
        """Create feature vector from feedback."""
        # This is a simplified implementation
        # In practice, you would use user embeddings, item embeddings, context features, etc.
        
        try:
            # Basic features: user_id hash, item_id hash, timestamp features
            user_hash = hash(feedback.user_id) % 1000  # Simple hash
            item_hash = hash(feedback.item_id) % 1000
            
            # Time features
            hour = feedback.timestamp.hour
            day_of_week = feedback.timestamp.weekday()
            
            # Context features
            context_features = []
            if feedback.context:
                # Extract numeric context features
                for key, value in feedback.context.items():
                    if isinstance(value, (int, float)):
                        context_features.append(value)
            
            # Pad or truncate context features to fixed size
            context_features = context_features[:10]  # Max 10 context features
            context_features.extend([0] * (10 - len(context_features)))  # Pad with zeros
            
            feature_vector = [user_hash, item_hash, hour, day_of_week] + context_features
            return np.array(feature_vector, dtype=float)
            
        except Exception as e:
            logger.error(f"Error creating feature vector: {e}")
            return None
    
    def _process_feedback_value(self, feedback: UserFeedback) -> Optional[float]:
        """Process feedback value to target format."""
        if feedback.feedback_type == FeedbackType.EXPLICIT:
            # Normalize explicit ratings to [0, 1]
            return (feedback.feedback_value - 1) / 4  # Assuming 1-5 scale
        elif feedback.feedback_type == FeedbackType.IMPLICIT:
            # Process implicit feedback (clicks, time, etc.)
            if feedback.feedback_value > 0:
                return min(1.0, feedback.feedback_value / 100)  # Normalize
            else:
                return 0.0
        elif feedback.feedback_type == FeedbackType.NEGATIVE:
            # Negative feedback
            return 0.0
        else:
            return None
    
    def _get_model_for_item(self, item_id: str) -> Optional[str]:
        """Determine which model should handle this item."""
        # Simple implementation - in practice, you would have item-to-model mapping
        if item_id.startswith('job_'):
            return 'job_recommendation_model'
        elif item_id.startswith('course_'):
            return 'course_recommendation_model'
        elif item_id.startswith('skill_'):
            return 'skill_classification_model'
        else:
            return 'general_recommendation_model'
    
    def _update_performance_metrics(self):
        """Update performance metrics for all models."""
        current_time = datetime.now()
        
        for model_id, model in self.models.items():
            # Get recent feedback for this model
            recent_feedback = self.feedback_processor.get_recent_feedback(hours=24)
            model_feedback = [fb for fb in recent_feedback 
                            if self._get_model_for_item(fb.item_id) == model_id]
            
            if not model_feedback:
                continue
            
            # Calculate performance metrics
            metrics = self._calculate_performance_metrics(model_id, model_feedback)
            
            if metrics:
                self.performance_tracker[model_id].append(metrics)
                
                # Keep only recent metrics (last 30 days)
                cutoff_time = current_time - timedelta(days=30)
                self.performance_tracker[model_id] = [
                    m for m in self.performance_tracker[model_id] 
                    if m.timestamp >= cutoff_time
                ]
    
    def _calculate_performance_metrics(self, model_id: str, 
                                     feedback_list: List[UserFeedback]) -> Optional[ModelPerformanceMetrics]:
        """Calculate performance metrics from feedback."""
        if not feedback_list:
            return None
        
        # Calculate basic metrics
        explicit_feedback = [fb for fb in feedback_list if fb.feedback_type == FeedbackType.EXPLICIT]
        implicit_feedback = [fb for fb in feedback_list if fb.feedback_type == FeedbackType.IMPLICIT]
        
        # User satisfaction (from explicit ratings)
        if explicit_feedback:
            satisfaction_scores = [fb.feedback_value for fb in explicit_feedback]
            user_satisfaction = np.mean(satisfaction_scores) / 5.0  # Normalize to [0, 1]
        else:
            user_satisfaction = 0.5  # Neutral
        
        # Engagement rate (from implicit feedback)
        if implicit_feedback:
            engagement_values = [fb.feedback_value for fb in implicit_feedback]
            engagement_rate = min(1.0, np.mean(engagement_values) / 100)
        else:
            engagement_rate = 0.0
        
        # Conversion rate (simplified)
        positive_feedback = [fb for fb in feedback_list 
                           if (fb.feedback_type == FeedbackType.EXPLICIT and fb.feedback_value >= 4) or
                              (fb.feedback_type == FeedbackType.IMPLICIT and fb.feedback_value > 0)]
        conversion_rate = len(positive_feedback) / len(feedback_list)
        
        return ModelPerformanceMetrics(
            model_id=model_id,
            accuracy=0.0,  # Would need ground truth for actual accuracy
            precision=0.0,  # Would need ground truth
            recall=0.0,     # Would need ground truth
            f1_score=0.0,   # Would need ground truth
            user_satisfaction=user_satisfaction,
            engagement_rate=engagement_rate,
            conversion_rate=conversion_rate,
            timestamp=datetime.now(),
            sample_size=len(feedback_list)
        )
    
    def get_model_performance(self, model_id: str, days: int = 7) -> List[ModelPerformanceMetrics]:
        """Get performance metrics for a model over the last N days."""
        if model_id not in self.performance_tracker:
            return []
        
        cutoff_time = datetime.now() - timedelta(days=days)
        return [m for m in self.performance_tracker[model_id] if m.timestamp >= cutoff_time]
    
    def get_learning_summary(self) -> Dict[str, Any]:
        """Get summary of continuous learning activity."""
        summary = {
            'is_running': self.is_running,
            'learning_mode': self.learning_mode.value,
            'registered_models': len(self.models),
            'feedback_queue_size': self.feedback_queue.qsize(),
            'pending_updates': {model_id: len(updates) 
                              for model_id, updates in self.pending_updates.items()},
            'last_batch_update': self.last_batch_update,
            'model_states': {model_id: model.get_model_state() 
                           for model_id, model in self.models.items()}
        }
        
        return summary
    
    def save_models(self, directory: str):
        """Save all models to disk."""
        import os
        os.makedirs(directory, exist_ok=True)
        
        for model_id, model in self.models.items():
            model_path = os.path.join(directory, f"{model_id}.pkl")
            
            # Save model state
            model_state = {
                'model': model.model,
                'scaler': model.scaler,
                'metadata': model.get_model_state()
            }
            
            with open(model_path, 'wb') as f:
                pickle.dump(model_state, f)
        
        logger.info(f"Saved {len(self.models)} models to {directory}")
    
    def load_models(self, directory: str):
        """Load models from disk."""
        import os
        
        for filename in os.listdir(directory):
            if filename.endswith('.pkl'):
                model_id = filename[:-4]  # Remove .pkl extension
                model_path = os.path.join(directory, filename)
                
                try:
                    with open(model_path, 'rb') as f:
                        model_state = pickle.load(f)
                    
                    # Recreate model
                    if 'SGDRegressor' in str(type(model_state['model'])):
                        model = OnlineRecommendationModel(model_id)
                    else:
                        model = OnlineClassificationModel(model_id)
                    
                    model.model = model_state['model']
                    model.scaler = model_state['scaler']
                    model.is_fitted = model_state['metadata']['is_fitted']
                    model.update_count = model_state['metadata']['update_count']
                    
                    self.register_model(model)
                    
                except Exception as e:
                    logger.error(f"Error loading model {model_id}: {e}")
        
        logger.info(f"Loaded {len(self.models)} models from {directory}")


def create_feedback_simulation(n_users: int = 100, n_items: int = 50, 
                             n_feedback: int = 1000) -> List[UserFeedback]:
    """Create simulated user feedback for testing."""
    np.random.seed(42)
    
    feedback_list = []
    
    for _ in range(n_feedback):
        user_id = f"user_{np.random.randint(0, n_users)}"
        item_id = f"job_{np.random.randint(0, n_items)}"
        
        # Random feedback type
        feedback_type = np.random.choice(list(FeedbackType))
        
        # Generate feedback value based on type
        if feedback_type == FeedbackType.EXPLICIT:
            feedback_value = np.random.randint(1, 6)  # 1-5 rating
        elif feedback_type == FeedbackType.IMPLICIT:
            feedback_value = np.random.exponential(10)  # Time spent, clicks, etc.
        else:
            feedback_value = np.random.choice([0, 1])  # Binary feedback
        
        # Random context
        context = {
            'device': np.random.choice(['mobile', 'desktop', 'tablet']),
            'location': np.random.choice(['home', 'work', 'other']),
            'time_of_day': np.random.choice(['morning', 'afternoon', 'evening'])
        }
        
        # Random timestamp within last 30 days
        timestamp = datetime.now() - timedelta(
            days=np.random.randint(0, 30),
            hours=np.random.randint(0, 24),
            minutes=np.random.randint(0, 60)
        )
        
        feedback = UserFeedback(
            user_id=user_id,
            item_id=item_id,
            feedback_type=feedback_type,
            feedback_value=feedback_value,
            context=context,
            timestamp=timestamp,
            session_id=f"session_{np.random.randint(0, 1000)}"
        )
        
        feedback_list.append(feedback)
    
    return feedback_list