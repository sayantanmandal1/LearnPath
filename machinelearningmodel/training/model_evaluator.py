"""
Model evaluation and validation frameworks.

This module provides comprehensive evaluation metrics, validation strategies,
and performance analysis tools for recommendation systems and ML models.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from datetime import datetime
import json
from dataclasses import dataclass, asdict
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings

from .data_preparation import UserItemMatrix, TrainingDataset
from .model_trainer import TrainingResult


logger = logging.getLogger(__name__)


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics."""
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    auc_roc: Optional[float] = None
    auc_pr: Optional[float] = None
    rmse: Optional[float] = None
    mae: Optional[float] = None
    r2: Optional[float] = None
    ndcg_at_k: Optional[Dict[int, float]] = None
    precision_at_k: Optional[Dict[int, float]] = None
    recall_at_k: Optional[Dict[int, float]] = None
    map_score: Optional[float] = None
    coverage: Optional[float] = None
    diversity: Optional[float] = None
    novelty: Optional[float] = None
    custom_metrics: Optional[Dict[str, float]] = None


@dataclass
class RecommendationMetrics:
    """Specialized metrics for recommendation systems."""
    precision_at_k: Dict[int, float]
    recall_at_k: Dict[int, float]
    ndcg_at_k: Dict[int, float]
    map_score: float
    mrr_score: float
    coverage: float
    diversity: float
    novelty: float
    serendipity: float
    catalog_coverage: float
    intra_list_diversity: float


@dataclass
class ModelComparison:
    """Results from model comparison."""
    model_names: List[str]
    metrics: Dict[str, List[float]]
    statistical_tests: Dict[str, Dict[str, float]]
    best_model: str
    ranking: List[str]
    significance_matrix: np.ndarray


class BaseEvaluator:
    """Base class for model evaluators."""
    
    def __init__(self, metrics: List[str] = None):
        self.metrics = metrics or ['accuracy', 'precision', 'recall', 'f1_score']
        self.evaluation_history = []
        
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray, 
                **kwargs) -> EvaluationMetrics:
        """Evaluate predictions against ground truth."""
        raise NotImplementedError("Subclasses must implement evaluate method")
    
    def cross_validate(self, model, X: np.ndarray, y: np.ndarray, 
                      cv: int = 5) -> Dict[str, float]:
        """Perform cross-validation."""
        raise NotImplementedError("Subclasses must implement cross_validate method")


class ClassificationEvaluator(BaseEvaluator):
    """Evaluator for classification models."""
    
    def __init__(self, metrics: List[str] = None, average: str = 'macro'):
        super().__init__(metrics)
        self.average = average
        
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray, 
                y_prob: Optional[np.ndarray] = None, **kwargs) -> EvaluationMetrics:
        """Evaluate classification predictions."""
        logger.info("Evaluating classification model")
        
        metrics = EvaluationMetrics()
        
        # Basic classification metrics
        if 'accuracy' in self.metrics:
            metrics.accuracy = accuracy_score(y_true, y_pred)
        
        if 'precision' in self.metrics:
            metrics.precision = precision_score(y_true, y_pred, average=self.average, zero_division=0)
        
        if 'recall' in self.metrics:
            metrics.recall = recall_score(y_true, y_pred, average=self.average, zero_division=0)
        
        if 'f1_score' in self.metrics:
            metrics.f1_score = f1_score(y_true, y_pred, average=self.average, zero_division=0)
        
        # Probabilistic metrics (if probabilities are provided)
        if y_prob is not None:
            try:
                if y_prob.ndim == 1 or y_prob.shape[1] == 2:
                    # Binary classification
                    y_prob_binary = y_prob[:, 1] if y_prob.ndim > 1 else y_prob
                    metrics.auc_roc = roc_auc_score(y_true, y_prob_binary)
                    metrics.auc_pr = average_precision_score(y_true, y_prob_binary)
                else:
                    # Multi-class classification
                    metrics.auc_roc = roc_auc_score(y_true, y_prob, multi_class='ovr', average=self.average)
            except ValueError as e:
                logger.warning(f"Could not calculate AUC metrics: {e}")
        
        # Custom metrics
        metrics.custom_metrics = {}
        
        # Confusion matrix analysis
        cm = confusion_matrix(y_true, y_pred)
        metrics.custom_metrics['confusion_matrix'] = cm.tolist()
        
        # Per-class metrics for multi-class problems
        if len(np.unique(y_true)) > 2:
            per_class_precision = precision_score(y_true, y_pred, average=None, zero_division=0)
            per_class_recall = recall_score(y_true, y_pred, average=None, zero_division=0)
            per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
            
            metrics.custom_metrics['per_class_precision'] = per_class_precision.tolist()
            metrics.custom_metrics['per_class_recall'] = per_class_recall.tolist()
            metrics.custom_metrics['per_class_f1'] = per_class_f1.tolist()
        
        return metrics
    
    def cross_validate(self, model, X: np.ndarray, y: np.ndarray, 
                      cv: int = 5) -> Dict[str, float]:
        """Perform cross-validation for classification."""
        logger.info(f"Performing {cv}-fold cross-validation")
        
        results = {}
        
        # Stratified K-fold for classification
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        
        for metric in self.metrics:
            if metric == 'accuracy':
                scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy')
            elif metric == 'precision':
                scores = cross_val_score(model, X, y, cv=skf, scoring=f'precision_{self.average}')
            elif metric == 'recall':
                scores = cross_val_score(model, X, y, cv=skf, scoring=f'recall_{self.average}')
            elif metric == 'f1_score':
                scores = cross_val_score(model, X, y, cv=skf, scoring=f'f1_{self.average}')
            else:
                continue
            
            results[f'{metric}_mean'] = np.mean(scores)
            results[f'{metric}_std'] = np.std(scores)
            results[f'{metric}_scores'] = scores.tolist()
        
        return results


class RegressionEvaluator(BaseEvaluator):
    """Evaluator for regression models."""
    
    def __init__(self, metrics: List[str] = None):
        default_metrics = ['rmse', 'mae', 'r2']
        super().__init__(metrics or default_metrics)
        
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray, 
                **kwargs) -> EvaluationMetrics:
        """Evaluate regression predictions."""
        logger.info("Evaluating regression model")
        
        metrics = EvaluationMetrics()
        
        if 'rmse' in self.metrics:
            metrics.rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        
        if 'mae' in self.metrics:
            metrics.mae = mean_absolute_error(y_true, y_pred)
        
        if 'r2' in self.metrics:
            metrics.r2 = r2_score(y_true, y_pred)
        
        # Custom regression metrics
        metrics.custom_metrics = {}
        
        # Mean Absolute Percentage Error
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
        metrics.custom_metrics['mape'] = mape
        
        # Explained variance
        explained_var = 1 - np.var(y_true - y_pred) / np.var(y_true)
        metrics.custom_metrics['explained_variance'] = explained_var
        
        # Residual analysis
        residuals = y_true - y_pred
        metrics.custom_metrics['residual_mean'] = np.mean(residuals)
        metrics.custom_metrics['residual_std'] = np.std(residuals)
        
        return metrics
    
    def cross_validate(self, model, X: np.ndarray, y: np.ndarray, 
                      cv: int = 5) -> Dict[str, float]:
        """Perform cross-validation for regression."""
        logger.info(f"Performing {cv}-fold cross-validation")
        
        results = {}
        
        for metric in self.metrics:
            if metric == 'rmse':
                scores = cross_val_score(model, X, y, cv=cv, scoring='neg_root_mean_squared_error')
                scores = -scores  # Convert back to positive RMSE
            elif metric == 'mae':
                scores = cross_val_score(model, X, y, cv=cv, scoring='neg_mean_absolute_error')
                scores = -scores  # Convert back to positive MAE
            elif metric == 'r2':
                scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
            else:
                continue
            
            results[f'{metric}_mean'] = np.mean(scores)
            results[f'{metric}_std'] = np.std(scores)
            results[f'{metric}_scores'] = scores.tolist()
        
        return results


class RecommendationEvaluator(BaseEvaluator):
    """Evaluator for recommendation systems."""
    
    def __init__(self, k_values: List[int] = None):
        self.k_values = k_values or [1, 5, 10, 20]
        super().__init__()
        
    def evaluate_recommendations(self, recommendations: Dict[str, List[Tuple[str, float]]], 
                               ground_truth: Dict[str, List[str]]) -> RecommendationMetrics:
        """
        Evaluate recommendation quality.
        
        Args:
            recommendations: Dict mapping user_id to list of (item_id, score) tuples
            ground_truth: Dict mapping user_id to list of relevant item_ids
        """
        logger.info("Evaluating recommendation system")
        
        # Calculate metrics for each k value
        precision_at_k = {}
        recall_at_k = {}
        ndcg_at_k = {}
        
        for k in self.k_values:
            precision_scores = []
            recall_scores = []
            ndcg_scores = []
            
            for user_id in recommendations:
                if user_id not in ground_truth:
                    continue
                
                # Get top-k recommendations
                top_k_recs = [item_id for item_id, _ in recommendations[user_id][:k]]
                relevant_items = set(ground_truth[user_id])
                
                # Precision@k
                if len(top_k_recs) > 0:
                    precision = len(set(top_k_recs) & relevant_items) / len(top_k_recs)
                    precision_scores.append(precision)
                
                # Recall@k
                if len(relevant_items) > 0:
                    recall = len(set(top_k_recs) & relevant_items) / len(relevant_items)
                    recall_scores.append(recall)
                
                # NDCG@k
                ndcg = self._calculate_ndcg(top_k_recs, relevant_items, k)
                ndcg_scores.append(ndcg)
            
            precision_at_k[k] = np.mean(precision_scores) if precision_scores else 0
            recall_at_k[k] = np.mean(recall_scores) if recall_scores else 0
            ndcg_at_k[k] = np.mean(ndcg_scores) if ndcg_scores else 0
        
        # Mean Average Precision (MAP)
        map_score = self._calculate_map(recommendations, ground_truth)
        
        # Mean Reciprocal Rank (MRR)
        mrr_score = self._calculate_mrr(recommendations, ground_truth)
        
        # Coverage metrics
        coverage = self._calculate_coverage(recommendations, ground_truth)
        catalog_coverage = self._calculate_catalog_coverage(recommendations)
        
        # Diversity metrics
        diversity = self._calculate_diversity(recommendations)
        intra_list_diversity = self._calculate_intra_list_diversity(recommendations)
        
        # Novelty and serendipity
        novelty = self._calculate_novelty(recommendations)
        serendipity = self._calculate_serendipity(recommendations, ground_truth)
        
        return RecommendationMetrics(
            precision_at_k=precision_at_k,
            recall_at_k=recall_at_k,
            ndcg_at_k=ndcg_at_k,
            map_score=map_score,
            mrr_score=mrr_score,
            coverage=coverage,
            diversity=diversity,
            novelty=novelty,
            serendipity=serendipity,
            catalog_coverage=catalog_coverage,
            intra_list_diversity=intra_list_diversity
        )
    
    def _calculate_ndcg(self, recommendations: List[str], relevant_items: set, k: int) -> float:
        """Calculate Normalized Discounted Cumulative Gain."""
        dcg = 0
        for i, item in enumerate(recommendations[:k]):
            if item in relevant_items:
                dcg += 1 / np.log2(i + 2)  # +2 because log2(1) = 0
        
        # Ideal DCG
        idcg = sum(1 / np.log2(i + 2) for i in range(min(len(relevant_items), k)))
        
        return dcg / idcg if idcg > 0 else 0
    
    def _calculate_map(self, recommendations: Dict[str, List[Tuple[str, float]]], 
                      ground_truth: Dict[str, List[str]]) -> float:
        """Calculate Mean Average Precision."""
        ap_scores = []
        
        for user_id in recommendations:
            if user_id not in ground_truth:
                continue
            
            user_recs = [item_id for item_id, _ in recommendations[user_id]]
            relevant_items = set(ground_truth[user_id])
            
            if not relevant_items:
                continue
            
            # Calculate Average Precision for this user
            ap = 0
            relevant_count = 0
            
            for i, item in enumerate(user_recs):
                if item in relevant_items:
                    relevant_count += 1
                    precision_at_i = relevant_count / (i + 1)
                    ap += precision_at_i
            
            if relevant_count > 0:
                ap /= len(relevant_items)
                ap_scores.append(ap)
        
        return np.mean(ap_scores) if ap_scores else 0
    
    def _calculate_mrr(self, recommendations: Dict[str, List[Tuple[str, float]]], 
                      ground_truth: Dict[str, List[str]]) -> float:
        """Calculate Mean Reciprocal Rank."""
        rr_scores = []
        
        for user_id in recommendations:
            if user_id not in ground_truth:
                continue
            
            user_recs = [item_id for item_id, _ in recommendations[user_id]]
            relevant_items = set(ground_truth[user_id])
            
            # Find rank of first relevant item
            for i, item in enumerate(user_recs):
                if item in relevant_items:
                    rr_scores.append(1 / (i + 1))
                    break
            else:
                rr_scores.append(0)  # No relevant item found
        
        return np.mean(rr_scores) if rr_scores else 0
    
    def _calculate_coverage(self, recommendations: Dict[str, List[Tuple[str, float]]], 
                          ground_truth: Dict[str, List[str]]) -> float:
        """Calculate user coverage (fraction of users with at least one relevant recommendation)."""
        covered_users = 0
        
        for user_id in recommendations:
            if user_id not in ground_truth:
                continue
            
            user_recs = set(item_id for item_id, _ in recommendations[user_id])
            relevant_items = set(ground_truth[user_id])
            
            if user_recs & relevant_items:  # Intersection is not empty
                covered_users += 1
        
        return covered_users / len(recommendations) if recommendations else 0
    
    def _calculate_catalog_coverage(self, recommendations: Dict[str, List[Tuple[str, float]]]) -> float:
        """Calculate catalog coverage (fraction of items that appear in recommendations)."""
        all_items = set()
        recommended_items = set()
        
        for user_recs in recommendations.values():
            for item_id, _ in user_recs:
                recommended_items.add(item_id)
                all_items.add(item_id)
        
        # In practice, you would have the full catalog size
        # For now, we use the items that appear in recommendations
        return len(recommended_items) / len(all_items) if all_items else 0
    
    def _calculate_diversity(self, recommendations: Dict[str, List[Tuple[str, float]]]) -> float:
        """Calculate overall diversity of recommendations."""
        # Simple diversity measure: average number of unique items per user
        diversity_scores = []
        
        for user_recs in recommendations.values():
            unique_items = set(item_id for item_id, _ in user_recs)
            diversity_scores.append(len(unique_items) / len(user_recs) if user_recs else 0)
        
        return np.mean(diversity_scores) if diversity_scores else 0
    
    def _calculate_intra_list_diversity(self, recommendations: Dict[str, List[Tuple[str, float]]]) -> float:
        """Calculate intra-list diversity (diversity within each user's recommendations)."""
        # Simplified measure - in practice, you would use item features for similarity
        diversity_scores = []
        
        for user_recs in recommendations.values():
            if len(user_recs) <= 1:
                diversity_scores.append(0)
                continue
            
            # Simple measure: ratio of unique items to total items
            unique_items = set(item_id for item_id, _ in user_recs)
            diversity = len(unique_items) / len(user_recs)
            diversity_scores.append(diversity)
        
        return np.mean(diversity_scores) if diversity_scores else 0
    
    def _calculate_novelty(self, recommendations: Dict[str, List[Tuple[str, float]]]) -> float:
        """Calculate novelty of recommendations."""
        # Simple novelty measure: average inverse popularity of recommended items
        item_popularity = {}
        
        # Count item frequencies
        for user_recs in recommendations.values():
            for item_id, _ in user_recs:
                item_popularity[item_id] = item_popularity.get(item_id, 0) + 1
        
        # Calculate novelty scores
        novelty_scores = []
        for user_recs in recommendations.values():
            user_novelty = []
            for item_id, _ in user_recs:
                popularity = item_popularity[item_id]
                novelty = 1 / (1 + popularity)  # Inverse popularity
                user_novelty.append(novelty)
            
            if user_novelty:
                novelty_scores.append(np.mean(user_novelty))
        
        return np.mean(novelty_scores) if novelty_scores else 0
    
    def _calculate_serendipity(self, recommendations: Dict[str, List[Tuple[str, float]]], 
                              ground_truth: Dict[str, List[str]]) -> float:
        """Calculate serendipity of recommendations."""
        # Simplified serendipity: relevant items that are also novel
        serendipity_scores = []
        
        # Calculate item popularity from ground truth
        item_popularity = {}
        for user_items in ground_truth.values():
            for item in user_items:
                item_popularity[item] = item_popularity.get(item, 0) + 1
        
        for user_id in recommendations:
            if user_id not in ground_truth:
                continue
            
            user_recs = [item_id for item_id, _ in recommendations[user_id]]
            relevant_items = set(ground_truth[user_id])
            
            serendipitous_items = 0
            for item in user_recs:
                if item in relevant_items:
                    popularity = item_popularity.get(item, 0)
                    if popularity < np.percentile(list(item_popularity.values()), 25):  # Bottom quartile
                        serendipitous_items += 1
            
            serendipity = serendipitous_items / len(user_recs) if user_recs else 0
            serendipity_scores.append(serendipity)
        
        return np.mean(serendipity_scores) if serendipity_scores else 0


class ModelComparator:
    """Compare multiple models and perform statistical significance tests."""
    
    def __init__(self, significance_level: float = 0.05):
        self.significance_level = significance_level
        
    def compare_models(self, model_results: Dict[str, List[float]], 
                      metric_name: str = 'accuracy') -> ModelComparison:
        """
        Compare multiple models using statistical tests.
        
        Args:
            model_results: Dict mapping model names to lists of metric values
            metric_name: Name of the metric being compared
        """
        logger.info(f"Comparing {len(model_results)} models on {metric_name}")
        
        model_names = list(model_results.keys())
        n_models = len(model_names)
        
        # Calculate summary statistics
        metrics = {}
        for model_name, scores in model_results.items():
            metrics[model_name] = {
                'mean': np.mean(scores),
                'std': np.std(scores),
                'median': np.median(scores),
                'min': np.min(scores),
                'max': np.max(scores),
                'scores': scores
            }
        
        # Perform pairwise statistical tests
        significance_matrix = np.ones((n_models, n_models))
        statistical_tests = {}
        
        for i, model1 in enumerate(model_names):
            for j, model2 in enumerate(model_names):
                if i != j:
                    scores1 = model_results[model1]
                    scores2 = model_results[model2]
                    
                    # Perform t-test
                    t_stat, p_value = stats.ttest_ind(scores1, scores2)
                    significance_matrix[i, j] = p_value
                    
                    test_key = f"{model1}_vs_{model2}"
                    statistical_tests[test_key] = {
                        't_statistic': t_stat,
                        'p_value': p_value,
                        'significant': p_value < self.significance_level,
                        'effect_size': (np.mean(scores1) - np.mean(scores2)) / np.sqrt((np.var(scores1) + np.var(scores2)) / 2)
                    }
        
        # Rank models by mean performance
        ranking = sorted(model_names, key=lambda x: metrics[x]['mean'], reverse=True)
        best_model = ranking[0]
        
        return ModelComparison(
            model_names=model_names,
            metrics=metrics,
            statistical_tests=statistical_tests,
            best_model=best_model,
            ranking=ranking,
            significance_matrix=significance_matrix
        )
    
    def plot_comparison(self, comparison: ModelComparison, metric_name: str = 'accuracy', 
                       save_path: Optional[str] = None):
        """Plot model comparison results."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Box plot of model performances
        model_scores = [comparison.metrics[model]['scores'] for model in comparison.model_names]
        ax1.boxplot(model_scores, labels=comparison.model_names)
        ax1.set_title(f'Model Comparison - {metric_name}')
        ax1.set_ylabel(metric_name)
        ax1.tick_params(axis='x', rotation=45)
        
        # Significance matrix heatmap
        im = ax2.imshow(comparison.significance_matrix, cmap='RdYlBu_r', vmin=0, vmax=1)
        ax2.set_xticks(range(len(comparison.model_names)))
        ax2.set_yticks(range(len(comparison.model_names)))
        ax2.set_xticklabels(comparison.model_names, rotation=45)
        ax2.set_yticklabels(comparison.model_names)
        ax2.set_title('Statistical Significance (p-values)')
        
        # Add colorbar
        plt.colorbar(im, ax=ax2)
        
        # Add significance annotations
        for i in range(len(comparison.model_names)):
            for j in range(len(comparison.model_names)):
                p_val = comparison.significance_matrix[i, j]
                if p_val < 0.001:
                    text = '***'
                elif p_val < 0.01:
                    text = '**'
                elif p_val < 0.05:
                    text = '*'
                else:
                    text = ''
                ax2.text(j, i, text, ha='center', va='center', color='black', fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig


class ValidationFramework:
    """Comprehensive validation framework for ML models."""
    
    def __init__(self):
        self.evaluators = {
            'classification': ClassificationEvaluator(),
            'regression': RegressionEvaluator(),
            'recommendation': RecommendationEvaluator()
        }
        self.comparator = ModelComparator()
        
    def validate_model(self, model, dataset: TrainingDataset, 
                      model_type: str = 'classification') -> Dict[str, Any]:
        """Comprehensive model validation."""
        logger.info(f"Validating {model_type} model")
        
        evaluator = self.evaluators[model_type]
        results = {}
        
        # Test set evaluation
        if model_type == 'classification':
            y_pred = model.predict(dataset.X_test)
            y_prob = None
            if hasattr(model, 'predict_proba'):
                y_prob = model.predict_proba(dataset.X_test)
            
            test_metrics = evaluator.evaluate(dataset.y_test, y_pred, y_prob=y_prob)
            results['test_metrics'] = asdict(test_metrics)
            
        elif model_type == 'regression':
            y_pred = model.predict(dataset.X_test)
            test_metrics = evaluator.evaluate(dataset.y_test, y_pred)
            results['test_metrics'] = asdict(test_metrics)
        
        # Cross-validation
        cv_results = evaluator.cross_validate(model, dataset.X_train, dataset.y_train)
        results['cross_validation'] = cv_results
        
        # Learning curves
        learning_curves = self._generate_learning_curves(model, dataset, model_type)
        results['learning_curves'] = learning_curves
        
        return results
    
    def _generate_learning_curves(self, model, dataset: TrainingDataset, 
                                 model_type: str) -> Dict[str, List[float]]:
        """Generate learning curves for model validation."""
        train_sizes = np.linspace(0.1, 1.0, 10)
        train_scores = []
        val_scores = []
        
        for train_size in train_sizes:
            # Sample training data
            n_samples = int(len(dataset.X_train) * train_size)
            indices = np.random.choice(len(dataset.X_train), n_samples, replace=False)
            
            X_train_sample = dataset.X_train[indices]
            y_train_sample = dataset.y_train[indices]
            
            # Train model on sample
            model.fit(X_train_sample, y_train_sample)
            
            # Evaluate on training and validation sets
            if model_type == 'classification':
                train_pred = model.predict(X_train_sample)
                val_pred = model.predict(dataset.X_val)
                
                train_score = accuracy_score(y_train_sample, train_pred)
                val_score = accuracy_score(dataset.y_val, val_pred)
                
            elif model_type == 'regression':
                train_pred = model.predict(X_train_sample)
                val_pred = model.predict(dataset.X_val)
                
                train_score = r2_score(y_train_sample, train_pred)
                val_score = r2_score(dataset.y_val, val_pred)
            
            train_scores.append(train_score)
            val_scores.append(val_score)
        
        return {
            'train_sizes': train_sizes.tolist(),
            'train_scores': train_scores,
            'val_scores': val_scores
        }
    
    def comprehensive_evaluation(self, models: Dict[str, Any], 
                               datasets: Dict[str, TrainingDataset]) -> Dict[str, Any]:
        """Perform comprehensive evaluation of multiple models."""
        logger.info("Performing comprehensive model evaluation")
        
        results = {}
        
        for model_name, model_info in models.items():
            model = model_info['model']
            model_type = model_info['type']
            dataset_name = model_info['dataset']
            
            if dataset_name not in datasets:
                logger.warning(f"Dataset {dataset_name} not found for model {model_name}")
                continue
            
            dataset = datasets[dataset_name]
            
            # Validate model
            validation_results = self.validate_model(model, dataset, model_type)
            results[model_name] = validation_results
        
        # Compare models of the same type
        model_comparisons = {}
        model_types = {}
        
        for model_name, model_info in models.items():
            model_type = model_info['type']
            if model_type not in model_types:
                model_types[model_type] = []
            model_types[model_type].append(model_name)
        
        for model_type, model_names in model_types.items():
            if len(model_names) > 1:
                # Extract scores for comparison
                model_scores = {}
                for model_name in model_names:
                    if model_name in results:
                        if model_type == 'classification':
                            scores = results[model_name]['cross_validation'].get('accuracy_scores', [])
                        elif model_type == 'regression':
                            scores = results[model_name]['cross_validation'].get('r2_scores', [])
                        else:
                            continue
                        
                        model_scores[model_name] = scores
                
                if model_scores:
                    comparison = self.comparator.compare_models(model_scores, model_type)
                    model_comparisons[model_type] = asdict(comparison)
        
        results['model_comparisons'] = model_comparisons
        
        return results