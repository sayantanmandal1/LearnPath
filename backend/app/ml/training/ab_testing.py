"""
A/B testing infrastructure for model comparison and experimentation.

This module provides tools for conducting A/B tests on ML models,
including experiment design, traffic splitting, statistical analysis,
and result interpretation.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from datetime import datetime, timedelta
import json
import hashlib
from dataclasses import dataclass, asdict
from enum import Enum
import uuid
from scipy import stats
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

from .model_evaluator import EvaluationMetrics, ModelComparator


logger = logging.getLogger(__name__)


class ExperimentStatus(str, Enum):
    """Status of A/B test experiment."""
    DRAFT = "draft"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class TrafficSplitMethod(str, Enum):
    """Methods for splitting traffic between variants."""
    RANDOM = "random"
    HASH_BASED = "hash_based"
    STRATIFIED = "stratified"
    GEOGRAPHIC = "geographic"
    TEMPORAL = "temporal"


@dataclass
class ExperimentVariant:
    """Configuration for an experiment variant."""
    variant_id: str
    name: str
    description: str
    model_config: Dict[str, Any]
    traffic_allocation: float  # Percentage of traffic (0.0 to 1.0)
    is_control: bool = False


@dataclass
class ExperimentConfig:
    """Configuration for A/B test experiment."""
    experiment_id: str
    name: str
    description: str
    variants: List[ExperimentVariant]
    primary_metric: str
    secondary_metrics: List[str]
    minimum_sample_size: int
    significance_level: float
    power: float
    traffic_split_method: TrafficSplitMethod
    start_date: datetime
    end_date: Optional[datetime]
    stratification_features: Optional[List[str]] = None
    guardrail_metrics: Optional[List[str]] = None


@dataclass
class ExperimentResult:
    """Results from A/B test experiment."""
    experiment_id: str
    variant_id: str
    sample_size: int
    metric_values: Dict[str, List[float]]
    summary_statistics: Dict[str, Dict[str, float]]
    timestamp: datetime


@dataclass
class StatisticalTest:
    """Results from statistical significance test."""
    test_name: str
    statistic: float
    p_value: float
    confidence_interval: Tuple[float, float]
    effect_size: float
    is_significant: bool
    power: float


@dataclass
class ExperimentAnalysis:
    """Complete analysis of A/B test experiment."""
    experiment_id: str
    status: ExperimentStatus
    duration_days: int
    total_samples: int
    variant_results: Dict[str, ExperimentResult]
    statistical_tests: Dict[str, StatisticalTest]
    recommendations: List[str]
    confidence_level: float
    winner: Optional[str]
    lift: Optional[float]


class TrafficSplitter:
    """Handle traffic splitting for A/B tests."""
    
    def __init__(self, method: TrafficSplitMethod = TrafficSplitMethod.HASH_BASED):
        self.method = method
        
    def assign_variant(self, user_id: str, experiment_config: ExperimentConfig) -> str:
        """Assign user to experiment variant."""
        if self.method == TrafficSplitMethod.RANDOM:
            return self._random_assignment(experiment_config)
        elif self.method == TrafficSplitMethod.HASH_BASED:
            return self._hash_based_assignment(user_id, experiment_config)
        elif self.method == TrafficSplitMethod.STRATIFIED:
            return self._stratified_assignment(user_id, experiment_config)
        else:
            raise ValueError(f"Unsupported traffic split method: {self.method}")
    
    def _random_assignment(self, experiment_config: ExperimentConfig) -> str:
        """Randomly assign user to variant."""
        rand_val = np.random.random()
        cumulative_allocation = 0
        
        for variant in experiment_config.variants:
            cumulative_allocation += variant.traffic_allocation
            if rand_val <= cumulative_allocation:
                return variant.variant_id
        
        # Fallback to control variant
        control_variant = next((v for v in experiment_config.variants if v.is_control), 
                              experiment_config.variants[0])
        return control_variant.variant_id
    
    def _hash_based_assignment(self, user_id: str, experiment_config: ExperimentConfig) -> str:
        """Assign user based on hash of user ID for consistency."""
        # Create hash of user_id + experiment_id for consistency
        hash_input = f"{user_id}_{experiment_config.experiment_id}"
        hash_value = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
        
        # Normalize to [0, 1]
        normalized_hash = (hash_value % 10000) / 10000.0
        
        cumulative_allocation = 0
        for variant in experiment_config.variants:
            cumulative_allocation += variant.traffic_allocation
            if normalized_hash <= cumulative_allocation:
                return variant.variant_id
        
        # Fallback to control variant
        control_variant = next((v for v in experiment_config.variants if v.is_control), 
                              experiment_config.variants[0])
        return control_variant.variant_id
    
    def _stratified_assignment(self, user_id: str, experiment_config: ExperimentConfig) -> str:
        """Assign user based on stratification features."""
        # This would require user features - simplified implementation
        # In practice, you would use user demographics, behavior, etc.
        return self._hash_based_assignment(user_id, experiment_config)


class ExperimentTracker:
    """Track experiment metrics and results."""
    
    def __init__(self):
        self.experiment_data = {}
        self.user_assignments = {}
        
    def track_user_assignment(self, user_id: str, experiment_id: str, variant_id: str):
        """Track user assignment to variant."""
        if experiment_id not in self.user_assignments:
            self.user_assignments[experiment_id] = {}
        
        self.user_assignments[experiment_id][user_id] = {
            'variant_id': variant_id,
            'assignment_time': datetime.now(),
            'user_id': user_id
        }
    
    def track_metric(self, user_id: str, experiment_id: str, metric_name: str, 
                    metric_value: float, timestamp: Optional[datetime] = None):
        """Track metric value for user in experiment."""
        if timestamp is None:
            timestamp = datetime.now()
        
        if experiment_id not in self.experiment_data:
            self.experiment_data[experiment_id] = {}
        
        if user_id not in self.experiment_data[experiment_id]:
            self.experiment_data[experiment_id][user_id] = {}
        
        if metric_name not in self.experiment_data[experiment_id][user_id]:
            self.experiment_data[experiment_id][user_id][metric_name] = []
        
        self.experiment_data[experiment_id][user_id][metric_name].append({
            'value': metric_value,
            'timestamp': timestamp
        })
    
    def get_experiment_results(self, experiment_id: str) -> Dict[str, ExperimentResult]:
        """Get aggregated results for experiment."""
        if experiment_id not in self.experiment_data:
            return {}
        
        if experiment_id not in self.user_assignments:
            return {}
        
        # Group users by variant
        variant_users = {}
        for user_id, assignment in self.user_assignments[experiment_id].items():
            variant_id = assignment['variant_id']
            if variant_id not in variant_users:
                variant_users[variant_id] = []
            variant_users[variant_id].append(user_id)
        
        # Aggregate metrics by variant
        results = {}
        for variant_id, users in variant_users.items():
            variant_metrics = {}
            
            for user_id in users:
                if user_id in self.experiment_data[experiment_id]:
                    user_data = self.experiment_data[experiment_id][user_id]
                    
                    for metric_name, metric_records in user_data.items():
                        if metric_name not in variant_metrics:
                            variant_metrics[metric_name] = []
                        
                        # Use latest value for each user
                        latest_value = max(metric_records, key=lambda x: x['timestamp'])['value']
                        variant_metrics[metric_name].append(latest_value)
            
            # Calculate summary statistics
            summary_stats = {}
            for metric_name, values in variant_metrics.items():
                if values:
                    summary_stats[metric_name] = {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'median': np.median(values),
                        'min': np.min(values),
                        'max': np.max(values),
                        'count': len(values)
                    }
            
            results[variant_id] = ExperimentResult(
                experiment_id=experiment_id,
                variant_id=variant_id,
                sample_size=len(users),
                metric_values=variant_metrics,
                summary_statistics=summary_stats,
                timestamp=datetime.now()
            )
        
        return results


class StatisticalAnalyzer:
    """Perform statistical analysis of A/B test results."""
    
    def __init__(self, significance_level: float = 0.05):
        self.significance_level = significance_level
        
    def analyze_experiment(self, experiment_config: ExperimentConfig, 
                          results: Dict[str, ExperimentResult]) -> ExperimentAnalysis:
        """Perform complete statistical analysis of experiment."""
        logger.info(f"Analyzing experiment {experiment_config.experiment_id}")
        
        # Find control and treatment variants
        control_variant = next((v for v in experiment_config.variants if v.is_control), None)
        if not control_variant:
            control_variant = experiment_config.variants[0]  # Use first as control
        
        control_result = results.get(control_variant.variant_id)
        if not control_result:
            raise ValueError("Control variant results not found")
        
        # Perform statistical tests for each variant vs control
        statistical_tests = {}
        winner = None
        best_lift = 0
        
        for variant in experiment_config.variants:
            if variant.variant_id == control_variant.variant_id:
                continue  # Skip control vs control
            
            variant_result = results.get(variant.variant_id)
            if not variant_result:
                continue
            
            # Test primary metric
            primary_metric = experiment_config.primary_metric
            test_result = self._perform_statistical_test(
                control_result, variant_result, primary_metric
            )
            
            statistical_tests[f"{variant.variant_id}_vs_control"] = test_result
            
            # Check if this is the best performing variant
            if test_result.is_significant and test_result.effect_size > best_lift:
                winner = variant.variant_id
                best_lift = test_result.effect_size
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            experiment_config, results, statistical_tests
        )
        
        # Calculate experiment duration
        start_date = experiment_config.start_date
        end_date = experiment_config.end_date or datetime.now()
        duration_days = (end_date - start_date).days
        
        # Calculate total samples
        total_samples = sum(result.sample_size for result in results.values())
        
        return ExperimentAnalysis(
            experiment_id=experiment_config.experiment_id,
            status=ExperimentStatus.COMPLETED,
            duration_days=duration_days,
            total_samples=total_samples,
            variant_results=results,
            statistical_tests=statistical_tests,
            recommendations=recommendations,
            confidence_level=1 - self.significance_level,
            winner=winner,
            lift=best_lift
        )
    
    def _perform_statistical_test(self, control_result: ExperimentResult, 
                                 variant_result: ExperimentResult, 
                                 metric_name: str) -> StatisticalTest:
        """Perform statistical test between control and variant."""
        control_values = control_result.metric_values.get(metric_name, [])
        variant_values = variant_result.metric_values.get(metric_name, [])
        
        if not control_values or not variant_values:
            return StatisticalTest(
                test_name="t_test",
                statistic=0,
                p_value=1.0,
                confidence_interval=(0, 0),
                effect_size=0,
                is_significant=False,
                power=0
            )
        
        # Perform two-sample t-test
        t_stat, p_value = stats.ttest_ind(variant_values, control_values)
        
        # Calculate confidence interval for difference in means
        control_mean = np.mean(control_values)
        variant_mean = np.mean(variant_values)
        
        pooled_std = np.sqrt(((len(control_values) - 1) * np.var(control_values) + 
                             (len(variant_values) - 1) * np.var(variant_values)) / 
                            (len(control_values) + len(variant_values) - 2))
        
        se_diff = pooled_std * np.sqrt(1/len(control_values) + 1/len(variant_values))
        
        # 95% confidence interval
        t_critical = stats.t.ppf(1 - self.significance_level/2, 
                                len(control_values) + len(variant_values) - 2)
        
        mean_diff = variant_mean - control_mean
        margin_error = t_critical * se_diff
        
        ci_lower = mean_diff - margin_error
        ci_upper = mean_diff + margin_error
        
        # Calculate effect size (Cohen's d)
        effect_size = mean_diff / pooled_std if pooled_std > 0 else 0
        
        # Calculate statistical power (simplified)
        power = self._calculate_power(len(control_values), len(variant_values), 
                                    effect_size, self.significance_level)
        
        return StatisticalTest(
            test_name="t_test",
            statistic=t_stat,
            p_value=p_value,
            confidence_interval=(ci_lower, ci_upper),
            effect_size=effect_size,
            is_significant=p_value < self.significance_level,
            power=power
        )
    
    def _calculate_power(self, n1: int, n2: int, effect_size: float, 
                       alpha: float) -> float:
        """Calculate statistical power (simplified approximation)."""
        # Simplified power calculation
        # In practice, you would use more sophisticated methods
        n_harmonic = 2 * n1 * n2 / (n1 + n2)
        ncp = effect_size * np.sqrt(n_harmonic / 2)  # Non-centrality parameter
        
        # Approximate power using normal distribution
        z_alpha = stats.norm.ppf(1 - alpha/2)
        z_beta = ncp - z_alpha
        power = stats.norm.cdf(z_beta)
        
        return max(0, min(1, power))
    
    def _generate_recommendations(self, experiment_config: ExperimentConfig,
                                results: Dict[str, ExperimentResult],
                                statistical_tests: Dict[str, StatisticalTest]) -> List[str]:
        """Generate recommendations based on experiment results."""
        recommendations = []
        
        # Check sample size adequacy
        min_sample_size = experiment_config.minimum_sample_size
        for variant_id, result in results.items():
            if result.sample_size < min_sample_size:
                recommendations.append(
                    f"Variant {variant_id} has insufficient sample size "
                    f"({result.sample_size} < {min_sample_size}). "
                    "Consider running the experiment longer."
                )
        
        # Check for significant results
        significant_tests = [test for test in statistical_tests.values() if test.is_significant]
        
        if not significant_tests:
            recommendations.append(
                "No statistically significant differences found. "
                "Consider running the experiment longer or increasing effect size."
            )
        else:
            best_test = max(significant_tests, key=lambda x: x.effect_size)
            recommendations.append(
                f"Significant improvement found with effect size {best_test.effect_size:.3f}. "
                "Consider implementing the winning variant."
            )
        
        # Check statistical power
        low_power_tests = [test for test in statistical_tests.values() if test.power < 0.8]
        if low_power_tests:
            recommendations.append(
                f"{len(low_power_tests)} tests have low statistical power (< 0.8). "
                "Results may not be reliable. Consider increasing sample size."
            )
        
        # Check for guardrail metrics
        if experiment_config.guardrail_metrics:
            recommendations.append(
                "Review guardrail metrics to ensure no negative impact on key business metrics."
            )
        
        return recommendations


class ABTestFramework:
    """Complete A/B testing framework."""
    
    def __init__(self):
        self.traffic_splitter = TrafficSplitter()
        self.experiment_tracker = ExperimentTracker()
        self.statistical_analyzer = StatisticalAnalyzer()
        self.active_experiments = {}
        
    def create_experiment(self, config: ExperimentConfig) -> str:
        """Create new A/B test experiment."""
        logger.info(f"Creating experiment: {config.name}")
        
        # Validate configuration
        self._validate_experiment_config(config)
        
        # Store experiment configuration
        self.active_experiments[config.experiment_id] = {
            'config': config,
            'status': ExperimentStatus.DRAFT,
            'created_at': datetime.now()
        }
        
        return config.experiment_id
    
    def start_experiment(self, experiment_id: str):
        """Start running an experiment."""
        if experiment_id not in self.active_experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        self.active_experiments[experiment_id]['status'] = ExperimentStatus.RUNNING
        self.active_experiments[experiment_id]['started_at'] = datetime.now()
        
        logger.info(f"Started experiment: {experiment_id}")
    
    def assign_user_to_experiment(self, user_id: str, experiment_id: str) -> str:
        """Assign user to experiment variant."""
        if experiment_id not in self.active_experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        experiment_info = self.active_experiments[experiment_id]
        if experiment_info['status'] != ExperimentStatus.RUNNING:
            raise ValueError(f"Experiment {experiment_id} is not running")
        
        config = experiment_info['config']
        variant_id = self.traffic_splitter.assign_variant(user_id, config)
        
        # Track assignment
        self.experiment_tracker.track_user_assignment(user_id, experiment_id, variant_id)
        
        return variant_id
    
    def track_metric(self, user_id: str, experiment_id: str, metric_name: str, 
                    metric_value: float):
        """Track metric for user in experiment."""
        self.experiment_tracker.track_metric(user_id, experiment_id, metric_name, metric_value)
    
    def analyze_experiment(self, experiment_id: str) -> ExperimentAnalysis:
        """Analyze experiment results."""
        if experiment_id not in self.active_experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        config = self.active_experiments[experiment_id]['config']
        results = self.experiment_tracker.get_experiment_results(experiment_id)
        
        return self.statistical_analyzer.analyze_experiment(config, results)
    
    def stop_experiment(self, experiment_id: str) -> ExperimentAnalysis:
        """Stop experiment and return final analysis."""
        if experiment_id not in self.active_experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        # Mark as completed
        self.active_experiments[experiment_id]['status'] = ExperimentStatus.COMPLETED
        self.active_experiments[experiment_id]['completed_at'] = datetime.now()
        
        # Perform final analysis
        analysis = self.analyze_experiment(experiment_id)
        
        logger.info(f"Completed experiment: {experiment_id}")
        return analysis
    
    def _validate_experiment_config(self, config: ExperimentConfig):
        """Validate experiment configuration."""
        # Check traffic allocation sums to 1.0
        total_allocation = sum(variant.traffic_allocation for variant in config.variants)
        if abs(total_allocation - 1.0) > 0.001:
            raise ValueError(f"Traffic allocation must sum to 1.0, got {total_allocation}")
        
        # Check at least one control variant
        control_variants = [v for v in config.variants if v.is_control]
        if len(control_variants) != 1:
            raise ValueError("Exactly one control variant must be specified")
        
        # Check significance level
        if not 0 < config.significance_level < 1:
            raise ValueError("Significance level must be between 0 and 1")
        
        # Check minimum sample size
        if config.minimum_sample_size <= 0:
            raise ValueError("Minimum sample size must be positive")
    
    def get_experiment_status(self, experiment_id: str) -> Dict[str, Any]:
        """Get current status of experiment."""
        if experiment_id not in self.active_experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        experiment_info = self.active_experiments[experiment_id]
        config = experiment_info['config']
        
        # Get current results
        results = self.experiment_tracker.get_experiment_results(experiment_id)
        
        status_info = {
            'experiment_id': experiment_id,
            'name': config.name,
            'status': experiment_info['status'].value,
            'created_at': experiment_info['created_at'],
            'variants': [
                {
                    'variant_id': variant.variant_id,
                    'name': variant.name,
                    'traffic_allocation': variant.traffic_allocation,
                    'is_control': variant.is_control,
                    'sample_size': results.get(variant.variant_id, ExperimentResult('', '', 0, {}, {}, datetime.now())).sample_size
                }
                for variant in config.variants
            ],
            'total_samples': sum(result.sample_size for result in results.values()),
            'primary_metric': config.primary_metric,
            'duration_days': (datetime.now() - experiment_info['created_at']).days
        }
        
        if 'started_at' in experiment_info:
            status_info['started_at'] = experiment_info['started_at']
        
        if 'completed_at' in experiment_info:
            status_info['completed_at'] = experiment_info['completed_at']
        
        return status_info
    
    def plot_experiment_results(self, experiment_id: str, metric_name: str, 
                               save_path: Optional[str] = None):
        """Plot experiment results."""
        results = self.experiment_tracker.get_experiment_results(experiment_id)
        config = self.active_experiments[experiment_id]['config']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Box plot of metric values by variant
        variant_data = []
        variant_labels = []
        
        for variant in config.variants:
            if variant.variant_id in results:
                values = results[variant.variant_id].metric_values.get(metric_name, [])
                if values:
                    variant_data.append(values)
                    variant_labels.append(f"{variant.name}\n(n={len(values)})")
        
        if variant_data:
            ax1.boxplot(variant_data, labels=variant_labels)
            ax1.set_title(f'{metric_name} by Variant')
            ax1.set_ylabel(metric_name)
            ax1.tick_params(axis='x', rotation=45)
        
        # Bar plot of mean values with confidence intervals
        means = []
        stds = []
        labels = []
        
        for variant in config.variants:
            if variant.variant_id in results:
                summary = results[variant.variant_id].summary_statistics.get(metric_name, {})
                if summary:
                    means.append(summary['mean'])
                    stds.append(summary['std'] / np.sqrt(summary['count']))  # Standard error
                    labels.append(variant.name)
        
        if means:
            x_pos = np.arange(len(labels))
            ax2.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7)
            ax2.set_xticks(x_pos)
            ax2.set_xticklabels(labels, rotation=45)
            ax2.set_title(f'Mean {metric_name} with 95% CI')
            ax2.set_ylabel(metric_name)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig


def create_model_comparison_experiment(model_configs: Dict[str, Dict[str, Any]], 
                                     experiment_name: str = "Model Comparison",
                                     traffic_split: Optional[Dict[str, float]] = None) -> ExperimentConfig:
    """Create A/B test experiment for comparing ML models."""
    experiment_id = str(uuid.uuid4())
    
    # Default equal traffic split
    if traffic_split is None:
        traffic_per_model = 1.0 / len(model_configs)
        traffic_split = {model_id: traffic_per_model for model_id in model_configs.keys()}
    
    # Create variants
    variants = []
    control_set = False
    
    for i, (model_id, config) in enumerate(model_configs.items()):
        is_control = not control_set  # First model is control
        if is_control:
            control_set = True
        
        variant = ExperimentVariant(
            variant_id=model_id,
            name=config.get('name', f'Model {i+1}'),
            description=config.get('description', f'Model variant {model_id}'),
            model_config=config,
            traffic_allocation=traffic_split[model_id],
            is_control=is_control
        )
        variants.append(variant)
    
    return ExperimentConfig(
        experiment_id=experiment_id,
        name=experiment_name,
        description=f"A/B test comparing {len(model_configs)} ML models",
        variants=variants,
        primary_metric='accuracy',
        secondary_metrics=['precision', 'recall', 'f1_score'],
        minimum_sample_size=1000,
        significance_level=0.05,
        power=0.8,
        traffic_split_method=TrafficSplitMethod.HASH_BASED,
        start_date=datetime.now(),
        end_date=None
    )