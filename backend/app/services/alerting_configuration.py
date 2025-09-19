"""
Alerting Configuration and Management Service
Manages alert rules, thresholds, and notification channels
"""
import json
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, asdict
from enum import Enum
import structlog

from app.core.monitoring import system_monitor, Alert, AlertType, AlertSeverity
from app.core.alerting import alerting_system, NotificationChannel, NotificationConfig
from app.core.redis import redis_manager

logger = structlog.get_logger()


class AlertRuleType(Enum):
    """Types of alert rules"""
    THRESHOLD = "threshold"
    RATE = "rate"
    ANOMALY = "anomaly"
    PATTERN = "pattern"
    COMPOSITE = "composite"


class ComparisonOperator(Enum):
    """Comparison operators for thresholds"""
    GREATER_THAN = "gt"
    LESS_THAN = "lt"
    EQUAL = "eq"
    NOT_EQUAL = "ne"
    GREATER_EQUAL = "ge"
    LESS_EQUAL = "le"


@dataclass
class AlertRule:
    """Alert rule configuration"""
    id: str
    name: str
    description: str
    rule_type: AlertRuleType
    metric_name: str
    threshold_value: float
    comparison_operator: ComparisonOperator
    evaluation_window_minutes: int
    severity: AlertSeverity
    enabled: bool = True
    notification_channels: Set[NotificationChannel] = None
    conditions: Dict[str, Any] = None
    tags: Dict[str, str] = None
    created_at: datetime = None
    updated_at: datetime = None
    
    def __post_init__(self):
        if self.notification_channels is None:
            self.notification_channels = {NotificationChannel.EMAIL}
        if self.conditions is None:
            self.conditions = {}
        if self.tags is None:
            self.tags = {}
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.updated_at is None:
            self.updated_at = datetime.utcnow()


@dataclass
class AlertRuleEvaluation:
    """Result of alert rule evaluation"""
    rule_id: str
    triggered: bool
    current_value: float
    threshold_value: float
    evaluation_time: datetime
    message: str
    metadata: Dict[str, Any] = None


class AlertingConfigurationService:
    """Service for managing alerting configuration"""
    
    def __init__(self):
        self.alert_rules: Dict[str, AlertRule] = {}
        self.evaluation_history: Dict[str, List[AlertRuleEvaluation]] = {}
        self.rule_states: Dict[str, Dict[str, Any]] = {}
        
        # Initialize default alert rules
        self._initialize_default_rules()
    
    def _initialize_default_rules(self):
        """Initialize default alert rules"""
        default_rules = [
            AlertRule(
                id="api_error_rate_high",
                name="High API Error Rate",
                description="Alert when API error rate exceeds 5%",
                rule_type=AlertRuleType.THRESHOLD,
                metric_name="api_error_rate_percent",
                threshold_value=5.0,
                comparison_operator=ComparisonOperator.GREATER_THAN,
                evaluation_window_minutes=5,
                severity=AlertSeverity.HIGH,
                notification_channels={NotificationChannel.EMAIL, NotificationChannel.SLACK},
                tags={"category": "api_performance", "priority": "high"}
            ),
            AlertRule(
                id="api_response_time_slow",
                name="Slow API Response Time",
                description="Alert when API response time exceeds 2 seconds",
                rule_type=AlertRuleType.THRESHOLD,
                metric_name="api_response_time_p95_ms",
                threshold_value=2000.0,
                comparison_operator=ComparisonOperator.GREATER_THAN,
                evaluation_window_minutes=10,
                severity=AlertSeverity.MEDIUM,
                notification_channels={NotificationChannel.SLACK},
                tags={"category": "api_performance", "priority": "medium"}
            ),
            AlertRule(
                id="external_api_failure_rate_high",
                name="High External API Failure Rate",
                description="Alert when external API failure rate exceeds 10%",
                rule_type=AlertRuleType.THRESHOLD,
                metric_name="external_api_failure_rate_percent",
                threshold_value=10.0,
                comparison_operator=ComparisonOperator.GREATER_THAN,
                evaluation_window_minutes=5,
                severity=AlertSeverity.HIGH,
                notification_channels={NotificationChannel.EMAIL, NotificationChannel.WEBHOOK},
                tags={"category": "external_apis", "priority": "high"}
            ),
            AlertRule(
                id="system_cpu_high",
                name="High System CPU Usage",
                description="Alert when system CPU usage exceeds 80%",
                rule_type=AlertRuleType.THRESHOLD,
                metric_name="system_cpu_percent",
                threshold_value=80.0,
                comparison_operator=ComparisonOperator.GREATER_THAN,
                evaluation_window_minutes=5,
                severity=AlertSeverity.MEDIUM,
                notification_channels={NotificationChannel.EMAIL},
                tags={"category": "system_resources", "priority": "medium"}
            ),
            AlertRule(
                id="system_memory_high",
                name="High System Memory Usage",
                description="Alert when system memory usage exceeds 85%",
                rule_type=AlertRuleType.THRESHOLD,
                metric_name="system_memory_percent",
                threshold_value=85.0,
                comparison_operator=ComparisonOperator.GREATER_THAN,
                evaluation_window_minutes=5,
                severity=AlertSeverity.HIGH,
                notification_channels={NotificationChannel.EMAIL, NotificationChannel.SLACK},
                tags={"category": "system_resources", "priority": "high"}
            ),
            AlertRule(
                id="cache_hit_rate_low",
                name="Low Cache Hit Rate",
                description="Alert when cache hit rate falls below 70%",
                rule_type=AlertRuleType.THRESHOLD,
                metric_name="cache_hit_rate_percent",
                threshold_value=70.0,
                comparison_operator=ComparisonOperator.LESS_THAN,
                evaluation_window_minutes=15,
                severity=AlertSeverity.LOW,
                notification_channels={NotificationChannel.SLACK},
                tags={"category": "caching", "priority": "low"}
            ),
            AlertRule(
                id="user_engagement_drop",
                name="User Engagement Drop",
                description="Alert when user engagement drops significantly",
                rule_type=AlertRuleType.RATE,
                metric_name="user_engagement_events_per_hour",
                threshold_value=-30.0,  # 30% decrease
                comparison_operator=ComparisonOperator.LESS_THAN,
                evaluation_window_minutes=60,
                severity=AlertSeverity.MEDIUM,
                notification_channels={NotificationChannel.EMAIL},
                tags={"category": "user_engagement", "priority": "medium"}
            ),
            AlertRule(
                id="database_connections_high",
                name="High Database Connection Usage",
                description="Alert when database connections exceed 80% of limit",
                rule_type=AlertRuleType.THRESHOLD,
                metric_name="database_connections_percent",
                threshold_value=80.0,
                comparison_operator=ComparisonOperator.GREATER_THAN,
                evaluation_window_minutes=5,
                severity=AlertSeverity.HIGH,
                notification_channels={NotificationChannel.EMAIL, NotificationChannel.WEBHOOK},
                tags={"category": "database", "priority": "high"}
            ),
            AlertRule(
                id="ml_model_accuracy_low",
                name="ML Model Accuracy Drop",
                description="Alert when ML model accuracy drops below acceptable threshold",
                rule_type=AlertRuleType.THRESHOLD,
                metric_name="ml_model_accuracy_percent",
                threshold_value=75.0,
                comparison_operator=ComparisonOperator.LESS_THAN,
                evaluation_window_minutes=30,
                severity=AlertSeverity.MEDIUM,
                notification_channels={NotificationChannel.EMAIL},
                tags={"category": "ml_models", "priority": "medium"}
            ),
            AlertRule(
                id="queue_size_high",
                name="High Task Queue Size",
                description="Alert when task queue size exceeds 1000 tasks",
                rule_type=AlertRuleType.THRESHOLD,
                metric_name="task_queue_size",
                threshold_value=1000.0,
                comparison_operator=ComparisonOperator.GREATER_THAN,
                evaluation_window_minutes=5,
                severity=AlertSeverity.HIGH,
                notification_channels={NotificationChannel.EMAIL, NotificationChannel.SLACK},
                tags={"category": "task_processing", "priority": "high"}
            )
        ]
        
        for rule in default_rules:
            self.alert_rules[rule.id] = rule
    
    async def create_alert_rule(self, rule: AlertRule) -> str:
        """Create a new alert rule"""
        try:
            rule.created_at = datetime.utcnow()
            rule.updated_at = datetime.utcnow()
            
            self.alert_rules[rule.id] = rule
            
            # Store in Redis for persistence
            await self._store_rule_in_redis(rule)
            
            logger.info(f"Created alert rule: {rule.name} ({rule.id})")
            return rule.id
            
        except Exception as e:
            logger.error(f"Failed to create alert rule: {e}")
            raise
    
    async def update_alert_rule(self, rule_id: str, updates: Dict[str, Any]) -> bool:
        """Update an existing alert rule"""
        try:
            if rule_id not in self.alert_rules:
                return False
            
            rule = self.alert_rules[rule_id]
            
            # Update fields
            for field, value in updates.items():
                if hasattr(rule, field):
                    setattr(rule, field, value)
            
            rule.updated_at = datetime.utcnow()
            
            # Store updated rule in Redis
            await self._store_rule_in_redis(rule)
            
            logger.info(f"Updated alert rule: {rule.name} ({rule.id})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update alert rule {rule_id}: {e}")
            raise
    
    async def delete_alert_rule(self, rule_id: str) -> bool:
        """Delete an alert rule"""
        try:
            if rule_id not in self.alert_rules:
                return False
            
            rule = self.alert_rules[rule_id]
            del self.alert_rules[rule_id]
            
            # Remove from Redis
            await redis_manager.delete(f"alert_rule:{rule_id}")
            
            # Clean up evaluation history
            if rule_id in self.evaluation_history:
                del self.evaluation_history[rule_id]
            
            if rule_id in self.rule_states:
                del self.rule_states[rule_id]
            
            logger.info(f"Deleted alert rule: {rule.name} ({rule.id})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete alert rule {rule_id}: {e}")
            raise
    
    def get_alert_rule(self, rule_id: str) -> Optional[AlertRule]:
        """Get an alert rule by ID"""
        return self.alert_rules.get(rule_id)
    
    def get_all_alert_rules(self, enabled_only: bool = False) -> List[AlertRule]:
        """Get all alert rules"""
        rules = list(self.alert_rules.values())
        if enabled_only:
            rules = [rule for rule in rules if rule.enabled]
        return rules
    
    def get_alert_rules_by_category(self, category: str) -> List[AlertRule]:
        """Get alert rules by category tag"""
        return [
            rule for rule in self.alert_rules.values()
            if rule.tags.get("category") == category
        ]
    
    async def evaluate_alert_rules(self, metrics: Dict[str, float]) -> List[AlertRuleEvaluation]:
        """Evaluate all enabled alert rules against current metrics"""
        evaluations = []
        
        try:
            for rule in self.alert_rules.values():
                if not rule.enabled:
                    continue
                
                evaluation = await self._evaluate_single_rule(rule, metrics)
                evaluations.append(evaluation)
                
                # Store evaluation in history
                if rule.id not in self.evaluation_history:
                    self.evaluation_history[rule.id] = []
                
                self.evaluation_history[rule.id].append(evaluation)
                
                # Keep only recent evaluations (last 1000)
                if len(self.evaluation_history[rule.id]) > 1000:
                    self.evaluation_history[rule.id] = self.evaluation_history[rule.id][-1000:]
                
                # Create alert if rule is triggered
                if evaluation.triggered:
                    await self._create_alert_from_evaluation(rule, evaluation)
            
            return evaluations
            
        except Exception as e:
            logger.error(f"Failed to evaluate alert rules: {e}")
            return []
    
    async def _evaluate_single_rule(self, rule: AlertRule, metrics: Dict[str, float]) -> AlertRuleEvaluation:
        """Evaluate a single alert rule"""
        try:
            current_value = metrics.get(rule.metric_name, 0.0)
            triggered = False
            message = ""
            
            # Evaluate based on comparison operator
            if rule.comparison_operator == ComparisonOperator.GREATER_THAN:
                triggered = current_value > rule.threshold_value
                message = f"{rule.metric_name} ({current_value}) > {rule.threshold_value}"
            elif rule.comparison_operator == ComparisonOperator.LESS_THAN:
                triggered = current_value < rule.threshold_value
                message = f"{rule.metric_name} ({current_value}) < {rule.threshold_value}"
            elif rule.comparison_operator == ComparisonOperator.EQUAL:
                triggered = current_value == rule.threshold_value
                message = f"{rule.metric_name} ({current_value}) == {rule.threshold_value}"
            elif rule.comparison_operator == ComparisonOperator.NOT_EQUAL:
                triggered = current_value != rule.threshold_value
                message = f"{rule.metric_name} ({current_value}) != {rule.threshold_value}"
            elif rule.comparison_operator == ComparisonOperator.GREATER_EQUAL:
                triggered = current_value >= rule.threshold_value
                message = f"{rule.metric_name} ({current_value}) >= {rule.threshold_value}"
            elif rule.comparison_operator == ComparisonOperator.LESS_EQUAL:
                triggered = current_value <= rule.threshold_value
                message = f"{rule.metric_name} ({current_value}) <= {rule.threshold_value}"
            
            # Check for rate-based rules
            if rule.rule_type == AlertRuleType.RATE:
                triggered = await self._evaluate_rate_rule(rule, current_value)
                message = f"Rate change for {rule.metric_name}: {current_value}%"
            
            return AlertRuleEvaluation(
                rule_id=rule.id,
                triggered=triggered,
                current_value=current_value,
                threshold_value=rule.threshold_value,
                evaluation_time=datetime.utcnow(),
                message=message,
                metadata={
                    "rule_name": rule.name,
                    "rule_type": rule.rule_type.value,
                    "comparison_operator": rule.comparison_operator.value,
                    "tags": rule.tags
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to evaluate rule {rule.id}: {e}")
            return AlertRuleEvaluation(
                rule_id=rule.id,
                triggered=False,
                current_value=0.0,
                threshold_value=rule.threshold_value,
                evaluation_time=datetime.utcnow(),
                message=f"Evaluation failed: {str(e)}"
            )
    
    async def _evaluate_rate_rule(self, rule: AlertRule, current_value: float) -> bool:
        """Evaluate rate-based alert rule"""
        try:
            # Get historical values for rate calculation
            history = self.evaluation_history.get(rule.id, [])
            if len(history) < 2:
                return False  # Need at least 2 data points
            
            # Calculate rate of change
            previous_value = history[-1].current_value
            if previous_value == 0:
                return False  # Avoid division by zero
            
            rate_change = ((current_value - previous_value) / previous_value) * 100
            
            # Check if rate change exceeds threshold
            return rate_change < rule.threshold_value  # Negative threshold for decrease
            
        except Exception as e:
            logger.error(f"Failed to evaluate rate rule {rule.id}: {e}")
            return False
    
    async def _create_alert_from_evaluation(self, rule: AlertRule, evaluation: AlertRuleEvaluation):
        """Create an alert from rule evaluation"""
        try:
            # Check if we already have an active alert for this rule
            active_alerts = system_monitor.get_active_alerts()
            existing_alert = any(
                alert for alert in active_alerts
                if alert.metadata.get("rule_id") == rule.id
            )
            
            if existing_alert:
                return  # Don't create duplicate alerts
            
            # Create alert
            alert = system_monitor.create_alert(
                alert_type=AlertType.RESOURCE_EXHAUSTION,  # Default type
                severity=rule.severity,
                title=rule.name,
                description=f"{rule.description}\n{evaluation.message}",
                service="alerting_system",
                metadata={
                    "rule_id": rule.id,
                    "rule_name": rule.name,
                    "metric_name": rule.metric_name,
                    "current_value": evaluation.current_value,
                    "threshold_value": evaluation.threshold_value,
                    "evaluation_time": evaluation.evaluation_time.isoformat(),
                    "tags": rule.tags
                }
            )
            
            logger.warning(f"Alert triggered by rule {rule.name}: {evaluation.message}")
            
        except Exception as e:
            logger.error(f"Failed to create alert from evaluation: {e}")
    
    async def _store_rule_in_redis(self, rule: AlertRule):
        """Store alert rule in Redis"""
        try:
            rule_data = asdict(rule)
            # Convert datetime objects to strings
            rule_data["created_at"] = rule.created_at.isoformat()
            rule_data["updated_at"] = rule.updated_at.isoformat()
            # Convert enums to strings
            rule_data["rule_type"] = rule.rule_type.value
            rule_data["comparison_operator"] = rule.comparison_operator.value
            rule_data["severity"] = rule.severity.value
            rule_data["notification_channels"] = [ch.value for ch in rule.notification_channels]
            
            await redis_manager.setex(
                f"alert_rule:{rule.id}",
                86400,  # 24 hours TTL
                json.dumps(rule_data)
            )
            
        except Exception as e:
            logger.error(f"Failed to store rule in Redis: {e}")
    
    async def load_rules_from_redis(self):
        """Load alert rules from Redis"""
        try:
            # This would typically scan for all alert rule keys
            # For now, we'll keep the in-memory rules
            pass
            
        except Exception as e:
            logger.error(f"Failed to load rules from Redis: {e}")
    
    def get_rule_evaluation_history(self, rule_id: str, limit: int = 100) -> List[AlertRuleEvaluation]:
        """Get evaluation history for a rule"""
        history = self.evaluation_history.get(rule_id, [])
        return history[-limit:] if limit > 0 else history
    
    def get_rule_statistics(self, rule_id: str) -> Dict[str, Any]:
        """Get statistics for a rule"""
        try:
            history = self.evaluation_history.get(rule_id, [])
            if not history:
                return {"error": "No evaluation history available"}
            
            total_evaluations = len(history)
            triggered_count = len([e for e in history if e.triggered])
            trigger_rate = (triggered_count / total_evaluations) * 100
            
            recent_history = history[-24:]  # Last 24 evaluations
            recent_triggered = len([e for e in recent_history if e.triggered])
            recent_trigger_rate = (recent_triggered / len(recent_history)) * 100 if recent_history else 0
            
            return {
                "rule_id": rule_id,
                "total_evaluations": total_evaluations,
                "triggered_count": triggered_count,
                "trigger_rate_percent": round(trigger_rate, 2),
                "recent_trigger_rate_percent": round(recent_trigger_rate, 2),
                "last_evaluation": history[-1].evaluation_time.isoformat() if history else None,
                "last_triggered": next(
                    (e.evaluation_time.isoformat() for e in reversed(history) if e.triggered),
                    None
                )
            }
            
        except Exception as e:
            logger.error(f"Failed to get rule statistics for {rule_id}: {e}")
            return {"error": str(e)}
    
    async def configure_notification_channel(self, config: NotificationConfig):
        """Configure a notification channel"""
        try:
            alerting_system.add_notification_config(config)
            logger.info(f"Configured notification channel: {config.channel.value}")
            
        except Exception as e:
            logger.error(f"Failed to configure notification channel: {e}")
            raise
    
    async def test_alert_rule(self, rule_id: str, test_metrics: Dict[str, float]) -> AlertRuleEvaluation:
        """Test an alert rule with provided metrics"""
        try:
            rule = self.get_alert_rule(rule_id)
            if not rule:
                raise ValueError(f"Alert rule {rule_id} not found")
            
            evaluation = await self._evaluate_single_rule(rule, test_metrics)
            
            logger.info(f"Test evaluation for rule {rule.name}: triggered={evaluation.triggered}")
            return evaluation
            
        except Exception as e:
            logger.error(f"Failed to test alert rule {rule_id}: {e}")
            raise


# Global instance
alerting_config_service = AlertingConfigurationService()