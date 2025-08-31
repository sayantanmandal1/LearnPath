"""
Comprehensive monitoring and alerting system
"""
import asyncio
import json
import time
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque

import structlog
from prometheus_client import Counter, Histogram, Gauge, Summary
from app.core.graceful_degradation import ServiceStatus, degradation_manager

logger = structlog.get_logger()


class AlertSeverity(Enum):
    """Alert severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertType(Enum):
    """Types of alerts"""
    SERVICE_DOWN = "service_down"
    HIGH_ERROR_RATE = "high_error_rate"
    SLOW_RESPONSE = "slow_response"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    SECURITY_INCIDENT = "security_incident"
    DATA_QUALITY = "data_quality"


@dataclass
class Alert:
    """Alert information"""
    id: str
    type: AlertType
    severity: AlertSeverity
    title: str
    description: str
    timestamp: datetime
    service: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary"""
        return {
            "id": self.id,
            "type": self.type.value,
            "severity": self.severity.value,
            "title": self.title,
            "description": self.description,
            "timestamp": self.timestamp.isoformat(),
            "service": self.service,
            "metadata": self.metadata,
            "resolved": self.resolved,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
        }


@dataclass
class MetricThreshold:
    """Metric threshold configuration"""
    metric_name: str
    threshold_value: float
    comparison: str  # 'gt', 'lt', 'eq'
    duration_seconds: int = 60
    severity: AlertSeverity = AlertSeverity.MEDIUM


class SystemMonitor:
    """Comprehensive system monitoring"""
    
    def __init__(self):
        self.alerts: List[Alert] = []
        self.alert_handlers: List[Callable[[Alert], None]] = []
        self.metric_thresholds: List[MetricThreshold] = []
        self.metric_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.monitoring_active = False
        self._monitoring_task: Optional[asyncio.Task] = None
        
        # Prometheus metrics
        self.setup_metrics()
        
        # Default thresholds
        self.setup_default_thresholds()
    
    def setup_metrics(self):
        """Setup Prometheus metrics"""
        self.error_rate_gauge = Gauge(
            'system_error_rate',
            'System error rate percentage',
            ['service']
        )
        
        self.response_time_histogram = Histogram(
            'system_response_time_seconds',
            'System response time in seconds',
            ['service', 'endpoint']
        )
        
        self.active_alerts_gauge = Gauge(
            'system_active_alerts',
            'Number of active alerts',
            ['severity']
        )
        
        self.service_health_gauge = Gauge(
            'service_health_status',
            'Service health status (1=healthy, 0.5=degraded, 0=down)',
            ['service']
        )
        
        self.database_connections_gauge = Gauge(
            'database_connections_active',
            'Active database connections'
        )
        
        self.cache_hit_rate_gauge = Gauge(
            'cache_hit_rate',
            'Cache hit rate percentage'
        )
        
        self.ml_model_accuracy_gauge = Gauge(
            'ml_model_accuracy',
            'ML model accuracy score',
            ['model_name']
        )
        
        self.external_api_calls_counter = Counter(
            'external_api_calls_total',
            'Total external API calls',
            ['service', 'status']
        )
    
    def setup_default_thresholds(self):
        """Setup default monitoring thresholds"""
        self.metric_thresholds = [
            MetricThreshold(
                metric_name="error_rate",
                threshold_value=5.0,  # 5% error rate
                comparison="gt",
                severity=AlertSeverity.HIGH
            ),
            MetricThreshold(
                metric_name="response_time_p95",
                threshold_value=5.0,  # 5 seconds
                comparison="gt",
                severity=AlertSeverity.MEDIUM
            ),
            MetricThreshold(
                metric_name="database_connections",
                threshold_value=80,  # 80% of max connections
                comparison="gt",
                severity=AlertSeverity.HIGH
            ),
            MetricThreshold(
                metric_name="cache_hit_rate",
                threshold_value=70.0,  # 70% hit rate
                comparison="lt",
                severity=AlertSeverity.LOW
            ),
        ]
    
    def add_alert_handler(self, handler: Callable[[Alert], None]):
        """Add alert handler"""
        self.alert_handlers.append(handler)
    
    def create_alert(
        self,
        alert_type: AlertType,
        severity: AlertSeverity,
        title: str,
        description: str,
        service: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Alert:
        """Create and process new alert"""
        alert = Alert(
            id=f"{alert_type.value}_{int(time.time())}",
            type=alert_type,
            severity=severity,
            title=title,
            description=description,
            timestamp=datetime.utcnow(),
            service=service,
            metadata=metadata or {}
        )
        
        self.alerts.append(alert)
        
        # Update metrics
        self.active_alerts_gauge.labels(severity=severity.value).inc()
        
        # Notify handlers
        for handler in self.alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Alert handler failed: {e}")
        
        logger.warning(
            "Alert created",
            alert_id=alert.id,
            type=alert_type.value,
            severity=severity.value,
            title=title,
            service=service
        )
        
        return alert
    
    def resolve_alert(self, alert_id: str):
        """Resolve an alert"""
        for alert in self.alerts:
            if alert.id == alert_id and not alert.resolved:
                alert.resolved = True
                alert.resolved_at = datetime.utcnow()
                
                # Update metrics
                self.active_alerts_gauge.labels(severity=alert.severity.value).dec()
                
                logger.info(f"Alert resolved: {alert_id}")
                break
    
    def get_active_alerts(self, severity: Optional[AlertSeverity] = None) -> List[Alert]:
        """Get active alerts"""
        alerts = [alert for alert in self.alerts if not alert.resolved]
        
        if severity:
            alerts = [alert for alert in alerts if alert.severity == severity]
        
        return sorted(alerts, key=lambda x: x.timestamp, reverse=True)
    
    def get_alert_summary(self) -> Dict[str, Any]:
        """Get alert summary"""
        active_alerts = self.get_active_alerts()
        
        summary = {
            "total_active": len(active_alerts),
            "by_severity": {
                "critical": len([a for a in active_alerts if a.severity == AlertSeverity.CRITICAL]),
                "high": len([a for a in active_alerts if a.severity == AlertSeverity.HIGH]),
                "medium": len([a for a in active_alerts if a.severity == AlertSeverity.MEDIUM]),
                "low": len([a for a in active_alerts if a.severity == AlertSeverity.LOW]),
            },
            "by_service": defaultdict(int)
        }
        
        for alert in active_alerts:
            if alert.service:
                summary["by_service"][alert.service] += 1
        
        return summary
    
    def record_metric(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Record a metric value"""
        timestamp = time.time()
        
        # Store in history
        self.metric_history[name].append({
            "timestamp": timestamp,
            "value": value,
            "labels": labels or {}
        })
        
        # Check thresholds
        self.check_metric_thresholds(name, value)
    
    def check_metric_thresholds(self, metric_name: str, value: float):
        """Check if metric value exceeds thresholds"""
        for threshold in self.metric_thresholds:
            if threshold.metric_name != metric_name:
                continue
            
            threshold_exceeded = False
            
            if threshold.comparison == "gt" and value > threshold.threshold_value:
                threshold_exceeded = True
            elif threshold.comparison == "lt" and value < threshold.threshold_value:
                threshold_exceeded = True
            elif threshold.comparison == "eq" and value == threshold.threshold_value:
                threshold_exceeded = True
            
            if threshold_exceeded:
                # Check if we already have an active alert for this threshold
                existing_alert = any(
                    alert for alert in self.get_active_alerts()
                    if alert.metadata.get("metric_name") == metric_name
                    and alert.metadata.get("threshold_value") == threshold.threshold_value
                )
                
                if not existing_alert:
                    self.create_alert(
                        alert_type=AlertType.RESOURCE_EXHAUSTION,
                        severity=threshold.severity,
                        title=f"Metric threshold exceeded: {metric_name}",
                        description=f"Metric {metric_name} value {value} exceeds threshold {threshold.threshold_value}",
                        metadata={
                            "metric_name": metric_name,
                            "metric_value": value,
                            "threshold_value": threshold.threshold_value,
                            "comparison": threshold.comparison
                        }
                    )
    
    def update_service_health_metrics(self):
        """Update service health metrics"""
        services_health = degradation_manager.get_all_services_health()
        
        for service_name, health in services_health.items():
            # Convert status to numeric value
            status_value = {
                ServiceStatus.HEALTHY: 1.0,
                ServiceStatus.DEGRADED: 0.5,
                ServiceStatus.UNAVAILABLE: 0.0,
                ServiceStatus.MAINTENANCE: 0.3
            }.get(health.status, 0.0)
            
            self.service_health_gauge.labels(service=service_name).set(status_value)
            
            # Create alerts for unhealthy services
            if health.status == ServiceStatus.UNAVAILABLE:
                existing_alert = any(
                    alert for alert in self.get_active_alerts()
                    if alert.type == AlertType.SERVICE_DOWN
                    and alert.service == service_name
                )
                
                if not existing_alert:
                    self.create_alert(
                        alert_type=AlertType.SERVICE_DOWN,
                        severity=AlertSeverity.CRITICAL,
                        title=f"Service unavailable: {service_name}",
                        description=f"Service {service_name} is currently unavailable. Last error: {health.last_error}",
                        service=service_name,
                        metadata={
                            "error_count": health.error_count,
                            "success_rate": health.success_rate,
                            "last_error": health.last_error
                        }
                    )
    
    async def start_monitoring(self):
        """Start background monitoring"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Started system monitoring")
    
    async def stop_monitoring(self):
        """Stop background monitoring"""
        self.monitoring_active = False
        
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
            self._monitoring_task = None
        
        logger.info("Stopped system monitoring")
    
    async def _monitoring_loop(self):
        """Background monitoring loop"""
        while self.monitoring_active:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds
                
                # Update service health metrics
                self.update_service_health_metrics()
                
                # Clean up old resolved alerts (older than 24 hours)
                cutoff_time = datetime.utcnow() - timedelta(hours=24)
                self.alerts = [
                    alert for alert in self.alerts
                    if not alert.resolved or alert.resolved_at > cutoff_time
                ]
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
    
    def get_system_health_report(self) -> Dict[str, Any]:
        """Generate comprehensive system health report"""
        services_health = degradation_manager.get_all_services_health()
        alert_summary = self.get_alert_summary()
        
        # Calculate overall system health
        healthy_services = sum(1 for h in services_health.values() if h.is_healthy)
        total_services = len(services_health)
        overall_health = (healthy_services / total_services * 100) if total_services > 0 else 100
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "overall_health_percentage": overall_health,
            "services": {
                name: {
                    "status": health.status.value,
                    "success_rate": health.success_rate,
                    "error_count": health.error_count,
                    "response_time": health.response_time,
                    "last_check": health.last_check,
                    "fallback_available": health.fallback_available
                }
                for name, health in services_health.items()
            },
            "alerts": alert_summary,
            "metrics": {
                "total_services": total_services,
                "healthy_services": healthy_services,
                "degraded_services": sum(1 for h in services_health.values() 
                                       if h.status == ServiceStatus.DEGRADED),
                "unavailable_services": sum(1 for h in services_health.values() 
                                          if h.status == ServiceStatus.UNAVAILABLE),
            }
        }


# Alert handlers
class AlertHandlers:
    """Collection of alert handlers"""
    
    @staticmethod
    def log_alert(alert: Alert):
        """Log alert to structured logs"""
        logger.warning(
            "System alert",
            alert_id=alert.id,
            type=alert.type.value,
            severity=alert.severity.value,
            title=alert.title,
            description=alert.description,
            service=alert.service,
            metadata=alert.metadata
        )
    
    @staticmethod
    def email_alert(alert: Alert):
        """Send alert via email (placeholder)"""
        # This would integrate with an email service
        logger.info(f"Email alert sent for: {alert.title}")
    
    @staticmethod
    def slack_alert(alert: Alert):
        """Send alert to Slack (placeholder)"""
        # This would integrate with Slack API
        logger.info(f"Slack alert sent for: {alert.title}")
    
    @staticmethod
    def webhook_alert(alert: Alert):
        """Send alert via webhook (placeholder)"""
        # This would send HTTP POST to configured webhook
        logger.info(f"Webhook alert sent for: {alert.title}")


# Global monitor instance
system_monitor = SystemMonitor()

# Register default alert handlers
system_monitor.add_alert_handler(AlertHandlers.log_alert)