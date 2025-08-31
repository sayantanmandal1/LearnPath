"""
Production Monitoring and Alerting System
Comprehensive monitoring setup for production deployment.
"""

import asyncio
import time
import psutil
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import json
import aiohttp
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dataclasses import dataclass, asdict
from enum import Enum

from app.core.database import get_db
from app.core.redis import get_redis
from app.core.logging import get_logger

logger = get_logger(__name__)


class AlertSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class MetricType(Enum):
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


@dataclass
class Metric:
    """Represents a monitoring metric"""
    name: str
    value: float
    metric_type: MetricType
    labels: Dict[str, str] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()
        if self.labels is None:
            self.labels = {}


@dataclass
class Alert:
    """Represents a monitoring alert"""
    alert_id: str
    severity: AlertSeverity
    title: str
    message: str
    source: str
    timestamp: datetime = None
    metadata: Dict[str, Any] = None
    resolved: bool = False
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()
        if self.metadata is None:
            self.metadata = {}


class SystemMetricsCollector:
    """Collects system-level metrics"""
    
    def __init__(self):
        self.metrics: List[Metric] = []
    
    async def collect_system_metrics(self) -> List[Metric]:
        """Collect comprehensive system metrics"""
        metrics = []
        
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        metrics.append(Metric("system_cpu_usage_percent", cpu_percent, MetricType.GAUGE))
        
        cpu_count = psutil.cpu_count()
        metrics.append(Metric("system_cpu_count", cpu_count, MetricType.GAUGE))
        
        # Memory metrics
        memory = psutil.virtual_memory()
        metrics.append(Metric("system_memory_total_bytes", memory.total, MetricType.GAUGE))
        metrics.append(Metric("system_memory_used_bytes", memory.used, MetricType.GAUGE))
        metrics.append(Metric("system_memory_usage_percent", memory.percent, MetricType.GAUGE))
        
        # Disk metrics
        disk = psutil.disk_usage('/')
        metrics.append(Metric("system_disk_total_bytes", disk.total, MetricType.GAUGE))
        metrics.append(Metric("system_disk_used_bytes", disk.used, MetricType.GAUGE))
        metrics.append(Metric("system_disk_usage_percent", (disk.used / disk.total) * 100, MetricType.GAUGE))
        
        # Network metrics
        network = psutil.net_io_counters()
        metrics.append(Metric("system_network_bytes_sent", network.bytes_sent, MetricType.COUNTER))
        metrics.append(Metric("system_network_bytes_recv", network.bytes_recv, MetricType.COUNTER))
        
        # Process metrics
        process = psutil.Process()
        metrics.append(Metric("process_memory_rss_bytes", process.memory_info().rss, MetricType.GAUGE))
        metrics.append(Metric("process_cpu_percent", process.cpu_percent(), MetricType.GAUGE))
        
        return metrics


class DatabaseMetricsCollector:
    """Collects database-related metrics"""
    
    def __init__(self):
        self.db_session = None
    
    async def collect_database_metrics(self) -> List[Metric]:
        """Collect database performance metrics"""
        metrics = []
        
        try:
            # Database connection metrics
            async with get_db() as db:
                # Connection count
                result = await db.execute("SELECT count(*) FROM pg_stat_activity WHERE datname = current_database()")
                connection_count = result.scalar()
                metrics.append(Metric("database_connections_active", connection_count, MetricType.GAUGE))
                
                # Database size
                result = await db.execute("SELECT pg_database_size(current_database())")
                db_size = result.scalar()
                metrics.append(Metric("database_size_bytes", db_size, MetricType.GAUGE))
                
                # Query performance metrics
                result = await db.execute("""
                    SELECT 
                        calls,
                        total_time,
                        mean_time,
                        max_time
                    FROM pg_stat_statements 
                    ORDER BY total_time DESC 
                    LIMIT 1
                """)
                
                if result:
                    row = result.fetchone()
                    if row:
                        metrics.append(Metric("database_query_calls_total", row[0], MetricType.COUNTER))
                        metrics.append(Metric("database_query_time_total_ms", row[1], MetricType.COUNTER))
                        metrics.append(Metric("database_query_time_mean_ms", row[2], MetricType.GAUGE))
                        metrics.append(Metric("database_query_time_max_ms", row[3], MetricType.GAUGE))
                
                # Table sizes
                result = await db.execute("""
                    SELECT 
                        schemaname,
                        tablename,
                        pg_total_relation_size(schemaname||'.'||tablename) as size
                    FROM pg_tables 
                    WHERE schemaname = 'public'
                    ORDER BY size DESC
                    LIMIT 5
                """)
                
                for row in result:
                    table_name = f"{row[0]}.{row[1]}"
                    metrics.append(Metric(
                        "database_table_size_bytes", 
                        row[2], 
                        MetricType.GAUGE,
                        labels={"table": table_name}
                    ))
        
        except Exception as e:
            logger.error(f"Failed to collect database metrics: {e}")
            metrics.append(Metric("database_metrics_collection_errors", 1, MetricType.COUNTER))
        
        return metrics


class RedisMetricsCollector:
    """Collects Redis cache metrics"""
    
    async def collect_redis_metrics(self) -> List[Metric]:
        """Collect Redis performance metrics"""
        metrics = []
        
        try:
            redis_manager = await get_redis()
            redis_client = redis_manager.redis
            
            # Redis info
            info = await redis_client.info()
            
            # Memory metrics
            metrics.append(Metric("redis_memory_used_bytes", info.get("used_memory", 0), MetricType.GAUGE))
            metrics.append(Metric("redis_memory_peak_bytes", info.get("used_memory_peak", 0), MetricType.GAUGE))
            metrics.append(Metric("redis_memory_rss_bytes", info.get("used_memory_rss", 0), MetricType.GAUGE))
            
            # Connection metrics
            metrics.append(Metric("redis_connected_clients", info.get("connected_clients", 0), MetricType.GAUGE))
            metrics.append(Metric("redis_blocked_clients", info.get("blocked_clients", 0), MetricType.GAUGE))
            
            # Command metrics
            metrics.append(Metric("redis_total_commands_processed", info.get("total_commands_processed", 0), MetricType.COUNTER))
            metrics.append(Metric("redis_instantaneous_ops_per_sec", info.get("instantaneous_ops_per_sec", 0), MetricType.GAUGE))
            
            # Key metrics
            metrics.append(Metric("redis_expired_keys", info.get("expired_keys", 0), MetricType.COUNTER))
            metrics.append(Metric("redis_evicted_keys", info.get("evicted_keys", 0), MetricType.COUNTER))
            
            # Hit rate
            keyspace_hits = info.get("keyspace_hits", 0)
            keyspace_misses = info.get("keyspace_misses", 0)
            total_requests = keyspace_hits + keyspace_misses
            
            if total_requests > 0:
                hit_rate = keyspace_hits / total_requests
                metrics.append(Metric("redis_hit_rate", hit_rate, MetricType.GAUGE))
            
            metrics.append(Metric("redis_keyspace_hits", keyspace_hits, MetricType.COUNTER))
            metrics.append(Metric("redis_keyspace_misses", keyspace_misses, MetricType.COUNTER))
        
        except Exception as e:
            logger.error(f"Failed to collect Redis metrics: {e}")
            metrics.append(Metric("redis_metrics_collection_errors", 1, MetricType.COUNTER))
        
        return metrics


class ApplicationMetricsCollector:
    """Collects application-specific metrics"""
    
    def __init__(self):
        self.request_count = 0
        self.error_count = 0
        self.response_times = []
    
    async def collect_application_metrics(self) -> List[Metric]:
        """Collect application performance metrics"""
        metrics = []
        
        try:
            # Request metrics
            metrics.append(Metric("http_requests_total", self.request_count, MetricType.COUNTER))
            metrics.append(Metric("http_errors_total", self.error_count, MetricType.COUNTER))
            
            # Response time metrics
            if self.response_times:
                avg_response_time = sum(self.response_times) / len(self.response_times)
                max_response_time = max(self.response_times)
                min_response_time = min(self.response_times)
                
                metrics.append(Metric("http_response_time_avg_seconds", avg_response_time, MetricType.GAUGE))
                metrics.append(Metric("http_response_time_max_seconds", max_response_time, MetricType.GAUGE))
                metrics.append(Metric("http_response_time_min_seconds", min_response_time, MetricType.GAUGE))
            
            # Error rate
            if self.request_count > 0:
                error_rate = self.error_count / self.request_count
                metrics.append(Metric("http_error_rate", error_rate, MetricType.GAUGE))
            
            # ML model metrics (simulated)
            metrics.append(Metric("ml_model_predictions_total", 1000, MetricType.COUNTER))
            metrics.append(Metric("ml_model_accuracy", 0.85, MetricType.GAUGE))
            metrics.append(Metric("ml_model_inference_time_ms", 150.5, MetricType.GAUGE))
            
            # Business metrics
            metrics.append(Metric("active_users_total", 500, MetricType.GAUGE))
            metrics.append(Metric("job_recommendations_generated", 2500, MetricType.COUNTER))
            metrics.append(Metric("profiles_created_total", 150, MetricType.COUNTER))
        
        except Exception as e:
            logger.error(f"Failed to collect application metrics: {e}")
            metrics.append(Metric("application_metrics_collection_errors", 1, MetricType.COUNTER))
        
        return metrics
    
    def record_request(self, response_time: float, status_code: int):
        """Record HTTP request metrics"""
        self.request_count += 1
        self.response_times.append(response_time)
        
        if status_code >= 400:
            self.error_count += 1
        
        # Keep only last 1000 response times
        if len(self.response_times) > 1000:
            self.response_times = self.response_times[-1000:]


class AlertManager:
    """Manages alerts and notifications"""
    
    def __init__(self):
        self.alerts: List[Alert] = []
        self.alert_rules = self._load_alert_rules()
        self.notification_channels = self._setup_notification_channels()
    
    def _load_alert_rules(self) -> List[Dict[str, Any]]:
        """Load alert rules configuration"""
        return [
            {
                "name": "high_cpu_usage",
                "condition": lambda metrics: any(m.name == "system_cpu_usage_percent" and m.value > 80 for m in metrics),
                "severity": AlertSeverity.WARNING,
                "message": "High CPU usage detected: {cpu_usage}%"
            },
            {
                "name": "high_memory_usage",
                "condition": lambda metrics: any(m.name == "system_memory_usage_percent" and m.value > 85 for m in metrics),
                "severity": AlertSeverity.ERROR,
                "message": "High memory usage detected: {memory_usage}%"
            },
            {
                "name": "database_connection_limit",
                "condition": lambda metrics: any(m.name == "database_connections_active" and m.value > 180 for m in metrics),
                "severity": AlertSeverity.WARNING,
                "message": "Database connection count is high: {connection_count}"
            },
            {
                "name": "high_error_rate",
                "condition": lambda metrics: any(m.name == "http_error_rate" and m.value > 0.05 for m in metrics),
                "severity": AlertSeverity.ERROR,
                "message": "High error rate detected: {error_rate:.2%}"
            },
            {
                "name": "slow_response_time",
                "condition": lambda metrics: any(m.name == "http_response_time_avg_seconds" and m.value > 2.0 for m in metrics),
                "severity": AlertSeverity.WARNING,
                "message": "Slow response time detected: {response_time:.2f}s"
            },
            {
                "name": "redis_memory_high",
                "condition": lambda metrics: any(m.name == "redis_memory_used_bytes" and m.value > 1024*1024*1024 for m in metrics),
                "severity": AlertSeverity.WARNING,
                "message": "Redis memory usage is high: {redis_memory_mb:.1f}MB"
            }
        ]
    
    def _setup_notification_channels(self) -> Dict[str, Any]:
        """Setup notification channels"""
        return {
            "email": {
                "enabled": True,
                "smtp_server": "smtp.gmail.com",
                "smtp_port": 587,
                "username": "alerts@career-recommender.com",
                "password": "app_password",
                "recipients": ["admin@career-recommender.com", "ops@career-recommender.com"]
            },
            "slack": {
                "enabled": True,
                "webhook_url": "https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK"
            },
            "pagerduty": {
                "enabled": False,
                "integration_key": "your_pagerduty_integration_key"
            }
        }
    
    async def evaluate_alerts(self, metrics: List[Metric]) -> List[Alert]:
        """Evaluate metrics against alert rules"""
        triggered_alerts = []
        
        for rule in self.alert_rules:
            try:
                if rule["condition"](metrics):
                    # Extract relevant metric values for message formatting
                    metric_values = {}
                    for metric in metrics:
                        if metric.name in rule["message"]:
                            if "percent" in metric.name:
                                metric_values[metric.name.replace("system_", "").replace("_percent", "")] = metric.value
                            elif "bytes" in metric.name and "memory" in metric.name:
                                metric_values["redis_memory_mb"] = metric.value / (1024 * 1024)
                            else:
                                metric_values[metric.name] = metric.value
                    
                    alert = Alert(
                        alert_id=f"{rule['name']}_{int(time.time())}",
                        severity=rule["severity"],
                        title=rule["name"].replace("_", " ").title(),
                        message=rule["message"].format(**metric_values),
                        source="monitoring_system",
                        metadata={"rule": rule["name"], "metrics": metric_values}
                    )
                    
                    triggered_alerts.append(alert)
                    self.alerts.append(alert)
            
            except Exception as e:
                logger.error(f"Failed to evaluate alert rule {rule['name']}: {e}")
        
        return triggered_alerts
    
    async def send_alerts(self, alerts: List[Alert]):
        """Send alerts through configured notification channels"""
        for alert in alerts:
            try:
                # Send email notification
                if self.notification_channels["email"]["enabled"]:
                    await self._send_email_alert(alert)
                
                # Send Slack notification
                if self.notification_channels["slack"]["enabled"]:
                    await self._send_slack_alert(alert)
                
                # Send PagerDuty notification for critical alerts
                if (self.notification_channels["pagerduty"]["enabled"] and 
                    alert.severity == AlertSeverity.CRITICAL):
                    await self._send_pagerduty_alert(alert)
            
            except Exception as e:
                logger.error(f"Failed to send alert {alert.alert_id}: {e}")
    
    async def _send_email_alert(self, alert: Alert):
        """Send email alert"""
        try:
            config = self.notification_channels["email"]
            
            msg = MIMEMultipart()
            msg['From'] = config["username"]
            msg['To'] = ", ".join(config["recipients"])
            msg['Subject'] = f"[{alert.severity.value.upper()}] {alert.title}"
            
            body = f"""
            Alert: {alert.title}
            Severity: {alert.severity.value.upper()}
            Time: {alert.timestamp}
            Message: {alert.message}
            
            Metadata:
            {json.dumps(alert.metadata, indent=2)}
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            server = smtplib.SMTP(config["smtp_server"], config["smtp_port"])
            server.starttls()
            server.login(config["username"], config["password"])
            server.send_message(msg)
            server.quit()
            
            logger.info(f"Email alert sent for {alert.alert_id}")
        
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
    
    async def _send_slack_alert(self, alert: Alert):
        """Send Slack alert"""
        try:
            webhook_url = self.notification_channels["slack"]["webhook_url"]
            
            color_map = {
                AlertSeverity.INFO: "good",
                AlertSeverity.WARNING: "warning", 
                AlertSeverity.ERROR: "danger",
                AlertSeverity.CRITICAL: "danger"
            }
            
            payload = {
                "attachments": [
                    {
                        "color": color_map.get(alert.severity, "warning"),
                        "title": f"{alert.severity.value.upper()}: {alert.title}",
                        "text": alert.message,
                        "fields": [
                            {
                                "title": "Time",
                                "value": alert.timestamp.strftime("%Y-%m-%d %H:%M:%S UTC"),
                                "short": True
                            },
                            {
                                "title": "Source",
                                "value": alert.source,
                                "short": True
                            }
                        ],
                        "footer": "Career Recommender Monitoring",
                        "ts": int(alert.timestamp.timestamp())
                    }
                ]
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(webhook_url, json=payload) as response:
                    if response.status == 200:
                        logger.info(f"Slack alert sent for {alert.alert_id}")
                    else:
                        logger.error(f"Failed to send Slack alert: {response.status}")
        
        except Exception as e:
            logger.error(f"Failed to send Slack alert: {e}")
    
    async def _send_pagerduty_alert(self, alert: Alert):
        """Send PagerDuty alert"""
        try:
            integration_key = self.notification_channels["pagerduty"]["integration_key"]
            
            payload = {
                "routing_key": integration_key,
                "event_action": "trigger",
                "dedup_key": alert.alert_id,
                "payload": {
                    "summary": f"{alert.title}: {alert.message}",
                    "source": alert.source,
                    "severity": alert.severity.value,
                    "timestamp": alert.timestamp.isoformat(),
                    "custom_details": alert.metadata
                }
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://events.pagerduty.com/v2/enqueue",
                    json=payload
                ) as response:
                    if response.status == 202:
                        logger.info(f"PagerDuty alert sent for {alert.alert_id}")
                    else:
                        logger.error(f"Failed to send PagerDuty alert: {response.status}")
        
        except Exception as e:
            logger.error(f"Failed to send PagerDuty alert: {e}")


class ProductionMonitor:
    """Main production monitoring system"""
    
    def __init__(self):
        self.system_collector = SystemMetricsCollector()
        self.database_collector = DatabaseMetricsCollector()
        self.redis_collector = RedisMetricsCollector()
        self.application_collector = ApplicationMetricsCollector()
        self.alert_manager = AlertManager()
        self.running = False
    
    async def start_monitoring(self, interval: int = 60):
        """Start the monitoring loop"""
        self.running = True
        logger.info("Starting production monitoring system")
        
        while self.running:
            try:
                # Collect all metrics
                all_metrics = []
                
                system_metrics = await self.system_collector.collect_system_metrics()
                all_metrics.extend(system_metrics)
                
                database_metrics = await self.database_collector.collect_database_metrics()
                all_metrics.extend(database_metrics)
                
                redis_metrics = await self.redis_collector.collect_redis_metrics()
                all_metrics.extend(redis_metrics)
                
                application_metrics = await self.application_collector.collect_application_metrics()
                all_metrics.extend(application_metrics)
                
                # Log metrics summary
                logger.info(f"Collected {len(all_metrics)} metrics")
                
                # Evaluate alerts
                triggered_alerts = await self.alert_manager.evaluate_alerts(all_metrics)
                
                if triggered_alerts:
                    logger.warning(f"Triggered {len(triggered_alerts)} alerts")
                    await self.alert_manager.send_alerts(triggered_alerts)
                
                # Export metrics (to Prometheus, etc.)
                await self._export_metrics(all_metrics)
                
                # Wait for next collection interval
                await asyncio.sleep(interval)
            
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(interval)
    
    def stop_monitoring(self):
        """Stop the monitoring system"""
        self.running = False
        logger.info("Stopping production monitoring system")
    
    async def _export_metrics(self, metrics: List[Metric]):
        """Export metrics to external systems"""
        try:
            # Export to Prometheus format
            prometheus_metrics = self._format_prometheus_metrics(metrics)
            
            # Store in Redis for /metrics endpoint
            redis_manager = await get_redis()
            await redis_manager.set("prometheus_metrics", prometheus_metrics, ttl=120)
            
            # Log key metrics
            for metric in metrics:
                if metric.name in ["system_cpu_usage_percent", "system_memory_usage_percent", "http_error_rate"]:
                    logger.info(f"Metric {metric.name}: {metric.value}")
        
        except Exception as e:
            logger.error(f"Failed to export metrics: {e}")
    
    def _format_prometheus_metrics(self, metrics: List[Metric]) -> str:
        """Format metrics in Prometheus format"""
        prometheus_lines = []
        
        for metric in metrics:
            # Add help text
            prometheus_lines.append(f"# HELP {metric.name} {metric.name.replace('_', ' ').title()}")
            prometheus_lines.append(f"# TYPE {metric.name} {metric.metric_type.value}")
            
            # Add labels if present
            labels_str = ""
            if metric.labels:
                labels_list = [f'{k}="{v}"' for k, v in metric.labels.items()]
                labels_str = "{" + ",".join(labels_list) + "}"
            
            # Add metric line
            prometheus_lines.append(f"{metric.name}{labels_str} {metric.value} {int(metric.timestamp.timestamp() * 1000)}")
        
        return "\n".join(prometheus_lines)
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Get current system health status"""
        try:
            # Collect current metrics
            system_metrics = await self.system_collector.collect_system_metrics()
            database_metrics = await self.database_collector.collect_database_metrics()
            redis_metrics = await self.redis_collector.collect_redis_metrics()
            
            # Determine overall health
            cpu_usage = next((m.value for m in system_metrics if m.name == "system_cpu_usage_percent"), 0)
            memory_usage = next((m.value for m in system_metrics if m.name == "system_memory_usage_percent"), 0)
            
            health_status = "healthy"
            if cpu_usage > 80 or memory_usage > 85:
                health_status = "degraded"
            if cpu_usage > 95 or memory_usage > 95:
                health_status = "unhealthy"
            
            return {
                "status": health_status,
                "timestamp": datetime.utcnow().isoformat(),
                "metrics": {
                    "cpu_usage_percent": cpu_usage,
                    "memory_usage_percent": memory_usage,
                    "active_alerts": len([a for a in self.alert_manager.alerts if not a.resolved])
                },
                "monitoring_active": self.running
            }
        
        except Exception as e:
            logger.error(f"Failed to get health status: {e}")
            return {
                "status": "unknown",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }


# Global monitoring instance
production_monitor = ProductionMonitor()


async def start_production_monitoring():
    """Start the production monitoring system"""
    await production_monitor.start_monitoring()


def stop_production_monitoring():
    """Stop the production monitoring system"""
    production_monitor.stop_monitoring()


async def get_monitoring_health() -> Dict[str, Any]:
    """Get monitoring system health"""
    return await production_monitor.get_health_status()


if __name__ == "__main__":
    # Run monitoring system
    asyncio.run(start_production_monitoring())