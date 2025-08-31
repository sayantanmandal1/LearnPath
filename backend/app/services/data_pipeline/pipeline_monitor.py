"""
Pipeline Monitoring and Alerting System
Monitors pipeline execution, performance, and data quality.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum
import json
import aiohttp
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from app.core.redis import get_redis
from app.core.logging import get_logger
from app.services.data_pipeline.pipeline_config import get_pipeline_config

logger = get_logger(__name__)


class AlertLevel(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class PipelineMetrics:
    """Pipeline execution metrics"""
    pipeline_name: str
    execution_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    status: str = "running"
    records_processed: int = 0
    records_failed: int = 0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    error_count: int = 0
    warnings_count: int = 0
    data_quality_score: float = 1.0
    throughput_per_second: float = 0.0


@dataclass
class Alert:
    """Pipeline alert"""
    alert_id: str
    pipeline_name: str
    level: AlertLevel
    title: str
    message: str
    timestamp: datetime
    metadata: Dict[str, Any] = None
    resolved: bool = False


class PipelineMonitor:
    """
    Monitors pipeline execution and sends alerts for issues
    """
    
    def __init__(self):
        self.config = get_pipeline_config()
        self.redis_client = None
        self.active_monitors: Dict[str, Dict] = {}
        
    async def initialize(self):
        """Initialize the monitor"""
        redis_manager = await get_redis()
        self.redis_client = redis_manager.redis
        logger.info("Pipeline monitor initialized")
    
    async def start_job_monitoring(self, execution_id: str, pipeline_name: str):
        """Start monitoring a pipeline job"""
        if not self.redis_client:
            await self.initialize()
        
        metrics = PipelineMetrics(
            pipeline_name=pipeline_name,
            execution_id=execution_id,
            start_time=datetime.utcnow()
        )
        
        self.active_monitors[execution_id] = {
            'metrics': metrics,
            'last_update': datetime.utcnow()
        }
        
        # Store initial metrics
        await self._store_metrics(metrics)
        
        logger.info(f"Started monitoring pipeline job {execution_id}")
    
    async def update_job_metrics(self, execution_id: str, **kwargs):
        """Update metrics for a running job"""
        if execution_id not in self.active_monitors:
            logger.warning(f"No active monitor found for execution {execution_id}")
            return
        
        monitor_data = self.active_monitors[execution_id]
        metrics = monitor_data['metrics']
        
        # Update metrics
        for key, value in kwargs.items():
            if hasattr(metrics, key):
                setattr(metrics, key, value)
        
        # Calculate derived metrics
        if metrics.end_time and metrics.start_time:
            metrics.duration_seconds = (metrics.end_time - metrics.start_time).total_seconds()
            
            if metrics.duration_seconds > 0:
                metrics.throughput_per_second = metrics.records_processed / metrics.duration_seconds
        
        # Update data quality score
        if metrics.records_processed > 0:
            success_rate = (metrics.records_processed - metrics.records_failed) / metrics.records_processed
            error_penalty = min(metrics.error_count * 0.01, 0.5)  # Max 50% penalty
            metrics.data_quality_score = max(0.0, success_rate - error_penalty)
        
        monitor_data['last_update'] = datetime.utcnow()
        
        # Store updated metrics
        await self._store_metrics(metrics)
        
        # Check for alerts
        await self._check_alerts(metrics)
    
    async def stop_job_monitoring(self, execution_id: str):
        """Stop monitoring a pipeline job"""
        if execution_id in self.active_monitors:
            monitor_data = self.active_monitors[execution_id]
            metrics = monitor_data['metrics']
            
            # Final metrics update
            metrics.end_time = datetime.utcnow()
            if metrics.start_time:
                metrics.duration_seconds = (metrics.end_time - metrics.start_time).total_seconds()
            
            # Store final metrics
            await self._store_metrics(metrics)
            
            # Remove from active monitors
            del self.active_monitors[execution_id]
            
            logger.info(f"Stopped monitoring pipeline job {execution_id}")
    
    async def get_job_metrics(self, execution_id: str) -> Optional[PipelineMetrics]:
        """Get metrics for a specific job execution"""
        try:
            metrics_key = f"pipeline_metrics:{execution_id}"
            metrics_data = await self.redis_client.get(metrics_key)
            
            if metrics_data:
                data = json.loads(metrics_data)
                # Convert datetime strings back to datetime objects
                data['start_time'] = datetime.fromisoformat(data['start_time'])
                if data.get('end_time'):
                    data['end_time'] = datetime.fromisoformat(data['end_time'])
                
                return PipelineMetrics(**data)
        except Exception as e:
            logger.error(f"Failed to get metrics for {execution_id}: {e}")
        
        return None
    
    async def get_pipeline_history(self, pipeline_name: str, days: int = 7) -> List[PipelineMetrics]:
        """Get execution history for a pipeline"""
        try:
            pattern = f"pipeline_metrics:*"
            keys = await self.redis_client.keys(pattern)
            
            history = []
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            
            for key in keys:
                try:
                    metrics_data = await self.redis_client.get(key)
                    if metrics_data:
                        data = json.loads(metrics_data)
                        
                        if data.get('pipeline_name') == pipeline_name:
                            start_time = datetime.fromisoformat(data['start_time'])
                            
                            if start_time >= cutoff_date:
                                data['start_time'] = start_time
                                if data.get('end_time'):
                                    data['end_time'] = datetime.fromisoformat(data['end_time'])
                                
                                history.append(PipelineMetrics(**data))
                except Exception as e:
                    logger.error(f"Failed to parse metrics from {key}: {e}")
            
            # Sort by start time, most recent first
            history.sort(key=lambda x: x.start_time, reverse=True)
            return history
            
        except Exception as e:
            logger.error(f"Failed to get pipeline history for {pipeline_name}: {e}")
            return []
    
    async def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health metrics"""
        try:
            health = {
                'active_jobs': len(self.active_monitors),
                'total_pipelines': 0,
                'success_rate_24h': 0.0,
                'avg_duration_24h': 0.0,
                'error_rate_24h': 0.0,
                'data_quality_score_24h': 0.0,
                'alerts_24h': 0
            }
            
            # Get metrics from last 24 hours
            cutoff_time = datetime.utcnow() - timedelta(hours=24)
            pattern = f"pipeline_metrics:*"
            keys = await self.redis_client.keys(pattern)
            
            recent_metrics = []
            for key in keys:
                try:
                    metrics_data = await self.redis_client.get(key)
                    if metrics_data:
                        data = json.loads(metrics_data)
                        start_time = datetime.fromisoformat(data['start_time'])
                        
                        if start_time >= cutoff_time:
                            recent_metrics.append(data)
                except Exception:
                    continue
            
            if recent_metrics:
                health['total_pipelines'] = len(recent_metrics)
                
                # Calculate success rate
                completed_jobs = [m for m in recent_metrics if m.get('status') == 'completed']
                health['success_rate_24h'] = len(completed_jobs) / len(recent_metrics)
                
                # Calculate average duration
                durations = [m.get('duration_seconds', 0) for m in completed_jobs if m.get('duration_seconds')]
                if durations:
                    health['avg_duration_24h'] = sum(durations) / len(durations)
                
                # Calculate error rate
                total_errors = sum(m.get('error_count', 0) for m in recent_metrics)
                total_records = sum(m.get('records_processed', 0) for m in recent_metrics)
                if total_records > 0:
                    health['error_rate_24h'] = total_errors / total_records
                
                # Calculate average data quality score
                quality_scores = [m.get('data_quality_score', 1.0) for m in recent_metrics]
                health['data_quality_score_24h'] = sum(quality_scores) / len(quality_scores)
            
            # Get alert count
            alert_pattern = f"pipeline_alert:*"
            alert_keys = await self.redis_client.keys(alert_pattern)
            
            recent_alerts = 0
            for key in alert_keys:
                try:
                    alert_data = await self.redis_client.get(key)
                    if alert_data:
                        alert = json.loads(alert_data)
                        alert_time = datetime.fromisoformat(alert['timestamp'])
                        if alert_time >= cutoff_time:
                            recent_alerts += 1
                except Exception:
                    continue
            
            health['alerts_24h'] = recent_alerts
            
            return health
            
        except Exception as e:
            logger.error(f"Failed to get system health: {e}")
            return {}
    
    async def _store_metrics(self, metrics: PipelineMetrics):
        """Store metrics in Redis"""
        try:
            metrics_key = f"pipeline_metrics:{metrics.execution_id}"
            
            # Convert to dict and handle datetime serialization
            metrics_dict = asdict(metrics)
            metrics_dict['start_time'] = metrics.start_time.isoformat()
            if metrics.end_time:
                metrics_dict['end_time'] = metrics.end_time.isoformat()
            
            await self.redis_client.set(
                metrics_key,
                json.dumps(metrics_dict),
                ex=86400 * 30  # Keep for 30 days
            )
        except Exception as e:
            logger.error(f"Failed to store metrics for {metrics.execution_id}: {e}")
    
    async def _check_alerts(self, metrics: PipelineMetrics):
        """Check metrics against alert thresholds"""
        alerts = []
        
        # Check data quality
        if metrics.data_quality_score < 0.8:
            alerts.append(Alert(
                alert_id=f"{metrics.execution_id}_quality",
                pipeline_name=metrics.pipeline_name,
                level=AlertLevel.WARNING if metrics.data_quality_score > 0.6 else AlertLevel.ERROR,
                title="Low Data Quality",
                message=f"Data quality score is {metrics.data_quality_score:.2f}",
                timestamp=datetime.utcnow(),
                metadata={'execution_id': metrics.execution_id, 'score': metrics.data_quality_score}
            ))
        
        # Check error rate
        if metrics.records_processed > 0:
            error_rate = metrics.records_failed / metrics.records_processed
            if error_rate > 0.1:  # 10% error rate threshold
                alerts.append(Alert(
                    alert_id=f"{metrics.execution_id}_errors",
                    pipeline_name=metrics.pipeline_name,
                    level=AlertLevel.WARNING if error_rate < 0.2 else AlertLevel.ERROR,
                    title="High Error Rate",
                    message=f"Error rate is {error_rate:.1%}",
                    timestamp=datetime.utcnow(),
                    metadata={'execution_id': metrics.execution_id, 'error_rate': error_rate}
                ))
        
        # Check duration (if we have historical data)
        if metrics.duration_seconds:
            # Get recent executions for comparison
            history = await self.get_pipeline_history(metrics.pipeline_name, days=7)
            if len(history) > 1:
                avg_duration = sum(h.duration_seconds for h in history[1:] if h.duration_seconds) / (len(history) - 1)
                
                if metrics.duration_seconds > avg_duration * 2:  # Taking more than 2x average time
                    alerts.append(Alert(
                        alert_id=f"{metrics.execution_id}_duration",
                        pipeline_name=metrics.pipeline_name,
                        level=AlertLevel.WARNING,
                        title="Long Execution Time",
                        message=f"Execution took {metrics.duration_seconds:.0f}s (avg: {avg_duration:.0f}s)",
                        timestamp=datetime.utcnow(),
                        metadata={'execution_id': metrics.execution_id, 'duration': metrics.duration_seconds}
                    ))
        
        # Send alerts
        for alert in alerts:
            await self._send_alert(alert)
    
    async def _send_alert(self, alert: Alert):
        """Send an alert via configured channels"""
        try:
            # Store alert
            alert_key = f"pipeline_alert:{alert.alert_id}"
            alert_dict = asdict(alert)
            alert_dict['timestamp'] = alert.timestamp.isoformat()
            alert_dict['level'] = alert.level.value
            
            await self.redis_client.set(
                alert_key,
                json.dumps(alert_dict),
                ex=86400 * 7  # Keep for 7 days
            )
            
            # Send webhook alert
            if self.config.alert_webhook_url:
                await self._send_webhook_alert(alert)
            
            # Send email alert
            if self.config.alert_email and alert.level in [AlertLevel.ERROR, AlertLevel.CRITICAL]:
                await self._send_email_alert(alert)
            
            logger.info(f"Sent {alert.level.value} alert: {alert.title}")
            
        except Exception as e:
            logger.error(f"Failed to send alert {alert.alert_id}: {e}")
    
    async def _send_webhook_alert(self, alert: Alert):
        """Send alert via webhook"""
        try:
            payload = {
                'alert_id': alert.alert_id,
                'pipeline_name': alert.pipeline_name,
                'level': alert.level.value,
                'title': alert.title,
                'message': alert.message,
                'timestamp': alert.timestamp.isoformat(),
                'metadata': alert.metadata
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.config.alert_webhook_url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status != 200:
                        logger.error(f"Webhook alert failed with status {response.status}")
                        
        except Exception as e:
            logger.error(f"Failed to send webhook alert: {e}")
    
    async def _send_email_alert(self, alert: Alert):
        """Send alert via email"""
        try:
            # This is a simplified email implementation
            # In production, you'd want to use a proper email service
            msg = MIMEMultipart()
            msg['From'] = "noreply@career-recommender.com"
            msg['To'] = self.config.alert_email
            msg['Subject'] = f"[{alert.level.value.upper()}] {alert.title}"
            
            body = f"""
            Pipeline Alert
            
            Pipeline: {alert.pipeline_name}
            Level: {alert.level.value}
            Title: {alert.title}
            Message: {alert.message}
            Time: {alert.timestamp}
            
            Metadata: {json.dumps(alert.metadata, indent=2) if alert.metadata else 'None'}
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            # Note: This would need proper SMTP configuration in production
            logger.info(f"Email alert prepared for {alert.alert_id}")
            
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")


# Global monitor instance
pipeline_monitor = PipelineMonitor()


async def get_pipeline_monitor() -> PipelineMonitor:
    """Get the global pipeline monitor instance"""
    return pipeline_monitor