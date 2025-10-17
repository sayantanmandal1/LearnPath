"""
Background Monitoring Service
Continuously monitors system performance and evaluates alert rules
"""
import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import structlog

from app.services.system_performance_analytics import (
    system_performance_analytics,
    SystemHealthSnapshot
)
from app.services.alerting_configuration import alerting_config_service
from app.services.performance_monitoring import performance_monitor
from app.core.monitoring import system_monitor
from app.core.graceful_degradation import degradation_manager
from app.core.redis import redis_manager
from app.core.database import AsyncSessionLocal
from sqlalchemy import text

logger = structlog.get_logger()


class BackgroundMonitoringService:
    """Background service for continuous system monitoring"""
    
    def __init__(self):
        self.monitoring_active = False
        self.monitoring_task: Optional[asyncio.Task] = None
        self.metrics_collection_interval = 60  # 1 minute
        self.alert_evaluation_interval = 300  # 5 minutes
        self.health_check_interval = 30  # 30 seconds
        
        # Metrics storage
        self.current_metrics: Dict[str, float] = {}
        self.last_metrics_collection = None
        self.last_alert_evaluation = None
        self.last_health_check = None
    
    async def start_monitoring(self):
        """Start background monitoring"""
        if self.monitoring_active:
            logger.warning("Background monitoring is already active")
            return
        
        self.monitoring_active = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        
        # Start system monitor
        await system_monitor.start_monitoring()
        
        logger.info("Background monitoring service started")
    
    async def stop_monitoring(self):
        """Stop background monitoring"""
        if not self.monitoring_active:
            return
        
        self.monitoring_active = False
        
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
            self.monitoring_task = None
        
        # Stop system monitor
        await system_monitor.stop_monitoring()
        
        logger.info("Background monitoring service stopped")
    
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        logger.info("Starting monitoring loop")
        
        while self.monitoring_active:
            try:
                current_time = time.time()
                
                # Health check (every 30 seconds)
                if (self.last_health_check is None or 
                    current_time - self.last_health_check >= self.health_check_interval):
                    await self._perform_health_check()
                    self.last_health_check = current_time
                
                # Metrics collection (every 1 minute)
                if (self.last_metrics_collection is None or 
                    current_time - self.last_metrics_collection >= self.metrics_collection_interval):
                    await self._collect_system_metrics()
                    self.last_metrics_collection = current_time
                
                # Alert evaluation (every 5 minutes)
                if (self.last_alert_evaluation is None or 
                    current_time - self.last_alert_evaluation >= self.alert_evaluation_interval):
                    await self._evaluate_alert_rules()
                    self.last_alert_evaluation = current_time
                
                # Sleep for a short interval
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(30)  # Wait before retrying
        
        logger.info("Monitoring loop stopped")
    
    async def _perform_health_check(self):
        """Perform system health check"""
        try:
            health_status = {
                "timestamp": datetime.utcnow(),
                "database": await self._check_database_health(),
                "redis": await self._check_redis_health(),
                "external_services": await self._check_external_services_health(),
                "system_resources": await self._check_system_resources()
            }
            
            # Store health status in Redis
            await redis_manager.setex(
                "system_health_status",
                300,  # 5 minutes TTL
                str(health_status)
            )
            
            # Log any critical issues
            for component, status in health_status.items():
                if isinstance(status, dict) and not status.get("healthy", True):
                    logger.warning(f"Health check failed for {component}: {status}")
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
    
    async def _check_database_health(self) -> Dict[str, Any]:
        """Check database health"""
        try:
            start_time = time.time()
            
            async with AsyncSessionLocal() as session:
                await session.execute(text("SELECT 1"))
            
            response_time = (time.time() - start_time) * 1000
            
            return {
                "healthy": True,
                "response_time_ms": response_time,
                "status": "connected"
            }
            
        except Exception as e:
            return {
                "healthy": False,
                "error": str(e),
                "status": "disconnected"
            }
    
    async def _check_redis_health(self) -> Dict[str, Any]:
        """Check Redis health"""
        try:
            start_time = time.time()
            
            await redis_manager.redis.ping()
            
            response_time = (time.time() - start_time) * 1000
            
            return {
                "healthy": True,
                "response_time_ms": response_time,
                "status": "connected"
            }
            
        except Exception as e:
            return {
                "healthy": False,
                "error": str(e),
                "status": "disconnected"
            }
    
    async def _check_external_services_health(self) -> Dict[str, Any]:
        """Check external services health"""
        try:
            services_health = degradation_manager.get_all_services_health()
            
            healthy_services = 0
            total_services = len(services_health)
            
            for service_name, health in services_health.items():
                if health.is_healthy:
                    healthy_services += 1
            
            overall_healthy = healthy_services == total_services
            
            return {
                "healthy": overall_healthy,
                "healthy_services": healthy_services,
                "total_services": total_services,
                "health_percentage": (healthy_services / total_services * 100) if total_services > 0 else 100,
                "services": {
                    name: {
                        "healthy": health.is_healthy,
                        "status": health.status.value,
                        "success_rate": health.success_rate
                    }
                    for name, health in services_health.items()
                }
            }
            
        except Exception as e:
            return {
                "healthy": False,
                "error": str(e)
            }
    
    async def _check_system_resources(self) -> Dict[str, Any]:
        """Check system resources"""
        try:
            # Get current performance metrics
            metrics = await performance_monitor.collect_metrics()
            
            # Define thresholds
            cpu_threshold = 80.0
            memory_threshold = 85.0
            
            cpu_healthy = metrics.cpu_percent < cpu_threshold
            memory_healthy = metrics.memory_percent < memory_threshold
            
            return {
                "healthy": cpu_healthy and memory_healthy,
                "cpu": {
                    "healthy": cpu_healthy,
                    "usage_percent": metrics.cpu_percent,
                    "threshold": cpu_threshold
                },
                "memory": {
                    "healthy": memory_healthy,
                    "usage_percent": metrics.memory_percent,
                    "threshold": memory_threshold
                },
                "cache": {
                    "hit_rate_percent": metrics.cache_hit_rate
                },
                "queue": {
                    "size": metrics.queue_size
                }
            }
            
        except Exception as e:
            return {
                "healthy": False,
                "error": str(e)
            }
    
    async def _collect_system_metrics(self):
        """Collect comprehensive system metrics"""
        try:
            # Collect performance metrics
            performance_metrics = await performance_monitor.collect_metrics()
            
            # Get API performance analytics
            api_analytics = await system_performance_analytics.get_api_performance_analytics(hours=1)
            
            # Get external API analytics
            external_api_analytics = await system_performance_analytics.get_external_api_analytics(hours=1)
            
            # Get user engagement analytics
            engagement_analytics = await system_performance_analytics.get_user_engagement_analytics(hours=1)
            
            # Compile metrics for alert evaluation
            self.current_metrics = {
                # System metrics
                "system_cpu_percent": performance_metrics.cpu_percent,
                "system_memory_percent": performance_metrics.memory_percent,
                "cache_hit_rate_percent": performance_metrics.cache_hit_rate,
                "task_queue_size": performance_metrics.queue_size,
                "database_connections_active": performance_metrics.active_connections,
                "api_response_time_ms": performance_metrics.response_time_ms,
                
                # API performance metrics
                "api_error_rate_percent": api_analytics.get("error_rate", 0) if isinstance(api_analytics, dict) else 0,
                "api_response_time_p95_ms": api_analytics.get("response_time_stats", {}).get("p95_ms", 0) if isinstance(api_analytics, dict) else 0,
                "api_success_rate_percent": api_analytics.get("success_rate", 100) if isinstance(api_analytics, dict) else 100,
                
                # External API metrics
                "external_api_failure_rate_percent": external_api_analytics.get("overall_stats", {}).get("overall_failure_rate", 0) if isinstance(external_api_analytics, dict) else 0,
                "external_api_success_rate_percent": external_api_analytics.get("overall_stats", {}).get("overall_success_rate", 100) if isinstance(external_api_analytics, dict) else 100,
                
                # User engagement metrics
                "user_engagement_events_per_hour": engagement_analytics.get("total_events", 0) if isinstance(engagement_analytics, dict) else 0,
                "active_users_count": engagement_analytics.get("unique_users", 0) if isinstance(engagement_analytics, dict) else 0,
                
                # Derived metrics
                "database_connections_percent": min(100, (performance_metrics.active_connections / 100) * 100),  # Assuming max 100 connections
                "ml_model_accuracy_percent": 85.0,  # Placeholder - would come from ML service
            }
            
            # Record system health snapshot
            snapshot = SystemHealthSnapshot(
                timestamp=datetime.utcnow(),
                cpu_percent=performance_metrics.cpu_percent,
                memory_percent=performance_metrics.memory_percent,
                disk_usage_percent=0.0,  # Not available in current metrics
                active_connections=performance_metrics.active_connections,
                cache_hit_rate=performance_metrics.cache_hit_rate,
                queue_size=performance_metrics.queue_size,
                error_rate=self.current_metrics["api_error_rate_percent"],
                response_time_p95=self.current_metrics["api_response_time_p95_ms"]
            )
            
            await system_performance_analytics.record_system_health_snapshot(snapshot)
            
            logger.debug(f"Collected {len(self.current_metrics)} system metrics")
            
        except Exception as e:
            logger.error(f"Failed to collect system metrics: {e}")
    
    async def _evaluate_alert_rules(self):
        """Evaluate all alert rules against current metrics"""
        try:
            if not self.current_metrics:
                logger.warning("No metrics available for alert evaluation")
                return
            
            # Evaluate alert rules
            evaluations = await alerting_config_service.evaluate_alert_rules(self.current_metrics)
            
            triggered_count = len([e for e in evaluations if e.triggered])
            total_count = len(evaluations)
            
            if triggered_count > 0:
                logger.warning(f"Alert evaluation: {triggered_count}/{total_count} rules triggered")
            else:
                logger.debug(f"Alert evaluation: {total_count} rules evaluated, none triggered")
            
            # Store evaluation results in Redis
            evaluation_summary = {
                "timestamp": datetime.utcnow().isoformat(),
                "total_rules": total_count,
                "triggered_rules": triggered_count,
                "evaluations": [
                    {
                        "rule_id": e.rule_id,
                        "triggered": e.triggered,
                        "current_value": e.current_value,
                        "threshold_value": e.threshold_value,
                        "message": e.message
                    }
                    for e in evaluations
                ]
            }
            
            await redis_manager.setex(
                "alert_evaluation_summary",
                3600,  # 1 hour TTL
                str(evaluation_summary)
            )
            
        except Exception as e:
            logger.error(f"Failed to evaluate alert rules: {e}")
    
    def get_monitoring_status(self) -> Dict[str, Any]:
        """Get current monitoring status"""
        return {
            "monitoring_active": self.monitoring_active,
            "last_metrics_collection": datetime.fromtimestamp(self.last_metrics_collection).isoformat() if self.last_metrics_collection else None,
            "last_alert_evaluation": datetime.fromtimestamp(self.last_alert_evaluation).isoformat() if self.last_alert_evaluation else None,
            "last_health_check": datetime.fromtimestamp(self.last_health_check).isoformat() if self.last_health_check else None,
            "current_metrics_count": len(self.current_metrics),
            "intervals": {
                "metrics_collection_seconds": self.metrics_collection_interval,
                "alert_evaluation_seconds": self.alert_evaluation_interval,
                "health_check_seconds": self.health_check_interval
            }
        }
    
    def get_current_metrics(self) -> Dict[str, float]:
        """Get current system metrics"""
        return self.current_metrics.copy()
    
    async def force_metrics_collection(self):
        """Force immediate metrics collection"""
        await self._collect_system_metrics()
        logger.info("Forced metrics collection completed")
    
    async def force_alert_evaluation(self):
        """Force immediate alert evaluation"""
        await self._evaluate_alert_rules()
        logger.info("Forced alert evaluation completed")
    
    async def force_health_check(self):
        """Force immediate health check"""
        await self._perform_health_check()
        logger.info("Forced health check completed")


# Global instance
background_monitoring_service = BackgroundMonitoringService()