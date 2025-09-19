"""
Enhanced performance monitoring service for caching and background jobs
"""
import asyncio
import time
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
import structlog
import psutil
import threading

from app.services.cache_service import get_cache_service
from app.core.redis import get_redis
from app.core.celery_app import task_manager
from app.core.config import settings

logger = structlog.get_logger()


@dataclass
class PerformanceMetric:
    """Individual performance metric"""
    name: str
    value: float
    unit: str
    timestamp: datetime
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class SystemMetrics:
    """System-level performance metrics"""
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    disk_usage_percent: float
    network_io: Dict[str, int]
    timestamp: datetime


@dataclass
class CacheMetrics:
    """Cache performance metrics"""
    hit_rate: float
    miss_rate: float
    total_requests: int
    memory_usage_mb: float
    key_count: int
    eviction_count: int
    timestamp: datetime


@dataclass
class BackgroundJobMetrics:
    """Background job performance metrics"""
    active_tasks: int
    completed_tasks: int
    failed_tasks: int
    average_execution_time: float
    queue_sizes: Dict[str, int]
    worker_count: int
    timestamp: datetime


class PerformanceMonitor:
    """Enhanced performance monitoring service"""
    
    def __init__(self, collection_interval: int = 60):
        self.collection_interval = collection_interval
        self.metrics_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1440))  # 24 hours of minutes
        self.is_monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        
        # Performance thresholds
        self.thresholds = {
            "cpu_percent": 80.0,
            "memory_percent": 85.0,
            "cache_hit_rate": 70.0,
            "response_time_ms": 1000.0,
            "error_rate": 5.0
        }
        
        # Alert callbacks
        self.alert_callbacks: List[Callable] = []
    
    def start_monitoring(self):
        """Start continuous performance monitoring"""
        if self.is_monitoring:
            logger.warning("Performance monitoring already running")
            return
        
        self.is_monitoring = True
        self._stop_event.clear()
        
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self.monitor_thread.start()
        
        logger.info("Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop performance monitoring"""
        if not self.is_monitoring:
            return
        
        self.is_monitoring = False
        self._stop_event.set()
        
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        
        logger.info("Performance monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while not self._stop_event.is_set():
            try:
                # Collect all metrics
                asyncio.run(self._collect_all_metrics())
                
                # Check thresholds and trigger alerts
                self._check_thresholds()
                
                # Wait for next collection interval
                self._stop_event.wait(self.collection_interval)
                
            except Exception as e:
                logger.error("Error in monitoring loop", error=str(e))
                time.sleep(5)  # Brief pause before retrying
    
    async def _collect_all_metrics(self):
        """Collect all performance metrics"""
        timestamp = datetime.utcnow()
        
        # Collect system metrics
        system_metrics = await self._collect_system_metrics(timestamp)
        self.metrics_history["system"].append(system_metrics)
        
        # Collect cache metrics
        cache_metrics = await self._collect_cache_metrics(timestamp)
        self.metrics_history["cache"].append(cache_metrics)
        
        # Collect background job metrics
        job_metrics = await self._collect_background_job_metrics(timestamp)
        self.metrics_history["background_jobs"].append(job_metrics)
        
        # Collect application-specific metrics
        app_metrics = await self._collect_application_metrics(timestamp)
        self.metrics_history["application"].extend(app_metrics)
    
    async def _collect_system_metrics(self, timestamp: datetime) -> SystemMetrics:
        """Collect system-level metrics"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_used_mb = memory.used / 1024 / 1024
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_usage_percent = disk.percent
            
            # Network I/O
            network = psutil.net_io_counters()
            network_io = {
                "bytes_sent": network.bytes_sent,
                "bytes_recv": network.bytes_recv,
                "packets_sent": network.packets_sent,
                "packets_recv": network.packets_recv
            }
            
            return SystemMetrics(
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                memory_used_mb=memory_used_mb,
                disk_usage_percent=disk_usage_percent,
                network_io=network_io,
                timestamp=timestamp
            )
            
        except Exception as e:
            logger.error("Failed to collect system metrics", error=str(e))
            return SystemMetrics(0, 0, 0, 0, {}, timestamp)
    
    async def _collect_cache_metrics(self, timestamp: datetime) -> CacheMetrics:
        """Collect cache performance metrics"""
        try:
            cache_service = await get_cache_service()
            redis_manager = await get_redis()
            
            # Get cache service stats
            cache_stats = cache_service.get_stats()
            
            # Get Redis info
            redis_info = await redis_manager.redis.info()
            memory_usage_mb = redis_info.get("used_memory", 0) / 1024 / 1024
            
            # Get key count
            db_info = redis_info.get("db0", {})
            key_count = db_info.get("keys", 0) if isinstance(db_info, dict) else 0
            
            return CacheMetrics(
                hit_rate=cache_stats.get("hit_rate_percent", 0),
                miss_rate=100 - cache_stats.get("hit_rate_percent", 0),
                total_requests=cache_stats.get("hits", 0) + cache_stats.get("misses", 0),
                memory_usage_mb=memory_usage_mb,
                key_count=key_count,
                eviction_count=redis_info.get("evicted_keys", 0),
                timestamp=timestamp
            )
            
        except Exception as e:
            logger.error("Failed to collect cache metrics", error=str(e))
            return CacheMetrics(0, 0, 0, 0, 0, 0, timestamp)
    
    async def _collect_background_job_metrics(self, timestamp: datetime) -> BackgroundJobMetrics:
        """Collect background job metrics"""
        try:
            # Get Celery worker stats
            worker_stats = task_manager.get_worker_stats()
            active_tasks = task_manager.get_active_tasks()
            
            # Calculate metrics
            total_workers = len(worker_stats) if worker_stats else 0
            total_active = len(active_tasks) if active_tasks else 0
            
            # Get queue sizes (simplified - would need more complex logic for real implementation)
            queue_sizes = {
                "default": 0,
                "ml_queue": 0,
                "data_queue": 0,
                "cache_queue": 0,
                "analytics_queue": 0
            }
            
            return BackgroundJobMetrics(
                active_tasks=total_active,
                completed_tasks=0,  # Would need to track this separately
                failed_tasks=0,     # Would need to track this separately
                average_execution_time=0.0,  # Would need to track this separately
                queue_sizes=queue_sizes,
                worker_count=total_workers,
                timestamp=timestamp
            )
            
        except Exception as e:
            logger.error("Failed to collect background job metrics", error=str(e))
            return BackgroundJobMetrics(0, 0, 0, 0.0, {}, 0, timestamp)
    
    async def _collect_application_metrics(self, timestamp: datetime) -> List[PerformanceMetric]:
        """Collect application-specific metrics"""
        metrics = []
        
        try:
            # Database connection pool metrics
            from app.core.database import engine
            if hasattr(engine.pool, 'size'):
                metrics.append(PerformanceMetric(
                    name="db_pool_size",
                    value=engine.pool.size(),
                    unit="connections",
                    timestamp=timestamp,
                    tags={"component": "database"}
                ))
            
            if hasattr(engine.pool, 'checked_in'):
                metrics.append(PerformanceMetric(
                    name="db_pool_checked_in",
                    value=engine.pool.checked_in(),
                    unit="connections",
                    timestamp=timestamp,
                    tags={"component": "database"}
                ))
            
            # API response time metrics (would be collected from middleware)
            # This is a placeholder - real implementation would track actual response times
            metrics.append(PerformanceMetric(
                name="api_response_time_avg",
                value=150.0,  # Placeholder value
                unit="milliseconds",
                timestamp=timestamp,
                tags={"component": "api"}
            ))
            
            # External API circuit breaker metrics
            from app.services.external_apis.circuit_breaker import circuit_breaker_manager
            breaker_stats = circuit_breaker_manager.get_all_stats()
            
            for breaker_name, stats in breaker_stats.items():
                metrics.append(PerformanceMetric(
                    name="circuit_breaker_failure_rate",
                    value=(stats.total_failures / max(stats.total_requests, 1)) * 100,
                    unit="percent",
                    timestamp=timestamp,
                    tags={"component": "circuit_breaker", "service": breaker_name}
                ))
            
        except Exception as e:
            logger.error("Failed to collect application metrics", error=str(e))
        
        return metrics
    
    def _check_thresholds(self):
        """Check performance thresholds and trigger alerts"""
        try:
            # Get latest metrics
            if "system" in self.metrics_history and self.metrics_history["system"]:
                latest_system = self.metrics_history["system"][-1]
                
                if latest_system.cpu_percent > self.thresholds["cpu_percent"]:
                    self._trigger_alert("high_cpu", {
                        "current": latest_system.cpu_percent,
                        "threshold": self.thresholds["cpu_percent"]
                    })
                
                if latest_system.memory_percent > self.thresholds["memory_percent"]:
                    self._trigger_alert("high_memory", {
                        "current": latest_system.memory_percent,
                        "threshold": self.thresholds["memory_percent"]
                    })
            
            if "cache" in self.metrics_history and self.metrics_history["cache"]:
                latest_cache = self.metrics_history["cache"][-1]
                
                if latest_cache.hit_rate < self.thresholds["cache_hit_rate"]:
                    self._trigger_alert("low_cache_hit_rate", {
                        "current": latest_cache.hit_rate,
                        "threshold": self.thresholds["cache_hit_rate"]
                    })
            
        except Exception as e:
            logger.error("Failed to check thresholds", error=str(e))
    
    def _trigger_alert(self, alert_type: str, data: Dict[str, Any]):
        """Trigger performance alert"""
        alert_data = {
            "type": alert_type,
            "timestamp": datetime.utcnow(),
            "data": data
        }
        
        logger.warning("Performance alert triggered", **alert_data)
        
        # Call registered alert callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert_data)
            except Exception as e:
                logger.error("Alert callback failed", error=str(e))
    
    def add_alert_callback(self, callback: Callable):
        """Add alert callback function"""
        self.alert_callbacks.append(callback)
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        current_metrics = {}
        
        for metric_type, history in self.metrics_history.items():
            if history:
                current_metrics[metric_type] = history[-1]
        
        return current_metrics
    
    def get_metrics_history(
        self,
        metric_type: str,
        duration_minutes: int = 60
    ) -> List[Any]:
        """Get metrics history for a specific duration"""
        if metric_type not in self.metrics_history:
            return []
        
        history = self.metrics_history[metric_type]
        cutoff_time = datetime.utcnow() - timedelta(minutes=duration_minutes)
        
        return [
            metric for metric in history
            if metric.timestamp > cutoff_time
        ]
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        summary = {
            "monitoring_active": self.is_monitoring,
            "collection_interval": self.collection_interval,
            "metrics_collected": {
                metric_type: len(history)
                for metric_type, history in self.metrics_history.items()
            }
        }
        
        # Add current metrics
        current = self.get_current_metrics()
        if current:
            summary["current_metrics"] = current
        
        # Add threshold status
        summary["threshold_status"] = {}
        if "system" in current:
            system = current["system"]
            summary["threshold_status"]["cpu"] = {
                "current": system.cpu_percent,
                "threshold": self.thresholds["cpu_percent"],
                "status": "ok" if system.cpu_percent <= self.thresholds["cpu_percent"] else "warning"
            }
            summary["threshold_status"]["memory"] = {
                "current": system.memory_percent,
                "threshold": self.thresholds["memory_percent"],
                "status": "ok" if system.memory_percent <= self.thresholds["memory_percent"] else "warning"
            }
        
        if "cache" in current:
            cache = current["cache"]
            summary["threshold_status"]["cache_hit_rate"] = {
                "current": cache.hit_rate,
                "threshold": self.thresholds["cache_hit_rate"],
                "status": "ok" if cache.hit_rate >= self.thresholds["cache_hit_rate"] else "warning"
            }
        
        return summary
    
    def update_threshold(self, metric_name: str, threshold_value: float):
        """Update performance threshold"""
        self.thresholds[metric_name] = threshold_value
        logger.info(f"Updated threshold for {metric_name} to {threshold_value}")
    
    def clear_metrics_history(self, metric_type: str = None):
        """Clear metrics history"""
        if metric_type:
            if metric_type in self.metrics_history:
                self.metrics_history[metric_type].clear()
                logger.info(f"Cleared metrics history for {metric_type}")
        else:
            self.metrics_history.clear()
            logger.info("Cleared all metrics history")


# Global performance monitor instance
_performance_monitor: Optional[PerformanceMonitor] = None


def get_performance_monitor() -> PerformanceMonitor:
    """Get global performance monitor instance"""
    global _performance_monitor
    if _performance_monitor is None:
        _performance_monitor = PerformanceMonitor()
    return _performance_monitor


# Alert callback example
def default_alert_callback(alert_data: Dict[str, Any]):
    """Default alert callback that logs alerts"""
    logger.critical(
        "PERFORMANCE ALERT",
        alert_type=alert_data["type"],
        timestamp=alert_data["timestamp"],
        data=alert_data["data"]
    )


# Initialize performance monitoring
def initialize_performance_monitoring():
    """Initialize performance monitoring with default settings"""
    monitor = get_performance_monitor()
    monitor.add_alert_callback(default_alert_callback)
    
    # Start monitoring if enabled in settings
    if getattr(settings, 'ENABLE_PERFORMANCE_MONITORING', True):
        monitor.start_monitoring()
        logger.info("Performance monitoring initialized and started")
    else:
        logger.info("Performance monitoring initialized but not started (disabled in settings)")