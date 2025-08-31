"""
Performance monitoring and bottleneck identification service
"""
import time
import psutil
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import structlog
from contextlib import asynccontextmanager

from app.core.database_optimization import db_monitor, get_database_health
from app.services.cache_service import get_cache_service
from app.core.celery_app import task_manager
from app.core.config import settings

logger = structlog.get_logger()


@dataclass
class PerformanceMetrics:
    """Performance metrics data structure"""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    disk_io_read_mb: float
    disk_io_write_mb: float
    network_sent_mb: float
    network_recv_mb: float
    active_connections: int
    response_time_ms: float
    cache_hit_rate: float
    queue_size: int


class PerformanceMonitor:
    """System performance monitoring and analysis"""
    
    def __init__(self):
        self.metrics_history: List[PerformanceMetrics] = []
        self.alerts: List[Dict[str, Any]] = []
        self.thresholds = {
            "cpu_percent": 80.0,
            "memory_percent": 85.0,
            "response_time_ms": 2000.0,
            "cache_hit_rate": 70.0,
            "disk_io_mb_per_sec": 100.0,
            "queue_size": 1000
        }
        self._last_disk_io = None
        self._last_network_io = None
    
    async def collect_metrics(self) -> PerformanceMetrics:
        """Collect current system performance metrics"""
        try:
            # System metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            # Disk I/O
            disk_io = psutil.disk_io_counters()
            disk_read_mb = disk_io.read_bytes / 1024 / 1024 if disk_io else 0
            disk_write_mb = disk_io.write_bytes / 1024 / 1024 if disk_io else 0
            
            # Network I/O
            network_io = psutil.net_io_counters()
            network_sent_mb = network_io.bytes_sent / 1024 / 1024 if network_io else 0
            network_recv_mb = network_io.bytes_recv / 1024 / 1024 if network_io else 0
            
            # Database metrics
            db_health = await get_database_health()
            active_connections = db_health.get("performance_report", {}).get("connection_stats", {}).get("active_connections", 0)
            
            # Cache metrics
            cache_service = await get_cache_service()
            cache_stats = cache_service.get_stats()
            cache_hit_rate = cache_stats.get("hit_rate_percent", 0)
            
            # Application metrics
            response_time_ms = await self._measure_response_time()
            
            # Queue metrics
            celery_stats = task_manager.get_worker_stats()
            queue_size = sum(
                len(worker_stats.get("active", {})) 
                for worker_stats in celery_stats.values()
            ) if celery_stats else 0
            
            metrics = PerformanceMetrics(
                timestamp=time.time(),
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                memory_used_mb=memory.used / 1024 / 1024,
                disk_io_read_mb=disk_read_mb,
                disk_io_write_mb=disk_write_mb,
                network_sent_mb=network_sent_mb,
                network_recv_mb=network_recv_mb,
                active_connections=active_connections,
                response_time_ms=response_time_ms,
                cache_hit_rate=cache_hit_rate,
                queue_size=queue_size
            )
            
            # Store metrics
            self.metrics_history.append(metrics)
            
            # Keep only last 1000 metrics (about 16 hours at 1-minute intervals)
            if len(self.metrics_history) > 1000:
                self.metrics_history = self.metrics_history[-1000:]
            
            # Check for alerts
            await self._check_alerts(metrics)
            
            return metrics
            
        except Exception as e:
            logger.error("Failed to collect performance metrics", error=str(e))
            raise
    
    async def _measure_response_time(self) -> float:
        """Measure application response time"""
        try:
            start_time = time.time()
            
            # Simple health check endpoint simulation
            from app.core.database import AsyncSessionLocal
            async with AsyncSessionLocal() as session:
                from sqlalchemy import text
                await session.execute(text("SELECT 1"))
            
            end_time = time.time()
            return (end_time - start_time) * 1000  # Convert to milliseconds
            
        except Exception as e:
            logger.error("Failed to measure response time", error=str(e))
            return 0.0
    
    async def _check_alerts(self, metrics: PerformanceMetrics) -> None:
        """Check metrics against thresholds and generate alerts"""
        alerts = []
        
        # CPU alert
        if metrics.cpu_percent > self.thresholds["cpu_percent"]:
            alerts.append({
                "type": "high_cpu",
                "severity": "warning" if metrics.cpu_percent < 90 else "critical",
                "message": f"High CPU usage: {metrics.cpu_percent:.1f}%",
                "value": metrics.cpu_percent,
                "threshold": self.thresholds["cpu_percent"],
                "timestamp": metrics.timestamp
            })
        
        # Memory alert
        if metrics.memory_percent > self.thresholds["memory_percent"]:
            alerts.append({
                "type": "high_memory",
                "severity": "warning" if metrics.memory_percent < 95 else "critical",
                "message": f"High memory usage: {metrics.memory_percent:.1f}%",
                "value": metrics.memory_percent,
                "threshold": self.thresholds["memory_percent"],
                "timestamp": metrics.timestamp
            })
        
        # Response time alert
        if metrics.response_time_ms > self.thresholds["response_time_ms"]:
            alerts.append({
                "type": "slow_response",
                "severity": "warning" if metrics.response_time_ms < 5000 else "critical",
                "message": f"Slow response time: {metrics.response_time_ms:.0f}ms",
                "value": metrics.response_time_ms,
                "threshold": self.thresholds["response_time_ms"],
                "timestamp": metrics.timestamp
            })
        
        # Cache hit rate alert
        if metrics.cache_hit_rate < self.thresholds["cache_hit_rate"]:
            alerts.append({
                "type": "low_cache_hit_rate",
                "severity": "warning",
                "message": f"Low cache hit rate: {metrics.cache_hit_rate:.1f}%",
                "value": metrics.cache_hit_rate,
                "threshold": self.thresholds["cache_hit_rate"],
                "timestamp": metrics.timestamp
            })
        
        # Queue size alert
        if metrics.queue_size > self.thresholds["queue_size"]:
            alerts.append({
                "type": "high_queue_size",
                "severity": "warning" if metrics.queue_size < 2000 else "critical",
                "message": f"High queue size: {metrics.queue_size} tasks",
                "value": metrics.queue_size,
                "threshold": self.thresholds["queue_size"],
                "timestamp": metrics.timestamp
            })
        
        # Add new alerts
        for alert in alerts:
            self.alerts.append(alert)
            logger.warning("Performance alert", **alert)
        
        # Keep only recent alerts (last 24 hours)
        cutoff_time = time.time() - 86400
        self.alerts = [
            alert for alert in self.alerts 
            if alert["timestamp"] > cutoff_time
        ]
    
    def get_performance_summary(self, hours: int = 1) -> Dict[str, Any]:
        """Get performance summary for the specified time period"""
        try:
            cutoff_time = time.time() - (hours * 3600)
            recent_metrics = [
                m for m in self.metrics_history 
                if m.timestamp > cutoff_time
            ]
            
            if not recent_metrics:
                return {"error": "No metrics available for the specified period"}
            
            # Calculate averages
            avg_cpu = sum(m.cpu_percent for m in recent_metrics) / len(recent_metrics)
            avg_memory = sum(m.memory_percent for m in recent_metrics) / len(recent_metrics)
            avg_response_time = sum(m.response_time_ms for m in recent_metrics) / len(recent_metrics)
            avg_cache_hit_rate = sum(m.cache_hit_rate for m in recent_metrics) / len(recent_metrics)
            
            # Find peaks
            max_cpu = max(m.cpu_percent for m in recent_metrics)
            max_memory = max(m.memory_percent for m in recent_metrics)
            max_response_time = max(m.response_time_ms for m in recent_metrics)
            
            # Recent alerts
            recent_alerts = [
                alert for alert in self.alerts 
                if alert["timestamp"] > cutoff_time
            ]
            
            return {
                "period_hours": hours,
                "metrics_count": len(recent_metrics),
                "averages": {
                    "cpu_percent": round(avg_cpu, 2),
                    "memory_percent": round(avg_memory, 2),
                    "response_time_ms": round(avg_response_time, 2),
                    "cache_hit_rate": round(avg_cache_hit_rate, 2)
                },
                "peaks": {
                    "max_cpu_percent": round(max_cpu, 2),
                    "max_memory_percent": round(max_memory, 2),
                    "max_response_time_ms": round(max_response_time, 2)
                },
                "alerts": {
                    "total_count": len(recent_alerts),
                    "critical_count": len([a for a in recent_alerts if a["severity"] == "critical"]),
                    "warning_count": len([a for a in recent_alerts if a["severity"] == "warning"]),
                    "recent_alerts": recent_alerts[-5:]  # Last 5 alerts
                },
                "current_status": self._get_current_status(recent_metrics[-1] if recent_metrics else None)
            }
            
        except Exception as e:
            logger.error("Failed to generate performance summary", error=str(e))
            return {"error": str(e)}
    
    def _get_current_status(self, latest_metrics: Optional[PerformanceMetrics]) -> str:
        """Determine current system status"""
        if not latest_metrics:
            return "unknown"
        
        critical_issues = 0
        warning_issues = 0
        
        if latest_metrics.cpu_percent > 90:
            critical_issues += 1
        elif latest_metrics.cpu_percent > self.thresholds["cpu_percent"]:
            warning_issues += 1
        
        if latest_metrics.memory_percent > 95:
            critical_issues += 1
        elif latest_metrics.memory_percent > self.thresholds["memory_percent"]:
            warning_issues += 1
        
        if latest_metrics.response_time_ms > 5000:
            critical_issues += 1
        elif latest_metrics.response_time_ms > self.thresholds["response_time_ms"]:
            warning_issues += 1
        
        if critical_issues > 0:
            return "critical"
        elif warning_issues > 0:
            return "warning"
        else:
            return "healthy"
    
    def get_bottleneck_analysis(self) -> Dict[str, Any]:
        """Analyze system bottlenecks and provide recommendations"""
        try:
            if len(self.metrics_history) < 10:
                return {"error": "Insufficient data for bottleneck analysis"}
            
            recent_metrics = self.metrics_history[-60:]  # Last hour of data
            bottlenecks = []
            recommendations = []
            
            # CPU bottleneck analysis
            avg_cpu = sum(m.cpu_percent for m in recent_metrics) / len(recent_metrics)
            if avg_cpu > 70:
                bottlenecks.append({
                    "type": "cpu",
                    "severity": "high" if avg_cpu > 85 else "medium",
                    "average_usage": round(avg_cpu, 2),
                    "description": "High CPU usage detected"
                })
                recommendations.append({
                    "type": "cpu_optimization",
                    "priority": "high",
                    "suggestion": "Consider optimizing CPU-intensive operations or scaling horizontally"
                })
            
            # Memory bottleneck analysis
            avg_memory = sum(m.memory_percent for m in recent_metrics) / len(recent_metrics)
            if avg_memory > 75:
                bottlenecks.append({
                    "type": "memory",
                    "severity": "high" if avg_memory > 90 else "medium",
                    "average_usage": round(avg_memory, 2),
                    "description": "High memory usage detected"
                })
                recommendations.append({
                    "type": "memory_optimization",
                    "priority": "high",
                    "suggestion": "Review memory usage patterns and consider increasing available memory"
                })
            
            # Response time bottleneck analysis
            avg_response_time = sum(m.response_time_ms for m in recent_metrics) / len(recent_metrics)
            if avg_response_time > 1000:
                bottlenecks.append({
                    "type": "response_time",
                    "severity": "high" if avg_response_time > 3000 else "medium",
                    "average_time_ms": round(avg_response_time, 2),
                    "description": "Slow response times detected"
                })
                recommendations.append({
                    "type": "performance_optimization",
                    "priority": "high",
                    "suggestion": "Optimize database queries and implement better caching strategies"
                })
            
            # Cache performance analysis
            avg_cache_hit_rate = sum(m.cache_hit_rate for m in recent_metrics) / len(recent_metrics)
            if avg_cache_hit_rate < 60:
                bottlenecks.append({
                    "type": "cache",
                    "severity": "medium",
                    "hit_rate": round(avg_cache_hit_rate, 2),
                    "description": "Low cache hit rate detected"
                })
                recommendations.append({
                    "type": "cache_optimization",
                    "priority": "medium",
                    "suggestion": "Review caching strategies and increase cache TTL for stable data"
                })
            
            # Queue size analysis
            avg_queue_size = sum(m.queue_size for m in recent_metrics) / len(recent_metrics)
            if avg_queue_size > 100:
                bottlenecks.append({
                    "type": "queue",
                    "severity": "high" if avg_queue_size > 500 else "medium",
                    "average_size": round(avg_queue_size, 2),
                    "description": "High task queue size detected"
                })
                recommendations.append({
                    "type": "queue_optimization",
                    "priority": "high",
                    "suggestion": "Scale up worker processes or optimize task processing"
                })
            
            return {
                "analysis_timestamp": time.time(),
                "data_points_analyzed": len(recent_metrics),
                "bottlenecks": bottlenecks,
                "recommendations": recommendations,
                "overall_health": self._get_current_status(recent_metrics[-1] if recent_metrics else None)
            }
            
        except Exception as e:
            logger.error("Failed to analyze bottlenecks", error=str(e))
            return {"error": str(e)}


@asynccontextmanager
async def performance_context(operation_name: str):
    """Context manager for measuring operation performance"""
    start_time = time.time()
    start_memory = psutil.Process().memory_info().rss / 1024 / 1024
    
    try:
        yield
    finally:
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        execution_time = (end_time - start_time) * 1000  # milliseconds
        memory_delta = end_memory - start_memory
        
        logger.info(
            "Operation performance",
            operation=operation_name,
            execution_time_ms=round(execution_time, 2),
            memory_delta_mb=round(memory_delta, 2)
        )


# Global performance monitor
performance_monitor = PerformanceMonitor()


async def get_performance_monitor() -> PerformanceMonitor:
    """Get global performance monitor instance"""
    return performance_monitor