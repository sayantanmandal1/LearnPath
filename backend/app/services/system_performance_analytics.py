"""
System Performance Analytics and Monitoring Service
Comprehensive monitoring for external API integrations, user engagement, and system health
"""
import asyncio
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
from enum import Enum
import structlog

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, or_, text
from prometheus_client import Counter, Histogram, Gauge, Summary, CollectorRegistry

# from app.core.database import get_db  # Removed to avoid import issues
from app.core.redis import redis_manager
from app.core.monitoring import system_monitor, Alert, AlertType, AlertSeverity
from app.core.graceful_degradation import degradation_manager
from app.services.performance_monitoring import performance_monitor
from app.models.user import User
from app.models.profile import UserProfile

logger = structlog.get_logger()


class MetricType(Enum):
    """Types of metrics to track"""
    API_PERFORMANCE = "api_performance"
    USER_ENGAGEMENT = "user_engagement"
    SYSTEM_HEALTH = "system_health"
    EXTERNAL_API = "external_api"
    ML_MODEL = "ml_model"
    DATABASE = "database"
    CACHE = "cache"


@dataclass
class APIPerformanceMetric:
    """API performance metric data"""
    endpoint: str
    method: str
    response_time_ms: float
    status_code: int
    timestamp: datetime
    user_id: Optional[str] = None
    error_message: Optional[str] = None
    request_size_bytes: Optional[int] = None
    response_size_bytes: Optional[int] = None


@dataclass
class UserEngagementMetric:
    """User engagement metric data"""
    user_id: str
    action: str
    feature: str
    timestamp: datetime
    session_id: Optional[str] = None
    duration_seconds: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ExternalAPIMetric:
    """External API performance metric"""
    service_name: str
    endpoint: str
    response_time_ms: float
    status_code: int
    timestamp: datetime
    success: bool
    error_message: Optional[str] = None
    retry_count: int = 0


@dataclass
class SystemHealthSnapshot:
    """System health snapshot"""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    disk_usage_percent: float
    active_connections: int
    cache_hit_rate: float
    queue_size: int
    error_rate: float
    response_time_p95: float


class SystemPerformanceAnalytics:
    """Comprehensive system performance analytics service"""
    
    def __init__(self):
        self.metrics_buffer: Dict[MetricType, deque] = {
            metric_type: deque(maxlen=10000) for metric_type in MetricType
        }
        self.analytics_cache = {}
        self.cache_ttl = 300  # 5 minutes
        
        # Prometheus metrics
        self.setup_prometheus_metrics()
        
        # Performance thresholds
        self.thresholds = {
            "api_response_time_ms": 2000,
            "external_api_response_time_ms": 5000,
            "error_rate_percent": 5.0,
            "cache_hit_rate_percent": 70.0,
            "user_session_duration_minutes": 30,
            "database_query_time_ms": 1000,
        }
    
    def setup_prometheus_metrics(self):
        """Setup Prometheus metrics for monitoring"""
        self.api_request_duration = Histogram(
            'api_request_duration_seconds',
            'API request duration in seconds',
            ['endpoint', 'method', 'status_code']
        )
        
        self.external_api_duration = Histogram(
            'external_api_duration_seconds',
            'External API request duration in seconds',
            ['service', 'endpoint', 'status']
        )
        
        self.user_engagement_counter = Counter(
            'user_engagement_total',
            'Total user engagement events',
            ['action', 'feature']
        )
        
        self.system_error_rate = Gauge(
            'system_error_rate_percent',
            'System error rate percentage'
        )
        
        self.active_users_gauge = Gauge(
            'active_users_total',
            'Number of active users',
            ['time_window']
        )
        
        self.ml_model_performance = Gauge(
            'ml_model_performance_score',
            'ML model performance score',
            ['model_name', 'metric_type']
        )
    
    async def record_api_performance(self, metric: APIPerformanceMetric):
        """Record API performance metric"""
        try:
            # Store in buffer
            self.metrics_buffer[MetricType.API_PERFORMANCE].append(metric)
            
            # Update Prometheus metrics
            self.api_request_duration.labels(
                endpoint=metric.endpoint,
                method=metric.method,
                status_code=str(metric.status_code)
            ).observe(metric.response_time_ms / 1000)
            
            # Check for performance issues
            if metric.response_time_ms > self.thresholds["api_response_time_ms"]:
                await self._create_performance_alert(
                    f"Slow API response: {metric.endpoint}",
                    f"Response time {metric.response_time_ms}ms exceeds threshold",
                    AlertSeverity.MEDIUM,
                    {"endpoint": metric.endpoint, "response_time": metric.response_time_ms}
                )
            
            # Store in Redis for real-time analytics
            await self._store_metric_in_redis("api_performance", asdict(metric))
            
        except Exception as e:
            logger.error(f"Failed to record API performance metric: {e}")
    
    async def record_user_engagement(self, metric: UserEngagementMetric):
        """Record user engagement metric"""
        try:
            # Store in buffer
            self.metrics_buffer[MetricType.USER_ENGAGEMENT].append(metric)
            
            # Update Prometheus metrics
            self.user_engagement_counter.labels(
                action=metric.action,
                feature=metric.feature
            ).inc()
            
            # Store in Redis
            await self._store_metric_in_redis("user_engagement", asdict(metric))
            
        except Exception as e:
            logger.error(f"Failed to record user engagement metric: {e}")
    
    async def record_external_api_performance(self, metric: ExternalAPIMetric):
        """Record external API performance metric"""
        try:
            # Store in buffer
            self.metrics_buffer[MetricType.EXTERNAL_API].append(metric)
            
            # Update Prometheus metrics
            status = "success" if metric.success else "error"
            self.external_api_duration.labels(
                service=metric.service_name,
                endpoint=metric.endpoint,
                status=status
            ).observe(metric.response_time_ms / 1000)
            
            # Check for issues
            if not metric.success or metric.response_time_ms > self.thresholds["external_api_response_time_ms"]:
                severity = AlertSeverity.HIGH if not metric.success else AlertSeverity.MEDIUM
                await self._create_performance_alert(
                    f"External API issue: {metric.service_name}",
                    f"{'Failed request' if not metric.success else 'Slow response'}: {metric.error_message or 'Timeout'}",
                    severity,
                    {
                        "service": metric.service_name,
                        "endpoint": metric.endpoint,
                        "response_time": metric.response_time_ms,
                        "retry_count": metric.retry_count
                    }
                )
            
            # Store in Redis
            await self._store_metric_in_redis("external_api", asdict(metric))
            
        except Exception as e:
            logger.error(f"Failed to record external API metric: {e}")
    
    async def record_system_health_snapshot(self, snapshot: SystemHealthSnapshot):
        """Record system health snapshot"""
        try:
            # Store in buffer
            self.metrics_buffer[MetricType.SYSTEM_HEALTH].append(snapshot)
            
            # Update Prometheus metrics
            self.system_error_rate.set(snapshot.error_rate)
            
            # Check thresholds
            if snapshot.error_rate > self.thresholds["error_rate_percent"]:
                await self._create_performance_alert(
                    "High system error rate",
                    f"Error rate {snapshot.error_rate:.2f}% exceeds threshold",
                    AlertSeverity.HIGH,
                    {"error_rate": snapshot.error_rate}
                )
            
            if snapshot.cache_hit_rate < self.thresholds["cache_hit_rate_percent"]:
                await self._create_performance_alert(
                    "Low cache hit rate",
                    f"Cache hit rate {snapshot.cache_hit_rate:.2f}% below threshold",
                    AlertSeverity.MEDIUM,
                    {"cache_hit_rate": snapshot.cache_hit_rate}
                )
            
            # Store in Redis
            await self._store_metric_in_redis("system_health", asdict(snapshot))
            
        except Exception as e:
            logger.error(f"Failed to record system health snapshot: {e}")
    
    async def get_api_performance_analytics(
        self, 
        hours: int = 24,
        endpoint_filter: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get API performance analytics"""
        cache_key = f"api_analytics_{hours}_{endpoint_filter or 'all'}"
        
        # Check cache
        cached = await self._get_cached_analytics(cache_key)
        if cached:
            return cached
        
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=hours)
            
            # Filter metrics
            metrics = [
                m for m in self.metrics_buffer[MetricType.API_PERFORMANCE]
                if m.timestamp > cutoff_time and (
                    not endpoint_filter or endpoint_filter in m.endpoint
                )
            ]
            
            if not metrics:
                return {"error": "No API performance data available"}
            
            # Calculate analytics
            total_requests = len(metrics)
            successful_requests = len([m for m in metrics if 200 <= m.status_code < 400])
            error_requests = total_requests - successful_requests
            
            response_times = [m.response_time_ms for m in metrics]
            avg_response_time = sum(response_times) / len(response_times)
            p95_response_time = sorted(response_times)[int(len(response_times) * 0.95)]
            p99_response_time = sorted(response_times)[int(len(response_times) * 0.99)]
            
            # Endpoint breakdown
            endpoint_stats = defaultdict(lambda: {"count": 0, "avg_time": 0, "errors": 0})
            for metric in metrics:
                endpoint_stats[metric.endpoint]["count"] += 1
                endpoint_stats[metric.endpoint]["avg_time"] += metric.response_time_ms
                if metric.status_code >= 400:
                    endpoint_stats[metric.endpoint]["errors"] += 1
            
            # Calculate averages
            for endpoint, stats in endpoint_stats.items():
                stats["avg_time"] = stats["avg_time"] / stats["count"]
                stats["error_rate"] = (stats["errors"] / stats["count"]) * 100
            
            # Slowest endpoints
            slowest_endpoints = sorted(
                endpoint_stats.items(),
                key=lambda x: x[1]["avg_time"],
                reverse=True
            )[:5]
            
            # Error-prone endpoints
            error_prone_endpoints = sorted(
                endpoint_stats.items(),
                key=lambda x: x[1]["error_rate"],
                reverse=True
            )[:5]
            
            analytics = {
                "period_hours": hours,
                "total_requests": total_requests,
                "successful_requests": successful_requests,
                "error_requests": error_requests,
                "success_rate": (successful_requests / total_requests) * 100,
                "error_rate": (error_requests / total_requests) * 100,
                "response_time_stats": {
                    "average_ms": round(avg_response_time, 2),
                    "p95_ms": round(p95_response_time, 2),
                    "p99_ms": round(p99_response_time, 2),
                    "min_ms": min(response_times),
                    "max_ms": max(response_times)
                },
                "endpoint_stats": dict(endpoint_stats),
                "slowest_endpoints": slowest_endpoints,
                "error_prone_endpoints": error_prone_endpoints,
                "generated_at": datetime.utcnow().isoformat()
            }
            
            # Cache results
            await self._cache_analytics(cache_key, analytics)
            return analytics
            
        except Exception as e:
            logger.error(f"Failed to generate API performance analytics: {e}")
            return {"error": str(e)}
    
    async def get_user_engagement_analytics(self, hours: int = 24) -> Dict[str, Any]:
        """Get user engagement analytics"""
        cache_key = f"engagement_analytics_{hours}"
        
        # Check cache
        cached = await self._get_cached_analytics(cache_key)
        if cached:
            return cached
        
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=hours)
            
            # Filter metrics
            metrics = [
                m for m in self.metrics_buffer[MetricType.USER_ENGAGEMENT]
                if m.timestamp > cutoff_time
            ]
            
            if not metrics:
                return {"error": "No user engagement data available"}
            
            # Calculate analytics
            total_events = len(metrics)
            unique_users = len(set(m.user_id for m in metrics))
            
            # Feature usage
            feature_usage = defaultdict(int)
            action_usage = defaultdict(int)
            user_sessions = defaultdict(list)
            
            for metric in metrics:
                feature_usage[metric.feature] += 1
                action_usage[metric.action] += 1
                user_sessions[metric.user_id].append(metric)
            
            # Session analytics
            session_durations = []
            for user_id, user_metrics in user_sessions.items():
                if len(user_metrics) > 1:
                    session_start = min(m.timestamp for m in user_metrics)
                    session_end = max(m.timestamp for m in user_metrics)
                    duration = (session_end - session_start).total_seconds() / 60  # minutes
                    session_durations.append(duration)
            
            avg_session_duration = sum(session_durations) / len(session_durations) if session_durations else 0
            
            # Most popular features and actions
            popular_features = sorted(feature_usage.items(), key=lambda x: x[1], reverse=True)[:10]
            popular_actions = sorted(action_usage.items(), key=lambda x: x[1], reverse=True)[:10]
            
            # User activity patterns (hourly)
            hourly_activity = defaultdict(int)
            for metric in metrics:
                hour = metric.timestamp.hour
                hourly_activity[hour] += 1
            
            analytics = {
                "period_hours": hours,
                "total_events": total_events,
                "unique_users": unique_users,
                "events_per_user": round(total_events / unique_users, 2) if unique_users > 0 else 0,
                "average_session_duration_minutes": round(avg_session_duration, 2),
                "feature_usage": dict(feature_usage),
                "action_usage": dict(action_usage),
                "popular_features": popular_features,
                "popular_actions": popular_actions,
                "hourly_activity": dict(hourly_activity),
                "generated_at": datetime.utcnow().isoformat()
            }
            
            # Update Prometheus metrics
            self.active_users_gauge.labels(time_window=f"{hours}h").set(unique_users)
            
            # Cache results
            await self._cache_analytics(cache_key, analytics)
            return analytics
            
        except Exception as e:
            logger.error(f"Failed to generate user engagement analytics: {e}")
            return {"error": str(e)}
    
    async def get_external_api_analytics(self, hours: int = 24) -> Dict[str, Any]:
        """Get external API performance analytics"""
        cache_key = f"external_api_analytics_{hours}"
        
        # Check cache
        cached = await self._get_cached_analytics(cache_key)
        if cached:
            return cached
        
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=hours)
            
            # Filter metrics
            metrics = [
                m for m in self.metrics_buffer[MetricType.EXTERNAL_API]
                if m.timestamp > cutoff_time
            ]
            
            if not metrics:
                return {"error": "No external API data available"}
            
            # Service-wise analytics
            service_stats = defaultdict(lambda: {
                "total_requests": 0,
                "successful_requests": 0,
                "failed_requests": 0,
                "total_response_time": 0,
                "max_response_time": 0,
                "retry_count": 0
            })
            
            for metric in metrics:
                stats = service_stats[metric.service_name]
                stats["total_requests"] += 1
                stats["total_response_time"] += metric.response_time_ms
                stats["max_response_time"] = max(stats["max_response_time"], metric.response_time_ms)
                stats["retry_count"] += metric.retry_count
                
                if metric.success:
                    stats["successful_requests"] += 1
                else:
                    stats["failed_requests"] += 1
            
            # Calculate derived metrics
            for service, stats in service_stats.items():
                stats["success_rate"] = (stats["successful_requests"] / stats["total_requests"]) * 100
                stats["failure_rate"] = (stats["failed_requests"] / stats["total_requests"]) * 100
                stats["avg_response_time"] = stats["total_response_time"] / stats["total_requests"]
            
            # Overall statistics
            total_requests = sum(stats["total_requests"] for stats in service_stats.values())
            total_successful = sum(stats["successful_requests"] for stats in service_stats.values())
            total_failed = sum(stats["failed_requests"] for stats in service_stats.values())
            
            # Most problematic services
            problematic_services = sorted(
                service_stats.items(),
                key=lambda x: x[1]["failure_rate"],
                reverse=True
            )[:5]
            
            # Slowest services
            slowest_services = sorted(
                service_stats.items(),
                key=lambda x: x[1]["avg_response_time"],
                reverse=True
            )[:5]
            
            analytics = {
                "period_hours": hours,
                "overall_stats": {
                    "total_requests": total_requests,
                    "successful_requests": total_successful,
                    "failed_requests": total_failed,
                    "overall_success_rate": (total_successful / total_requests) * 100 if total_requests > 0 else 0,
                    "overall_failure_rate": (total_failed / total_requests) * 100 if total_requests > 0 else 0
                },
                "service_stats": dict(service_stats),
                "problematic_services": problematic_services,
                "slowest_services": slowest_services,
                "generated_at": datetime.utcnow().isoformat()
            }
            
            # Cache results
            await self._cache_analytics(cache_key, analytics)
            return analytics
            
        except Exception as e:
            logger.error(f"Failed to generate external API analytics: {e}")
            return {"error": str(e)}
    
    async def get_system_health_analytics(self, hours: int = 24) -> Dict[str, Any]:
        """Get system health analytics"""
        cache_key = f"system_health_analytics_{hours}"
        
        # Check cache
        cached = await self._get_cached_analytics(cache_key)
        if cached:
            return cached
        
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=hours)
            
            # Filter metrics
            metrics = [
                m for m in self.metrics_buffer[MetricType.SYSTEM_HEALTH]
                if m.timestamp > cutoff_time
            ]
            
            if not metrics:
                return {"error": "No system health data available"}
            
            # Calculate averages and trends
            cpu_values = [m.cpu_percent for m in metrics]
            memory_values = [m.memory_percent for m in metrics]
            cache_hit_rates = [m.cache_hit_rate for m in metrics]
            error_rates = [m.error_rate for m in metrics]
            response_times = [m.response_time_p95 for m in metrics]
            
            analytics = {
                "period_hours": hours,
                "data_points": len(metrics),
                "cpu_stats": {
                    "average": round(sum(cpu_values) / len(cpu_values), 2),
                    "max": max(cpu_values),
                    "min": min(cpu_values),
                    "current": cpu_values[-1] if cpu_values else 0
                },
                "memory_stats": {
                    "average": round(sum(memory_values) / len(memory_values), 2),
                    "max": max(memory_values),
                    "min": min(memory_values),
                    "current": memory_values[-1] if memory_values else 0
                },
                "cache_stats": {
                    "average_hit_rate": round(sum(cache_hit_rates) / len(cache_hit_rates), 2),
                    "max_hit_rate": max(cache_hit_rates),
                    "min_hit_rate": min(cache_hit_rates),
                    "current_hit_rate": cache_hit_rates[-1] if cache_hit_rates else 0
                },
                "error_stats": {
                    "average_error_rate": round(sum(error_rates) / len(error_rates), 2),
                    "max_error_rate": max(error_rates),
                    "min_error_rate": min(error_rates),
                    "current_error_rate": error_rates[-1] if error_rates else 0
                },
                "response_time_stats": {
                    "average_p95": round(sum(response_times) / len(response_times), 2),
                    "max_p95": max(response_times),
                    "min_p95": min(response_times),
                    "current_p95": response_times[-1] if response_times else 0
                },
                "generated_at": datetime.utcnow().isoformat()
            }
            
            # Cache results
            await self._cache_analytics(cache_key, analytics)
            return analytics
            
        except Exception as e:
            logger.error(f"Failed to generate system health analytics: {e}")
            return {"error": str(e)}
    
    async def get_comprehensive_performance_report(self, hours: int = 24) -> Dict[str, Any]:
        """Get comprehensive performance report"""
        try:
            # Get all analytics concurrently
            api_analytics, engagement_analytics, external_api_analytics, health_analytics = await asyncio.gather(
                self.get_api_performance_analytics(hours),
                self.get_user_engagement_analytics(hours),
                self.get_external_api_analytics(hours),
                self.get_system_health_analytics(hours),
                return_exceptions=True
            )
            
            # Get current alerts
            active_alerts = system_monitor.get_active_alerts()
            alert_summary = system_monitor.get_alert_summary()
            
            # Calculate overall health score
            health_score = await self._calculate_overall_health_score(
                api_analytics, external_api_analytics, health_analytics
            )
            
            # Generate recommendations
            recommendations = await self._generate_performance_recommendations(
                api_analytics, external_api_analytics, health_analytics
            )
            
            return {
                "report_period_hours": hours,
                "generated_at": datetime.utcnow().isoformat(),
                "overall_health_score": health_score,
                "api_performance": api_analytics if not isinstance(api_analytics, Exception) else {"error": str(api_analytics)},
                "user_engagement": engagement_analytics if not isinstance(engagement_analytics, Exception) else {"error": str(engagement_analytics)},
                "external_api_performance": external_api_analytics if not isinstance(external_api_analytics, Exception) else {"error": str(external_api_analytics)},
                "system_health": health_analytics if not isinstance(health_analytics, Exception) else {"error": str(health_analytics)},
                "alerts": {
                    "active_alerts": len(active_alerts),
                    "alert_summary": alert_summary,
                    "recent_alerts": [
                        {
                            "id": alert.id,
                            "type": alert.type.value,
                            "severity": alert.severity.value,
                            "title": alert.title,
                            "timestamp": alert.timestamp.isoformat()
                        }
                        for alert in active_alerts[:10]
                    ]
                },
                "recommendations": recommendations
            }
            
        except Exception as e:
            logger.error(f"Failed to generate comprehensive performance report: {e}")
            return {"error": str(e)}
    
    async def _create_performance_alert(
        self, 
        title: str, 
        description: str, 
        severity: AlertSeverity,
        metadata: Dict[str, Any]
    ):
        """Create performance-related alert"""
        system_monitor.create_alert(
            alert_type=AlertType.SLOW_RESPONSE,
            severity=severity,
            title=title,
            description=description,
            service="performance_monitoring",
            metadata=metadata
        )
    
    async def _store_metric_in_redis(self, metric_type: str, metric_data: Dict[str, Any]):
        """Store metric in Redis for real-time access"""
        try:
            key = f"metrics:{metric_type}:{int(time.time())}"
            await redis_manager.setex(key, 3600, json.dumps(metric_data, default=str))  # 1 hour TTL
        except Exception as e:
            logger.error(f"Failed to store metric in Redis: {e}")
    
    async def _get_cached_analytics(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get cached analytics result"""
        try:
            cached_data = self.analytics_cache.get(cache_key)
            if cached_data and time.time() - cached_data["timestamp"] < self.cache_ttl:
                return cached_data["data"]
        except Exception as e:
            logger.error(f"Failed to get cached analytics: {e}")
        return None
    
    async def _cache_analytics(self, cache_key: str, data: Dict[str, Any]):
        """Cache analytics result"""
        try:
            self.analytics_cache[cache_key] = {
                "data": data,
                "timestamp": time.time()
            }
        except Exception as e:
            logger.error(f"Failed to cache analytics: {e}")
    
    async def _calculate_overall_health_score(
        self, 
        api_analytics: Dict[str, Any],
        external_api_analytics: Dict[str, Any],
        health_analytics: Dict[str, Any]
    ) -> float:
        """Calculate overall system health score (0-100)"""
        try:
            score = 100.0
            
            # API performance impact
            if isinstance(api_analytics, dict) and "error_rate" in api_analytics:
                error_rate = api_analytics["error_rate"]
                if error_rate > 10:
                    score -= 30
                elif error_rate > 5:
                    score -= 15
                elif error_rate > 1:
                    score -= 5
            
            # External API impact
            if isinstance(external_api_analytics, dict) and "overall_stats" in external_api_analytics:
                failure_rate = external_api_analytics["overall_stats"].get("overall_failure_rate", 0)
                if failure_rate > 20:
                    score -= 25
                elif failure_rate > 10:
                    score -= 15
                elif failure_rate > 5:
                    score -= 10
            
            # System health impact
            if isinstance(health_analytics, dict):
                cpu_avg = health_analytics.get("cpu_stats", {}).get("average", 0)
                memory_avg = health_analytics.get("memory_stats", {}).get("average", 0)
                error_rate_avg = health_analytics.get("error_stats", {}).get("average_error_rate", 0)
                
                if cpu_avg > 90:
                    score -= 20
                elif cpu_avg > 80:
                    score -= 10
                
                if memory_avg > 90:
                    score -= 20
                elif memory_avg > 80:
                    score -= 10
                
                if error_rate_avg > 5:
                    score -= 15
            
            return max(0.0, min(100.0, score))
            
        except Exception as e:
            logger.error(f"Failed to calculate health score: {e}")
            return 50.0  # Default neutral score
    
    async def _generate_performance_recommendations(
        self,
        api_analytics: Dict[str, Any],
        external_api_analytics: Dict[str, Any],
        health_analytics: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate performance improvement recommendations"""
        recommendations = []
        
        try:
            # API performance recommendations
            if isinstance(api_analytics, dict):
                if api_analytics.get("error_rate", 0) > 5:
                    recommendations.append({
                        "category": "API Performance",
                        "priority": "high",
                        "title": "High API Error Rate",
                        "description": f"API error rate is {api_analytics['error_rate']:.2f}%, which exceeds the 5% threshold",
                        "actions": [
                            "Review error logs for common failure patterns",
                            "Implement better error handling and retry logic",
                            "Consider circuit breaker patterns for external dependencies"
                        ]
                    })
                
                avg_response_time = api_analytics.get("response_time_stats", {}).get("average_ms", 0)
                if avg_response_time > 2000:
                    recommendations.append({
                        "category": "API Performance",
                        "priority": "medium",
                        "title": "Slow API Response Times",
                        "description": f"Average response time is {avg_response_time:.0f}ms, which exceeds the 2000ms threshold",
                        "actions": [
                            "Optimize database queries and add proper indexing",
                            "Implement caching for frequently accessed data",
                            "Consider API response pagination for large datasets"
                        ]
                    })
            
            # External API recommendations
            if isinstance(external_api_analytics, dict) and "overall_stats" in external_api_analytics:
                failure_rate = external_api_analytics["overall_stats"].get("overall_failure_rate", 0)
                if failure_rate > 10:
                    recommendations.append({
                        "category": "External APIs",
                        "priority": "high",
                        "title": "High External API Failure Rate",
                        "description": f"External API failure rate is {failure_rate:.2f}%, indicating reliability issues",
                        "actions": [
                            "Implement robust retry mechanisms with exponential backoff",
                            "Add circuit breakers to prevent cascade failures",
                            "Consider fallback data sources or cached responses"
                        ]
                    })
            
            # System health recommendations
            if isinstance(health_analytics, dict):
                cpu_avg = health_analytics.get("cpu_stats", {}).get("average", 0)
                memory_avg = health_analytics.get("memory_stats", {}).get("average", 0)
                cache_hit_rate = health_analytics.get("cache_stats", {}).get("average_hit_rate", 100)
                
                if cpu_avg > 80:
                    recommendations.append({
                        "category": "System Resources",
                        "priority": "high",
                        "title": "High CPU Usage",
                        "description": f"Average CPU usage is {cpu_avg:.1f}%, indicating potential performance bottlenecks",
                        "actions": [
                            "Profile application to identify CPU-intensive operations",
                            "Consider horizontal scaling or load balancing",
                            "Optimize algorithms and data processing logic"
                        ]
                    })
                
                if memory_avg > 80:
                    recommendations.append({
                        "category": "System Resources",
                        "priority": "high",
                        "title": "High Memory Usage",
                        "description": f"Average memory usage is {memory_avg:.1f}%, which may lead to performance degradation",
                        "actions": [
                            "Review memory usage patterns and identify leaks",
                            "Optimize data structures and caching strategies",
                            "Consider increasing available memory or implementing memory limits"
                        ]
                    })
                
                if cache_hit_rate < 70:
                    recommendations.append({
                        "category": "Caching",
                        "priority": "medium",
                        "title": "Low Cache Hit Rate",
                        "description": f"Cache hit rate is {cache_hit_rate:.1f}%, indicating inefficient caching",
                        "actions": [
                            "Review caching strategies and TTL settings",
                            "Identify frequently accessed data that should be cached",
                            "Consider implementing cache warming strategies"
                        ]
                    })
            
        except Exception as e:
            logger.error(f"Failed to generate recommendations: {e}")
        
        return recommendations


# Global instance
system_performance_analytics = SystemPerformanceAnalytics()