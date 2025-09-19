"""
System Monitoring and Analytics API Endpoints
"""
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import asyncio

from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
import structlog

from app.core.database import get_db
from app.core.monitoring import system_monitor, AlertSeverity
from app.services.system_performance_analytics import (
    system_performance_analytics,
    APIPerformanceMetric,
    UserEngagementMetric,
    ExternalAPIMetric,
    SystemHealthSnapshot
)
from app.services.performance_monitoring import performance_monitor
from app.core.graceful_degradation import degradation_manager
from app.schemas.health import (
    SystemMetricsResponse,
    AlertResponse,
    AlertSummaryResponse
)

logger = structlog.get_logger()
router = APIRouter()


@router.get("/performance/api")
async def get_api_performance_analytics(
    hours: int = Query(24, ge=1, le=168, description="Hours to analyze (1-168)"),
    endpoint_filter: Optional[str] = Query(None, description="Filter by endpoint pattern")
):
    """Get API performance analytics"""
    try:
        analytics = await system_performance_analytics.get_api_performance_analytics(
            hours=hours,
            endpoint_filter=endpoint_filter
        )
        return analytics
    except Exception as e:
        logger.error(f"Failed to get API performance analytics: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve API performance analytics")


@router.get("/performance/user-engagement")
async def get_user_engagement_analytics(
    hours: int = Query(24, ge=1, le=168, description="Hours to analyze (1-168)")
):
    """Get user engagement analytics"""
    try:
        analytics = await system_performance_analytics.get_user_engagement_analytics(hours=hours)
        return analytics
    except Exception as e:
        logger.error(f"Failed to get user engagement analytics: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve user engagement analytics")


@router.get("/performance/external-apis")
async def get_external_api_analytics(
    hours: int = Query(24, ge=1, le=168, description="Hours to analyze (1-168)")
):
    """Get external API performance analytics"""
    try:
        analytics = await system_performance_analytics.get_external_api_analytics(hours=hours)
        return analytics
    except Exception as e:
        logger.error(f"Failed to get external API analytics: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve external API analytics")


@router.get("/performance/system-health")
async def get_system_health_analytics(
    hours: int = Query(24, ge=1, le=168, description="Hours to analyze (1-168)")
):
    """Get system health analytics"""
    try:
        analytics = await system_performance_analytics.get_system_health_analytics(hours=hours)
        return analytics
    except Exception as e:
        logger.error(f"Failed to get system health analytics: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve system health analytics")


@router.get("/performance/comprehensive-report")
async def get_comprehensive_performance_report(
    hours: int = Query(24, ge=1, le=168, description="Hours to analyze (1-168)")
):
    """Get comprehensive performance report"""
    try:
        report = await system_performance_analytics.get_comprehensive_performance_report(hours=hours)
        return report
    except Exception as e:
        logger.error(f"Failed to get comprehensive performance report: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve comprehensive performance report")


@router.post("/metrics/api-performance")
async def record_api_performance_metric(
    endpoint: str,
    method: str,
    response_time_ms: float,
    status_code: int,
    user_id: Optional[str] = None,
    error_message: Optional[str] = None,
    request_size_bytes: Optional[int] = None,
    response_size_bytes: Optional[int] = None
):
    """Record API performance metric"""
    try:
        metric = APIPerformanceMetric(
            endpoint=endpoint,
            method=method,
            response_time_ms=response_time_ms,
            status_code=status_code,
            timestamp=datetime.utcnow(),
            user_id=user_id,
            error_message=error_message,
            request_size_bytes=request_size_bytes,
            response_size_bytes=response_size_bytes
        )
        
        await system_performance_analytics.record_api_performance(metric)
        return {"message": "API performance metric recorded successfully"}
        
    except Exception as e:
        logger.error(f"Failed to record API performance metric: {e}")
        raise HTTPException(status_code=500, detail="Failed to record API performance metric")


@router.post("/metrics/user-engagement")
async def record_user_engagement_metric(
    user_id: str,
    action: str,
    feature: str,
    session_id: Optional[str] = None,
    duration_seconds: Optional[float] = None,
    metadata: Optional[Dict[str, Any]] = None
):
    """Record user engagement metric"""
    try:
        metric = UserEngagementMetric(
            user_id=user_id,
            action=action,
            feature=feature,
            timestamp=datetime.utcnow(),
            session_id=session_id,
            duration_seconds=duration_seconds,
            metadata=metadata
        )
        
        await system_performance_analytics.record_user_engagement(metric)
        return {"message": "User engagement metric recorded successfully"}
        
    except Exception as e:
        logger.error(f"Failed to record user engagement metric: {e}")
        raise HTTPException(status_code=500, detail="Failed to record user engagement metric")


@router.post("/metrics/external-api")
async def record_external_api_metric(
    service_name: str,
    endpoint: str,
    response_time_ms: float,
    status_code: int,
    success: bool,
    error_message: Optional[str] = None,
    retry_count: int = 0
):
    """Record external API performance metric"""
    try:
        metric = ExternalAPIMetric(
            service_name=service_name,
            endpoint=endpoint,
            response_time_ms=response_time_ms,
            status_code=status_code,
            timestamp=datetime.utcnow(),
            success=success,
            error_message=error_message,
            retry_count=retry_count
        )
        
        await system_performance_analytics.record_external_api_performance(metric)
        return {"message": "External API metric recorded successfully"}
        
    except Exception as e:
        logger.error(f"Failed to record external API metric: {e}")
        raise HTTPException(status_code=500, detail="Failed to record external API metric")


@router.post("/metrics/system-health")
async def record_system_health_snapshot(
    cpu_percent: float,
    memory_percent: float,
    disk_usage_percent: float,
    active_connections: int,
    cache_hit_rate: float,
    queue_size: int,
    error_rate: float,
    response_time_p95: float
):
    """Record system health snapshot"""
    try:
        snapshot = SystemHealthSnapshot(
            timestamp=datetime.utcnow(),
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            disk_usage_percent=disk_usage_percent,
            active_connections=active_connections,
            cache_hit_rate=cache_hit_rate,
            queue_size=queue_size,
            error_rate=error_rate,
            response_time_p95=response_time_p95
        )
        
        await system_performance_analytics.record_system_health_snapshot(snapshot)
        return {"message": "System health snapshot recorded successfully"}
        
    except Exception as e:
        logger.error(f"Failed to record system health snapshot: {e}")
        raise HTTPException(status_code=500, detail="Failed to record system health snapshot")


@router.get("/alerts/active")
async def get_active_alerts(
    severity: Optional[str] = Query(None, description="Filter by severity"),
    service: Optional[str] = Query(None, description="Filter by service"),
    limit: int = Query(50, ge=1, le=200, description="Maximum number of alerts to return")
):
    """Get active system alerts"""
    try:
        severity_filter = None
        if severity:
            try:
                severity_filter = AlertSeverity(severity.lower())
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid severity: {severity}")
        
        alerts = system_monitor.get_active_alerts(severity_filter)
        
        # Filter by service if specified
        if service:
            alerts = [alert for alert in alerts if alert.service == service]
        
        # Limit results
        alerts = alerts[:limit]
        
        return [
            AlertResponse(
                id=alert.id,
                type=alert.type.value,
                severity=alert.severity.value,
                title=alert.title,
                description=alert.description,
                timestamp=alert.timestamp,
                service=alert.service,
                metadata=alert.metadata,
                resolved=alert.resolved,
                resolved_at=alert.resolved_at
            )
            for alert in alerts
        ]
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get active alerts: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve active alerts")


@router.get("/alerts/summary")
async def get_alerts_summary():
    """Get alerts summary"""
    try:
        summary = system_monitor.get_alert_summary()
        return AlertSummaryResponse(
            total_active=summary["total_active"],
            by_severity=summary["by_severity"],
            by_service=dict(summary["by_service"])
        )
    except Exception as e:
        logger.error(f"Failed to get alerts summary: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve alerts summary")


@router.post("/alerts/{alert_id}/resolve")
async def resolve_alert(alert_id: str):
    """Resolve a specific alert"""
    try:
        system_monitor.resolve_alert(alert_id)
        return {"message": f"Alert {alert_id} resolved successfully"}
    except Exception as e:
        logger.error(f"Failed to resolve alert {alert_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to resolve alert")


@router.get("/services/health")
async def get_services_health():
    """Get health status of all monitored services"""
    try:
        services_health = degradation_manager.get_all_services_health()
        
        return {
            service_name: {
                "status": health.status.value,
                "success_rate": health.success_rate,
                "error_count": health.error_count,
                "success_count": health.success_count,
                "response_time": health.response_time,
                "last_check": datetime.fromtimestamp(health.last_check).isoformat(),
                "last_error": health.last_error,
                "fallback_available": health.fallback_available,
                "is_healthy": health.is_healthy
            }
            for service_name, health in services_health.items()
        }
        
    except Exception as e:
        logger.error(f"Failed to get services health: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve services health")


@router.get("/services/{service_name}/health")
async def get_service_health(service_name: str):
    """Get health status of a specific service"""
    try:
        health = degradation_manager.get_service_health(service_name)
        
        if not health:
            raise HTTPException(status_code=404, detail=f"Service '{service_name}' not found")
        
        return {
            "name": health.name,
            "status": health.status.value,
            "success_rate": health.success_rate,
            "error_count": health.error_count,
            "success_count": health.success_count,
            "response_time": health.response_time,
            "last_check": datetime.fromtimestamp(health.last_check).isoformat(),
            "last_error": health.last_error,
            "fallback_available": health.fallback_available,
            "is_healthy": health.is_healthy
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get service health for {service_name}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve service health")


@router.get("/system/overview")
async def get_system_overview():
    """Get comprehensive system overview"""
    try:
        # Get system health report
        health_report = system_monitor.get_system_health_report()
        
        # Get performance summary
        performance_summary = performance_monitor.get_performance_summary(hours=1)
        
        # Get bottleneck analysis
        bottleneck_analysis = performance_monitor.get_bottleneck_analysis()
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "health_report": health_report,
            "performance_summary": performance_summary,
            "bottleneck_analysis": bottleneck_analysis
        }
        
    except Exception as e:
        logger.error(f"Failed to get system overview: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve system overview")


@router.post("/monitoring/start")
async def start_monitoring(background_tasks: BackgroundTasks):
    """Start system monitoring"""
    try:
        await system_monitor.start_monitoring()
        
        # Start performance monitoring collection
        background_tasks.add_task(start_performance_collection)
        
        return {"message": "System monitoring started successfully"}
        
    except Exception as e:
        logger.error(f"Failed to start monitoring: {e}")
        raise HTTPException(status_code=500, detail="Failed to start monitoring")


@router.post("/monitoring/stop")
async def stop_monitoring():
    """Stop system monitoring"""
    try:
        await system_monitor.stop_monitoring()
        return {"message": "System monitoring stopped successfully"}
        
    except Exception as e:
        logger.error(f"Failed to stop monitoring: {e}")
        raise HTTPException(status_code=500, detail="Failed to stop monitoring")


@router.get("/monitoring/status")
async def get_monitoring_status():
    """Get monitoring system status"""
    try:
        return {
            "monitoring_active": system_monitor.monitoring_active,
            "total_alerts": len(system_monitor.alerts),
            "active_alerts": len(system_monitor.get_active_alerts()),
            "alert_handlers": len(system_monitor.alert_handlers),
            "metric_thresholds": len(system_monitor.metric_thresholds),
            "services_monitored": len(degradation_manager.get_all_services_health())
        }
        
    except Exception as e:
        logger.error(f"Failed to get monitoring status: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve monitoring status")


@router.post("/test/create-alert")
async def create_test_alert(
    severity: str = Query("medium", description="Alert severity"),
    service: Optional[str] = Query(None, description="Service name")
):
    """Create a test alert for testing monitoring system"""
    try:
        from app.core.monitoring import AlertType
        
        alert_severity = AlertSeverity(severity.lower())
        
        alert = system_monitor.create_alert(
            alert_type=AlertType.SECURITY_INCIDENT,
            severity=alert_severity,
            title="Test Alert",
            description="This is a test alert created via API for monitoring system testing",
            service=service,
            metadata={"test": True, "created_via": "api"}
        )
        
        return {
            "message": "Test alert created successfully",
            "alert_id": alert.id,
            "severity": alert.severity.value,
            "service": alert.service
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid severity: {severity}")
    except Exception as e:
        logger.error(f"Failed to create test alert: {e}")
        raise HTTPException(status_code=500, detail="Failed to create test alert")


async def start_performance_collection():
    """Background task to collect performance metrics"""
    try:
        while True:
            # Collect current performance metrics
            metrics = await performance_monitor.collect_metrics()
            
            # Convert to system health snapshot
            snapshot = SystemHealthSnapshot(
                timestamp=datetime.utcnow(),
                cpu_percent=metrics.cpu_percent,
                memory_percent=metrics.memory_percent,
                disk_usage_percent=0.0,  # Not available in current metrics
                active_connections=metrics.active_connections,
                cache_hit_rate=metrics.cache_hit_rate,
                queue_size=metrics.queue_size,
                error_rate=0.0,  # Calculate from recent API metrics
                response_time_p95=metrics.response_time_ms
            )
            
            # Record in analytics system
            await system_performance_analytics.record_system_health_snapshot(snapshot)
            
            # Wait 60 seconds before next collection
            await asyncio.sleep(60)
            
    except Exception as e:
        logger.error(f"Performance collection task failed: {e}")