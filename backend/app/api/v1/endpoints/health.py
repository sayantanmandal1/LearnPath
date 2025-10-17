"""
Health check and system status endpoints
"""
import asyncio
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.redis import redis_manager
from app.core.monitoring import system_monitor, AlertSeverity
from app.core.graceful_degradation import degradation_manager, ServiceStatus
from app.core.exceptions import SystemHealthError
from app.schemas.health import (
    HealthCheckResponse,
    SystemStatusResponse,
    ServiceHealthResponse,
    AlertResponse,
    SystemMetricsResponse
)

logger = structlog.get_logger()
router = APIRouter()


@router.get("/", response_model=HealthCheckResponse)
async def basic_health_check():
    """Basic health check endpoint"""
    return HealthCheckResponse(
        status="healthy",
        timestamp=datetime.utcnow(),
        version="0.1.0"
    )


@router.get("/detailed", response_model=SystemStatusResponse)
async def detailed_health_check(db: AsyncSession = Depends(get_db)):
    """Detailed system health check"""
    start_time = time.time()
    checks = {}
    overall_status = "healthy"
    
    # Database check
    try:
        await db.execute("SELECT 1")
        checks["database"] = {
            "status": "healthy",
            "response_time": time.time() - start_time,
            "details": "Database connection successful"
        }
    except Exception as e:
        checks["database"] = {
            "status": "unhealthy",
            "response_time": time.time() - start_time,
            "details": f"Database connection failed: {str(e)}"
        }
        overall_status = "unhealthy"
    
    # Redis check
    try:
        redis_start = time.time()
        await redis_manager.redis.ping()
        checks["redis"] = {
            "status": "healthy",
            "response_time": time.time() - redis_start,
            "details": "Redis connection successful"
        }
    except Exception as e:
        checks["redis"] = {
            "status": "unhealthy",
            "response_time": time.time() - redis_start,
            "details": f"Redis connection failed: {str(e)}"
        }
        overall_status = "unhealthy"
    
    # External services check
    services_health = degradation_manager.get_all_services_health()
    for service_name, health in services_health.items():
        status_map = {
            ServiceStatus.HEALTHY: "healthy",
            ServiceStatus.DEGRADED: "degraded",
            ServiceStatus.UNAVAILABLE: "unhealthy",
            ServiceStatus.MAINTENANCE: "maintenance"
        }
        
        checks[f"external_{service_name}"] = {
            "status": status_map.get(health.status, "unknown"),
            "response_time": health.response_time,
            "details": f"Success rate: {health.success_rate:.2%}, Errors: {health.error_count}",
            "last_error": health.last_error
        }
        
        if health.status == ServiceStatus.UNAVAILABLE:
            overall_status = "degraded"  # External services don't make system unhealthy
    
    total_response_time = time.time() - start_time
    
    return SystemStatusResponse(
        status=overall_status,
        timestamp=datetime.utcnow(),
        response_time=total_response_time,
        checks=checks,
        version="0.1.0"
    )


@router.get("/services", response_model=List[ServiceHealthResponse])
async def get_services_health():
    """Get health status of all external services"""
    services_health = degradation_manager.get_all_services_health()
    
    return [
        ServiceHealthResponse(
            name=name,
            status=health.status.value,
            success_rate=health.success_rate,
            error_count=health.error_count,
            success_count=health.success_count,
            response_time=health.response_time,
            last_check=datetime.fromtimestamp(health.last_check),
            last_error=health.last_error,
            fallback_available=health.fallback_available
        )
        for name, health in services_health.items()
    ]


@router.get("/services/{service_name}", response_model=ServiceHealthResponse)
async def get_service_health(service_name: str):
    """Get health status of a specific service"""
    health = degradation_manager.get_service_health(service_name)
    
    if not health:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Service '{service_name}' not found"
        )
    
    return ServiceHealthResponse(
        name=health.name,
        status=health.status.value,
        success_rate=health.success_rate,
        error_count=health.error_count,
        success_count=health.success_count,
        response_time=health.response_time,
        last_check=datetime.fromtimestamp(health.last_check),
        last_error=health.last_error,
        fallback_available=health.fallback_available
    )


@router.get("/alerts", response_model=List[AlertResponse])
async def get_alerts(
    severity: Optional[str] = Query(None, description="Filter by severity"),
    active_only: bool = Query(True, description="Show only active alerts")
):
    """Get system alerts"""
    severity_filter = None
    if severity:
        try:
            severity_filter = AlertSeverity(severity.lower())
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid severity: {severity}"
            )
    
    if active_only:
        alerts = system_monitor.get_active_alerts(severity_filter)
    else:
        alerts = system_monitor.alerts
        if severity_filter:
            alerts = [a for a in alerts if a.severity == severity_filter]
    
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


@router.post("/alerts/{alert_id}/resolve")
async def resolve_alert(alert_id: str):
    """Resolve an alert"""
    system_monitor.resolve_alert(alert_id)
    return {"message": f"Alert {alert_id} resolved"}


@router.get("/alerts/summary")
async def get_alerts_summary():
    """Get alerts summary"""
    return system_monitor.get_alert_summary()


@router.get("/metrics", response_model=SystemMetricsResponse)
async def get_system_metrics():
    """Get system metrics and performance data"""
    health_report = system_monitor.get_system_health_report()
    
    return SystemMetricsResponse(
        timestamp=datetime.utcnow(),
        overall_health_percentage=health_report["overall_health_percentage"],
        services_count=health_report["metrics"]["total_services"],
        healthy_services_count=health_report["metrics"]["healthy_services"],
        degraded_services_count=health_report["metrics"]["degraded_services"],
        unavailable_services_count=health_report["metrics"]["unavailable_services"],
        active_alerts_count=health_report["alerts"]["total_active"],
        critical_alerts_count=health_report["alerts"]["by_severity"]["critical"],
        services_health=health_report["services"]
    )


@router.get("/status")
async def get_system_status():
    """Get comprehensive system status"""
    return system_monitor.get_system_health_report()


@router.post("/test-alert")
async def create_test_alert(
    severity: str = Query("medium", description="Alert severity"),
    service: Optional[str] = Query(None, description="Service name")
):
    """Create a test alert (for testing purposes)"""
    try:
        alert_severity = AlertSeverity(severity.lower())
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid severity: {severity}"
        )
    
    from app.core.monitoring import AlertType
    
    alert = system_monitor.create_alert(
        alert_type=AlertType.SECURITY_INCIDENT,
        severity=alert_severity,
        title="Test Alert",
        description="This is a test alert created via API",
        service=service,
        metadata={"test": True, "created_via": "api"}
    )
    
    return {"message": "Test alert created", "alert_id": alert.id}


@router.get("/readiness")
async def readiness_check(db: AsyncSession = Depends(get_db)):
    """Kubernetes readiness probe"""
    try:
        # Check database
        await db.execute("SELECT 1")
        # Check Redis
        await redis_manager.redis.ping()
        return {"status": "ready"}
    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service not ready"
        )


@router.get("/liveness")
async def liveness_check():
    """Kubernetes liveness probe"""
    return {"status": "alive", "timestamp": datetime.utcnow()}


@router.get("/startup")
async def startup_check():
    """Kubernetes startup probe"""
    # Check if critical services are initialized
    try:
        # This could check if ML models are loaded, etc.
        return {"status": "started", "timestamp": datetime.utcnow()}
    except Exception as e:
        logger.error(f"Startup check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service not started"
        )


@router.get("/dependencies")
async def check_dependencies():
    """Check all system dependencies"""
    dependencies = {}
    overall_healthy = True
    
    # Check database
    try:
        from app.core.database import engine
        async with engine.begin() as conn:
            await conn.execute("SELECT 1")
        dependencies["database"] = {"status": "healthy", "type": "postgresql"}
    except Exception as e:
        dependencies["database"] = {"status": "unhealthy", "error": str(e), "type": "postgresql"}
        overall_healthy = False
    
    # Check Redis
    try:
        await redis_manager.redis.ping()
        dependencies["redis"] = {"status": "healthy", "type": "redis"}
    except Exception as e:
        dependencies["redis"] = {"status": "unhealthy", "error": str(e), "type": "redis"}
        overall_healthy = False
    
    # Check external services
    services_health = degradation_manager.get_all_services_health()
    for service_name, health in services_health.items():
        dependencies[f"external_{service_name}"] = {
            "status": health.status.value,
            "type": "external_api",
            "success_rate": health.success_rate,
            "fallback_available": health.fallback_available
        }
    
    return {
        "overall_status": "healthy" if overall_healthy else "unhealthy",
        "dependencies": dependencies,
        "timestamp": datetime.utcnow()
    }