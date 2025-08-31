"""
Health check and monitoring schemas
"""
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class HealthCheckResponse(BaseModel):
    """Basic health check response"""
    status: str = Field(..., description="Health status")
    timestamp: datetime = Field(..., description="Check timestamp")
    version: str = Field(..., description="Application version")


class SystemStatusResponse(BaseModel):
    """Detailed system status response"""
    status: str = Field(..., description="Overall system status")
    timestamp: datetime = Field(..., description="Check timestamp")
    response_time: float = Field(..., description="Total response time in seconds")
    checks: Dict[str, Dict[str, Any]] = Field(..., description="Individual component checks")
    version: str = Field(..., description="Application version")


class ServiceHealthResponse(BaseModel):
    """Service health status response"""
    name: str = Field(..., description="Service name")
    status: str = Field(..., description="Service status")
    success_rate: float = Field(..., description="Success rate percentage")
    error_count: int = Field(..., description="Total error count")
    success_count: int = Field(..., description="Total success count")
    response_time: Optional[float] = Field(None, description="Last response time in seconds")
    last_check: datetime = Field(..., description="Last health check timestamp")
    last_error: Optional[str] = Field(None, description="Last error message")
    fallback_available: bool = Field(..., description="Whether fallback is available")


class AlertResponse(BaseModel):
    """Alert response schema"""
    id: str = Field(..., description="Alert ID")
    type: str = Field(..., description="Alert type")
    severity: str = Field(..., description="Alert severity")
    title: str = Field(..., description="Alert title")
    description: str = Field(..., description="Alert description")
    timestamp: datetime = Field(..., description="Alert timestamp")
    service: Optional[str] = Field(None, description="Related service")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    resolved: bool = Field(..., description="Whether alert is resolved")
    resolved_at: Optional[datetime] = Field(None, description="Resolution timestamp")


class SystemMetricsResponse(BaseModel):
    """System metrics response"""
    timestamp: datetime = Field(..., description="Metrics timestamp")
    overall_health_percentage: float = Field(..., description="Overall system health percentage")
    services_count: int = Field(..., description="Total number of services")
    healthy_services_count: int = Field(..., description="Number of healthy services")
    degraded_services_count: int = Field(..., description="Number of degraded services")
    unavailable_services_count: int = Field(..., description="Number of unavailable services")
    active_alerts_count: int = Field(..., description="Number of active alerts")
    critical_alerts_count: int = Field(..., description="Number of critical alerts")
    services_health: Dict[str, Dict[str, Any]] = Field(..., description="Detailed services health")


class AlertSummaryResponse(BaseModel):
    """Alert summary response"""
    total_active: int = Field(..., description="Total active alerts")
    by_severity: Dict[str, int] = Field(..., description="Alerts by severity")
    by_service: Dict[str, int] = Field(..., description="Alerts by service")


class DependencyCheckResponse(BaseModel):
    """Dependency check response"""
    overall_status: str = Field(..., description="Overall dependencies status")
    dependencies: Dict[str, Dict[str, Any]] = Field(..., description="Individual dependency status")
    timestamp: datetime = Field(..., description="Check timestamp")