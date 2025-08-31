"""
Tests for comprehensive error handling and monitoring system
"""
import asyncio
import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock

from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from sqlalchemy.exc import SQLAlchemyError
from redis.exceptions import RedisError

from app.core.exceptions import (
    APIException, ExternalAPIException, MLModelError, DatabaseError,
    InsufficientDataError, ProcessingTimeoutError
)
from app.core.graceful_degradation import (
    GracefulDegradationManager, ServiceStatus, CircuitBreaker, CircuitBreakerConfig
)
from app.core.monitoring import SystemMonitor, Alert, AlertType, AlertSeverity
from app.core.alerting import AlertingSystem, NotificationChannel, NotificationConfig
from app.middleware.error_handler import ErrorHandlerMiddleware


class TestCustomExceptions:
    """Test custom exception classes"""
    
    def test_api_exception_creation(self):
        """Test APIException creation with all parameters"""
        exc = APIException(
            status_code=400,
            detail="Test error",
            error_code="TEST_ERROR",
            recovery_suggestions=["Try again", "Contact support"],
            user_friendly_message="Something went wrong"
        )
        
        assert exc.status_code == 400
        assert exc.detail == "Test error"
        assert exc.error_code == "TEST_ERROR"
        assert len(exc.recovery_suggestions) == 2
        assert exc.user_friendly_message == "Something went wrong"
        assert isinstance(exc.timestamp, datetime)
    
    def test_api_exception_to_dict(self):
        """Test APIException to_dict method"""
        exc = APIException(
            status_code=500,
            detail="Internal error",
            error_code="INTERNAL_ERROR",
            recovery_suggestions=["Retry"],
            user_friendly_message="Please try again"
        )
        
        result = exc.to_dict()
        
        assert result["error"]["code"] == "INTERNAL_ERROR"
        assert result["error"]["message"] == "Please try again"
        assert result["error"]["technical_detail"] == "Internal error"
        assert result["error"]["recovery_suggestions"] == ["Retry"]
        assert "timestamp" in result["error"]
    
    def test_external_api_exception(self):
        """Test ExternalAPIException with service-specific details"""
        exc = ExternalAPIException(
            service="GitHub",
            detail="Rate limit exceeded",
            retry_after=60
        )
        
        assert exc.status_code == 502
        assert "GitHub" in exc.detail
        assert exc.details["service"] == "GitHub"
        assert exc.details["retry_after"] == 60
        assert len(exc.recovery_suggestions) > 0
    
    def test_ml_model_error_with_fallback(self):
        """Test MLModelError with fallback availability"""
        exc = MLModelError(
            model_name="Recommendation Engine",
            detail="Model loading failed",
            fallback_available=True
        )
        
        assert exc.status_code == 503
        assert exc.details["model_name"] == "Recommendation Engine"
        assert exc.details["fallback_available"] is True
        assert "simplified version" in " ".join(exc.recovery_suggestions)
    
    def test_insufficient_data_error(self):
        """Test InsufficientDataError with specific missing data"""
        required = ["resume", "skills", "experience"]
        missing = ["resume", "experience"]
        
        exc = InsufficientDataError(required, missing)
        
        assert exc.status_code == 400
        assert exc.error_code == "INSUFFICIENT_DATA"
        assert exc.details["required_data"] == required
        assert exc.details["missing_data"] == missing
        assert any("resume" in suggestion for suggestion in exc.recovery_suggestions)


class TestGracefulDegradation:
    """Test graceful degradation system"""
    
    @pytest.fixture
    def degradation_manager(self):
        """Create degradation manager for testing"""
        return GracefulDegradationManager()
    
    def test_service_registration(self, degradation_manager):
        """Test service registration"""
        def fallback():
            return "fallback_result"
        
        degradation_manager.register_service("test_service", fallback)
        
        health = degradation_manager.get_service_health("test_service")
        assert health is not None
        assert health.name == "test_service"
        assert health.status == ServiceStatus.HEALTHY
        assert health.fallback_available is True
    
    def test_circuit_breaker_states(self):
        """Test circuit breaker state transitions"""
        config = CircuitBreakerConfig(failure_threshold=2, recovery_timeout=60)
        breaker = CircuitBreaker("test", config)
        
        # Initially closed
        assert breaker.state.value == "closed"
        assert breaker.can_execute() is True
        
        # Record failures
        breaker.record_failure()
        breaker.record_failure()
        
        # Should open after threshold
        assert breaker.state.value == "open"
        assert breaker.can_execute() is False
    
    @pytest.mark.asyncio
    async def test_with_fallback_success(self, degradation_manager):
        """Test successful execution with fallback"""
        async def primary_func():
            return "primary_result"
        
        def fallback_func():
            return "fallback_result"
        
        degradation_manager.register_service("test_service", fallback_func)
        
        result = await degradation_manager.with_fallback(
            "test_service", primary_func
        )
        
        assert result == "primary_result"
        
        health = degradation_manager.get_service_health("test_service")
        assert health.status == ServiceStatus.HEALTHY
        assert health.success_count == 1
    
    @pytest.mark.asyncio
    async def test_with_fallback_failure(self, degradation_manager):
        """Test fallback execution on primary failure"""
        async def primary_func():
            raise Exception("Primary failed")
        
        def fallback_func():
            return "fallback_result"
        
        degradation_manager.register_service("test_service", fallback_func)
        
        result = await degradation_manager.with_fallback(
            "test_service", primary_func
        )
        
        assert result == "fallback_result"
        
        health = degradation_manager.get_service_health("test_service")
        assert health.status == ServiceStatus.UNAVAILABLE
        assert health.error_count == 1


class TestSystemMonitoring:
    """Test system monitoring"""
    
    @pytest.fixture
    def monitor(self):
        """Create system monitor for testing"""
        from prometheus_client import CollectorRegistry
        # Use a separate registry for testing to avoid conflicts
        monitor = SystemMonitor()
        monitor.setup_metrics = lambda: None  # Skip metrics setup in tests
        return monitor
    
    def test_alert_creation(self, monitor):
        """Test alert creation and handling"""
        alert = monitor.create_alert(
            alert_type=AlertType.SERVICE_DOWN,
            severity=AlertSeverity.HIGH,
            title="Test Alert",
            description="Test description",
            service="test_service"
        )
        
        assert alert.type == AlertType.SERVICE_DOWN
        assert alert.severity == AlertSeverity.HIGH
        assert alert.title == "Test Alert"
        assert alert.service == "test_service"
        assert not alert.resolved
    
    def test_alert_resolution(self, monitor):
        """Test alert resolution"""
        alert = monitor.create_alert(
            alert_type=AlertType.HIGH_ERROR_RATE,
            severity=AlertSeverity.MEDIUM,
            title="High Error Rate",
            description="Error rate exceeded threshold"
        )
        
        # Resolve alert
        monitor.resolve_alert(alert.id)
        
        # Check resolution
        resolved_alert = next(a for a in monitor.alerts if a.id == alert.id)
        assert resolved_alert.resolved is True
        assert resolved_alert.resolved_at is not None
    
    def test_metric_threshold_checking(self, monitor):
        """Test metric threshold checking"""
        # Record metric that exceeds threshold
        monitor.record_metric("error_rate", 10.0)  # Above 5% threshold
        
        # Check if alert was created
        active_alerts = monitor.get_active_alerts()
        threshold_alerts = [
            alert for alert in active_alerts
            if alert.metadata.get("metric_name") == "error_rate"
        ]
        
        assert len(threshold_alerts) > 0
        assert threshold_alerts[0].severity == AlertSeverity.HIGH
    
    def test_alert_summary(self, monitor):
        """Test alert summary generation"""
        # Create alerts of different severities
        monitor.create_alert(AlertType.SERVICE_DOWN, AlertSeverity.CRITICAL, "Critical", "Desc")
        monitor.create_alert(AlertType.HIGH_ERROR_RATE, AlertSeverity.HIGH, "High", "Desc")
        monitor.create_alert(AlertType.SLOW_RESPONSE, AlertSeverity.MEDIUM, "Medium", "Desc")
        
        summary = monitor.get_alert_summary()
        
        assert summary["total_active"] == 3
        assert summary["by_severity"]["critical"] == 1
        assert summary["by_severity"]["high"] == 1
        assert summary["by_severity"]["medium"] == 1


class TestAlertingSystem:
    """Test alerting system"""
    
    @pytest.fixture
    def alerting_system(self):
        """Create alerting system for testing"""
        return AlertingSystem()
    
    def test_notification_config_addition(self, alerting_system):
        """Test adding notification configurations"""
        config = NotificationConfig(
            channel=NotificationChannel.EMAIL,
            enabled=True,
            config={"to_emails": ["test@example.com"]},
            severity_filter={AlertSeverity.HIGH}
        )
        
        alerting_system.add_notification_config(config)
        
        assert len(alerting_system.notification_configs) > 0
        assert config in alerting_system.notification_configs
    
    def test_rate_limiting(self, alerting_system):
        """Test alert rate limiting"""
        # Create many alerts quickly
        alert = Alert(
            id="test_alert",
            type=AlertType.HIGH_ERROR_RATE,
            severity=AlertSeverity.HIGH,
            title="Test Alert",
            description="Test",
            timestamp=datetime.utcnow()
        )
        
        # Fill up rate limit window
        for i in range(alerting_system.max_alerts_per_window):
            alerting_system.alert_history[f"alert_{i}"] = datetime.utcnow()
        
        # This alert should be rate limited
        assert alerting_system.is_rate_limited(alert) is True
    
    @pytest.mark.asyncio
    async def test_email_notification_creation(self, alerting_system):
        """Test email notification body creation"""
        alert = Alert(
            id="test_alert",
            type=AlertType.SERVICE_DOWN,
            severity=AlertSeverity.CRITICAL,
            title="Service Down",
            description="Test service is down",
            timestamp=datetime.utcnow(),
            service="test_service"
        )
        alert.recovery_suggestions = ["Restart service", "Check logs"]
        
        email_body = alerting_system.create_email_body(alert)
        
        assert "Service Down" in email_body
        assert "CRITICAL" in email_body
        assert "test_service" in email_body
        assert "Restart service" in email_body
        assert "Check logs" in email_body


class TestErrorHandlerMiddleware:
    """Test error handler middleware"""
    
    @pytest.fixture
    def app(self):
        """Create test FastAPI app with error handler"""
        app = FastAPI()
        app.add_middleware(ErrorHandlerMiddleware)
        
        @app.get("/test-api-exception")
        async def test_api_exception():
            raise APIException(
                status_code=400,
                detail="Test API error",
                error_code="TEST_ERROR",
                recovery_suggestions=["Try again"]
            )
        
        @app.get("/test-database-error")
        async def test_database_error():
            raise SQLAlchemyError("Database connection failed")
        
        @app.get("/test-redis-error")
        async def test_redis_error():
            raise RedisError("Redis connection failed")
        
        @app.get("/test-unexpected-error")
        async def test_unexpected_error():
            raise ValueError("Unexpected error")
        
        return app
    
    def test_api_exception_handling(self, app):
        """Test API exception handling"""
        client = TestClient(app)
        response = client.get("/test-api-exception")
        
        assert response.status_code == 400
        data = response.json()
        assert data["error"]["code"] == "TEST_ERROR"
        assert "recovery_suggestions" in data["error"]
    
    def test_database_error_handling(self, app):
        """Test database error handling"""
        client = TestClient(app)
        response = client.get("/test-database-error")
        
        assert response.status_code == 503
        data = response.json()
        assert data["error"]["code"] == "DATABASE_ERROR"
        assert "database connectivity issues" in data["error"]["message"]
        assert "error_id" in data["error"]
    
    def test_redis_error_handling(self, app):
        """Test Redis error handling"""
        client = TestClient(app)
        response = client.get("/test-redis-error")
        
        assert response.status_code == 503
        data = response.json()
        assert data["error"]["code"] == "CACHE_ERROR"
        assert "cache" in data["error"]["message"].lower() or "redis" in data["error"]["message"].lower()
    
    def test_unexpected_error_handling(self, app):
        """Test unexpected error handling"""
        client = TestClient(app)
        response = client.get("/test-unexpected-error")
        
        assert response.status_code in [400, 500]  # ValueError can be classified as validation error
        data = response.json()
        assert data["error"]["code"] == "INTERNAL_SERVER_ERROR"
        assert "error_id" in data["error"]
        assert "recovery_suggestions" in data["error"]


@pytest.mark.asyncio
async def test_health_monitoring_integration():
    """Test integration between monitoring components"""
    # Create components
    degradation_manager = GracefulDegradationManager()
    monitor = SystemMonitor()
    monitor.setup_metrics = lambda: None  # Skip metrics setup in tests
    
    # Register service
    degradation_manager.register_service("test_service")
    
    # Simulate service failure
    degradation_manager.update_service_health(
        "test_service",
        ServiceStatus.UNAVAILABLE,
        error="Service connection failed"
    )
    
    # Update monitoring metrics
    monitor.update_service_health_metrics()
    
    # Check if alert was created
    active_alerts = monitor.get_active_alerts()
    service_down_alerts = [
        alert for alert in active_alerts
        if alert.type == AlertType.SERVICE_DOWN and alert.service == "test_service"
    ]
    
    assert len(service_down_alerts) > 0
    assert service_down_alerts[0].severity == AlertSeverity.CRITICAL


if __name__ == "__main__":
    pytest.main([__file__])