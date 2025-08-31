"""
Demo script for comprehensive error handling and monitoring system
"""
import asyncio
import time
from datetime import datetime

from app.core.exceptions import (
    APIException, ExternalAPIException, MLModelError, DatabaseError,
    InsufficientDataError, ProcessingTimeoutError
)
from app.core.graceful_degradation import (
    degradation_manager, ServiceStatus, with_graceful_degradation
)
from app.core.monitoring import system_monitor, AlertType, AlertSeverity
from app.core.alerting import alerting_system


async def demo_custom_exceptions():
    """Demonstrate custom exception classes"""
    print("=== Custom Exception Classes Demo ===\n")
    
    # API Exception with recovery suggestions
    try:
        raise APIException(
            status_code=400,
            detail="Invalid user input provided",
            error_code="VALIDATION_ERROR",
            recovery_suggestions=[
                "Check that all required fields are filled",
                "Ensure email format is valid",
                "Verify password meets requirements"
            ],
            user_friendly_message="Please check your input and try again"
        )
    except APIException as e:
        print("API Exception:")
        print(f"  Status: {e.status_code}")
        print(f"  User Message: {e.user_friendly_message}")
        print(f"  Recovery Suggestions: {e.recovery_suggestions}")
        print(f"  Error Dict: {e.to_dict()}")
        print()
    
    # External API Exception
    try:
        raise ExternalAPIException(
            service="GitHub",
            detail="Rate limit exceeded",
            retry_after=300
        )
    except ExternalAPIException as e:
        print("External API Exception:")
        print(f"  Service: {e.details['service']}")
        print(f"  Retry After: {e.details['retry_after']} seconds")
        print(f"  Recovery Suggestions: {e.recovery_suggestions}")
        print()
    
    # ML Model Error with fallback
    try:
        raise MLModelError(
            model_name="Career Recommendation Engine",
            detail="Model inference timeout",
            fallback_available=True
        )
    except MLModelError as e:
        print("ML Model Error:")
        print(f"  Model: {e.details['model_name']}")
        print(f"  Fallback Available: {e.details['fallback_available']}")
        print(f"  User Message: {e.user_friendly_message}")
        print()
    
    # Insufficient Data Error
    try:
        raise InsufficientDataError(
            required_data=["resume", "skills", "experience"],
            missing_data=["resume", "experience"]
        )
    except InsufficientDataError as e:
        print("Insufficient Data Error:")
        print(f"  Missing Data: {e.details['missing_data']}")
        print(f"  Recovery Suggestions: {e.recovery_suggestions}")
        print()


async def demo_graceful_degradation():
    """Demonstrate graceful degradation system"""
    print("=== Graceful Degradation Demo ===\n")
    
    # Register services with fallbacks
    def github_fallback():
        return {
            "repositories": [],
            "languages": ["Python", "JavaScript"],
            "note": "GitHub data temporarily unavailable - using cached data"
        }
    
    def recommendation_fallback():
        return [
            {
                "title": "Software Developer",
                "match_score": 0.7,
                "note": "Basic recommendation - AI analysis temporarily unavailable"
            }
        ]
    
    degradation_manager.register_service("github", github_fallback)
    degradation_manager.register_service("recommendations", recommendation_fallback)
    
    print("Registered services:")
    for name, health in degradation_manager.get_all_services_health().items():
        print(f"  {name}: {health.status.value} (fallback: {health.fallback_available})")
    print()
    
    # Simulate service failure and recovery
    async def failing_github_service():
        raise Exception("GitHub API rate limit exceeded")
    
    async def working_github_service():
        return {
            "repositories": ["ai-career-app", "ml-models"],
            "languages": ["Python", "TypeScript", "Go"],
            "contributions": 150
        }
    
    # Test with failing service (should use fallback)
    print("Testing with failing GitHub service:")
    try:
        result = await degradation_manager.with_fallback(
            "github", failing_github_service
        )
        print(f"  Result: {result}")
    except Exception as e:
        print(f"  Error: {e}")
    
    # Check service health after failure
    health = degradation_manager.get_service_health("github")
    print(f"  Service health: {health.status.value} (errors: {health.error_count})")
    print()
    
    # Test with working service
    print("Testing with working GitHub service:")
    result = await degradation_manager.with_fallback(
        "github", working_github_service
    )
    print(f"  Result: {result}")
    
    # Check service health after success
    health = degradation_manager.get_service_health("github")
    print(f"  Service health: {health.status.value} (success rate: {health.success_rate:.2%})")
    print()


async def demo_system_monitoring():
    """Demonstrate system monitoring"""
    print("=== System Monitoring Demo ===\n")
    
    # Create various types of alerts
    alerts = [
        {
            "type": AlertType.SERVICE_DOWN,
            "severity": AlertSeverity.CRITICAL,
            "title": "Database Connection Lost",
            "description": "PostgreSQL database is unreachable",
            "service": "database"
        },
        {
            "type": AlertType.HIGH_ERROR_RATE,
            "severity": AlertSeverity.HIGH,
            "title": "High Error Rate Detected",
            "description": "Error rate exceeded 10% in the last 5 minutes",
            "service": "api"
        },
        {
            "type": AlertType.SLOW_RESPONSE,
            "severity": AlertSeverity.MEDIUM,
            "title": "Slow Response Times",
            "description": "95th percentile response time exceeded 3 seconds",
            "service": "recommendations"
        },
        {
            "type": AlertType.RESOURCE_EXHAUSTION,
            "severity": AlertSeverity.LOW,
            "title": "High Memory Usage",
            "description": "Memory usage is at 85% of available capacity",
            "service": "ml_models"
        }
    ]
    
    print("Creating test alerts:")
    created_alerts = []
    for alert_data in alerts:
        alert = system_monitor.create_alert(**alert_data)
        created_alerts.append(alert)
        print(f"  Created: [{alert.severity.value.upper()}] {alert.title}")
    
    print()
    
    # Show alert summary
    summary = system_monitor.get_alert_summary()
    print("Alert Summary:")
    print(f"  Total Active: {summary['total_active']}")
    print(f"  By Severity: {summary['by_severity']}")
    print(f"  By Service: {dict(summary['by_service'])}")
    print()
    
    # Test metric recording and threshold checking
    print("Testing metric thresholds:")
    
    # Record metrics that exceed thresholds
    test_metrics = [
        ("error_rate", 8.5),  # Above 5% threshold
        ("response_time_p95", 6.2),  # Above 5s threshold
        ("cache_hit_rate", 65.0),  # Below 70% threshold
    ]
    
    for metric_name, value in test_metrics:
        system_monitor.record_metric(metric_name, value)
        print(f"  Recorded {metric_name}: {value}")
    
    print()
    
    # Show updated alert summary
    updated_summary = system_monitor.get_alert_summary()
    print("Updated Alert Summary:")
    print(f"  Total Active: {updated_summary['total_active']}")
    print()
    
    # Resolve some alerts
    print("Resolving alerts:")
    for alert in created_alerts[:2]:
        system_monitor.resolve_alert(alert.id)
        print(f"  Resolved: {alert.title}")
    
    print()
    
    # Show final summary
    final_summary = system_monitor.get_alert_summary()
    print("Final Alert Summary:")
    print(f"  Total Active: {final_summary['total_active']}")
    print()


async def demo_alerting_system():
    """Demonstrate alerting system"""
    print("=== Alerting System Demo ===\n")
    
    # Show notification configurations
    print("Notification Configurations:")
    for config in alerting_system.notification_configs:
        print(f"  {config.channel.value}: {'enabled' if config.enabled else 'disabled'}")
        print(f"    Severity Filter: {[s.value for s in config.severity_filter]}")
    print()
    
    # Create test alert
    from app.core.monitoring import Alert
    
    test_alert = Alert(
        id="demo_alert_" + str(int(time.time())),
        type=AlertType.SERVICE_DOWN,
        severity=AlertSeverity.HIGH,
        title="Demo Service Failure",
        description="This is a demonstration of the alerting system",
        timestamp=datetime.utcnow(),
        service="demo_service",
        recovery_suggestions=[
            "This is a demo alert",
            "No action required",
            "System will recover automatically"
        ]
    )
    
    print(f"Created test alert: {test_alert.title}")
    print(f"  ID: {test_alert.id}")
    print(f"  Severity: {test_alert.severity.value}")
    print(f"  Recovery Suggestions: {test_alert.recovery_suggestions}")
    print()
    
    # Test email body generation
    email_body = alerting_system.create_email_body(test_alert)
    print("Generated email notification body:")
    print("  (HTML content generated - would be sent via email)")
    print(f"  Length: {len(email_body)} characters")
    print()
    
    # Show rate limiting
    print("Rate Limiting Demo:")
    print(f"  Max alerts per window: {alerting_system.max_alerts_per_window}")
    print(f"  Window duration: {alerting_system.rate_limit_window} seconds")
    print(f"  Current alert count: {len(alerting_system.alert_history)}")
    print()


async def demo_health_report():
    """Demonstrate system health reporting"""
    print("=== System Health Report Demo ===\n")
    
    # Generate comprehensive health report
    health_report = system_monitor.get_system_health_report()
    
    print("System Health Report:")
    print(f"  Overall Health: {health_report['overall_health_percentage']:.1f}%")
    print(f"  Timestamp: {health_report['timestamp']}")
    print()
    
    print("Services:")
    for service_name, service_data in health_report['services'].items():
        print(f"  {service_name}:")
        print(f"    Status: {service_data['status']}")
        print(f"    Success Rate: {service_data['success_rate']:.2%}")
        print(f"    Error Count: {service_data['error_count']}")
        print(f"    Fallback Available: {service_data['fallback_available']}")
    print()
    
    print("Metrics:")
    metrics = health_report['metrics']
    print(f"  Total Services: {metrics['total_services']}")
    print(f"  Healthy Services: {metrics['healthy_services']}")
    print(f"  Degraded Services: {metrics['degraded_services']}")
    print(f"  Unavailable Services: {metrics['unavailable_services']}")
    print()
    
    print("Alerts:")
    alerts = health_report['alerts']
    print(f"  Total Active: {alerts['total_active']}")
    print(f"  By Severity: {alerts['by_severity']}")
    print()


async def main():
    """Run all demos"""
    print("AI Career Recommender - Error Handling & Monitoring System Demo")
    print("=" * 70)
    print()
    
    # Start monitoring systems
    await system_monitor.start_monitoring()
    await degradation_manager.start_health_monitoring()
    
    try:
        await demo_custom_exceptions()
        await asyncio.sleep(1)
        
        await demo_graceful_degradation()
        await asyncio.sleep(1)
        
        await demo_system_monitoring()
        await asyncio.sleep(1)
        
        await demo_alerting_system()
        await asyncio.sleep(1)
        
        await demo_health_report()
        
    finally:
        # Stop monitoring systems
        await system_monitor.stop_monitoring()
        await degradation_manager.stop_health_monitoring()
    
    print("Demo completed successfully!")


if __name__ == "__main__":
    asyncio.run(main())