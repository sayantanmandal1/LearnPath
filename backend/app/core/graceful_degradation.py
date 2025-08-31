"""
Graceful degradation system for external API failures
"""
import asyncio
import time
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union
from functools import wraps
from dataclasses import dataclass, field

import structlog
from app.core.exceptions import ExternalAPIException, SystemHealthError

logger = structlog.get_logger()

T = TypeVar('T')


class ServiceStatus(Enum):
    """Service status enumeration"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"
    MAINTENANCE = "maintenance"


@dataclass
class ServiceHealth:
    """Service health information"""
    name: str
    status: ServiceStatus
    last_check: float
    error_count: int = 0
    success_count: int = 0
    response_time: Optional[float] = None
    last_error: Optional[str] = None
    fallback_available: bool = False
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate"""
        total = self.success_count + self.error_count
        return self.success_count / total if total > 0 else 0.0
    
    @property
    def is_healthy(self) -> bool:
        """Check if service is healthy"""
        return self.status == ServiceStatus.HEALTHY


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration"""
    failure_threshold: int = 5
    recovery_timeout: int = 60
    success_threshold: int = 3
    timeout: int = 30


class CircuitBreakerState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class CircuitBreaker:
    """Circuit breaker for external services"""
    name: str
    config: CircuitBreakerConfig
    state: CircuitBreakerState = CircuitBreakerState.CLOSED
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: Optional[float] = None
    next_attempt_time: Optional[float] = None
    
    def can_execute(self) -> bool:
        """Check if request can be executed"""
        now = time.time()
        
        if self.state == CircuitBreakerState.CLOSED:
            return True
        elif self.state == CircuitBreakerState.OPEN:
            if self.next_attempt_time and now >= self.next_attempt_time:
                self.state = CircuitBreakerState.HALF_OPEN
                self.success_count = 0
                return True
            return False
        elif self.state == CircuitBreakerState.HALF_OPEN:
            return True
        
        return False
    
    def record_success(self):
        """Record successful execution"""
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                self.state = CircuitBreakerState.CLOSED
                self.failure_count = 0
        elif self.state == CircuitBreakerState.CLOSED:
            self.failure_count = max(0, self.failure_count - 1)
    
    def record_failure(self):
        """Record failed execution"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.state = CircuitBreakerState.OPEN
            self.next_attempt_time = time.time() + self.config.recovery_timeout
        elif (self.state == CircuitBreakerState.CLOSED and 
              self.failure_count >= self.config.failure_threshold):
            self.state = CircuitBreakerState.OPEN
            self.next_attempt_time = time.time() + self.config.recovery_timeout


class GracefulDegradationManager:
    """Manager for graceful degradation of external services"""
    
    def __init__(self):
        self.services: Dict[str, ServiceHealth] = {}
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.fallback_handlers: Dict[str, Callable] = {}
        self.health_check_interval = 60  # seconds
        self._health_check_task: Optional[asyncio.Task] = None
    
    def register_service(
        self,
        name: str,
        fallback_handler: Optional[Callable] = None,
        circuit_breaker_config: Optional[CircuitBreakerConfig] = None
    ):
        """Register a service for monitoring"""
        self.services[name] = ServiceHealth(
            name=name,
            status=ServiceStatus.HEALTHY,
            last_check=time.time(),
            fallback_available=fallback_handler is not None
        )
        
        if fallback_handler:
            self.fallback_handlers[name] = fallback_handler
        
        self.circuit_breakers[name] = CircuitBreaker(
            name=name,
            config=circuit_breaker_config or CircuitBreakerConfig()
        )
        
        logger.info(f"Registered service for monitoring: {name}")
    
    def get_service_health(self, name: str) -> Optional[ServiceHealth]:
        """Get health status of a service"""
        return self.services.get(name)
    
    def get_all_services_health(self) -> Dict[str, ServiceHealth]:
        """Get health status of all services"""
        return self.services.copy()
    
    def update_service_health(
        self,
        name: str,
        status: ServiceStatus,
        error: Optional[str] = None,
        response_time: Optional[float] = None
    ):
        """Update service health status"""
        if name not in self.services:
            return
        
        service = self.services[name]
        service.status = status
        service.last_check = time.time()
        service.response_time = response_time
        
        if error:
            service.error_count += 1
            service.last_error = error
        else:
            service.success_count += 1
        
        logger.info(
            f"Updated service health: {name}",
            status=status.value,
            error_count=service.error_count,
            success_count=service.success_count,
            success_rate=service.success_rate
        )
    
    async def with_fallback(
        self,
        service_name: str,
        primary_func: Callable[..., T],
        *args,
        fallback_func: Optional[Callable[..., T]] = None,
        **kwargs
    ) -> T:
        """Execute function with fallback on failure"""
        circuit_breaker = self.circuit_breakers.get(service_name)
        
        # Check circuit breaker
        if circuit_breaker and not circuit_breaker.can_execute():
            logger.warning(f"Circuit breaker open for {service_name}, using fallback")
            return await self._execute_fallback(service_name, fallback_func, *args, **kwargs)
        
        start_time = time.time()
        
        try:
            # Execute primary function
            if asyncio.iscoroutinefunction(primary_func):
                result = await primary_func(*args, **kwargs)
            else:
                result = primary_func(*args, **kwargs)
            
            # Record success
            response_time = time.time() - start_time
            self.update_service_health(service_name, ServiceStatus.HEALTHY, response_time=response_time)
            
            if circuit_breaker:
                circuit_breaker.record_success()
            
            return result
        
        except Exception as e:
            # Record failure
            response_time = time.time() - start_time
            error_msg = str(e)
            
            self.update_service_health(
                service_name,
                ServiceStatus.UNAVAILABLE,
                error=error_msg,
                response_time=response_time
            )
            
            if circuit_breaker:
                circuit_breaker.record_failure()
            
            logger.error(
                f"Service {service_name} failed, attempting fallback",
                error=error_msg,
                response_time=response_time
            )
            
            # Try fallback
            try:
                return await self._execute_fallback(service_name, fallback_func, *args, **kwargs)
            except Exception as fallback_error:
                logger.error(f"Fallback also failed for {service_name}: {fallback_error}")
                raise ExternalAPIException(
                    service=service_name,
                    detail=f"Primary service failed: {error_msg}. Fallback failed: {fallback_error}",
                    original_error=e
                )
    
    async def _execute_fallback(
        self,
        service_name: str,
        fallback_func: Optional[Callable],
        *args,
        **kwargs
    ) -> Any:
        """Execute fallback function"""
        # Use provided fallback or registered fallback
        fallback = fallback_func or self.fallback_handlers.get(service_name)
        
        if not fallback:
            raise ExternalAPIException(
                service=service_name,
                detail="Service unavailable and no fallback configured"
            )
        
        logger.info(f"Executing fallback for {service_name}")
        
        if asyncio.iscoroutinefunction(fallback):
            return await fallback(*args, **kwargs)
        else:
            return fallback(*args, **kwargs)
    
    def with_circuit_breaker(self, service_name: str):
        """Decorator for circuit breaker pattern"""
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            async def wrapper(*args, **kwargs):
                return await self.with_fallback(service_name, func, *args, **kwargs)
            return wrapper
        return decorator
    
    async def start_health_monitoring(self):
        """Start background health monitoring"""
        if self._health_check_task:
            return
        
        self._health_check_task = asyncio.create_task(self._health_check_loop())
        logger.info("Started health monitoring")
    
    async def stop_health_monitoring(self):
        """Stop background health monitoring"""
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
            self._health_check_task = None
            logger.info("Stopped health monitoring")
    
    async def _health_check_loop(self):
        """Background health check loop"""
        while True:
            try:
                await asyncio.sleep(self.health_check_interval)
                await self._perform_health_checks()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check loop error: {e}")
    
    async def _perform_health_checks(self):
        """Perform health checks on all services"""
        for service_name in self.services:
            try:
                # Simple health check - could be extended with actual service pings
                service = self.services[service_name]
                
                # Check if service hasn't been used recently
                time_since_check = time.time() - service.last_check
                if time_since_check > 300:  # 5 minutes
                    # Mark as potentially stale
                    if service.status == ServiceStatus.HEALTHY:
                        service.status = ServiceStatus.DEGRADED
                        logger.warning(f"Service {service_name} marked as degraded due to inactivity")
                
            except Exception as e:
                logger.error(f"Health check failed for {service_name}: {e}")


# Global instance
degradation_manager = GracefulDegradationManager()


def with_graceful_degradation(service_name: str, fallback_func: Optional[Callable] = None):
    """Decorator for graceful degradation"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await degradation_manager.with_fallback(
                service_name, func, *args, fallback_func=fallback_func, **kwargs
            )
        return wrapper
    return decorator


# Fallback implementations
class FallbackImplementations:
    """Common fallback implementations"""
    
    @staticmethod
    def empty_profile_data():
        """Fallback for profile data extraction"""
        return {
            "skills": [],
            "experience": [],
            "education": [],
            "projects": [],
            "note": "Profile data temporarily unavailable"
        }
    
    @staticmethod
    def basic_recommendations(user_profile: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Fallback for recommendations"""
        return [
            {
                "title": "Software Developer",
                "match_score": 0.7,
                "note": "Basic recommendation - full AI analysis temporarily unavailable"
            },
            {
                "title": "Data Analyst",
                "match_score": 0.6,
                "note": "Basic recommendation - full AI analysis temporarily unavailable"
            }
        ]
    
    @staticmethod
    def cached_job_data() -> List[Dict[str, Any]]:
        """Fallback for job market data"""
        return [
            {
                "title": "Software Engineer",
                "company": "Various Companies",
                "location": "Remote",
                "note": "Cached data - live job data temporarily unavailable"
            }
        ]