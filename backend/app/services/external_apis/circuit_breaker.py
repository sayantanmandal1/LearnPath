"""Circuit breaker pattern for external API calls."""

import logging
import asyncio
from typing import Dict, Optional, Callable, Any
from datetime import datetime, timedelta
from enum import Enum
from pydantic import BaseModel


logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Circuit is open, calls are blocked
    HALF_OPEN = "half_open"  # Testing if service is back


class CircuitBreakerConfig(BaseModel):
    """Configuration for circuit breaker."""
    failure_threshold: int = 5  # Number of failures before opening
    recovery_timeout: int = 60  # Seconds before trying half-open
    success_threshold: int = 3  # Successes needed to close from half-open
    timeout: float = 30.0  # Request timeout in seconds


class CircuitBreakerStats(BaseModel):
    """Statistics for circuit breaker."""
    state: CircuitState
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: Optional[datetime] = None
    last_success_time: Optional[datetime] = None
    total_requests: int = 0
    total_failures: int = 0
    total_successes: int = 0


class CircuitBreakerError(Exception):
    """Exception raised when circuit breaker is open."""
    pass


class CircuitBreaker:
    """Circuit breaker implementation for external API calls."""
    
    def __init__(self, name: str, config: Optional[CircuitBreakerConfig] = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.stats = CircuitBreakerStats(state=CircuitState.CLOSED)
        self._lock = asyncio.Lock()
    
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection."""
        async with self._lock:
            # Check if circuit should be opened
            if self._should_open_circuit():
                self.stats.state = CircuitState.OPEN
                self.stats.last_failure_time = datetime.utcnow()
                logger.warning(f"Circuit breaker {self.name} opened due to failures")
            
            # Check if circuit should transition to half-open
            elif self._should_attempt_reset():
                self.stats.state = CircuitState.HALF_OPEN
                self.stats.success_count = 0
                logger.info(f"Circuit breaker {self.name} transitioning to half-open")
        
        # Block calls if circuit is open
        if self.stats.state == CircuitState.OPEN:
            raise CircuitBreakerError(f"Circuit breaker {self.name} is open")
        
        # Execute the function
        self.stats.total_requests += 1
        
        try:
            # Apply timeout
            result = await asyncio.wait_for(
                func(*args, **kwargs),
                timeout=self.config.timeout
            )
            
            # Record success
            await self._record_success()
            return result
            
        except asyncio.TimeoutError:
            await self._record_failure()
            raise CircuitBreakerError(f"Request to {self.name} timed out after {self.config.timeout}s")
        
        except Exception as e:
            await self._record_failure()
            raise e
    
    async def _record_success(self):
        """Record a successful call."""
        async with self._lock:
            self.stats.success_count += 1
            self.stats.total_successes += 1
            self.stats.last_success_time = datetime.utcnow()
            
            # Reset failure count on success
            if self.stats.state == CircuitState.CLOSED:
                self.stats.failure_count = 0
            
            # Close circuit if enough successes in half-open state
            elif self.stats.state == CircuitState.HALF_OPEN:
                if self.stats.success_count >= self.config.success_threshold:
                    self.stats.state = CircuitState.CLOSED
                    self.stats.failure_count = 0
                    logger.info(f"Circuit breaker {self.name} closed after successful recovery")
    
    async def _record_failure(self):
        """Record a failed call."""
        async with self._lock:
            self.stats.failure_count += 1
            self.stats.total_failures += 1
            self.stats.last_failure_time = datetime.utcnow()
            
            # Reset success count on failure
            self.stats.success_count = 0
    
    def _should_open_circuit(self) -> bool:
        """Check if circuit should be opened."""
        return (
            self.stats.state == CircuitState.CLOSED and
            self.stats.failure_count >= self.config.failure_threshold
        )
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit should attempt to reset to half-open."""
        if self.stats.state != CircuitState.OPEN:
            return False
        
        if not self.stats.last_failure_time:
            return False
        
        time_since_failure = datetime.utcnow() - self.stats.last_failure_time
        return time_since_failure.total_seconds() >= self.config.recovery_timeout
    
    def get_stats(self) -> CircuitBreakerStats:
        """Get current circuit breaker statistics."""
        return self.stats.copy()
    
    async def reset(self):
        """Manually reset the circuit breaker."""
        async with self._lock:
            self.stats.state = CircuitState.CLOSED
            self.stats.failure_count = 0
            self.stats.success_count = 0
            logger.info(f"Circuit breaker {self.name} manually reset")


class CircuitBreakerManager:
    """Manager for multiple circuit breakers."""
    
    def __init__(self):
        self._breakers: Dict[str, CircuitBreaker] = {}
        self._configs: Dict[str, CircuitBreakerConfig] = {}
    
    def get_breaker(self, name: str, config: Optional[CircuitBreakerConfig] = None) -> CircuitBreaker:
        """Get or create a circuit breaker."""
        if name not in self._breakers:
            breaker_config = config or self._configs.get(name, CircuitBreakerConfig())
            self._breakers[name] = CircuitBreaker(name, breaker_config)
            logger.info(f"Created circuit breaker for {name}")
        
        return self._breakers[name]
    
    def configure_breaker(self, name: str, config: CircuitBreakerConfig):
        """Configure a circuit breaker."""
        self._configs[name] = config
        if name in self._breakers:
            self._breakers[name].config = config
    
    def get_all_stats(self) -> Dict[str, CircuitBreakerStats]:
        """Get statistics for all circuit breakers."""
        return {name: breaker.get_stats() for name, breaker in self._breakers.items()}
    
    async def reset_all(self):
        """Reset all circuit breakers."""
        for breaker in self._breakers.values():
            await breaker.reset()
        logger.info("All circuit breakers reset")
    
    async def reset_breaker(self, name: str):
        """Reset a specific circuit breaker."""
        if name in self._breakers:
            await self._breakers[name].reset()
        else:
            logger.warning(f"Circuit breaker {name} not found")


# Global circuit breaker manager
circuit_breaker_manager = CircuitBreakerManager()

# Configure circuit breakers for different services
circuit_breaker_manager.configure_breaker("github", CircuitBreakerConfig(
    failure_threshold=3,
    recovery_timeout=120,  # 2 minutes
    success_threshold=2,
    timeout=30.0
))

circuit_breaker_manager.configure_breaker("leetcode", CircuitBreakerConfig(
    failure_threshold=2,  # More sensitive due to anti-bot measures
    recovery_timeout=300,  # 5 minutes
    success_threshold=1,
    timeout=45.0
))

circuit_breaker_manager.configure_breaker("linkedin", CircuitBreakerConfig(
    failure_threshold=2,  # Very sensitive due to ToS restrictions
    recovery_timeout=600,  # 10 minutes
    success_threshold=1,
    timeout=60.0
))