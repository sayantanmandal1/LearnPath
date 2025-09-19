"""
Intelligent retry mechanism service for external API failures
"""
import asyncio
import random
import time
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import structlog

from app.services.cache_service import get_cache_service
from app.services.external_apis.circuit_breaker import circuit_breaker_manager, CircuitBreakerError

logger = structlog.get_logger()


class RetryStrategy(Enum):
    """Different retry strategies"""
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    LINEAR_BACKOFF = "linear_backoff"
    FIXED_DELAY = "fixed_delay"
    FIBONACCI_BACKOFF = "fibonacci_backoff"
    ADAPTIVE = "adaptive"


class FailureType(Enum):
    """Types of failures that can occur"""
    TIMEOUT = "timeout"
    RATE_LIMIT = "rate_limit"
    NETWORK_ERROR = "network_error"
    AUTH_ERROR = "auth_error"
    SERVER_ERROR = "server_error"
    CLIENT_ERROR = "client_error"
    UNKNOWN = "unknown"


@dataclass
class RetryConfig:
    """Configuration for retry behavior"""
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 300.0  # 5 minutes
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF
    jitter: bool = True
    backoff_multiplier: float = 2.0
    retry_on_exceptions: List[type] = field(default_factory=lambda: [Exception])
    stop_on_exceptions: List[type] = field(default_factory=list)


@dataclass
class RetryAttempt:
    """Information about a retry attempt"""
    attempt_number: int
    delay: float
    timestamp: datetime
    exception: Optional[Exception] = None
    success: bool = False


@dataclass
class RetryHistory:
    """History of retry attempts for a specific operation"""
    operation_id: str
    platform: str
    total_attempts: int = 0
    successful_attempts: int = 0
    failed_attempts: int = 0
    attempts: List[RetryAttempt] = field(default_factory=list)
    first_attempt: Optional[datetime] = None
    last_attempt: Optional[datetime] = None
    total_delay: float = 0.0


class IntelligentRetryService:
    """Service for intelligent retry mechanisms with adaptive strategies"""
    
    def __init__(self):
        self.retry_histories: Dict[str, RetryHistory] = {}
        self.platform_configs: Dict[str, RetryConfig] = {}
        self._setup_default_configs()
    
    def _setup_default_configs(self):
        """Setup default retry configurations for different platforms"""
        
        # GitHub API - generally reliable but has rate limits
        self.platform_configs["github"] = RetryConfig(
            max_retries=3,
            base_delay=2.0,
            max_delay=60.0,
            strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
            backoff_multiplier=2.0
        )
        
        # LeetCode - has anti-bot measures, needs careful handling
        self.platform_configs["leetcode"] = RetryConfig(
            max_retries=2,
            base_delay=5.0,
            max_delay=300.0,
            strategy=RetryStrategy.FIBONACCI_BACKOFF,
            jitter=True
        )
        
        # LinkedIn - very strict, minimal retries
        self.platform_configs["linkedin"] = RetryConfig(
            max_retries=1,
            base_delay=10.0,
            max_delay=600.0,
            strategy=RetryStrategy.FIXED_DELAY,
            jitter=True
        )
        
        # Codeforces - competitive programming platform
        self.platform_configs["codeforces"] = RetryConfig(
            max_retries=3,
            base_delay=1.5,
            max_delay=120.0,
            strategy=RetryStrategy.EXPONENTIAL_BACKOFF
        )
        
        # AtCoder - Japanese competitive programming platform
        self.platform_configs["atcoder"] = RetryConfig(
            max_retries=3,
            base_delay=2.0,
            max_delay=180.0,
            strategy=RetryStrategy.LINEAR_BACKOFF
        )
        
        # HackerRank - coding challenges platform
        self.platform_configs["hackerrank"] = RetryConfig(
            max_retries=3,
            base_delay=1.0,
            max_delay=90.0,
            strategy=RetryStrategy.EXPONENTIAL_BACKOFF
        )
        
        # Kaggle - data science platform
        self.platform_configs["kaggle"] = RetryConfig(
            max_retries=2,
            base_delay=3.0,
            max_delay=240.0,
            strategy=RetryStrategy.ADAPTIVE
        )
        
        # Job portals
        self.platform_configs["naukri"] = RetryConfig(
            max_retries=4,
            base_delay=2.0,
            max_delay=120.0,
            strategy=RetryStrategy.EXPONENTIAL_BACKOFF
        )
        
        self.platform_configs["linkedin_jobs"] = RetryConfig(
            max_retries=2,
            base_delay=5.0,
            max_delay=300.0,
            strategy=RetryStrategy.FIBONACCI_BACKOFF
        )
    
    async def retry_with_intelligence(
        self,
        operation_func: Callable,
        platform: str,
        operation_id: str,
        *args,
        **kwargs
    ) -> Any:
        """
        Execute operation with intelligent retry logic
        
        Args:
            operation_func: The function to execute
            platform: Platform name for configuration lookup
            operation_id: Unique identifier for this operation
            *args, **kwargs: Arguments to pass to operation_func
            
        Returns:
            Result of the operation
            
        Raises:
            Exception: If all retry attempts fail
        """
        config = self.platform_configs.get(platform, RetryConfig())
        history = self._get_or_create_history(operation_id, platform)
        
        last_exception = None
        
        for attempt in range(config.max_retries + 1):  # +1 for initial attempt
            try:
                # Record attempt start
                attempt_start = datetime.utcnow()
                
                # Execute the operation
                result = await operation_func(*args, **kwargs)
                
                # Record successful attempt
                self._record_attempt(
                    history,
                    attempt + 1,
                    0.0,
                    attempt_start,
                    success=True
                )
                
                logger.info(
                    "Operation succeeded",
                    platform=platform,
                    operation_id=operation_id,
                    attempt=attempt + 1
                )
                
                return result
                
            except Exception as e:
                last_exception = e
                failure_type = self._classify_failure(e)
                
                # Record failed attempt
                self._record_attempt(
                    history,
                    attempt + 1,
                    0.0,
                    attempt_start,
                    exception=e,
                    success=False
                )
                
                # Check if we should stop retrying
                if self._should_stop_retrying(e, config, attempt, config.max_retries):
                    logger.error(
                        "Stopping retries",
                        platform=platform,
                        operation_id=operation_id,
                        attempt=attempt + 1,
                        error=str(e),
                        failure_type=failure_type.value
                    )
                    break
                
                # Calculate delay for next attempt
                if attempt < config.max_retries:
                    delay = self._calculate_delay(
                        config,
                        attempt + 1,
                        failure_type,
                        history
                    )
                    
                    logger.warning(
                        "Operation failed, retrying",
                        platform=platform,
                        operation_id=operation_id,
                        attempt=attempt + 1,
                        delay=delay,
                        error=str(e),
                        failure_type=failure_type.value
                    )
                    
                    # Update attempt with actual delay
                    if history.attempts:
                        history.attempts[-1].delay = delay
                        history.total_delay += delay
                    
                    # Wait before retry
                    await asyncio.sleep(delay)
        
        # All retries exhausted
        logger.error(
            "All retry attempts exhausted",
            platform=platform,
            operation_id=operation_id,
            total_attempts=history.total_attempts,
            total_delay=history.total_delay
        )
        
        raise last_exception
    
    def _get_or_create_history(self, operation_id: str, platform: str) -> RetryHistory:
        """Get or create retry history for an operation"""
        if operation_id not in self.retry_histories:
            self.retry_histories[operation_id] = RetryHistory(
                operation_id=operation_id,
                platform=platform
            )
        return self.retry_histories[operation_id]
    
    def _record_attempt(
        self,
        history: RetryHistory,
        attempt_number: int,
        delay: float,
        timestamp: datetime,
        exception: Optional[Exception] = None,
        success: bool = False
    ):
        """Record a retry attempt"""
        attempt = RetryAttempt(
            attempt_number=attempt_number,
            delay=delay,
            timestamp=timestamp,
            exception=exception,
            success=success
        )
        
        history.attempts.append(attempt)
        history.total_attempts += 1
        
        if success:
            history.successful_attempts += 1
        else:
            history.failed_attempts += 1
        
        if history.first_attempt is None:
            history.first_attempt = timestamp
        history.last_attempt = timestamp
    
    def _classify_failure(self, exception: Exception) -> FailureType:
        """Classify the type of failure"""
        error_str = str(exception).lower()
        
        if "timeout" in error_str or isinstance(exception, asyncio.TimeoutError):
            return FailureType.TIMEOUT
        elif "rate limit" in error_str or "429" in error_str:
            return FailureType.RATE_LIMIT
        elif "network" in error_str or "connection" in error_str:
            return FailureType.NETWORK_ERROR
        elif "auth" in error_str or "401" in error_str or "403" in error_str:
            return FailureType.AUTH_ERROR
        elif "500" in error_str or "502" in error_str or "503" in error_str:
            return FailureType.SERVER_ERROR
        elif "400" in error_str or "404" in error_str:
            return FailureType.CLIENT_ERROR
        else:
            return FailureType.UNKNOWN
    
    def _should_stop_retrying(
        self,
        exception: Exception,
        config: RetryConfig,
        current_attempt: int,
        max_retries: int
    ) -> bool:
        """Determine if we should stop retrying"""
        
        # Check if we've reached max retries
        if current_attempt >= max_retries:
            return True
        
        # Check if exception type should stop retries
        for stop_exception in config.stop_on_exceptions:
            if isinstance(exception, stop_exception):
                return True
        
        # Check for specific failure types that shouldn't be retried
        failure_type = self._classify_failure(exception)
        
        if failure_type == FailureType.AUTH_ERROR:
            return True  # Don't retry auth errors
        
        if failure_type == FailureType.CLIENT_ERROR:
            return True  # Don't retry client errors (4xx)
        
        # Check circuit breaker
        if isinstance(exception, CircuitBreakerError):
            return True  # Circuit breaker is open, don't retry
        
        return False
    
    def _calculate_delay(
        self,
        config: RetryConfig,
        attempt: int,
        failure_type: FailureType,
        history: RetryHistory
    ) -> float:
        """Calculate delay before next retry attempt"""
        
        if config.strategy == RetryStrategy.FIXED_DELAY:
            delay = config.base_delay
            
        elif config.strategy == RetryStrategy.LINEAR_BACKOFF:
            delay = config.base_delay * attempt
            
        elif config.strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
            delay = config.base_delay * (config.backoff_multiplier ** (attempt - 1))
            
        elif config.strategy == RetryStrategy.FIBONACCI_BACKOFF:
            delay = config.base_delay * self._fibonacci(attempt)
            
        elif config.strategy == RetryStrategy.ADAPTIVE:
            delay = self._calculate_adaptive_delay(config, attempt, failure_type, history)
            
        else:
            delay = config.base_delay
        
        # Apply failure type specific adjustments
        if failure_type == FailureType.RATE_LIMIT:
            delay *= 2.0  # Double delay for rate limits
        elif failure_type == FailureType.SERVER_ERROR:
            delay *= 1.5  # Increase delay for server errors
        
        # Apply jitter if enabled
        if config.jitter:
            jitter_range = delay * 0.1  # 10% jitter
            delay += random.uniform(-jitter_range, jitter_range)
        
        # Ensure delay is within bounds
        delay = max(0.1, min(delay, config.max_delay))
        
        return delay
    
    def _calculate_adaptive_delay(
        self,
        config: RetryConfig,
        attempt: int,
        failure_type: FailureType,
        history: RetryHistory
    ) -> float:
        """Calculate adaptive delay based on historical performance"""
        
        # Start with exponential backoff
        base_delay = config.base_delay * (config.backoff_multiplier ** (attempt - 1))
        
        # Adjust based on recent success rate
        if history.total_attempts > 0:
            success_rate = history.successful_attempts / history.total_attempts
            
            if success_rate < 0.3:  # Low success rate
                base_delay *= 2.0
            elif success_rate > 0.8:  # High success rate
                base_delay *= 0.7
        
        # Adjust based on recent failures
        recent_failures = [
            a for a in history.attempts[-5:]  # Last 5 attempts
            if not a.success and a.timestamp > datetime.utcnow() - timedelta(minutes=10)
        ]
        
        if len(recent_failures) >= 3:
            base_delay *= 1.5  # Increase delay if many recent failures
        
        return base_delay
    
    def _fibonacci(self, n: int) -> int:
        """Calculate nth Fibonacci number"""
        if n <= 1:
            return n
        
        a, b = 0, 1
        for _ in range(2, n + 1):
            a, b = b, a + b
        
        return b
    
    async def get_retry_statistics(self, platform: str = None) -> Dict[str, Any]:
        """Get retry statistics"""
        
        if platform:
            # Statistics for specific platform
            platform_histories = [
                h for h in self.retry_histories.values()
                if h.platform == platform
            ]
        else:
            # Overall statistics
            platform_histories = list(self.retry_histories.values())
        
        if not platform_histories:
            return {
                "total_operations": 0,
                "success_rate": 0.0,
                "average_attempts": 0.0,
                "average_delay": 0.0
            }
        
        total_operations = len(platform_histories)
        successful_operations = len([h for h in platform_histories if h.successful_attempts > 0])
        total_attempts = sum(h.total_attempts for h in platform_histories)
        total_delay = sum(h.total_delay for h in platform_histories)
        
        return {
            "total_operations": total_operations,
            "successful_operations": successful_operations,
            "success_rate": (successful_operations / total_operations) * 100,
            "average_attempts": total_attempts / total_operations,
            "average_delay": total_delay / total_operations,
            "platform": platform
        }
    
    def configure_platform(self, platform: str, config: RetryConfig):
        """Configure retry behavior for a specific platform"""
        self.platform_configs[platform] = config
        logger.info(f"Updated retry configuration for platform: {platform}")
    
    def clear_history(self, platform: str = None, older_than: timedelta = None):
        """Clear retry history"""
        if older_than:
            cutoff_time = datetime.utcnow() - older_than
            to_remove = [
                op_id for op_id, history in self.retry_histories.items()
                if (not platform or history.platform == platform) and
                (history.last_attempt and history.last_attempt < cutoff_time)
            ]
        else:
            to_remove = [
                op_id for op_id, history in self.retry_histories.items()
                if not platform or history.platform == platform
            ]
        
        for op_id in to_remove:
            del self.retry_histories[op_id]
        
        logger.info(f"Cleared {len(to_remove)} retry history entries")


# Global intelligent retry service instance
_retry_service: Optional[IntelligentRetryService] = None


def get_intelligent_retry_service() -> IntelligentRetryService:
    """Get global intelligent retry service instance"""
    global _retry_service
    if _retry_service is None:
        _retry_service = IntelligentRetryService()
    return _retry_service