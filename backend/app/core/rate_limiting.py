"""
Rate limiting and DDoS protection mechanisms
"""
import time
from typing import Dict, Optional, Tuple
from datetime import datetime, timedelta
import asyncio
import structlog
from fastapi import Request, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
import redis.asyncio as redis

from app.core.config import settings
from app.core.monitoring import system_monitor, AlertType, AlertSeverity

logger = structlog.get_logger()


class RateLimitExceeded(HTTPException):
    """Rate limit exceeded exception"""
    def __init__(self, retry_after: int = None):
        detail = "Rate limit exceeded. Please try again later."
        if retry_after:
            detail += f" Retry after {retry_after} seconds."
        
        super().__init__(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=detail,
            headers={"Retry-After": str(retry_after)} if retry_after else None
        )


class RateLimiter:
    """Redis-based rate limiter with sliding window"""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
    
    async def is_allowed(self, key: str, limit: int, window: int) -> Tuple[bool, int]:
        """
        Check if request is allowed using sliding window algorithm
        
        Args:
            key: Unique identifier for the rate limit (e.g., IP address, user ID)
            limit: Maximum number of requests allowed
            window: Time window in seconds
        
        Returns:
            Tuple of (is_allowed, retry_after_seconds)
        """
        now = time.time()
        pipeline = self.redis.pipeline()
        
        # Remove expired entries
        pipeline.zremrangebyscore(key, 0, now - window)
        
        # Count current requests
        pipeline.zcard(key)
        
        # Add current request
        pipeline.zadd(key, {str(now): now})
        
        # Set expiration
        pipeline.expire(key, window)
        
        results = await pipeline.execute()
        current_requests = results[1]
        
        if current_requests >= limit:
            # Calculate retry after time
            oldest_request = await self.redis.zrange(key, 0, 0, withscores=True)
            if oldest_request:
                retry_after = int(oldest_request[0][1] + window - now)
                return False, max(retry_after, 1)
            return False, window
        
        return True, 0
    
    async def get_usage(self, key: str, window: int) -> int:
        """Get current usage count for a key"""
        now = time.time()
        await self.redis.zremrangebyscore(key, 0, now - window)
        return await self.redis.zcard(key)


class DDoSProtection:
    """Advanced DDoS protection with multiple detection methods"""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.suspicious_ips = set()
        self.blocked_ips = {}  # IP -> block_until_timestamp
    
    async def analyze_request_pattern(self, ip: str, user_agent: str, 
                                    endpoint: str) -> Dict[str, any]:
        """Analyze request patterns for suspicious behavior"""
        now = time.time()
        analysis = {
            'is_suspicious': False,
            'risk_score': 0,
            'reasons': []
        }
        
        # Check request frequency
        freq_key = f"ddos:freq:{ip}"
        frequency = await self.redis.get(freq_key)
        if frequency:
            freq_count = int(frequency)
            if freq_count > 100:  # More than 100 requests per minute
                analysis['is_suspicious'] = True
                analysis['risk_score'] += 30
                analysis['reasons'].append("High request frequency")
        
        # Increment frequency counter
        await self.redis.incr(freq_key)
        await self.redis.expire(freq_key, 60)  # 1 minute window
        
        # Check for bot-like user agents
        bot_indicators = [
            'bot', 'crawler', 'spider', 'scraper', 'curl', 'wget',
            'python-requests', 'go-http-client'
        ]
        if any(indicator in user_agent.lower() for indicator in bot_indicators):
            analysis['risk_score'] += 20
            analysis['reasons'].append("Bot-like user agent")
        
        # Check for endpoint abuse
        endpoint_key = f"ddos:endpoint:{ip}:{endpoint}"
        endpoint_count = await self.redis.get(endpoint_key)
        if endpoint_count and int(endpoint_count) > 20:  # Same endpoint > 20 times per minute
            analysis['is_suspicious'] = True
            analysis['risk_score'] += 25
            analysis['reasons'].append("Endpoint abuse")
        
        await self.redis.incr(endpoint_key)
        await self.redis.expire(endpoint_key, 60)
        
        # Check for distributed attack patterns
        global_key = "ddos:global_requests"
        global_count = await self.redis.get(global_key)
        if global_count and int(global_count) > 1000:  # More than 1000 global requests per minute
            analysis['risk_score'] += 15
            analysis['reasons'].append("High global traffic")
        
        await self.redis.incr(global_key)
        await self.redis.expire(global_key, 60)
        
        # Determine if suspicious
        if analysis['risk_score'] >= 50:
            analysis['is_suspicious'] = True
        
        return analysis
    
    async def block_ip(self, ip: str, duration: int = 3600):
        """Block IP address for specified duration"""
        block_until = time.time() + duration
        self.blocked_ips[ip] = block_until
        
        # Store in Redis for persistence across instances
        await self.redis.setex(f"blocked_ip:{ip}", duration, block_until)
        
        logger.warning("IP blocked for suspicious activity", 
                      ip=ip, duration=duration, block_until=block_until)
        
        # Create security alert
        system_monitor.create_alert(
            alert_type=AlertType.SECURITY_INCIDENT,
            severity=AlertSeverity.HIGH,
            title="IP Address Blocked",
            description=f"IP {ip} blocked for suspicious activity",
            metadata={"ip": ip, "duration": duration, "block_until": block_until}
        )
    
    async def is_blocked(self, ip: str) -> bool:
        """Check if IP is currently blocked"""
        now = time.time()
        
        # Check local cache first
        if ip in self.blocked_ips:
            if self.blocked_ips[ip] > now:
                return True
            else:
                del self.blocked_ips[ip]
        
        # Check Redis
        block_until = await self.redis.get(f"blocked_ip:{ip}")
        if block_until:
            block_until_time = float(block_until)
            if block_until_time > now:
                self.blocked_ips[ip] = block_until_time
                return True
            else:
                await self.redis.delete(f"blocked_ip:{ip}")
        
        return False
    
    async def cleanup_expired_blocks(self):
        """Clean up expired IP blocks"""
        now = time.time()
        expired_ips = [ip for ip, block_until in self.blocked_ips.items() 
                      if block_until <= now]
        
        for ip in expired_ips:
            del self.blocked_ips[ip]
            await self.redis.delete(f"blocked_ip:{ip}")


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware with DDoS protection"""
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.redis_client = None
        self.rate_limiter = None
        self.ddos_protection = None
        self._initialized = False
    
    async def _initialize(self):
        """Initialize Redis connection and components"""
        if not self._initialized:
            try:
                self.redis_client = redis.from_url(settings.REDIS_URL)
                self.rate_limiter = RateLimiter(self.redis_client)
                self.ddos_protection = DDoSProtection(self.redis_client)
                self._initialized = True
            except Exception as e:
                logger.error("Failed to initialize rate limiting", error=str(e))
    
    def _get_client_ip(self, request: Request) -> str:
        """Get client IP address from request"""
        # Check for forwarded headers (behind proxy/load balancer)
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        # Fallback to direct connection
        return request.client.host if request.client else "unknown"
    
    def _get_rate_limit_key(self, request: Request, ip: str) -> str:
        """Generate rate limit key based on request"""
        # Use different keys for different types of requests
        if request.url.path.startswith("/api/v1/auth"):
            return f"rate_limit:auth:{ip}"
        elif request.url.path.startswith("/api/v1/profiles"):
            return f"rate_limit:profiles:{ip}"
        elif request.url.path.startswith("/api/v1/recommendations"):
            return f"rate_limit:recommendations:{ip}"
        else:
            return f"rate_limit:general:{ip}"
    
    def _get_rate_limits(self, request: Request) -> Tuple[int, int]:
        """Get rate limits based on endpoint"""
        path = request.url.path
        
        # Authentication endpoints - stricter limits
        if path.startswith("/api/v1/auth"):
            return 5, 60  # 5 requests per minute
        
        # File upload endpoints - very strict
        elif "upload" in path:
            return 3, 300  # 3 requests per 5 minutes
        
        # ML/AI endpoints - moderate limits
        elif any(endpoint in path for endpoint in ["/recommendations", "/analytics", "/learning-paths"]):
            return 20, 60  # 20 requests per minute
        
        # General API endpoints
        elif path.startswith("/api/v1"):
            return 100, 60  # 100 requests per minute
        
        # Health checks and static content
        else:
            return 1000, 60  # 1000 requests per minute
    
    async def dispatch(self, request: Request, call_next):
        """Process request with rate limiting and DDoS protection"""
        await self._initialize()
        
        if not self._initialized:
            # If Redis is not available, allow request but log warning
            logger.warning("Rate limiting disabled - Redis not available")
            return await call_next(request)
        
        ip = self._get_client_ip(request)
        user_agent = request.headers.get("User-Agent", "")
        
        try:
            # Check if IP is blocked
            if await self.ddos_protection.is_blocked(ip):
                logger.warning("Blocked IP attempted access", ip=ip, path=request.url.path)
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Access denied. Your IP has been temporarily blocked."
                )
            
            # Analyze request patterns for DDoS
            analysis = await self.ddos_protection.analyze_request_pattern(
                ip, user_agent, request.url.path
            )
            
            # Block suspicious IPs
            if analysis['is_suspicious'] and analysis['risk_score'] >= 70:
                await self.ddos_protection.block_ip(ip, duration=3600)  # Block for 1 hour
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Access denied due to suspicious activity."
                )
            
            # Apply rate limiting
            rate_limit_key = self._get_rate_limit_key(request, ip)
            limit, window = self._get_rate_limits(request)
            
            is_allowed, retry_after = await self.rate_limiter.is_allowed(
                rate_limit_key, limit, window
            )
            
            if not is_allowed:
                logger.warning("Rate limit exceeded", 
                             ip=ip, path=request.url.path, 
                             limit=limit, window=window)
                
                # Create alert for high rate limit violations
                current_usage = await self.rate_limiter.get_usage(rate_limit_key, window)
                if current_usage > limit * 2:  # More than double the limit
                    system_monitor.create_alert(
                        alert_type=AlertType.SECURITY_INCIDENT,
                        severity=AlertSeverity.MEDIUM,
                        title="High Rate Limit Violations",
                        description=f"IP {ip} exceeded rate limit by {current_usage - limit} requests",
                        metadata={"ip": ip, "path": request.url.path, "usage": current_usage}
                    )
                
                raise RateLimitExceeded(retry_after)
            
            # Add rate limit headers to response
            response = await call_next(request)
            
            current_usage = await self.rate_limiter.get_usage(rate_limit_key, window)
            response.headers["X-RateLimit-Limit"] = str(limit)
            response.headers["X-RateLimit-Remaining"] = str(max(0, limit - current_usage))
            response.headers["X-RateLimit-Reset"] = str(int(time.time() + window))
            
            return response
        
        except HTTPException:
            raise
        except Exception as e:
            logger.error("Rate limiting error", error=str(e), ip=ip)
            # Allow request to proceed if rate limiting fails
            return await call_next(request)


# Background task for cleanup
async def cleanup_rate_limiting():
    """Background task to clean up expired rate limiting data"""
    try:
        redis_client = redis.from_url(settings.REDIS_URL)
        ddos_protection = DDoSProtection(redis_client)
        
        while True:
            await ddos_protection.cleanup_expired_blocks()
            await asyncio.sleep(300)  # Run every 5 minutes
    
    except Exception as e:
        logger.error("Rate limiting cleanup error", error=str(e))