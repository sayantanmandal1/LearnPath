"""
Advanced caching service with intelligent cache invalidation strategies and analysis result optimization
"""
import json
import hashlib
from typing import Any, Dict, List, Optional, Union, Callable
from datetime import datetime, timedelta
import structlog
from functools import wraps
import asyncio

from app.core.redis import RedisManager, get_redis
from app.core.config import settings

logger = structlog.get_logger()


class CacheKeyBuilder:
    """Utility class for building consistent cache keys"""
    
    @staticmethod
    def user_profile(user_id: str) -> str:
        return f"user_profile:{user_id}"
    
    @staticmethod
    def career_recommendations(user_id: str, params_hash: str) -> str:
        return f"career_rec:{user_id}:{params_hash}"
    
    @staticmethod
    def learning_path(user_id: str, skill_gaps_hash: str) -> str:
        return f"learning_path:{user_id}:{skill_gaps_hash}"
    
    @staticmethod
    def job_market_data(location: str, role: str) -> str:
        return f"job_market:{location}:{role}"
    
    @staticmethod
    def ml_prediction(model_name: str, input_hash: str) -> str:
        return f"ml_pred:{model_name}:{input_hash}"
    
    @staticmethod
    def api_response(endpoint: str, params_hash: str) -> str:
        return f"api_resp:{endpoint}:{params_hash}"
    
    @staticmethod
    def external_api(service: str, identifier: str) -> str:
        return f"ext_api:{service}:{identifier}"
    
    @staticmethod
    def analytics_data(user_id: str, report_type: str) -> str:
        return f"analytics:{user_id}:{report_type}"
    
    @staticmethod
    def analysis_result(user_id: str, analysis_type: str, params_hash: str = None) -> str:
        """Cache key for analysis results (skill assessment, career recommendations, etc.)"""
        if params_hash:
            return f"analysis:{user_id}:{analysis_type}:{params_hash}"
        return f"analysis:{user_id}:{analysis_type}"
    
    @staticmethod
    def resume_analysis(user_id: str) -> str:
        return f"resume_analysis:{user_id}"
    
    @staticmethod
    def platform_data(user_id: str, platform: str) -> str:
        return f"platform_data:{user_id}:{platform}"
    
    @staticmethod
    def job_recommendations(user_id: str, location: str = None) -> str:
        if location:
            return f"job_rec:{user_id}:{location}"
        return f"job_rec:{user_id}"
    
    @staticmethod
    def dashboard_data(user_id: str) -> str:
        return f"dashboard:{user_id}"


class CacheInvalidationStrategy:
    """Intelligent cache invalidation strategies"""
    
    def __init__(self, redis_manager: RedisManager):
        self.redis = redis_manager
    
    async def invalidate_user_cache(self, user_id: str) -> None:
        """Invalidate all cache entries for a specific user"""
        patterns = [
            f"user_profile:{user_id}",
            f"career_rec:{user_id}:*",
            f"learning_path:{user_id}:*",
            f"analytics:{user_id}:*",
        ]
        
        for pattern in patterns:
            await self._delete_pattern(pattern)
    
    async def invalidate_market_data_cache(self) -> None:
        """Invalidate job market data cache"""
        await self._delete_pattern("job_market:*")
    
    async def invalidate_ml_model_cache(self, model_name: str) -> None:
        """Invalidate ML model prediction cache"""
        await self._delete_pattern(f"ml_pred:{model_name}:*")
    
    async def invalidate_external_api_cache(self, service: str, identifier: str = None) -> None:
        """Invalidate external API cache"""
        if identifier:
            await self.redis.delete(f"ext_api:{service}:{identifier}")
        else:
            await self._delete_pattern(f"ext_api:{service}:*")
    
    async def invalidate_analysis_cache(self, user_id: str, analysis_type: str = None) -> None:
        """Invalidate analysis result cache for a user"""
        if analysis_type:
            await self._delete_pattern(f"analysis:{user_id}:{analysis_type}:*")
        else:
            await self._delete_pattern(f"analysis:{user_id}:*")
    
    async def invalidate_dashboard_cache(self, user_id: str) -> None:
        """Invalidate dashboard cache for a user"""
        patterns = [
            f"dashboard:{user_id}",
            f"job_rec:{user_id}:*",
            f"analytics:{user_id}:*"
        ]
        
        for pattern in patterns:
            await self._delete_pattern(pattern)
    
    async def _delete_pattern(self, pattern: str) -> None:
        """Delete all keys matching a pattern"""
        try:
            # Use SCAN to find matching keys (more efficient than KEYS)
            cursor = 0
            while True:
                cursor, keys = await self.redis.redis.scan(cursor, match=pattern, count=100)
                if keys:
                    await self.redis.redis.delete(*keys)
                if cursor == 0:
                    break
        except Exception as e:
            logger.error("Failed to delete cache pattern", pattern=pattern, error=str(e))


class AdvancedCacheService:
    """Advanced caching service with multiple cache layers and strategies"""
    
    def __init__(self, redis_manager: RedisManager):
        self.redis = redis_manager
        self.invalidation = CacheInvalidationStrategy(redis_manager)
        self._local_cache: Dict[str, Dict[str, Any]] = {}
        self._cache_stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "invalidations": 0
        }
    
    async def get(
        self,
        key: str,
        default: Any = None,
        use_local_cache: bool = True
    ) -> Any:
        """Get value from cache with multi-layer support"""
        try:
            # Check local cache first (L1)
            if use_local_cache and key in self._local_cache:
                cache_entry = self._local_cache[key]
                if cache_entry["expires_at"] > datetime.utcnow():
                    self._cache_stats["hits"] += 1
                    return cache_entry["value"]
                else:
                    # Remove expired entry
                    del self._local_cache[key]
            
            # Check Redis cache (L2)
            value = await self.redis.get(key)
            if value is not None:
                self._cache_stats["hits"] += 1
                
                # Store in local cache if enabled
                if use_local_cache:
                    self._local_cache[key] = {
                        "value": value,
                        "expires_at": datetime.utcnow() + timedelta(seconds=60)  # 1 minute local cache
                    }
                
                return value
            
            self._cache_stats["misses"] += 1
            return default
            
        except Exception as e:
            logger.error("Cache GET operation failed", key=key, error=str(e))
            return default
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        use_local_cache: bool = True
    ) -> bool:
        """Set value in cache with multi-layer support"""
        try:
            # Set in Redis (L2)
            success = await self.redis.set(key, value, ttl)
            
            if success:
                self._cache_stats["sets"] += 1
                
                # Set in local cache if enabled
                if use_local_cache:
                    local_ttl = min(ttl or 300, 60)  # Max 1 minute for local cache
                    self._local_cache[key] = {
                        "value": value,
                        "expires_at": datetime.utcnow() + timedelta(seconds=local_ttl)
                    }
            
            return success
            
        except Exception as e:
            logger.error("Cache SET operation failed", key=key, error=str(e))
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete value from all cache layers"""
        try:
            # Remove from local cache
            if key in self._local_cache:
                del self._local_cache[key]
            
            # Remove from Redis
            success = await self.redis.delete(key)
            if success:
                self._cache_stats["invalidations"] += 1
            
            return success
            
        except Exception as e:
            logger.error("Cache DELETE operation failed", key=key, error=str(e))
            return False
    
    async def get_or_set(
        self,
        key: str,
        factory: Callable,
        ttl: Optional[int] = None,
        use_local_cache: bool = True
    ) -> Any:
        """Get value from cache or set it using factory function"""
        value = await self.get(key, use_local_cache=use_local_cache)
        
        if value is None:
            # Generate value using factory
            if asyncio.iscoroutinefunction(factory):
                value = await factory()
            else:
                value = factory()
            
            # Cache the generated value
            await self.set(key, value, ttl, use_local_cache)
        
        return value
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics"""
        total_requests = self._cache_stats["hits"] + self._cache_stats["misses"]
        hit_rate = (self._cache_stats["hits"] / total_requests * 100) if total_requests > 0 else 0
        
        return {
            **self._cache_stats,
            "hit_rate_percent": round(hit_rate, 2),
            "local_cache_size": len(self._local_cache)
        }
    
    async def clear_expired_local_cache(self) -> None:
        """Clean up expired entries from local cache"""
        now = datetime.utcnow()
        expired_keys = [
            key for key, entry in self._local_cache.items()
            if entry["expires_at"] <= now
        ]
        
        for key in expired_keys:
            del self._local_cache[key]
        
        if expired_keys:
            logger.debug("Cleared expired local cache entries", count=len(expired_keys))


def cache_key_hash(*args, **kwargs) -> str:
    """Generate a hash for cache key from arguments"""
    key_data = {
        "args": args,
        "kwargs": sorted(kwargs.items())
    }
    key_string = json.dumps(key_data, sort_keys=True, default=str)
    return hashlib.md5(key_string.encode()).hexdigest()


def cached(
    ttl: int = 300,
    key_prefix: str = "",
    use_local_cache: bool = True,
    invalidate_on_error: bool = False
):
    """Decorator for caching function results"""
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Get cache service
            redis_manager = await get_redis()
            cache_service = AdvancedCacheService(redis_manager)
            
            # Generate cache key
            args_hash = cache_key_hash(*args, **kwargs)
            cache_key = f"{key_prefix}:{func.__name__}:{args_hash}" if key_prefix else f"{func.__name__}:{args_hash}"
            
            try:
                # Try to get from cache
                cached_result = await cache_service.get(cache_key, use_local_cache=use_local_cache)
                if cached_result is not None:
                    return cached_result
                
                # Execute function
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                
                # Cache the result
                await cache_service.set(cache_key, result, ttl, use_local_cache)
                
                return result
                
            except Exception as e:
                logger.error("Cached function execution failed", func=func.__name__, error=str(e))
                
                if invalidate_on_error:
                    await cache_service.delete(cache_key)
                
                raise
        
        return wrapper
    return decorator


# Global cache service instance
_cache_service: Optional[AdvancedCacheService] = None


async def get_cache_service() -> AdvancedCacheService:
    """Get global cache service instance"""
    global _cache_service
    if _cache_service is None:
        redis_manager = await get_redis()
        _cache_service = AdvancedCacheService(redis_manager)
    return _cache_service