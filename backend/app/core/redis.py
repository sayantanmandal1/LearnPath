"""
Redis configuration and connection management
"""
import json
from typing import Any, Optional, Union
import structlog
import redis.asyncio as redis
from redis.asyncio import Redis

from app.core.config import settings

logger = structlog.get_logger()


class RedisManager:
    """Redis connection and operations manager"""
    
    def __init__(self):
        self._redis: Optional[Redis] = None
    
    async def connect(self) -> None:
        """Initialize Redis connection"""
        try:
            self._redis = redis.from_url(
                settings.REDIS_URL,
                encoding="utf-8",
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True,
            )
            # Test connection
            await self._redis.ping()
            logger.info("Redis connection established")
        except Exception as e:
            logger.error("Failed to connect to Redis", error=str(e))
            raise
    
    async def disconnect(self) -> None:
        """Close Redis connection"""
        if self._redis:
            await self._redis.close()
            logger.info("Redis connection closed")
    
    @property
    def redis(self) -> Redis:
        """Get Redis client instance"""
        if not self._redis:
            raise RuntimeError("Redis not connected. Call connect() first.")
        return self._redis
    
    async def set(
        self,
        key: str,
        value: Union[str, dict, list],
        ttl: Optional[int] = None,
    ) -> bool:
        """Set a key-value pair with optional TTL"""
        try:
            if isinstance(value, (dict, list)):
                value = json.dumps(value)
            
            ttl = ttl or settings.REDIS_CACHE_TTL
            result = await self.redis.setex(key, ttl, value)
            return result
        except Exception as e:
            logger.error("Redis SET operation failed", key=key, error=str(e))
            return False
    
    async def setex(
        self,
        key: str,
        ttl: int,
        value: Union[str, dict, list],
    ) -> bool:
        """Set a key-value pair with TTL (Redis setex format)"""
        try:
            if isinstance(value, (dict, list)):
                value = json.dumps(value)
            
            result = await self.redis.setex(key, ttl, value)
            return result
        except Exception as e:
            logger.error("Redis SETEX operation failed", key=key, error=str(e))
            return False
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value by key"""
        try:
            value = await self.redis.get(key)
            if value is None:
                return None
            
            # Try to parse as JSON, fallback to string
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return value
        except Exception as e:
            logger.error("Redis GET operation failed", key=key, error=str(e))
            return None
    
    async def delete(self, key: str) -> bool:
        """Delete a key"""
        try:
            result = await self.redis.delete(key)
            return bool(result)
        except Exception as e:
            logger.error("Redis DELETE operation failed", key=key, error=str(e))
            return False
    
    async def exists(self, key: str) -> bool:
        """Check if key exists"""
        try:
            result = await self.redis.exists(key)
            return bool(result)
        except Exception as e:
            logger.error("Redis EXISTS operation failed", key=key, error=str(e))
            return False
    
    async def expire(self, key: str, ttl: int) -> bool:
        """Set TTL for existing key"""
        try:
            result = await self.redis.expire(key, ttl)
            return bool(result)
        except Exception as e:
            logger.error("Redis EXPIRE operation failed", key=key, error=str(e))
            return False
    
    async def increment(self, key: str, amount: int = 1) -> Optional[int]:
        """Increment a numeric value"""
        try:
            result = await self.redis.incrby(key, amount)
            return result
        except Exception as e:
            logger.error("Redis INCR operation failed", key=key, error=str(e))
            return None
    
    async def set_hash(self, key: str, mapping: dict, ttl: Optional[int] = None) -> bool:
        """Set hash fields"""
        try:
            # Convert dict values to strings
            str_mapping = {k: json.dumps(v) if isinstance(v, (dict, list)) else str(v) 
                          for k, v in mapping.items()}
            
            await self.redis.hset(key, mapping=str_mapping)
            
            if ttl:
                await self.redis.expire(key, ttl)
            
            return True
        except Exception as e:
            logger.error("Redis HSET operation failed", key=key, error=str(e))
            return False
    
    async def get_hash(self, key: str) -> Optional[dict]:
        """Get all hash fields"""
        try:
            result = await self.redis.hgetall(key)
            if not result:
                return None
            
            # Try to parse JSON values
            parsed_result = {}
            for k, v in result.items():
                try:
                    parsed_result[k] = json.loads(v)
                except json.JSONDecodeError:
                    parsed_result[k] = v
            
            return parsed_result
        except Exception as e:
            logger.error("Redis HGETALL operation failed", key=key, error=str(e))
            return None


# Global Redis manager instance
redis_manager = RedisManager()


async def get_redis() -> RedisManager:
    """Dependency to get Redis manager"""
    if not redis_manager._redis:
        await redis_manager.connect()
    return redis_manager
# Global Redis manager instance
redis_manager = RedisManager()

async def get_redis_client() -> Redis:
    """Get Redis client instance"""
    if not redis_manager._redis:
        await redis_manager.connect()
    return redis_manager.redis