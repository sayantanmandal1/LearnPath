"""
Cache management background tasks
"""
import asyncio
from typing import Dict, List, Any
import structlog
from datetime import datetime, timedelta

from app.core.celery_app import celery_app, TASK_PRIORITIES
from app.services.cache_service import get_cache_service
from app.core.redis import get_redis

logger = structlog.get_logger()


@celery_app.task(
    name="app.tasks.cache_tasks.cleanup_expired_cache",
    priority=TASK_PRIORITIES["LOW"]
)
def cleanup_expired_cache() -> Dict[str, Any]:
    """Clean up expired cache entries and optimize memory usage"""
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            cache_service = loop.run_until_complete(get_cache_service())
            
            # Clean up local cache
            loop.run_until_complete(cache_service.clear_expired_local_cache())
            
            # Get Redis client for advanced cleanup
            redis_manager = loop.run_until_complete(get_redis())
            
            # Get memory usage before cleanup
            memory_info_before = loop.run_until_complete(
                redis_manager.redis.info("memory")
            )
            
            # Run Redis memory optimization
            loop.run_until_complete(redis_manager.redis.memory_purge())
            
            # Get memory usage after cleanup
            memory_info_after = loop.run_until_complete(
                redis_manager.redis.info("memory")
            )
            
            # Calculate memory saved
            memory_saved = (
                memory_info_before.get("used_memory", 0) - 
                memory_info_after.get("used_memory", 0)
            )
            
            logger.info(
                "Cache cleanup completed",
                memory_saved_bytes=memory_saved,
                memory_used_mb=memory_info_after.get("used_memory", 0) / 1024 / 1024
            )
            
            return {
                "status": "success",
                "memory_saved_bytes": memory_saved,
                "memory_used_mb": memory_info_after.get("used_memory", 0) / 1024 / 1024,
                "cleanup_timestamp": datetime.utcnow().isoformat()
            }
            
        finally:
            loop.close()
            
    except Exception as e:
        logger.error("Cache cleanup failed", error=str(e))
        raise


@celery_app.task(
    name="app.tasks.cache_tasks.warm_cache",
    priority=TASK_PRIORITIES["MEDIUM"]
)
def warm_cache(cache_keys: List[str], data_source: str = "database") -> Dict[str, Any]:
    """Warm up cache with frequently accessed data"""
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            cache_service = loop.run_until_complete(get_cache_service())
            warmed_keys = []
            failed_keys = []
            
            for cache_key in cache_keys:
                try:
                    # Check if key already exists
                    exists = loop.run_until_complete(cache_service.get(cache_key))
                    
                    if exists is None:
                        # Generate data based on cache key pattern
                        data = loop.run_until_complete(
                            _generate_cache_data(cache_key, data_source)
                        )
                        
                        if data:
                            # Cache the data
                            success = loop.run_until_complete(
                                cache_service.set(cache_key, data, ttl=3600)
                            )
                            
                            if success:
                                warmed_keys.append(cache_key)
                            else:
                                failed_keys.append(cache_key)
                    
                except Exception as e:
                    logger.error("Failed to warm cache key", key=cache_key, error=str(e))
                    failed_keys.append(cache_key)
            
            return {
                "status": "success",
                "warmed_keys": len(warmed_keys),
                "failed_keys": len(failed_keys),
                "total_keys": len(cache_keys)
            }
            
        finally:
            loop.close()
            
    except Exception as e:
        logger.error("Cache warming failed", error=str(e))
        raise


@celery_app.task(
    name="app.tasks.cache_tasks.invalidate_user_cache",
    priority=TASK_PRIORITIES["HIGH"]
)
def invalidate_user_cache(user_id: str) -> Dict[str, Any]:
    """Invalidate all cache entries for a specific user"""
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            cache_service = loop.run_until_complete(get_cache_service())
            
            # Invalidate user-specific cache
            loop.run_until_complete(
                cache_service.invalidation.invalidate_user_cache(user_id)
            )
            
            logger.info("User cache invalidated", user_id=user_id)
            
            return {
                "status": "success",
                "user_id": user_id,
                "invalidated_at": datetime.utcnow().isoformat()
            }
            
        finally:
            loop.close()
            
    except Exception as e:
        logger.error("User cache invalidation failed", user_id=user_id, error=str(e))
        raise


@celery_app.task(
    name="app.tasks.cache_tasks.cache_statistics",
    priority=TASK_PRIORITIES["LOW"]
)
def cache_statistics() -> Dict[str, Any]:
    """Generate cache performance statistics"""
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            cache_service = loop.run_until_complete(get_cache_service())
            redis_manager = loop.run_until_complete(get_redis())
            
            # Get cache service stats
            cache_stats = cache_service.get_stats()
            
            # Get Redis info
            redis_info = loop.run_until_complete(redis_manager.redis.info())
            
            # Get key count by pattern
            key_patterns = [
                "user_profile:*",
                "career_rec:*",
                "learning_path:*",
                "job_market:*",
                "ml_pred:*",
                "api_resp:*"
            ]
            
            key_counts = {}
            for pattern in key_patterns:
                cursor = 0
                count = 0
                while True:
                    cursor, keys = loop.run_until_complete(
                        redis_manager.redis.scan(cursor, match=pattern, count=100)
                    )
                    count += len(keys)
                    if cursor == 0:
                        break
                key_counts[pattern] = count
            
            return {
                "cache_service_stats": cache_stats,
                "redis_memory_mb": redis_info.get("used_memory", 0) / 1024 / 1024,
                "redis_keys_total": redis_info.get("db0", {}).get("keys", 0),
                "key_counts_by_pattern": key_counts,
                "generated_at": datetime.utcnow().isoformat()
            }
            
        finally:
            loop.close()
            
    except Exception as e:
        logger.error("Cache statistics generation failed", error=str(e))
        raise


async def _generate_cache_data(cache_key: str, data_source: str) -> Any:
    """Generate data for cache warming based on cache key pattern"""
    try:
        if cache_key.startswith("user_profile:"):
            # Generate user profile data
            user_id = cache_key.split(":")[-1]
            return await _fetch_user_profile_data(user_id)
        
        elif cache_key.startswith("job_market:"):
            # Generate job market data
            parts = cache_key.split(":")
            location = parts[1] if len(parts) > 1 else "remote"
            role = parts[2] if len(parts) > 2 else "software_engineer"
            return await _fetch_job_market_data(location, role)
        
        elif cache_key.startswith("ml_pred:"):
            # Skip ML predictions for cache warming (too expensive)
            return None
        
        else:
            # Unknown pattern
            return None
            
    except Exception as e:
        logger.error("Failed to generate cache data", key=cache_key, error=str(e))
        return None


async def _fetch_user_profile_data(user_id: str) -> Dict[str, Any]:
    """Fetch user profile data from database"""
    try:
        from app.core.database import AsyncSessionLocal
        from app.repositories.profile import ProfileRepository
        
        async with AsyncSessionLocal() as session:
            profile_repo = ProfileRepository(session)
            profile = await profile_repo.get_by_user_id(user_id)
            
            if profile:
                return {
                    "user_id": user_id,
                    "skills": profile.skills or {},
                    "experience_years": profile.experience_years,
                    "dream_job": profile.dream_job,
                    "cached_at": datetime.utcnow().isoformat()
                }
        
        return None
        
    except Exception as e:
        logger.error("Failed to fetch user profile data", user_id=user_id, error=str(e))
        return None


async def _fetch_job_market_data(location: str, role: str) -> Dict[str, Any]:
    """Fetch job market data from database"""
    try:
        from app.core.database import AsyncSessionLocal
        from app.repositories.job import JobRepository
        
        async with AsyncSessionLocal() as session:
            job_repo = JobRepository(session)
            jobs = await job_repo.get_by_location_and_role(location, role, limit=50)
            
            return {
                "location": location,
                "role": role,
                "job_count": len(jobs),
                "jobs": [
                    {
                        "title": job.title,
                        "company": job.company,
                        "salary_range": job.salary_range,
                        "required_skills": job.required_skills
                    }
                    for job in jobs[:10]  # Limit to top 10 for cache
                ],
                "cached_at": datetime.utcnow().isoformat()
            }
        
    except Exception as e:
        logger.error("Failed to fetch job market data", location=location, role=role, error=str(e))
        return None