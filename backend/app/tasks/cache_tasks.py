"""
Enhanced cache management background tasks with intelligent optimization
"""
import asyncio
from typing import Dict, List, Any, Optional
import structlog
from datetime import datetime, timedelta
import json

from app.core.celery_app import celery_app, TASK_PRIORITIES
from app.services.cache_service import get_cache_service, CacheKeyBuilder
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


@celery_app.task(
    name="app.tasks.cache_tasks.optimize_cache_performance",
    priority=TASK_PRIORITIES["MEDIUM"]
)
def optimize_cache_performance() -> Dict[str, Any]:
    """Optimize cache performance by analyzing usage patterns"""
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            cache_service = loop.run_until_complete(get_cache_service())
            redis_manager = loop.run_until_complete(get_redis())
            
            # Analyze cache hit patterns
            cache_stats = cache_service.get_stats()
            
            # Get memory usage
            memory_info = loop.run_until_complete(redis_manager.redis.info("memory"))
            
            # Identify frequently accessed keys
            frequent_keys = loop.run_until_complete(_identify_frequent_keys(redis_manager))
            
            # Identify rarely accessed keys for potential eviction
            rare_keys = loop.run_until_complete(_identify_rare_keys(redis_manager))
            
            # Optimize TTL for frequently accessed keys
            optimized_keys = loop.run_until_complete(
                _optimize_key_ttls(redis_manager, frequent_keys)
            )
            
            # Clean up rarely accessed keys
            cleaned_keys = loop.run_until_complete(
                _cleanup_rare_keys(redis_manager, rare_keys)
            )
            
            optimization_results = {
                "status": "success",
                "cache_hit_rate": cache_stats.get("hit_rate_percent", 0),
                "memory_usage_mb": memory_info.get("used_memory", 0) / 1024 / 1024,
                "frequent_keys_count": len(frequent_keys),
                "optimized_keys_count": len(optimized_keys),
                "cleaned_keys_count": len(cleaned_keys),
                "optimization_timestamp": datetime.utcnow().isoformat()
            }
            
            logger.info(
                "Cache performance optimization completed",
                **optimization_results
            )
            
            return optimization_results
            
        finally:
            loop.close()
            
    except Exception as e:
        logger.error("Cache performance optimization failed", error=str(e))
        raise


@celery_app.task(
    name="app.tasks.cache_tasks.preload_analysis_cache",
    priority=TASK_PRIORITIES["LOW"]
)
def preload_analysis_cache(user_ids: List[str]) -> Dict[str, Any]:
    """Preload cache with analysis results for active users"""
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            cache_service = loop.run_until_complete(get_cache_service())
            preloaded_count = 0
            failed_count = 0
            
            for user_id in user_ids:
                try:
                    # Preload user profile cache
                    profile_key = CacheKeyBuilder.user_profile(user_id)
                    profile_exists = loop.run_until_complete(cache_service.get(profile_key))
                    
                    if not profile_exists:
                        profile_data = loop.run_until_complete(_fetch_user_profile_data(user_id))
                        if profile_data:
                            loop.run_until_complete(
                                cache_service.set(profile_key, profile_data, ttl=3600)
                            )
                            preloaded_count += 1
                    
                    # Preload analytics cache
                    analytics_key = CacheKeyBuilder.analytics_data(user_id, "dashboard")
                    analytics_exists = loop.run_until_complete(cache_service.get(analytics_key))
                    
                    if not analytics_exists:
                        analytics_data = loop.run_until_complete(_generate_analytics_data(user_id))
                        if analytics_data:
                            loop.run_until_complete(
                                cache_service.set(analytics_key, analytics_data, ttl=1800)
                            )
                            preloaded_count += 1
                    
                except Exception as e:
                    logger.error("Failed to preload cache for user", user_id=user_id, error=str(e))
                    failed_count += 1
            
            return {
                "status": "success",
                "preloaded_count": preloaded_count,
                "failed_count": failed_count,
                "total_users": len(user_ids)
            }
            
        finally:
            loop.close()
            
    except Exception as e:
        logger.error("Cache preloading failed", error=str(e))
        raise


@celery_app.task(
    name="app.tasks.cache_tasks.intelligent_cache_eviction",
    priority=TASK_PRIORITIES["LOW"]
)
def intelligent_cache_eviction() -> Dict[str, Any]:
    """Intelligently evict cache entries based on usage patterns"""
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            redis_manager = loop.run_until_complete(get_redis())
            
            # Get memory usage
            memory_info = loop.run_until_complete(redis_manager.redis.info("memory"))
            memory_usage_mb = memory_info.get("used_memory", 0) / 1024 / 1024
            
            # Only proceed if memory usage is high
            if memory_usage_mb < 100:  # Less than 100MB
                return {
                    "status": "skipped",
                    "reason": "Memory usage below threshold",
                    "memory_usage_mb": memory_usage_mb
                }
            
            # Identify candidates for eviction
            eviction_candidates = loop.run_until_complete(
                _identify_eviction_candidates(redis_manager)
            )
            
            # Evict selected keys
            evicted_count = 0
            for key in eviction_candidates[:100]:  # Limit to 100 keys per run
                success = loop.run_until_complete(redis_manager.delete(key))
                if success:
                    evicted_count += 1
            
            # Get memory usage after eviction
            memory_info_after = loop.run_until_complete(redis_manager.redis.info("memory"))
            memory_after_mb = memory_info_after.get("used_memory", 0) / 1024 / 1024
            memory_freed_mb = memory_usage_mb - memory_after_mb
            
            return {
                "status": "success",
                "evicted_keys": evicted_count,
                "memory_freed_mb": memory_freed_mb,
                "memory_usage_before_mb": memory_usage_mb,
                "memory_usage_after_mb": memory_after_mb
            }
            
        finally:
            loop.close()
            
    except Exception as e:
        logger.error("Intelligent cache eviction failed", error=str(e))
        raise


async def _identify_frequent_keys(redis_manager) -> List[str]:
    """Identify frequently accessed cache keys"""
    try:
        # This is a simplified approach - in production, you might use Redis modules
        # or external monitoring to track access patterns
        frequent_patterns = [
            "user_profile:*",
            "career_rec:*",
            "api_resp:dashboard:*"
        ]
        
        frequent_keys = []
        for pattern in frequent_patterns:
            cursor = 0
            while True:
                cursor, keys = await redis_manager.redis.scan(cursor, match=pattern, count=50)
                frequent_keys.extend(keys)
                if cursor == 0:
                    break
        
        return frequent_keys[:100]  # Limit to top 100
        
    except Exception as e:
        logger.error("Failed to identify frequent keys", error=str(e))
        return []


async def _identify_rare_keys(redis_manager) -> List[str]:
    """Identify rarely accessed cache keys"""
    try:
        # Look for keys that might be stale or rarely used
        rare_patterns = [
            "ml_pred:*",
            "ext_api:*"
        ]
        
        rare_keys = []
        for pattern in rare_patterns:
            cursor = 0
            while True:
                cursor, keys = await redis_manager.redis.scan(cursor, match=pattern, count=50)
                
                # Check TTL to identify potentially stale keys
                for key in keys:
                    ttl = await redis_manager.redis.ttl(key)
                    if ttl > 3600:  # Keys with more than 1 hour TTL
                        rare_keys.append(key)
                
                if cursor == 0:
                    break
        
        return rare_keys[:50]  # Limit to 50 keys
        
    except Exception as e:
        logger.error("Failed to identify rare keys", error=str(e))
        return []


async def _optimize_key_ttls(redis_manager, frequent_keys: List[str]) -> List[str]:
    """Optimize TTL for frequently accessed keys"""
    optimized_keys = []
    
    try:
        for key in frequent_keys:
            current_ttl = await redis_manager.redis.ttl(key)
            
            # Extend TTL for frequently accessed keys
            if 0 < current_ttl < 1800:  # Less than 30 minutes
                new_ttl = min(current_ttl * 2, 7200)  # Double TTL, max 2 hours
                success = await redis_manager.expire(key, new_ttl)
                if success:
                    optimized_keys.append(key)
        
        return optimized_keys
        
    except Exception as e:
        logger.error("Failed to optimize key TTLs", error=str(e))
        return []


async def _cleanup_rare_keys(redis_manager, rare_keys: List[str]) -> List[str]:
    """Clean up rarely accessed keys"""
    cleaned_keys = []
    
    try:
        for key in rare_keys:
            # Only clean up keys that are close to expiration anyway
            ttl = await redis_manager.redis.ttl(key)
            if ttl > 0 and ttl < 300:  # Less than 5 minutes remaining
                success = await redis_manager.delete(key)
                if success:
                    cleaned_keys.append(key)
        
        return cleaned_keys
        
    except Exception as e:
        logger.error("Failed to cleanup rare keys", error=str(e))
        return []


async def _identify_eviction_candidates(redis_manager) -> List[str]:
    """Identify keys that are good candidates for eviction"""
    candidates = []
    
    try:
        # Look for keys with specific patterns that might be safe to evict
        eviction_patterns = [
            "api_resp:*",  # API response cache can be regenerated
            "ml_pred:*",   # ML predictions can be recalculated
            "ext_api:*"    # External API data can be refetched
        ]
        
        for pattern in eviction_patterns:
            cursor = 0
            while True:
                cursor, keys = await redis_manager.redis.scan(cursor, match=pattern, count=100)
                
                for key in keys:
                    # Check if key is close to expiration or has been around for a while
                    ttl = await redis_manager.redis.ttl(key)
                    if ttl > 1800:  # Keys with more than 30 minutes TTL
                        candidates.append(key)
                
                if cursor == 0:
                    break
        
        return candidates
        
    except Exception as e:
        logger.error("Failed to identify eviction candidates", error=str(e))
        return []


async def _generate_analytics_data(user_id: str) -> Optional[Dict[str, Any]]:
    """Generate analytics data for cache preloading"""
    try:
        # This would typically call your analytics service
        return {
            "user_id": user_id,
            "dashboard_metrics": {
                "profile_completeness": 85,
                "skill_score": 78,
                "market_alignment": 82
            },
            "generated_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error("Failed to generate analytics data", user_id=user_id, error=str(e))
        return None