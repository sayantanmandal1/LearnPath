"""
Performance monitoring and optimization API endpoints
"""
from typing import Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import JSONResponse
import structlog

from app.services.performance_monitoring import get_performance_monitor, performance_context
from app.services.cache_service import get_cache_service
from app.core.database_optimization import get_database_health, db_optimizer
from app.core.celery_app import task_manager, celery_health_check
from app.api.dependencies import get_current_user
from app.models.user import User

logger = structlog.get_logger()
router = APIRouter()


@router.get("/health", response_model=Dict[str, Any])
async def get_system_health():
    """Get comprehensive system health status"""
    try:
        async with performance_context("system_health_check"):
            # Get performance monitor
            monitor = await get_performance_monitor()
            
            # Collect current metrics
            current_metrics = await monitor.collect_metrics()
            
            # Get database health
            db_health = await get_database_health()
            
            # Get cache service stats
            cache_service = await get_cache_service()
            cache_stats = cache_service.get_stats()
            
            # Get Celery health
            celery_health = celery_health_check()
            
            # Determine overall health status
            health_status = "healthy"
            if (current_metrics.cpu_percent > 90 or 
                current_metrics.memory_percent > 95 or 
                current_metrics.response_time_ms > 5000):
                health_status = "critical"
            elif (current_metrics.cpu_percent > 80 or 
                  current_metrics.memory_percent > 85 or 
                  current_metrics.response_time_ms > 2000):
                health_status = "warning"
            
            return {
                "status": health_status,
                "timestamp": current_metrics.timestamp,
                "system_metrics": {
                    "cpu_percent": current_metrics.cpu_percent,
                    "memory_percent": current_metrics.memory_percent,
                    "memory_used_mb": current_metrics.memory_used_mb,
                    "response_time_ms": current_metrics.response_time_ms,
                    "active_connections": current_metrics.active_connections
                },
                "database_health": {
                    "status": db_health.get("connection_status", "unknown"),
                    "total_queries": db_health.get("performance_report", {}).get("total_queries", 0),
                    "pool_status": db_health.get("pool_optimization", {}).get("pool_status", {})
                },
                "cache_health": {
                    "hit_rate_percent": cache_stats.get("hit_rate_percent", 0),
                    "local_cache_size": cache_stats.get("local_cache_size", 0),
                    "total_hits": cache_stats.get("hits", 0),
                    "total_misses": cache_stats.get("misses", 0)
                },
                "celery_health": celery_health
            }
            
    except Exception as e:
        logger.error("Failed to get system health", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to retrieve system health")


@router.get("/metrics", response_model=Dict[str, Any])
async def get_performance_metrics(
    hours: int = Query(1, ge=1, le=24, description="Hours of metrics to retrieve"),
    current_user: User = Depends(get_current_user)
):
    """Get performance metrics for the specified time period"""
    try:
        monitor = await get_performance_monitor()
        summary = monitor.get_performance_summary(hours=hours)
        
        return {
            "performance_summary": summary,
            "requested_hours": hours,
            "user_id": current_user.id
        }
        
    except Exception as e:
        logger.error("Failed to get performance metrics", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to retrieve performance metrics")


@router.get("/bottlenecks", response_model=Dict[str, Any])
async def analyze_bottlenecks(current_user: User = Depends(get_current_user)):
    """Analyze system bottlenecks and get optimization recommendations"""
    try:
        monitor = await get_performance_monitor()
        analysis = monitor.get_bottleneck_analysis()
        
        return {
            "bottleneck_analysis": analysis,
            "user_id": current_user.id
        }
        
    except Exception as e:
        logger.error("Failed to analyze bottlenecks", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to analyze system bottlenecks")


@router.get("/database/stats", response_model=Dict[str, Any])
async def get_database_stats(current_user: User = Depends(get_current_user)):
    """Get detailed database performance statistics"""
    try:
        # Get table statistics
        table_stats = await db_optimizer.analyze_table_stats()
        
        # Get index usage analysis
        index_analysis = await db_optimizer.analyze_index_usage()
        
        # Get optimization suggestions
        index_suggestions = await db_optimizer.suggest_indexes()
        
        # Get connection pool optimization
        pool_optimization = await db_optimizer.optimize_connection_pool()
        
        return {
            "table_statistics": table_stats,
            "index_analysis": index_analysis,
            "optimization_suggestions": index_suggestions,
            "connection_pool": pool_optimization,
            "user_id": current_user.id
        }
        
    except Exception as e:
        logger.error("Failed to get database stats", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to retrieve database statistics")


@router.get("/cache/stats", response_model=Dict[str, Any])
async def get_cache_stats(current_user: User = Depends(get_current_user)):
    """Get detailed cache performance statistics"""
    try:
        cache_service = await get_cache_service()
        cache_stats = cache_service.get_stats()
        
        # Get Redis memory info
        redis_info = await cache_service.redis.redis.info("memory")
        
        return {
            "cache_service_stats": cache_stats,
            "redis_memory": {
                "used_memory_mb": redis_info.get("used_memory", 0) / 1024 / 1024,
                "used_memory_peak_mb": redis_info.get("used_memory_peak", 0) / 1024 / 1024,
                "fragmentation_ratio": redis_info.get("mem_fragmentation_ratio", 0)
            },
            "user_id": current_user.id
        }
        
    except Exception as e:
        logger.error("Failed to get cache stats", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to retrieve cache statistics")


@router.get("/celery/stats", response_model=Dict[str, Any])
async def get_celery_stats(current_user: User = Depends(get_current_user)):
    """Get Celery task queue statistics"""
    try:
        # Get worker statistics
        worker_stats = task_manager.get_worker_stats()
        
        # Get active tasks
        active_tasks = task_manager.get_active_tasks()
        
        # Get Celery health
        health = celery_health_check()
        
        return {
            "worker_statistics": worker_stats,
            "active_tasks": active_tasks,
            "health_status": health,
            "user_id": current_user.id
        }
        
    except Exception as e:
        logger.error("Failed to get Celery stats", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to retrieve Celery statistics")


@router.post("/cache/clear", response_model=Dict[str, Any])
async def clear_cache(
    pattern: Optional[str] = Query(None, description="Cache key pattern to clear"),
    current_user: User = Depends(get_current_user)
):
    """Clear cache entries (admin only)"""
    try:
        # Note: In a real application, you'd want to check if user is admin
        cache_service = await get_cache_service()
        
        if pattern:
            # Clear specific pattern
            await cache_service.invalidation._delete_pattern(pattern)
            message = f"Cleared cache entries matching pattern: {pattern}"
        else:
            # Clear all cache
            await cache_service.redis.redis.flushdb()
            message = "Cleared all cache entries"
        
        logger.info("Cache cleared", pattern=pattern, user_id=current_user.id)
        
        return {
            "status": "success",
            "message": message,
            "user_id": current_user.id
        }
        
    except Exception as e:
        logger.error("Failed to clear cache", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to clear cache")


@router.post("/cache/warm", response_model=Dict[str, Any])
async def warm_cache(
    cache_keys: list[str],
    current_user: User = Depends(get_current_user)
):
    """Warm up cache with specified keys"""
    try:
        from app.tasks.cache_tasks import warm_cache
        
        # Start background task to warm cache
        task = warm_cache.delay(cache_keys)
        
        return {
            "status": "started",
            "task_id": task.id,
            "cache_keys_count": len(cache_keys),
            "user_id": current_user.id
        }
        
    except Exception as e:
        logger.error("Failed to start cache warming", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to start cache warming")


@router.get("/tasks/{task_id}", response_model=Dict[str, Any])
async def get_task_status(
    task_id: str,
    current_user: User = Depends(get_current_user)
):
    """Get status of a background task"""
    try:
        task_status = task_manager.get_task_status(task_id)
        
        return {
            "task_status": task_status,
            "user_id": current_user.id
        }
        
    except Exception as e:
        logger.error("Failed to get task status", task_id=task_id, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to retrieve task status")


@router.delete("/tasks/{task_id}", response_model=Dict[str, Any])
async def cancel_task(
    task_id: str,
    current_user: User = Depends(get_current_user)
):
    """Cancel a running background task"""
    try:
        success = task_manager.cancel_task(task_id)
        
        if success:
            return {
                "status": "cancelled",
                "task_id": task_id,
                "user_id": current_user.id
            }
        else:
            raise HTTPException(status_code=404, detail="Task not found or cannot be cancelled")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to cancel task", task_id=task_id, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to cancel task")