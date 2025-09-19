"""
API endpoints for performance optimization and monitoring
"""
from typing import Dict, List, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from pydantic import BaseModel
import structlog

from app.services.cache_service import get_cache_service
from app.services.concurrent_processing_service import get_concurrent_processing_service
from app.services.intelligent_retry_service import get_intelligent_retry_service
from app.services.enhanced_performance_monitoring import get_performance_monitor
from app.core.celery_app import celery_app, task_manager
from app.api.dependencies import get_current_user
from app.schemas.auth import UserResponse

logger = structlog.get_logger()

router = APIRouter()


class CacheOptimizationRequest(BaseModel):
    """Request model for cache optimization"""
    operation: str  # "cleanup", "warm", "invalidate", "optimize"
    target: Optional[str] = None  # user_id, platform, or pattern
    parameters: Optional[Dict[str, Any]] = None


class PerformanceOptimizationRequest(BaseModel):
    """Request model for performance optimization"""
    user_id: str
    platforms: Optional[List[str]] = None
    analysis_types: Optional[List[str]] = None
    use_concurrent_processing: bool = True
    use_intelligent_retry: bool = True


class RetryConfigurationRequest(BaseModel):
    """Request model for retry configuration"""
    platform: str
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 300.0
    strategy: str = "exponential_backoff"
    jitter: bool = True


@router.get("/cache/stats")
async def get_cache_statistics(
    current_user: UserResponse = Depends(get_current_user)
) -> Dict[str, Any]:
    """Get cache performance statistics"""
    try:
        cache_service = await get_cache_service()
        cache_stats = cache_service.get_stats()
        
        # Get Redis memory info
        from app.core.redis import get_redis
        redis_manager = await get_redis()
        redis_info = await redis_manager.redis.info("memory")
        
        return {
            "cache_service_stats": cache_stats,
            "redis_memory_mb": redis_info.get("used_memory", 0) / 1024 / 1024,
            "redis_peak_memory_mb": redis_info.get("used_memory_peak", 0) / 1024 / 1024,
            "memory_fragmentation_ratio": redis_info.get("mem_fragmentation_ratio", 0),
            "keyspace_hits": redis_info.get("keyspace_hits", 0),
            "keyspace_misses": redis_info.get("keyspace_misses", 0),
            "evicted_keys": redis_info.get("evicted_keys", 0),
            "expired_keys": redis_info.get("expired_keys", 0)
        }
        
    except Exception as e:
        logger.error("Failed to get cache statistics", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to retrieve cache statistics")


@router.post("/cache/warm/intelligent")
async def intelligent_cache_warming(
    user_ids: List[str],
    analysis_types: Optional[List[str]] = None,
    current_user: UserResponse = Depends(get_current_user)
) -> Dict[str, Any]:
    """Intelligently warm cache for active users"""
    try:
        # Schedule intelligent cache warming task
        task = celery_app.send_task(
            "app.tasks.data_collection_tasks.intelligent_cache_warming",
            args=[user_ids, analysis_types]
        )
        
        return {
            "status": "scheduled",
            "operation": "intelligent_cache_warming",
            "user_count": len(user_ids),
            "analysis_types": analysis_types or ["skill_assessment", "career_recommendations", "dashboard_data"],
            "task_id": task.id
        }
        
    except Exception as e:
        logger.error("Failed to schedule intelligent cache warming", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to schedule intelligent cache warming")


@router.post("/cache/optimize")
async def optimize_cache(
    request: CacheOptimizationRequest,
    background_tasks: BackgroundTasks,
    current_user: UserResponse = Depends(get_current_user)
) -> Dict[str, Any]:
    """Optimize cache performance"""
    try:
        if request.operation == "cleanup":
            # Schedule cache cleanup task
            task = celery_app.send_task(
                "app.tasks.cache_tasks.cleanup_expired_cache"
            )
            return {
                "status": "scheduled",
                "operation": "cleanup",
                "task_id": task.id
            }
            
        elif request.operation == "warm":
            # Schedule cache warming task
            cache_keys = request.parameters.get("cache_keys", []) if request.parameters else []
            task = celery_app.send_task(
                "app.tasks.cache_tasks.warm_cache",
                args=[cache_keys]
            )
            return {
                "status": "scheduled",
                "operation": "warm",
                "task_id": task.id,
                "cache_keys_count": len(cache_keys)
            }
            
        elif request.operation == "invalidate":
            if request.target:
                # Schedule user cache invalidation
                task = celery_app.send_task(
                    "app.tasks.cache_tasks.invalidate_user_cache",
                    args=[request.target]
                )
                return {
                    "status": "scheduled",
                    "operation": "invalidate",
                    "target": request.target,
                    "task_id": task.id
                }
            else:
                raise HTTPException(status_code=400, detail="Target required for invalidation")
                
        elif request.operation == "optimize":
            # Schedule cache performance optimization
            task = celery_app.send_task(
                "app.tasks.cache_tasks.optimize_cache_performance"
            )
            return {
                "status": "scheduled",
                "operation": "optimize",
                "task_id": task.id
            }
            
        else:
            raise HTTPException(status_code=400, detail="Invalid cache operation")
            
    except Exception as e:
        logger.error("Failed to optimize cache", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to optimize cache")


@router.post("/processing/concurrent")
async def run_concurrent_processing(
    request: PerformanceOptimizationRequest,
    current_user: UserResponse = Depends(get_current_user)
) -> Dict[str, Any]:
    """Run concurrent processing for user data collection and analysis"""
    try:
        if request.use_concurrent_processing:
            # Schedule concurrent analysis pipeline
            task = celery_app.send_task(
                "app.tasks.data_collection_tasks.concurrent_analysis_pipeline",
                args=[request.user_id, request.analysis_types]
            )
            
            return {
                "status": "scheduled",
                "user_id": request.user_id,
                "analysis_types": request.analysis_types,
                "task_id": task.id,
                "concurrent_processing": True
            }
        else:
            # Run traditional sequential processing
            return {
                "status": "error",
                "message": "Sequential processing not implemented in this endpoint"
            }
            
    except Exception as e:
        logger.error("Failed to run concurrent processing", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to run concurrent processing")


@router.get("/processing/stats")
async def get_processing_statistics(
    current_user: UserResponse = Depends(get_current_user)
) -> Dict[str, Any]:
    """Get concurrent processing statistics"""
    try:
        concurrent_service = await get_concurrent_processing_service()
        processing_stats = concurrent_service.get_processing_stats()
        
        # Get Celery worker stats
        worker_stats = task_manager.get_worker_stats()
        active_tasks = task_manager.get_active_tasks()
        
        return {
            "concurrent_processing": processing_stats,
            "celery_workers": len(worker_stats) if worker_stats else 0,
            "active_tasks": len(active_tasks) if active_tasks else 0,
            "worker_stats": worker_stats
        }
        
    except Exception as e:
        logger.error("Failed to get processing statistics", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to retrieve processing statistics")


@router.get("/retry/stats")
async def get_retry_statistics(
    platform: Optional[str] = None,
    current_user: UserResponse = Depends(get_current_user)
) -> Dict[str, Any]:
    """Get intelligent retry statistics"""
    try:
        retry_service = get_intelligent_retry_service()
        stats = await retry_service.get_retry_statistics(platform)
        
        return {
            "retry_statistics": stats,
            "platform_filter": platform
        }
        
    except Exception as e:
        logger.error("Failed to get retry statistics", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to retrieve retry statistics")


@router.post("/retry/configure")
async def configure_retry_behavior(
    request: RetryConfigurationRequest,
    current_user: UserResponse = Depends(get_current_user)
) -> Dict[str, Any]:
    """Configure retry behavior for a specific platform"""
    try:
        from app.services.intelligent_retry_service import RetryConfig, RetryStrategy
        
        retry_service = get_intelligent_retry_service()
        
        # Create retry configuration
        config = RetryConfig(
            max_retries=request.max_retries,
            base_delay=request.base_delay,
            max_delay=request.max_delay,
            strategy=RetryStrategy(request.strategy),
            jitter=request.jitter
        )
        
        # Configure platform
        retry_service.configure_platform(request.platform, config)
        
        return {
            "status": "success",
            "platform": request.platform,
            "configuration": {
                "max_retries": request.max_retries,
                "base_delay": request.base_delay,
                "max_delay": request.max_delay,
                "strategy": request.strategy,
                "jitter": request.jitter
            }
        }
        
    except Exception as e:
        logger.error("Failed to configure retry behavior", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to configure retry behavior")


@router.get("/monitoring/metrics")
async def get_performance_metrics(
    duration_minutes: int = 60,
    current_user: UserResponse = Depends(get_current_user)
) -> Dict[str, Any]:
    """Get performance monitoring metrics"""
    try:
        monitor = get_performance_monitor()
        
        # Get current metrics
        current_metrics = monitor.get_current_metrics()
        
        # Get performance summary
        summary = monitor.get_performance_summary()
        
        # Get metrics history
        system_history = monitor.get_metrics_history("system", duration_minutes)
        cache_history = monitor.get_metrics_history("cache", duration_minutes)
        
        return {
            "current_metrics": current_metrics,
            "performance_summary": summary,
            "history": {
                "system": [
                    {
                        "timestamp": metric.timestamp.isoformat(),
                        "cpu_percent": metric.cpu_percent,
                        "memory_percent": metric.memory_percent,
                        "memory_used_mb": metric.memory_used_mb
                    }
                    for metric in system_history
                ],
                "cache": [
                    {
                        "timestamp": metric.timestamp.isoformat(),
                        "hit_rate": metric.hit_rate,
                        "memory_usage_mb": metric.memory_usage_mb,
                        "key_count": metric.key_count
                    }
                    for metric in cache_history
                ]
            },
            "duration_minutes": duration_minutes
        }
        
    except Exception as e:
        logger.error("Failed to get performance metrics", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to retrieve performance metrics")


@router.post("/monitoring/start")
async def start_performance_monitoring(
    current_user: UserResponse = Depends(get_current_user)
) -> Dict[str, Any]:
    """Start performance monitoring"""
    try:
        monitor = get_performance_monitor()
        monitor.start_monitoring()
        
        return {
            "status": "success",
            "message": "Performance monitoring started",
            "monitoring_active": monitor.is_monitoring
        }
        
    except Exception as e:
        logger.error("Failed to start performance monitoring", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to start performance monitoring")


@router.post("/monitoring/stop")
async def stop_performance_monitoring(
    current_user: UserResponse = Depends(get_current_user)
) -> Dict[str, Any]:
    """Stop performance monitoring"""
    try:
        monitor = get_performance_monitor()
        monitor.stop_monitoring()
        
        return {
            "status": "success",
            "message": "Performance monitoring stopped",
            "monitoring_active": monitor.is_monitoring
        }
        
    except Exception as e:
        logger.error("Failed to stop performance monitoring", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to stop performance monitoring")


@router.get("/tasks/status/{task_id}")
async def get_task_status(
    task_id: str,
    current_user: UserResponse = Depends(get_current_user)
) -> Dict[str, Any]:
    """Get status of a background task"""
    try:
        task_status = task_manager.get_task_status(task_id)
        return task_status
        
    except Exception as e:
        logger.error("Failed to get task status", task_id=task_id, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to retrieve task status")


@router.post("/tasks/cancel/{task_id}")
async def cancel_task(
    task_id: str,
    current_user: UserResponse = Depends(get_current_user)
) -> Dict[str, Any]:
    """Cancel a background task"""
    try:
        success = task_manager.cancel_task(task_id)
        
        return {
            "status": "success" if success else "failed",
            "task_id": task_id,
            "cancelled": success
        }
        
    except Exception as e:
        logger.error("Failed to cancel task", task_id=task_id, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to cancel task")


@router.get("/health")
async def performance_health_check() -> Dict[str, Any]:
    """Health check for performance optimization services"""
    try:
        health_status = {
            "cache_service": "unknown",
            "concurrent_processing": "unknown",
            "retry_service": "unknown",
            "performance_monitoring": "unknown",
            "background_jobs": "unknown"
        }
        
        # Check cache service
        try:
            cache_service = await get_cache_service()
            cache_stats = cache_service.get_stats()
            health_status["cache_service"] = "healthy"
        except Exception:
            health_status["cache_service"] = "unhealthy"
        
        # Check concurrent processing
        try:
            concurrent_service = await get_concurrent_processing_service()
            processing_stats = concurrent_service.get_processing_stats()
            health_status["concurrent_processing"] = "healthy"
        except Exception:
            health_status["concurrent_processing"] = "unhealthy"
        
        # Check retry service
        try:
            retry_service = get_intelligent_retry_service()
            retry_stats = await retry_service.get_retry_statistics()
            health_status["retry_service"] = "healthy"
        except Exception:
            health_status["retry_service"] = "unhealthy"
        
        # Check performance monitoring
        try:
            monitor = get_performance_monitor()
            summary = monitor.get_performance_summary()
            health_status["performance_monitoring"] = "healthy"
        except Exception:
            health_status["performance_monitoring"] = "unhealthy"
        
        # Check background jobs
        try:
            from app.core.celery_app import celery_health_check
            celery_health = celery_health_check()
            health_status["background_jobs"] = celery_health["status"]
        except Exception:
            health_status["background_jobs"] = "unhealthy"
        
        overall_status = "healthy" if all(
            status == "healthy" for status in health_status.values()
        ) else "degraded"
        
        return {
            "overall_status": overall_status,
            "services": health_status,
            "timestamp": "2024-01-01T00:00:00Z"  # Would use actual timestamp
        }
        
    except Exception as e:
        logger.error("Performance health check failed", error=str(e))
        raise HTTPException(status_code=500, detail="Health check failed")