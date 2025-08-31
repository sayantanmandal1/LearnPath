"""
Celery configuration for background job processing
"""
import os
from celery import Celery
from kombu import Queue
import structlog

from app.core.config import settings

logger = structlog.get_logger()

# Create Celery instance
celery_app = Celery(
    "ai_career_recommender",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
    include=[
        "app.tasks.ml_tasks",
        "app.tasks.data_collection_tasks",
        "app.tasks.cache_tasks",
        "app.tasks.analytics_tasks",
    ]
)

# Celery configuration
celery_app.conf.update(
    # Task routing
    task_routes={
        "app.tasks.ml_tasks.*": {"queue": "ml_queue"},
        "app.tasks.data_collection_tasks.*": {"queue": "data_queue"},
        "app.tasks.cache_tasks.*": {"queue": "cache_queue"},
        "app.tasks.analytics_tasks.*": {"queue": "analytics_queue"},
    },
    
    # Queue configuration
    task_default_queue="default",
    task_queues=(
        Queue("default", routing_key="default"),
        Queue("ml_queue", routing_key="ml_queue"),
        Queue("data_queue", routing_key="data_queue"),
        Queue("cache_queue", routing_key="cache_queue"),
        Queue("analytics_queue", routing_key="analytics_queue"),
    ),
    
    # Task execution settings
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    
    # Task result settings
    result_expires=3600,  # 1 hour
    result_backend_transport_options={
        "master_name": "mymaster",
        "visibility_timeout": 3600,
    },
    
    # Worker settings
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    worker_max_tasks_per_child=1000,
    worker_concurrency=settings.CELERY_WORKER_CONCURRENCY,
    task_soft_time_limit=settings.CELERY_TASK_SOFT_TIME_LIMIT,
    task_time_limit=settings.CELERY_TASK_TIME_LIMIT,
    
    # Task retry settings
    task_default_retry_delay=60,  # 1 minute
    task_max_retries=3,
    
    # Monitoring
    worker_send_task_events=True,
    task_send_sent_event=True,
    
    # Performance settings
    task_compression="gzip",
    result_compression="gzip",
    
    # Security
    task_reject_on_worker_lost=True,
    task_ignore_result=False,
    
    # Beat schedule for periodic tasks
    beat_schedule={
        "refresh-job-market-data": {
            "task": "app.tasks.data_collection_tasks.refresh_job_market_data",
            "schedule": 3600.0,  # Every hour
        },
        "cleanup-expired-cache": {
            "task": "app.tasks.cache_tasks.cleanup_expired_cache",
            "schedule": 1800.0,  # Every 30 minutes
        },
        "update-ml-models": {
            "task": "app.tasks.ml_tasks.update_ml_models",
            "schedule": 86400.0,  # Daily
        },
        "generate-analytics-reports": {
            "task": "app.tasks.analytics_tasks.generate_daily_analytics",
            "schedule": 86400.0,  # Daily
        },
    },
)

# Task priority levels
TASK_PRIORITIES = {
    "HIGH": 9,
    "MEDIUM": 5,
    "LOW": 1,
}


class CeleryTaskManager:
    """Manager for Celery task operations"""
    
    @staticmethod
    def get_task_status(task_id: str) -> dict:
        """Get status of a specific task"""
        try:
            result = celery_app.AsyncResult(task_id)
            return {
                "task_id": task_id,
                "status": result.status,
                "result": result.result if result.ready() else None,
                "traceback": result.traceback if result.failed() else None,
            }
        except Exception as e:
            logger.error("Failed to get task status", task_id=task_id, error=str(e))
            return {"task_id": task_id, "status": "UNKNOWN", "error": str(e)}
    
    @staticmethod
    def cancel_task(task_id: str) -> bool:
        """Cancel a running task"""
        try:
            celery_app.control.revoke(task_id, terminate=True)
            return True
        except Exception as e:
            logger.error("Failed to cancel task", task_id=task_id, error=str(e))
            return False
    
    @staticmethod
    def get_active_tasks() -> list:
        """Get list of active tasks"""
        try:
            inspect = celery_app.control.inspect()
            active_tasks = inspect.active()
            return active_tasks or []
        except Exception as e:
            logger.error("Failed to get active tasks", error=str(e))
            return []
    
    @staticmethod
    def get_worker_stats() -> dict:
        """Get worker statistics"""
        try:
            inspect = celery_app.control.inspect()
            stats = inspect.stats()
            return stats or {}
        except Exception as e:
            logger.error("Failed to get worker stats", error=str(e))
            return {}
    
    @staticmethod
    def purge_queue(queue_name: str) -> int:
        """Purge all tasks from a specific queue"""
        try:
            return celery_app.control.purge()
        except Exception as e:
            logger.error("Failed to purge queue", queue=queue_name, error=str(e))
            return 0


# Task manager instance
task_manager = CeleryTaskManager()


def create_task_signature(task_name: str, args: tuple = (), kwargs: dict = None, **options):
    """Create a task signature for delayed execution"""
    kwargs = kwargs or {}
    return celery_app.signature(task_name, args=args, kwargs=kwargs, **options)


def chain_tasks(*tasks):
    """Chain multiple tasks to execute sequentially"""
    from celery import chain
    return chain(*tasks)


def group_tasks(*tasks):
    """Group multiple tasks to execute in parallel"""
    from celery import group
    return group(*tasks)


def chord_tasks(header, callback):
    """Execute tasks in parallel and call callback with results"""
    from celery import chord
    return chord(header)(callback)


# Health check for Celery
def celery_health_check() -> dict:
    """Check Celery health status"""
    try:
        # Check if broker is accessible
        inspect = celery_app.control.inspect()
        stats = inspect.stats()
        
        if stats:
            return {
                "status": "healthy",
                "workers": len(stats),
                "broker": "connected"
            }
        else:
            return {
                "status": "unhealthy",
                "workers": 0,
                "broker": "disconnected"
            }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "broker": "error"
        }