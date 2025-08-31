"""
Data Pipeline Scheduler
Manages automated scheduling and execution of data collection and processing pipelines.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.jobstores.redis import RedisJobStore
from apscheduler.executors.asyncio import AsyncIOExecutor

from app.core.redis import get_redis
from app.core.logging import get_logger
from app.services.data_pipeline.pipeline_monitor import PipelineMonitor
from app.services.data_pipeline.pipeline_config import PipelineConfig

logger = get_logger(__name__)


class PipelineStatus(Enum):
    SCHEDULED = "scheduled"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class PipelineJob:
    """Represents a scheduled pipeline job"""
    job_id: str
    pipeline_name: str
    schedule_type: str  # 'cron', 'interval'
    schedule_config: Dict[str, Any]
    enabled: bool = True
    max_retries: int = 3
    retry_delay: int = 300  # seconds
    timeout: int = 3600  # seconds
    metadata: Dict[str, Any] = None


class PipelineScheduler:
    """
    Manages automated scheduling and execution of data pipelines
    """
    
    def __init__(self):
        self.scheduler = None
        self.monitor = PipelineMonitor()
        self.config = PipelineConfig()
        self.redis_client = None
        self.running_jobs: Dict[str, Dict] = {}
        
    async def initialize(self):
        """Initialize the scheduler with Redis backend"""
        try:
            redis_manager = await get_redis()
            self.redis_client = redis_manager.redis
            
            # Configure job store and executor
            jobstores = {
                'default': RedisJobStore(
                    host=self.config.redis_host,
                    port=self.config.redis_port,
                    db=self.config.redis_db,
                    password=self.config.redis_password
                )
            }
            
            executors = {
                'default': AsyncIOExecutor()
            }
            
            job_defaults = {
                'coalesce': False,
                'max_instances': 1,
                'misfire_grace_time': 300
            }
            
            self.scheduler = AsyncIOScheduler(
                jobstores=jobstores,
                executors=executors,
                job_defaults=job_defaults,
                timezone='UTC'
            )
            
            logger.info("Pipeline scheduler initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize pipeline scheduler: {e}")
            raise
    
    async def start(self):
        """Start the scheduler"""
        if not self.scheduler:
            await self.initialize()
        
        self.scheduler.start()
        logger.info("Pipeline scheduler started")
        
        # Load and schedule existing jobs
        await self.load_scheduled_jobs()
    
    async def stop(self):
        """Stop the scheduler gracefully"""
        if self.scheduler:
            self.scheduler.shutdown(wait=True)
            logger.info("Pipeline scheduler stopped")
    
    async def schedule_job(self, job: PipelineJob) -> bool:
        """Schedule a new pipeline job"""
        try:
            # Create trigger based on schedule type
            if job.schedule_type == 'cron':
                trigger = CronTrigger(**job.schedule_config)
            elif job.schedule_type == 'interval':
                trigger = IntervalTrigger(**job.schedule_config)
            else:
                raise ValueError(f"Unsupported schedule type: {job.schedule_type}")
            
            # Add job to scheduler
            self.scheduler.add_job(
                func=self._execute_pipeline_job,
                trigger=trigger,
                id=job.job_id,
                name=f"Pipeline: {job.pipeline_name}",
                args=[job],
                max_instances=1,
                coalesce=True,
                misfire_grace_time=300
            )
            
            # Store job configuration
            await self._store_job_config(job)
            
            logger.info(f"Scheduled job {job.job_id} for pipeline {job.pipeline_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to schedule job {job.job_id}: {e}")
            return False
    
    async def unschedule_job(self, job_id: str) -> bool:
        """Remove a scheduled job"""
        try:
            self.scheduler.remove_job(job_id)
            await self._remove_job_config(job_id)
            logger.info(f"Unscheduled job {job_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to unschedule job {job_id}: {e}")
            return False
    
    async def get_scheduled_jobs(self) -> List[Dict]:
        """Get list of all scheduled jobs"""
        jobs = []
        for job in self.scheduler.get_jobs():
            job_info = {
                'id': job.id,
                'name': job.name,
                'next_run_time': job.next_run_time.isoformat() if job.next_run_time else None,
                'trigger': str(job.trigger)
            }
            jobs.append(job_info)
        return jobs
    
    async def get_job_status(self, job_id: str) -> Optional[Dict]:
        """Get status of a specific job"""
        if job_id in self.running_jobs:
            return self.running_jobs[job_id]
        
        # Check Redis for historical data
        try:
            status_key = f"pipeline_job_status:{job_id}"
            status_data = await self.redis_client.get(status_key)
            if status_data:
                return json.loads(status_data)
        except Exception as e:
            logger.error(f"Failed to get job status for {job_id}: {e}")
        
        return None
    
    async def _execute_pipeline_job(self, job: PipelineJob):
        """Execute a pipeline job with monitoring and error handling"""
        execution_id = f"{job.job_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        job_status = {
            'job_id': job.job_id,
            'execution_id': execution_id,
            'pipeline_name': job.pipeline_name,
            'status': PipelineStatus.RUNNING.value,
            'start_time': datetime.utcnow().isoformat(),
            'end_time': None,
            'duration': None,
            'error': None,
            'retry_count': 0
        }
        
        self.running_jobs[job.job_id] = job_status
        
        try:
            # Start monitoring
            await self.monitor.start_job_monitoring(execution_id, job.pipeline_name)
            
            # Execute the pipeline
            await self._run_pipeline(job.pipeline_name, job.metadata or {})
            
            # Update status on success
            job_status.update({
                'status': PipelineStatus.COMPLETED.value,
                'end_time': datetime.utcnow().isoformat()
            })
            
            logger.info(f"Pipeline job {job.job_id} completed successfully")
            
        except Exception as e:
            logger.error(f"Pipeline job {job.job_id} failed: {e}")
            
            job_status.update({
                'status': PipelineStatus.FAILED.value,
                'end_time': datetime.utcnow().isoformat(),
                'error': str(e)
            })
            
            # Handle retries
            if job_status['retry_count'] < job.max_retries:
                await self._schedule_retry(job, job_status)
        
        finally:
            # Calculate duration
            if job_status['end_time']:
                start_time = datetime.fromisoformat(job_status['start_time'])
                end_time = datetime.fromisoformat(job_status['end_time'])
                job_status['duration'] = (end_time - start_time).total_seconds()
            
            # Store final status
            await self._store_job_status(job_status)
            
            # Stop monitoring
            await self.monitor.stop_job_monitoring(execution_id)
            
            # Remove from running jobs
            if job.job_id in self.running_jobs:
                del self.running_jobs[job.job_id]
    
    async def _run_pipeline(self, pipeline_name: str, metadata: Dict[str, Any]):
        """Execute the actual pipeline based on its name"""
        from app.services.data_pipeline.job_collection_pipeline import JobCollectionPipeline
        from app.services.data_pipeline.skill_taxonomy_pipeline import SkillTaxonomyPipeline
        from app.services.data_pipeline.profile_refresh_pipeline import ProfileRefreshPipeline
        from app.services.data_pipeline.model_training_pipeline import ModelTrainingPipeline
        
        pipelines = {
            'job_collection': JobCollectionPipeline(),
            'skill_taxonomy_update': SkillTaxonomyPipeline(),
            'profile_refresh': ProfileRefreshPipeline(),
            'model_training': ModelTrainingPipeline()
        }
        
        if pipeline_name not in pipelines:
            raise ValueError(f"Unknown pipeline: {pipeline_name}")
        
        pipeline = pipelines[pipeline_name]
        await pipeline.execute(metadata)
    
    async def _schedule_retry(self, job: PipelineJob, job_status: Dict):
        """Schedule a retry for a failed job"""
        retry_count = job_status['retry_count'] + 1
        retry_delay = job.retry_delay * (2 ** (retry_count - 1))  # Exponential backoff
        
        retry_time = datetime.utcnow() + timedelta(seconds=retry_delay)
        
        self.scheduler.add_job(
            func=self._execute_pipeline_job,
            trigger='date',
            run_date=retry_time,
            id=f"{job.job_id}_retry_{retry_count}",
            args=[job],
            max_instances=1
        )
        
        job_status['retry_count'] = retry_count
        logger.info(f"Scheduled retry {retry_count} for job {job.job_id} at {retry_time}")
    
    async def _store_job_config(self, job: PipelineJob):
        """Store job configuration in Redis"""
        try:
            config_key = f"pipeline_job_config:{job.job_id}"
            config_data = {
                'job_id': job.job_id,
                'pipeline_name': job.pipeline_name,
                'schedule_type': job.schedule_type,
                'schedule_config': job.schedule_config,
                'enabled': job.enabled,
                'max_retries': job.max_retries,
                'retry_delay': job.retry_delay,
                'timeout': job.timeout,
                'metadata': job.metadata
            }
            await self.redis_client.set(config_key, json.dumps(config_data))
        except Exception as e:
            logger.error(f"Failed to store job config for {job.job_id}: {e}")
    
    async def _remove_job_config(self, job_id: str):
        """Remove job configuration from Redis"""
        try:
            config_key = f"pipeline_job_config:{job_id}"
            await self.redis_client.delete(config_key)
        except Exception as e:
            logger.error(f"Failed to remove job config for {job_id}: {e}")
    
    async def _store_job_status(self, job_status: Dict):
        """Store job execution status in Redis"""
        try:
            status_key = f"pipeline_job_status:{job_status['job_id']}"
            await self.redis_client.set(
                status_key, 
                json.dumps(job_status),
                ex=86400 * 7  # Keep for 7 days
            )
        except Exception as e:
            logger.error(f"Failed to store job status: {e}")
    
    async def load_scheduled_jobs(self):
        """Load and schedule jobs from configuration"""
        try:
            # Get all job configurations from Redis
            pattern = "pipeline_job_config:*"
            keys = await self.redis_client.keys(pattern)
            
            for key in keys:
                try:
                    config_data = await self.redis_client.get(key)
                    if config_data:
                        config = json.loads(config_data)
                        if config.get('enabled', True):
                            job = PipelineJob(**config)
                            await self.schedule_job(job)
                except Exception as e:
                    logger.error(f"Failed to load job from {key}: {e}")
                    
        except Exception as e:
            logger.error(f"Failed to load scheduled jobs: {e}")


# Global scheduler instance
pipeline_scheduler = PipelineScheduler()


async def get_pipeline_scheduler() -> PipelineScheduler:
    """Get the global pipeline scheduler instance"""
    return pipeline_scheduler