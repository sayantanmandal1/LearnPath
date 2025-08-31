"""
Pipeline Automation Initializer
Initializes and starts the data pipeline automation system.
"""

import asyncio
from datetime import datetime
from typing import Dict, List, Any

from app.core.logging import get_logger
from app.services.data_pipeline.pipeline_scheduler import get_pipeline_scheduler, PipelineJob
from app.services.data_pipeline.pipeline_config import get_config_manager, DefaultPipelineSchedules
from app.services.data_pipeline.pipeline_monitor import get_pipeline_monitor

logger = get_logger(__name__)


class PipelineInitializer:
    """
    Initializes the data pipeline automation system with default schedules
    """
    
    def __init__(self):
        self.config_manager = get_config_manager()
        self.default_schedules = DefaultPipelineSchedules()
        
    async def initialize_pipeline_system(self):
        """Initialize the complete pipeline automation system"""
        try:
            logger.info("Initializing pipeline automation system")
            
            # Initialize scheduler
            scheduler = await get_pipeline_scheduler()
            await scheduler.initialize()
            
            # Initialize monitor
            monitor = await get_pipeline_monitor()
            await monitor.initialize()
            
            # Set up default pipeline schedules
            await self._setup_default_schedules()
            
            # Start the scheduler
            await scheduler.start()
            
            logger.info("Pipeline automation system initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize pipeline system: {e}")
            raise
    
    async def _setup_default_schedules(self):
        """Set up default pipeline schedules"""
        try:
            scheduler = await get_pipeline_scheduler()
            
            # Define default pipeline jobs
            default_jobs = [
                {
                    'job_id': 'job_collection_daily',
                    'pipeline_name': 'job_collection',
                    'description': 'Daily job posting collection',
                    **self.default_schedules.get_job_collection_schedule()
                },
                {
                    'job_id': 'skill_taxonomy_weekly',
                    'pipeline_name': 'skill_taxonomy_update',
                    'description': 'Weekly skill taxonomy update',
                    **self.default_schedules.get_skill_taxonomy_schedule()
                },
                {
                    'job_id': 'profile_refresh_6h',
                    'pipeline_name': 'profile_refresh',
                    'description': 'Profile refresh every 6 hours',
                    **self.default_schedules.get_profile_refresh_schedule()
                },
                {
                    'job_id': 'model_training_weekly',
                    'pipeline_name': 'model_training',
                    'description': 'Weekly model retraining',
                    **self.default_schedules.get_model_training_schedule()
                },
                {
                    'job_id': 'data_quality_check_4h',
                    'pipeline_name': 'data_quality_check',
                    'description': 'Data quality checks every 4 hours',
                    **self.default_schedules.get_data_quality_check_schedule()
                },
                {
                    'job_id': 'backup_daily',
                    'pipeline_name': 'backup',
                    'description': 'Daily system backup',
                    **self.default_schedules.get_backup_schedule()
                }
            ]
            
            # Schedule each default job
            for job_config in default_jobs:
                try:
                    # Check if job already exists
                    existing_status = await scheduler.get_job_status(job_config['job_id'])
                    
                    if not existing_status:
                        # Create new pipeline job
                        pipeline_job = PipelineJob(
                            job_id=job_config['job_id'],
                            pipeline_name=job_config['pipeline_name'],
                            schedule_type=job_config['schedule_type'],
                            schedule_config=job_config['schedule_config'],
                            enabled=True,
                            max_retries=job_config['max_retries'],
                            retry_delay=job_config['retry_delay'],
                            timeout=job_config['timeout'],
                            metadata={'description': job_config['description']}
                        )
                        
                        success = await scheduler.schedule_job(pipeline_job)
                        
                        if success:
                            logger.info(f"Scheduled default job: {job_config['job_id']}")
                        else:
                            logger.warning(f"Failed to schedule default job: {job_config['job_id']}")
                    else:
                        logger.info(f"Default job already exists: {job_config['job_id']}")
                        
                except Exception as e:
                    logger.error(f"Failed to schedule job {job_config['job_id']}: {e}")
                    continue
            
            logger.info("Default pipeline schedules set up successfully")
            
        except Exception as e:
            logger.error(f"Failed to setup default schedules: {e}")
            raise
    
    async def shutdown_pipeline_system(self):
        """Gracefully shutdown the pipeline system"""
        try:
            logger.info("Shutting down pipeline automation system")
            
            scheduler = await get_pipeline_scheduler()
            await scheduler.stop()
            
            logger.info("Pipeline automation system shut down successfully")
            
        except Exception as e:
            logger.error(f"Failed to shutdown pipeline system: {e}")
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on the pipeline system"""
        try:
            health_status = {
                'timestamp': datetime.utcnow().isoformat(),
                'scheduler_status': 'unknown',
                'monitor_status': 'unknown',
                'scheduled_jobs_count': 0,
                'active_executions': 0,
                'overall_health': 'unhealthy'
            }
            
            # Check scheduler
            try:
                scheduler = await get_pipeline_scheduler()
                if scheduler.scheduler and scheduler.scheduler.running:
                    health_status['scheduler_status'] = 'running'
                    
                    # Get scheduled jobs count
                    jobs = await scheduler.get_scheduled_jobs()
                    health_status['scheduled_jobs_count'] = len(jobs)
                else:
                    health_status['scheduler_status'] = 'stopped'
            except Exception as e:
                health_status['scheduler_status'] = f'error: {str(e)}'
            
            # Check monitor
            try:
                monitor = await get_pipeline_monitor()
                health_status['monitor_status'] = 'running'
                health_status['active_executions'] = len(monitor.active_monitors)
            except Exception as e:
                health_status['monitor_status'] = f'error: {str(e)}'
            
            # Determine overall health
            if (health_status['scheduler_status'] == 'running' and 
                health_status['monitor_status'] == 'running'):
                health_status['overall_health'] = 'healthy'
            elif (health_status['scheduler_status'] in ['running', 'stopped'] and 
                  health_status['monitor_status'] == 'running'):
                health_status['overall_health'] = 'degraded'
            
            return health_status
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                'timestamp': datetime.utcnow().isoformat(),
                'overall_health': 'unhealthy',
                'error': str(e)
            }
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        try:
            # Get health check
            health = await self.health_check()
            
            # Get system metrics
            monitor = await get_pipeline_monitor()
            system_health = await monitor.get_system_health()
            
            # Combine status information
            status = {
                'health_check': health,
                'system_metrics': system_health,
                'timestamp': datetime.utcnow().isoformat()
            }
            
            return status
            
        except Exception as e:
            logger.error(f"Failed to get system status: {e}")
            return {
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }
    
    async def restart_pipeline_system(self):
        """Restart the pipeline system"""
        try:
            logger.info("Restarting pipeline automation system")
            
            # Shutdown current system
            await self.shutdown_pipeline_system()
            
            # Wait a moment
            await asyncio.sleep(2)
            
            # Initialize system again
            await self.initialize_pipeline_system()
            
            logger.info("Pipeline automation system restarted successfully")
            
        except Exception as e:
            logger.error(f"Failed to restart pipeline system: {e}")
            raise
    
    async def update_job_schedule(self, job_id: str, new_schedule: Dict[str, Any]):
        """Update schedule for an existing job"""
        try:
            scheduler = await get_pipeline_scheduler()
            
            # Get current job configuration
            current_config = await scheduler.get_job_config(job_id)
            if not current_config:
                raise ValueError(f"Job {job_id} not found")
            
            # Remove existing job
            await scheduler.unschedule_job(job_id)
            
            # Create updated job
            updated_job = PipelineJob(
                job_id=job_id,
                pipeline_name=current_config['pipeline_name'],
                schedule_type=new_schedule['schedule_type'],
                schedule_config=new_schedule['schedule_config'],
                enabled=new_schedule.get('enabled', True),
                max_retries=new_schedule.get('max_retries', current_config['max_retries']),
                retry_delay=new_schedule.get('retry_delay', current_config['retry_delay']),
                timeout=new_schedule.get('timeout', current_config['timeout']),
                metadata=current_config.get('metadata', {})
            )
            
            # Schedule updated job
            success = await scheduler.schedule_job(updated_job)
            
            if success:
                logger.info(f"Updated schedule for job: {job_id}")
                return True
            else:
                logger.error(f"Failed to update schedule for job: {job_id}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to update job schedule: {e}")
            raise
    
    async def pause_job(self, job_id: str):
        """Pause a scheduled job"""
        try:
            scheduler = await get_pipeline_scheduler()
            
            # Get job from scheduler
            job = scheduler.scheduler.get_job(job_id)
            if job:
                job.pause()
                logger.info(f"Paused job: {job_id}")
                return True
            else:
                logger.warning(f"Job not found: {job_id}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to pause job: {e}")
            raise
    
    async def resume_job(self, job_id: str):
        """Resume a paused job"""
        try:
            scheduler = await get_pipeline_scheduler()
            
            # Get job from scheduler
            job = scheduler.scheduler.get_job(job_id)
            if job:
                job.resume()
                logger.info(f"Resumed job: {job_id}")
                return True
            else:
                logger.warning(f"Job not found: {job_id}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to resume job: {e}")
            raise


# Global initializer instance
pipeline_initializer = PipelineInitializer()


async def get_pipeline_initializer() -> PipelineInitializer:
    """Get the global pipeline initializer instance"""
    return pipeline_initializer


async def initialize_pipeline_automation():
    """Initialize the pipeline automation system"""
    initializer = await get_pipeline_initializer()
    await initializer.initialize_pipeline_system()


async def shutdown_pipeline_automation():
    """Shutdown the pipeline automation system"""
    initializer = await get_pipeline_initializer()
    await initializer.shutdown_pipeline_system()