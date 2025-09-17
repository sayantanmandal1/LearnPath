"""
Pipeline Automation API Endpoints
Provides REST API for managing data pipeline automation and scheduling.
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from typing import Dict, List, Any, Optional
from datetime import datetime
import json

from app.core.logging import get_logger
from app.services.data_pipeline.pipeline_scheduler import get_pipeline_scheduler, PipelineJob
from app.services.data_pipeline.pipeline_monitor import get_pipeline_monitor
from app.services.data_pipeline.pipeline_config import get_config_manager
from app.services.data_pipeline.data_quality_validator import DataQualityValidator
from app.services.data_pipeline.backup_recovery import BackupRecoveryManager
from app.schemas.pipeline import (
    PipelineJobCreate, PipelineJobResponse, PipelineExecutionResponse,
    DataQualityReport, BackupRequest, BackupResponse
)

logger = get_logger(__name__)
router = APIRouter()


@router.get("/jobs", response_model=List[PipelineJobResponse])
async def get_scheduled_jobs():
    """Get all scheduled pipeline jobs"""
    try:
        scheduler = await get_pipeline_scheduler()
        jobs = await scheduler.get_scheduled_jobs()
        
        return [
            PipelineJobResponse(
                job_id=job['id'],
                name=job['name'],
                next_run_time=job['next_run_time'],
                trigger=job['trigger'],
                status="scheduled"
            )
            for job in jobs
        ]
        
    except Exception as e:
        logger.error(f"Failed to get scheduled jobs: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/jobs", response_model=Dict[str, Any])
async def schedule_pipeline_job(job_data: PipelineJobCreate):
    """Schedule a new pipeline job"""
    try:
        scheduler = await get_pipeline_scheduler()
        
        # Create pipeline job
        pipeline_job = PipelineJob(
            job_id=job_data.job_id,
            pipeline_name=job_data.pipeline_name,
            schedule_type=job_data.schedule_type,
            schedule_config=job_data.schedule_config,
            enabled=job_data.enabled,
            max_retries=job_data.max_retries,
            retry_delay=job_data.retry_delay,
            timeout=job_data.timeout,
            metadata=job_data.metadata
        )
        
        success = await scheduler.schedule_job(pipeline_job)
        
        if success:
            return {
                "success": True,
                "message": f"Pipeline job {job_data.job_id} scheduled successfully",
                "job_id": job_data.job_id
            }
        else:
            raise HTTPException(status_code=400, detail="Failed to schedule job")
            
    except Exception as e:
        logger.error(f"Failed to schedule pipeline job: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/jobs/{job_id}")
async def unschedule_pipeline_job(job_id: str):
    """Remove a scheduled pipeline job"""
    try:
        scheduler = await get_pipeline_scheduler()
        success = await scheduler.unschedule_job(job_id)
        
        if success:
            return {
                "success": True,
                "message": f"Pipeline job {job_id} unscheduled successfully"
            }
        else:
            raise HTTPException(status_code=404, detail="Job not found")
            
    except Exception as e:
        logger.error(f"Failed to unschedule pipeline job: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/jobs/{job_id}/status")
async def get_job_status(job_id: str):
    """Get status of a specific pipeline job"""
    try:
        scheduler = await get_pipeline_scheduler()
        status = await scheduler.get_job_status(job_id)
        
        if status:
            return status
        else:
            raise HTTPException(status_code=404, detail="Job status not found")
            
    except Exception as e:
        logger.error(f"Failed to get job status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/jobs/{job_id}/execute")
async def execute_pipeline_job(job_id: str, background_tasks: BackgroundTasks):
    """Execute a pipeline job immediately"""
    try:
        scheduler = await get_pipeline_scheduler()
        
        # Get job configuration
        job_config = await scheduler.get_job_config(job_id)
        if not job_config:
            raise HTTPException(status_code=404, detail="Job not found")
        
        # Create execution metadata
        execution_id = f"{job_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        metadata = {
            'execution_id': execution_id,
            'triggered_manually': True,
            **(job_config.get('metadata', {}))
        }
        
        # Execute in background
        background_tasks.add_task(
            scheduler._run_pipeline,
            job_config['pipeline_name'],
            metadata
        )
        
        return {
            "success": True,
            "message": f"Pipeline job {job_id} execution started",
            "execution_id": execution_id
        }
        
    except Exception as e:
        logger.error(f"Failed to execute pipeline job: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/executions/{execution_id}")
async def get_execution_status(execution_id: str):
    """Get status of a pipeline execution"""
    try:
        monitor = await get_pipeline_monitor()
        metrics = await monitor.get_job_metrics(execution_id)
        
        if metrics:
            return {
                "execution_id": execution_id,
                "pipeline_name": metrics.pipeline_name,
                "status": metrics.status,
                "start_time": metrics.start_time.isoformat(),
                "end_time": metrics.end_time.isoformat() if metrics.end_time else None,
                "duration_seconds": metrics.duration_seconds,
                "records_processed": metrics.records_processed,
                "records_failed": metrics.records_failed,
                "data_quality_score": metrics.data_quality_score,
                "error_count": metrics.error_count
            }
        else:
            raise HTTPException(status_code=404, detail="Execution not found")
            
    except Exception as e:
        logger.error(f"Failed to get execution status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/pipelines/{pipeline_name}/history")
async def get_pipeline_history(pipeline_name: str, days: int = 7):
    """Get execution history for a pipeline"""
    try:
        monitor = await get_pipeline_monitor()
        history = await monitor.get_pipeline_history(pipeline_name, days)
        
        return [
            {
                "execution_id": metrics.execution_id,
                "start_time": metrics.start_time.isoformat(),
                "end_time": metrics.end_time.isoformat() if metrics.end_time else None,
                "duration_seconds": metrics.duration_seconds,
                "status": metrics.status,
                "records_processed": metrics.records_processed,
                "records_failed": metrics.records_failed,
                "data_quality_score": metrics.data_quality_score
            }
            for metrics in history
        ]
        
    except Exception as e:
        logger.error(f"Failed to get pipeline history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/system/health")
async def get_system_health():
    """Get overall system health metrics"""
    try:
        monitor = await get_pipeline_monitor()
        health = await monitor.get_system_health()
        
        return health
        
    except Exception as e:
        logger.error(f"Failed to get system health: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/data-quality/report", response_model=DataQualityReport)
async def get_data_quality_report():
    """Get comprehensive data quality report"""
    try:
        validator = DataQualityValidator()
        report = await validator.run_data_quality_checks()
        
        return DataQualityReport(
            timestamp=report['timestamp'],
            overall_quality_score=report['overall_quality_score'],
            checks=report['checks'],
            issues=report['issues'],
            warnings=report['warnings'],
            recommendations=report['recommendations']
        )
        
    except Exception as e:
        logger.error(f"Failed to get data quality report: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/data-quality/history")
async def get_data_quality_history(days: int = 30):
    """Get data quality history"""
    try:
        validator = DataQualityValidator()
        history = await validator.get_quality_history(days)
        
        return history
        
    except Exception as e:
        logger.error(f"Failed to get data quality history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/backup", response_model=BackupResponse)
async def create_backup(backup_request: BackupRequest, background_tasks: BackgroundTasks):
    """Create a system backup"""
    try:
        backup_manager = BackupRecoveryManager()
        
        # Generate backup ID
        backup_id = f"backup_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        metadata = {
            'execution_id': backup_id,
            'backup_type': backup_request.backup_type,
            'include_ml_models': backup_request.include_ml_models,
            'compress_backup': backup_request.compress_backup
        }
        
        # Execute backup in background
        background_tasks.add_task(
            backup_manager.execute_backup,
            metadata
        )
        
        return BackupResponse(
            backup_id=backup_id,
            status="started",
            message="Backup operation started",
            timestamp=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Failed to create backup: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/backup/history")
async def get_backup_history(limit: int = 20):
    """Get backup history"""
    try:
        backup_manager = BackupRecoveryManager()
        history = await backup_manager.get_backup_history(limit)
        
        return history
        
    except Exception as e:
        logger.error(f"Failed to get backup history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/backup/{backup_id}/restore")
async def restore_from_backup(backup_id: str, background_tasks: BackgroundTasks, components: List[str] = None):
    """Restore system from backup"""
    try:
        backup_manager = BackupRecoveryManager()
        
        # Execute restore in background
        background_tasks.add_task(
            backup_manager.restore_from_backup,
            backup_id,
            components
        )
        
        return {
            "success": True,
            "message": f"Restore operation from backup {backup_id} started",
            "backup_id": backup_id
        }
        
    except Exception as e:
        logger.error(f"Failed to restore from backup: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/backup/{backup_id}/verify")
async def verify_backup_integrity(backup_id: str):
    """Verify backup integrity"""
    try:
        backup_manager = BackupRecoveryManager()
        verification_result = await backup_manager.verify_backup_integrity(backup_id)
        
        return verification_result
        
    except Exception as e:
        logger.error(f"Failed to verify backup integrity: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/config/schedules")
async def get_pipeline_schedules():
    """Get all pipeline schedule configurations"""
    try:
        config_manager = get_config_manager()
        schedules = config_manager.get_all_pipeline_configs()
        
        return schedules
        
    except Exception as e:
        logger.error(f"Failed to get pipeline schedules: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/config/schedules/{pipeline_name}")
async def update_pipeline_schedule(pipeline_name: str, schedule_config: Dict[str, Any]):
    """Update schedule configuration for a pipeline"""
    try:
        config_manager = get_config_manager()
        config_manager.update_schedule_config(pipeline_name, schedule_config)
        
        # Reschedule the job if it exists
        scheduler = await get_pipeline_scheduler()
        job_id = f"{pipeline_name}_scheduled"
        
        # Remove existing job
        await scheduler.unschedule_job(job_id)
        
        # Create new job with updated config
        pipeline_job = PipelineJob(
            job_id=job_id,
            pipeline_name=pipeline_name,
            schedule_type=schedule_config['schedule_type'],
            schedule_config=schedule_config['schedule_config'],
            enabled=schedule_config.get('enabled', True),
            max_retries=schedule_config.get('max_retries', 3),
            retry_delay=schedule_config.get('retry_delay', 300),
            timeout=schedule_config.get('timeout', 3600)
        )
        
        await scheduler.schedule_job(pipeline_job)
        
        return {
            "success": True,
            "message": f"Schedule updated for pipeline {pipeline_name}"
        }
        
    except Exception as e:
        logger.error(f"Failed to update pipeline schedule: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats/job-collection")
async def get_job_collection_stats(days: int = 7):
    """Get job collection pipeline statistics"""
    try:
        from app.services.data_pipeline.job_collection_pipeline import JobCollectionPipeline
        
        pipeline = JobCollectionPipeline()
        stats = await pipeline.get_collection_stats(days)
        
        return stats
        
    except Exception as e:
        logger.error(f"Failed to get job collection stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats/skill-taxonomy")
async def get_skill_taxonomy_stats(months: int = 6):
    """Get skill taxonomy update statistics"""
    try:
        from app.services.data_pipeline.skill_taxonomy_pipeline import SkillTaxonomyPipeline
        
        pipeline = SkillTaxonomyPipeline()
        stats = await pipeline.get_taxonomy_stats(months)
        
        return stats
        
    except Exception as e:
        logger.error(f"Failed to get skill taxonomy stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats/profile-refresh")
async def get_profile_refresh_stats(days: int = 7):
    """Get profile refresh pipeline statistics"""
    try:
        from app.services.data_pipeline.profile_refresh_pipeline import ProfileRefreshPipeline
        
        pipeline = ProfileRefreshPipeline()
        stats = await pipeline.get_refresh_stats(days)
        
        return stats
        
    except Exception as e:
        logger.error(f"Failed to get profile refresh stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats/model-training")
async def get_model_training_stats(months: int = 6):
    """Get model training pipeline statistics"""
    try:
        from app.services.data_pipeline.model_training_pipeline import ModelTrainingPipeline
        
        pipeline = ModelTrainingPipeline()
        stats = await pipeline.get_training_stats(months)
        
        return stats
        
    except Exception as e:
        logger.error(f"Failed to get model training stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/pipelines/{pipeline_name}/trigger")
async def trigger_pipeline(pipeline_name: str, background_tasks: BackgroundTasks, metadata: Dict[str, Any] = None):
    """Trigger a specific pipeline manually"""
    try:
        scheduler = await get_pipeline_scheduler()
        
        # Create execution metadata
        execution_id = f"{pipeline_name}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        execution_metadata = {
            'execution_id': execution_id,
            'triggered_manually': True,
            **(metadata or {})
        }
        
        # Execute pipeline in background
        background_tasks.add_task(
            scheduler._run_pipeline,
            pipeline_name,
            execution_metadata
        )
        
        return {
            "success": True,
            "message": f"Pipeline {pipeline_name} triggered successfully",
            "execution_id": execution_id
        }
        
    except Exception as e:
        logger.error(f"Failed to trigger pipeline: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/models/{model_name}/retrain")
async def trigger_model_retraining(model_name: str, background_tasks: BackgroundTasks, force: bool = False):
    """Trigger retraining for a specific model"""
    try:
        from app.services.data_pipeline.model_training_pipeline import ModelTrainingPipeline
        
        pipeline = ModelTrainingPipeline()
        
        # Execute retraining in background
        background_tasks.add_task(
            pipeline.trigger_model_retraining,
            model_name,
            force
        )
        
        return {
            "success": True,
            "message": f"Model {model_name} retraining triggered",
            "model_name": model_name,
            "forced": force
        }
        
    except Exception as e:
        logger.error(f"Failed to trigger model retraining: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/profiles/{profile_id}/refresh")
async def refresh_single_profile(profile_id: str, background_tasks: BackgroundTasks, platforms: List[str] = None):
    """Refresh a single user profile"""
    try:
        from app.services.data_pipeline.profile_refresh_pipeline import ProfileRefreshPipeline
        
        pipeline = ProfileRefreshPipeline()
        
        # Execute refresh in background
        background_tasks.add_task(
            pipeline.refresh_single_profile,
            profile_id,
            platforms
        )
        
        return {
            "success": True,
            "message": f"Profile {profile_id} refresh triggered",
            "profile_id": profile_id,
            "platforms": platforms or ['github', 'linkedin', 'leetcode']
        }
        
    except Exception as e:
        logger.error(f"Failed to refresh profile: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/profiles/{profile_id}/freshness")
async def get_profile_freshness(profile_id: str):
    """Get profile data freshness information"""
    try:
        from app.services.data_pipeline.profile_refresh_pipeline import ProfileRefreshPipeline
        
        pipeline = ProfileRefreshPipeline()
        freshness = await pipeline.get_profile_freshness(profile_id)
        
        return freshness
        
    except Exception as e:
        logger.error(f"Failed to get profile freshness: {e}")
        raise HTTPException(status_code=500, detail=str(e))