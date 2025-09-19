"""
API endpoints for data synchronization management
"""
from typing import List, Optional
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, status
from sqlalchemy.ext.asyncio import AsyncSession
import structlog

from app.core.database import get_db
from app.core.exceptions import DataSyncError, ConflictResolutionError
from app.services.data_sync_service import (
    data_sync_service,
    SupabaseUser,
    ProfileUpdates,
    DataConflicts,
    Resolution,
    IntegrityReport
)
from app.services.data_sync_monitor import data_sync_monitor, SyncMetrics
from app.tasks.data_sync_tasks import data_sync_task_manager
from app.core.supabase_client import supabase_client
from app.api.dependencies import get_current_user
from app.schemas.auth import UserResponse

logger = structlog.get_logger()

router = APIRouter()


@router.post("/sync/user/{user_id}", response_model=dict)
async def sync_user(
    user_id: str,
    background_tasks: BackgroundTasks,
    current_user: UserResponse = Depends(get_current_user)
):
    """
    Trigger synchronization for a specific user
    
    Args:
        user_id: User ID to synchronize
        background_tasks: FastAPI background tasks
        current_user: Current authenticated user
    
    Returns:
        dict: Sync status
    """
    try:
        logger.info("Triggering user sync", user_id=user_id, triggered_by=current_user.id)
        
        # Trigger background sync task
        data_sync_task_manager.trigger_user_sync(user_id)
        
        return {
            "status": "success",
            "message": f"User synchronization triggered for {user_id}",
            "user_id": user_id,
            "triggered_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error("Failed to trigger user sync", user_id=user_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to trigger user synchronization: {str(e)}"
        )


@router.post("/sync/full", response_model=dict)
async def trigger_full_sync(
    background_tasks: BackgroundTasks,
    current_user: UserResponse = Depends(get_current_user)
):
    """
    Trigger full system synchronization
    
    Args:
        background_tasks: FastAPI background tasks
        current_user: Current authenticated user
    
    Returns:
        dict: Sync status
    """
    try:
        logger.info("Triggering full sync", triggered_by=current_user.id)
        
        # Trigger background sync tasks
        data_sync_task_manager.trigger_full_sync()
        
        return {
            "status": "success",
            "message": "Full system synchronization triggered",
            "triggered_at": datetime.utcnow().isoformat(),
            "triggered_by": current_user.id
        }
        
    except Exception as e:
        logger.error("Failed to trigger full sync", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to trigger full synchronization: {str(e)}"
        )


@router.post("/sync/conflicts/resolve", response_model=dict)
async def resolve_conflicts(
    background_tasks: BackgroundTasks,
    current_user: UserResponse = Depends(get_current_user)
):
    """
    Trigger conflict resolution for all users
    
    Args:
        background_tasks: FastAPI background tasks
        current_user: Current authenticated user
    
    Returns:
        dict: Resolution status
    """
    try:
        logger.info("Triggering conflict resolution", triggered_by=current_user.id)
        
        # Trigger background conflict resolution task
        data_sync_task_manager.trigger_conflict_resolution()
        
        return {
            "status": "success",
            "message": "Conflict resolution triggered",
            "triggered_at": datetime.utcnow().isoformat(),
            "triggered_by": current_user.id
        }
        
    except Exception as e:
        logger.error("Failed to trigger conflict resolution", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to trigger conflict resolution: {str(e)}"
        )


@router.get("/sync/user/{user_id}/conflicts", response_model=Optional[DataConflicts])
async def get_user_conflicts(
    user_id: str,
    current_user: UserResponse = Depends(get_current_user)
):
    """
    Get data conflicts for a specific user
    
    Args:
        user_id: User ID to check
        current_user: Current authenticated user
    
    Returns:
        DataConflicts: Detected conflicts or None
    """
    try:
        logger.info("Getting user conflicts", user_id=user_id, requested_by=current_user.id)
        
        # Get Supabase user data
        supabase_user_data = await supabase_client.get_user_by_id(user_id)
        if not supabase_user_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"User {user_id} not found in Supabase"
            )
        
        supabase_user = SupabaseUser(**supabase_user_data)
        
        # Detect conflicts
        conflicts = await data_sync_service.detect_conflicts(user_id, supabase_user)
        
        return conflicts
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get user conflicts", user_id=user_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get user conflicts: {str(e)}"
        )


@router.post("/sync/user/{user_id}/conflicts/resolve", response_model=Resolution)
async def resolve_user_conflicts(
    user_id: str,
    current_user: UserResponse = Depends(get_current_user)
):
    """
    Resolve conflicts for a specific user
    
    Args:
        user_id: User ID to resolve conflicts for
        current_user: Current authenticated user
    
    Returns:
        Resolution: Conflict resolution result
    """
    try:
        logger.info("Resolving user conflicts", user_id=user_id, requested_by=current_user.id)
        
        # Get Supabase user data
        supabase_user_data = await supabase_client.get_user_by_id(user_id)
        if not supabase_user_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"User {user_id} not found in Supabase"
            )
        
        supabase_user = SupabaseUser(**supabase_user_data)
        
        # Detect conflicts
        conflicts = await data_sync_service.detect_conflicts(user_id, supabase_user)
        if not conflicts or not conflicts.conflicts:
            return Resolution(
                user_id=user_id,
                resolved_conflicts=[],
                failed_resolutions=[],
                resolution_strategy="no_conflicts"
            )
        
        # Resolve conflicts
        resolution = await data_sync_service.resolve_data_conflicts(conflicts)
        
        logger.info("User conflicts resolved", 
                   user_id=user_id,
                   resolved=len(resolution.resolved_conflicts),
                   failed=len(resolution.failed_resolutions))
        
        return resolution
        
    except HTTPException:
        raise
    except ConflictResolutionError as e:
        logger.error("Conflict resolution error", user_id=user_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e)
        )
    except Exception as e:
        logger.error("Failed to resolve user conflicts", user_id=user_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to resolve user conflicts: {str(e)}"
        )


@router.get("/sync/user/{user_id}/integrity", response_model=IntegrityReport)
async def validate_user_integrity(
    user_id: str,
    current_user: UserResponse = Depends(get_current_user)
):
    """
    Validate data integrity for a specific user
    
    Args:
        user_id: User ID to validate
        current_user: Current authenticated user
    
    Returns:
        IntegrityReport: Data integrity report
    """
    try:
        logger.info("Validating user data integrity", user_id=user_id, requested_by=current_user.id)
        
        # Validate data integrity
        report = await data_sync_service.validate_data_integrity(user_id)
        
        logger.info("User data integrity validated", 
                   user_id=user_id,
                   is_valid=report.is_valid,
                   issues_count=len(report.issues),
                   warnings_count=len(report.warnings))
        
        return report
        
    except Exception as e:
        logger.error("Failed to validate user integrity", user_id=user_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to validate user integrity: {str(e)}"
        )


@router.post("/sync/integrity/validate", response_model=dict)
async def trigger_integrity_validation(
    background_tasks: BackgroundTasks,
    current_user: UserResponse = Depends(get_current_user)
):
    """
    Trigger data integrity validation for all users
    
    Args:
        background_tasks: FastAPI background tasks
        current_user: Current authenticated user
    
    Returns:
        dict: Validation status
    """
    try:
        logger.info("Triggering integrity validation", triggered_by=current_user.id)
        
        # Trigger background integrity validation task
        data_sync_task_manager.trigger_integrity_validation()
        
        return {
            "status": "success",
            "message": "Data integrity validation triggered",
            "triggered_at": datetime.utcnow().isoformat(),
            "triggered_by": current_user.id
        }
        
    except Exception as e:
        logger.error("Failed to trigger integrity validation", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to trigger integrity validation: {str(e)}"
        )


@router.get("/sync/status", response_model=dict)
async def get_sync_status(
    current_user: UserResponse = Depends(get_current_user)
):
    """
    Get synchronization system status
    
    Args:
        current_user: Current authenticated user
    
    Returns:
        dict: Sync system status
    """
    try:
        logger.info("Getting sync status", requested_by=current_user.id)
        
        # Check Supabase connection
        supabase_healthy = await supabase_client.health_check()
        
        # Get basic stats (this would be expanded with actual metrics)
        status_info = {
            "status": "operational" if supabase_healthy else "degraded",
            "supabase_connection": "healthy" if supabase_healthy else "unhealthy",
            "sync_enabled": True,  # From settings
            "last_check": datetime.utcnow().isoformat(),
            "services": {
                "data_sync_service": "operational",
                "supabase_client": "healthy" if supabase_healthy else "unhealthy",
                "background_tasks": "operational"
            }
        }
        
        return status_info
        
    except Exception as e:
        logger.error("Failed to get sync status", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get sync status: {str(e)}"
        )


@router.post("/sync/user/{user_id}/profile", response_model=dict)
async def sync_user_profile_updates(
    user_id: str,
    updates: dict,
    source: str = "manual",
    current_user: UserResponse = Depends(get_current_user)
):
    """
    Sync profile updates for a specific user
    
    Args:
        user_id: User ID
        updates: Profile updates to apply
        source: Source of updates (manual, supabase, external_api)
        current_user: Current authenticated user
    
    Returns:
        dict: Sync result
    """
    try:
        logger.info("Syncing user profile updates", 
                   user_id=user_id, source=source, requested_by=current_user.id)
        
        # Create profile updates object
        profile_updates = ProfileUpdates(
            user_id=user_id,
            updates=updates,
            source=source
        )
        
        # Sync profile updates
        success = await data_sync_service.sync_profile_updates(user_id, profile_updates)
        
        if success:
            return {
                "status": "success",
                "message": f"Profile updates synced for user {user_id}",
                "user_id": user_id,
                "source": source,
                "synced_at": datetime.utcnow().isoformat()
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Failed to sync profile updates"
            )
        
    except HTTPException:
        raise
    except DataSyncError as e:
        logger.error("Data sync error", user_id=user_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e)
        )
    except Exception as e:
        logger.error("Failed to sync user profile updates", user_id=user_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to sync profile updates: {str(e)}"
        )


@router.get("/monitoring/metrics", response_model=dict)
async def get_sync_metrics(
    current_user: UserResponse = Depends(get_current_user)
):
    """
    Get current synchronization metrics
    
    Args:
        current_user: Current authenticated user
    
    Returns:
        dict: Current sync metrics
    """
    try:
        logger.info("Getting sync metrics", requested_by=current_user.id)
        
        # Collect current metrics
        metrics = await data_sync_monitor.collect_sync_metrics()
        
        return {
            "status": "success",
            "metrics": {
                "total_users_postgresql": metrics.total_users_postgresql,
                "total_users_supabase": metrics.total_users_supabase,
                "users_in_sync": metrics.users_in_sync,
                "users_with_conflicts": metrics.users_with_conflicts,
                "users_missing_in_postgresql": metrics.users_missing_in_postgresql,
                "users_missing_in_supabase": metrics.users_missing_in_supabase,
                "sync_success_rate": metrics.sync_success_rate,
                "average_sync_time": metrics.average_sync_time,
                "last_sync_check": metrics.last_sync_check.isoformat()
            },
            "collected_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error("Failed to get sync metrics", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get sync metrics: {str(e)}"
        )


@router.get("/monitoring/health", response_model=dict)
async def get_sync_health(
    current_user: UserResponse = Depends(get_current_user)
):
    """
    Get synchronization system health status
    
    Args:
        current_user: Current authenticated user
    
    Returns:
        dict: Health status information
    """
    try:
        logger.info("Getting sync health status", requested_by=current_user.id)
        
        health_status = await data_sync_monitor.get_sync_health_status()
        
        return {
            "status": "success",
            "health": health_status,
            "checked_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error("Failed to get sync health status", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get sync health status: {str(e)}"
        )


@router.get("/monitoring/metrics/history", response_model=dict)
async def get_metrics_history(
    hours: int = 24,
    current_user: UserResponse = Depends(get_current_user)
):
    """
    Get synchronization metrics history
    
    Args:
        hours: Number of hours of history to return
        current_user: Current authenticated user
    
    Returns:
        dict: Metrics history
    """
    try:
        logger.info("Getting metrics history", hours=hours, requested_by=current_user.id)
        
        history = await data_sync_monitor.get_sync_metrics_history(hours)
        
        return {
            "status": "success",
            "history": [
                {
                    "total_users_postgresql": m.total_users_postgresql,
                    "total_users_supabase": m.total_users_supabase,
                    "users_in_sync": m.users_in_sync,
                    "users_with_conflicts": m.users_with_conflicts,
                    "sync_success_rate": m.sync_success_rate,
                    "timestamp": m.last_sync_check.isoformat()
                }
                for m in history
            ],
            "period_hours": hours,
            "retrieved_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error("Failed to get metrics history", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get metrics history: {str(e)}"
        )


@router.get("/monitoring/alerts", response_model=dict)
async def get_sync_alerts(
    hours: int = 24,
    current_user: UserResponse = Depends(get_current_user)
):
    """
    Get recent synchronization alerts
    
    Args:
        hours: Number of hours of alerts to return
        current_user: Current authenticated user
    
    Returns:
        dict: Recent alerts
    """
    try:
        logger.info("Getting sync alerts", hours=hours, requested_by=current_user.id)
        
        alerts = await data_sync_monitor.get_recent_alerts(hours)
        
        return {
            "status": "success",
            "alerts": [
                {
                    "alert_type": a.alert_type,
                    "severity": a.severity,
                    "message": a.message,
                    "details": a.details,
                    "timestamp": a.timestamp.isoformat(),
                    "user_id": a.user_id
                }
                for a in alerts
            ],
            "period_hours": hours,
            "retrieved_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error("Failed to get sync alerts", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get sync alerts: {str(e)}"
        )


@router.get("/monitoring/user/{user_id}", response_model=dict)
async def get_user_sync_status(
    user_id: str,
    current_user: UserResponse = Depends(get_current_user)
):
    """
    Get detailed synchronization status for a specific user
    
    Args:
        user_id: User ID to check
        current_user: Current authenticated user
    
    Returns:
        dict: User sync status details
    """
    try:
        logger.info("Getting user sync status", user_id=user_id, requested_by=current_user.id)
        
        status = await data_sync_monitor.check_user_sync_status(user_id)
        
        return {
            "status": "success",
            "user_sync_status": status,
            "checked_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error("Failed to get user sync status", user_id=user_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get user sync status: {str(e)}"
        )
