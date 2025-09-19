"""
Background tasks for data synchronization between Supabase and PostgreSQL
"""
import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Any

import structlog
from celery import Celery
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.celery_app import celery_app
from app.core.config import settings
from app.core.database import AsyncSessionLocal
from app.core.supabase_client import supabase_client
from app.services.data_sync_service import (
    data_sync_service, 
    SupabaseUser, 
    ProfileUpdates,
    DataConflicts
)
from app.core.exceptions import DataSyncError

logger = structlog.get_logger()


@celery_app.task(bind=True, max_retries=settings.MAX_SYNC_RETRIES)
def sync_new_users_task(self):
    """
    Celery task to sync new users from Supabase to PostgreSQL
    """
    try:
        asyncio.run(_sync_new_users())
    except Exception as e:
        logger.error("Failed to sync new users", error=str(e))
        raise self.retry(countdown=60, exc=e)


@celery_app.task(bind=True, max_retries=settings.MAX_SYNC_RETRIES)
def sync_user_updates_task(self):
    """
    Celery task to sync user updates between systems
    """
    try:
        asyncio.run(_sync_user_updates())
    except Exception as e:
        logger.error("Failed to sync user updates", error=str(e))
        raise self.retry(countdown=60, exc=e)


@celery_app.task(bind=True, max_retries=settings.MAX_SYNC_RETRIES)
def resolve_data_conflicts_task(self):
    """
    Celery task to detect and resolve data conflicts
    """
    try:
        asyncio.run(_resolve_data_conflicts())
    except Exception as e:
        logger.error("Failed to resolve data conflicts", error=str(e))
        raise self.retry(countdown=60, exc=e)


@celery_app.task(bind=True, max_retries=settings.MAX_SYNC_RETRIES)
def validate_data_integrity_task(self):
    """
    Celery task to validate data integrity across systems
    """
    try:
        asyncio.run(_validate_data_integrity())
    except Exception as e:
        logger.error("Failed to validate data integrity", error=str(e))
        raise self.retry(countdown=60, exc=e)


@celery_app.task(bind=True)
def sync_single_user_task(self, user_id: str):
    """
    Celery task to sync a single user
    
    Args:
        user_id: User ID to sync
    """
    try:
        asyncio.run(_sync_single_user(user_id))
    except Exception as e:
        logger.error("Failed to sync single user", user_id=user_id, error=str(e))
        raise self.retry(countdown=30, exc=e)


async def _sync_new_users():
    """Sync new users from Supabase to PostgreSQL"""
    logger.info("Starting new user synchronization")
    
    if not settings.ENABLE_DATA_SYNC:
        logger.info("Data sync disabled, skipping new user sync")
        return
    
    try:
        # Get users created in the last sync interval
        since = datetime.utcnow() - timedelta(minutes=settings.DATA_SYNC_INTERVAL_MINUTES)
        new_users = await supabase_client.get_users_created_since(since)
        
        if not new_users:
            logger.info("No new users to sync")
            return
        
        logger.info("Found new users to sync", count=len(new_users))
        
        synced_count = 0
        failed_count = 0
        
        for user_data in new_users:
            try:
                supabase_user = SupabaseUser(**user_data)
                await data_sync_service.sync_new_user(supabase_user)
                synced_count += 1
                
            except Exception as e:
                logger.error("Failed to sync individual user", 
                           user_id=user_data.get("id"), error=str(e))
                failed_count += 1
        
        logger.info("New user synchronization completed", 
                   synced=synced_count, failed=failed_count)
        
    except Exception as e:
        logger.error("Failed to sync new users", error=str(e))
        raise


async def _sync_user_updates():
    """Sync user updates between systems"""
    logger.info("Starting user update synchronization")
    
    if not settings.ENABLE_DATA_SYNC:
        logger.info("Data sync disabled, skipping user update sync")
        return
    
    try:
        # This would typically involve checking for updates in both systems
        # For now, we'll focus on detecting conflicts and resolving them
        await _detect_and_resolve_conflicts()
        
    except Exception as e:
        logger.error("Failed to sync user updates", error=str(e))
        raise


async def _resolve_data_conflicts():
    """Detect and resolve data conflicts between systems"""
    logger.info("Starting data conflict resolution")
    
    try:
        async with AsyncSessionLocal() as db:
            # Get all users from PostgreSQL
            from app.models.user import User
            from sqlalchemy import select
            
            result = await db.execute(select(User.id))
            user_ids = [row[0] for row in result.fetchall()]
            
            conflicts_resolved = 0
            conflicts_failed = 0
            
            for user_id in user_ids:
                try:
                    # Get Supabase user data
                    supabase_user_data = await supabase_client.get_user_by_id(user_id)
                    if not supabase_user_data:
                        continue
                    
                    supabase_user = SupabaseUser(**supabase_user_data)
                    
                    # Detect conflicts
                    conflicts = await data_sync_service.detect_conflicts(user_id, supabase_user)
                    if conflicts and conflicts.conflicts:
                        # Resolve conflicts
                        resolution = await data_sync_service.resolve_data_conflicts(conflicts)
                        if resolution.resolved_conflicts:
                            conflicts_resolved += len(resolution.resolved_conflicts)
                        if resolution.failed_resolutions:
                            conflicts_failed += len(resolution.failed_resolutions)
                
                except Exception as e:
                    logger.error("Failed to resolve conflicts for user", 
                               user_id=user_id, error=str(e))
                    conflicts_failed += 1
            
            logger.info("Data conflict resolution completed", 
                       resolved=conflicts_resolved, failed=conflicts_failed)
    
    except Exception as e:
        logger.error("Failed to resolve data conflicts", error=str(e))
        raise


async def _validate_data_integrity():
    """Validate data integrity across all users"""
    logger.info("Starting data integrity validation")
    
    try:
        async with AsyncSessionLocal() as db:
            # Get all users from PostgreSQL
            from app.models.user import User
            from sqlalchemy import select
            
            result = await db.execute(select(User.id))
            user_ids = [row[0] for row in result.fetchall()]
            
            valid_count = 0
            invalid_count = 0
            
            for user_id in user_ids:
                try:
                    report = await data_sync_service.validate_data_integrity(user_id)
                    if report.is_valid:
                        valid_count += 1
                    else:
                        invalid_count += 1
                        logger.warning("Data integrity issues found", 
                                     user_id=user_id, 
                                     issues=report.issues,
                                     warnings=report.warnings)
                
                except Exception as e:
                    logger.error("Failed to validate integrity for user", 
                               user_id=user_id, error=str(e))
                    invalid_count += 1
            
            logger.info("Data integrity validation completed", 
                       valid=valid_count, invalid=invalid_count)
    
    except Exception as e:
        logger.error("Failed to validate data integrity", error=str(e))
        raise


async def _sync_single_user(user_id: str):
    """Sync a single user between systems"""
    logger.info("Syncing single user", user_id=user_id)
    
    try:
        # Get Supabase user data
        supabase_user_data = await supabase_client.get_user_by_id(user_id)
        if not supabase_user_data:
            logger.warning("User not found in Supabase", user_id=user_id)
            return
        
        supabase_user = SupabaseUser(**supabase_user_data)
        
        # Check if user exists in PostgreSQL
        async with AsyncSessionLocal() as db:
            from app.models.user import User
            from sqlalchemy import select
            
            result = await db.execute(select(User).where(User.id == user_id))
            postgresql_user = result.scalar_one_or_none()
            
            if not postgresql_user:
                # New user, sync from Supabase
                await data_sync_service.sync_new_user(supabase_user)
                logger.info("Synced new user from Supabase", user_id=user_id)
            else:
                # Existing user, check for conflicts and resolve
                conflicts = await data_sync_service.detect_conflicts(user_id, supabase_user)
                if conflicts and conflicts.conflicts:
                    resolution = await data_sync_service.resolve_data_conflicts(conflicts)
                    logger.info("Resolved conflicts for user", 
                              user_id=user_id,
                              resolved=len(resolution.resolved_conflicts),
                              failed=len(resolution.failed_resolutions))
                else:
                    logger.info("No conflicts found for user", user_id=user_id)
        
        # Validate data integrity
        report = await data_sync_service.validate_data_integrity(user_id)
        if not report.is_valid:
            logger.warning("Data integrity issues after sync", 
                         user_id=user_id, issues=report.issues)
    
    except Exception as e:
        logger.error("Failed to sync single user", user_id=user_id, error=str(e))
        raise


async def _detect_and_resolve_conflicts():
    """Helper function to detect and resolve conflicts for all users"""
    logger.info("Detecting and resolving conflicts for all users")
    
    try:
        async with AsyncSessionLocal() as db:
            # Get all users from PostgreSQL
            from app.models.user import User
            from sqlalchemy import select
            
            result = await db.execute(select(User.id))
            user_ids = [row[0] for row in result.fetchall()]
            
            for user_id in user_ids[:settings.SYNC_BATCH_SIZE]:  # Process in batches
                try:
                    # Get Supabase user data
                    supabase_user_data = await supabase_client.get_user_by_id(user_id)
                    if not supabase_user_data:
                        continue
                    
                    supabase_user = SupabaseUser(**supabase_user_data)
                    
                    # Detect conflicts
                    conflicts = await data_sync_service.detect_conflicts(user_id, supabase_user)
                    if conflicts and conflicts.conflicts:
                        # Resolve conflicts
                        await data_sync_service.resolve_data_conflicts(conflicts)
                
                except Exception as e:
                    logger.error("Failed to process user in conflict detection", 
                               user_id=user_id, error=str(e))
    
    except Exception as e:
        logger.error("Failed to detect and resolve conflicts", error=str(e))
        raise


# Periodic task setup
@celery_app.on_after_configure.connect
def setup_periodic_tasks(sender, **kwargs):
    """Set up periodic tasks for data synchronization"""
    if settings.ENABLE_DATA_SYNC:
        # Sync new users every hour
        sender.add_periodic_task(
            settings.DATA_SYNC_INTERVAL_MINUTES * 60.0,
            sync_new_users_task.s(),
            name='sync_new_users'
        )
        
        # Sync user updates every 2 hours
        sender.add_periodic_task(
            settings.DATA_SYNC_INTERVAL_MINUTES * 2 * 60.0,
            sync_user_updates_task.s(),
            name='sync_user_updates'
        )
        
        # Resolve conflicts every 4 hours
        sender.add_periodic_task(
            settings.DATA_SYNC_INTERVAL_MINUTES * 4 * 60.0,
            resolve_data_conflicts_task.s(),
            name='resolve_data_conflicts'
        )
        
        # Validate data integrity daily
        sender.add_periodic_task(
            24 * 60 * 60.0,  # 24 hours
            validate_data_integrity_task.s(),
            name='validate_data_integrity'
        )
        
        logger.info("Data synchronization periodic tasks configured")


class DataSyncTaskManager:
    """Manager for data synchronization tasks"""
    
    @staticmethod
    def trigger_user_sync(user_id: str):
        """Trigger synchronization for a specific user"""
        sync_single_user_task.delay(user_id)
    
    @staticmethod
    def trigger_full_sync():
        """Trigger full synchronization"""
        sync_new_users_task.delay()
        sync_user_updates_task.delay()
    
    @staticmethod
    def trigger_conflict_resolution():
        """Trigger conflict resolution"""
        resolve_data_conflicts_task.delay()
    
    @staticmethod
    def trigger_integrity_validation():
        """Trigger data integrity validation"""
        validate_data_integrity_task.delay()


# Global instance
data_sync_task_manager = DataSyncTaskManager()