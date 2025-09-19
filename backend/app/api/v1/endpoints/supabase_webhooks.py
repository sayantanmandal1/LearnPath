"""
Webhook handlers for Supabase events
"""
import hmac
import hashlib
from typing import Dict, Any

from fastapi import APIRouter, Request, HTTPException, status, BackgroundTasks
import structlog

from app.core.config import settings
from app.services.data_sync_service import data_sync_service, SupabaseUser
from app.tasks.data_sync_tasks import data_sync_task_manager
from app.core.exceptions import DataSyncError

logger = structlog.get_logger()

router = APIRouter()


def verify_webhook_signature(payload: bytes, signature: str, secret: str) -> bool:
    """
    Verify webhook signature from Supabase
    
    Args:
        payload: Raw request payload
        signature: Signature from webhook header
        secret: Webhook secret
    
    Returns:
        bool: True if signature is valid
    """
    if not secret:
        logger.warning("Webhook secret not configured, skipping signature verification")
        return True
    
    try:
        # Supabase uses HMAC-SHA256
        expected_signature = hmac.new(
            secret.encode('utf-8'),
            payload,
            hashlib.sha256
        ).hexdigest()
        
        # Remove 'sha256=' prefix if present
        if signature.startswith('sha256='):
            signature = signature[7:]
        
        return hmac.compare_digest(expected_signature, signature)
        
    except Exception as e:
        logger.error("Failed to verify webhook signature", error=str(e))
        return False


@router.post("/webhooks/supabase/auth")
async def handle_auth_webhook(
    request: Request,
    background_tasks: BackgroundTasks
):
    """
    Handle Supabase authentication webhooks
    
    This endpoint receives notifications when users sign up, sign in, update their profile, etc.
    """
    try:
        # Get raw payload for signature verification
        payload = await request.body()
        
        # Verify webhook signature if secret is configured
        signature = request.headers.get('x-supabase-signature', '')
        if settings.SUPABASE_JWT_SECRET:
            if not verify_webhook_signature(payload, signature, settings.SUPABASE_JWT_SECRET):
                logger.warning("Invalid webhook signature")
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid webhook signature"
                )
        
        # Parse JSON payload
        webhook_data = await request.json()
        
        event_type = webhook_data.get('type')
        user_data = webhook_data.get('record', {})
        old_record = webhook_data.get('old_record', {})
        
        logger.info("Received Supabase auth webhook", 
                   event_type=event_type, 
                   user_id=user_data.get('id'))
        
        # Handle different event types
        if event_type == 'INSERT':
            # New user signup
            await handle_user_signup(user_data, background_tasks)
            
        elif event_type == 'UPDATE':
            # User profile update
            await handle_user_update(user_data, old_record, background_tasks)
            
        elif event_type == 'DELETE':
            # User deletion (rare, but possible)
            await handle_user_deletion(user_data, background_tasks)
            
        else:
            logger.warning("Unknown webhook event type", event_type=event_type)
        
        return {"status": "success", "message": "Webhook processed"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to process auth webhook", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process webhook"
        )


async def handle_user_signup(user_data: Dict[str, Any], background_tasks: BackgroundTasks):
    """
    Handle new user signup event
    
    Args:
        user_data: User data from Supabase
        background_tasks: FastAPI background tasks
    """
    try:
        user_id = user_data.get('id')
        if not user_id:
            logger.error("No user ID in signup webhook data")
            return
        
        logger.info("Processing user signup", user_id=user_id)
        
        # Create SupabaseUser object
        supabase_user = SupabaseUser(
            id=user_data['id'],
            email=user_data.get('email', ''),
            email_confirmed_at=user_data.get('email_confirmed_at'),
            phone=user_data.get('phone'),
            phone_confirmed_at=user_data.get('phone_confirmed_at'),
            created_at=user_data.get('created_at'),
            updated_at=user_data.get('updated_at'),
            last_sign_in_at=user_data.get('last_sign_in_at'),
            raw_user_meta_data=user_data.get('raw_user_meta_data', {}),
            user_metadata=user_data.get('user_metadata', {}),
            is_anonymous=user_data.get('is_anonymous', False),
            role=user_data.get('role', 'authenticated')
        )
        
        # Sync new user to PostgreSQL
        try:
            await data_sync_service.sync_new_user(supabase_user)
            logger.info("Successfully synced new user", user_id=user_id)
        except DataSyncError as e:
            logger.error("Failed to sync new user", user_id=user_id, error=str(e))
            # Trigger background retry
            background_tasks.add_task(
                data_sync_task_manager.trigger_user_sync,
                user_id
            )
        
    except Exception as e:
        logger.error("Failed to handle user signup", error=str(e))


async def handle_user_update(
    user_data: Dict[str, Any], 
    old_record: Dict[str, Any], 
    background_tasks: BackgroundTasks
):
    """
    Handle user profile update event
    
    Args:
        user_data: Updated user data from Supabase
        old_record: Previous user data
        background_tasks: FastAPI background tasks
    """
    try:
        user_id = user_data.get('id')
        if not user_id:
            logger.error("No user ID in update webhook data")
            return
        
        logger.info("Processing user update", user_id=user_id)
        
        # Determine what changed
        changes = {}
        for key, new_value in user_data.items():
            old_value = old_record.get(key)
            if new_value != old_value:
                changes[key] = new_value
        
        if not changes:
            logger.info("No changes detected in user update", user_id=user_id)
            return
        
        logger.info("User changes detected", user_id=user_id, changes=list(changes.keys()))
        
        # Trigger user sync to handle the updates
        background_tasks.add_task(
            data_sync_task_manager.trigger_user_sync,
            user_id
        )
        
    except Exception as e:
        logger.error("Failed to handle user update", error=str(e))


async def handle_user_deletion(user_data: Dict[str, Any], background_tasks: BackgroundTasks):
    """
    Handle user deletion event
    
    Args:
        user_data: Deleted user data from Supabase
        background_tasks: FastAPI background tasks
    """
    try:
        user_id = user_data.get('id')
        if not user_id:
            logger.error("No user ID in deletion webhook data")
            return
        
        logger.info("Processing user deletion", user_id=user_id)
        
        # For now, we'll just log this event
        # In the future, we might want to soft-delete or archive the user data
        logger.warning("User deleted in Supabase", user_id=user_id)
        
        # Could trigger cleanup tasks here if needed
        
    except Exception as e:
        logger.error("Failed to handle user deletion", error=str(e))


@router.post("/webhooks/supabase/database")
async def handle_database_webhook(
    request: Request,
    background_tasks: BackgroundTasks
):
    """
    Handle Supabase database webhooks (for profile table changes)
    """
    try:
        # Get raw payload for signature verification
        payload = await request.body()
        
        # Verify webhook signature if secret is configured
        signature = request.headers.get('x-supabase-signature', '')
        if settings.SUPABASE_JWT_SECRET:
            if not verify_webhook_signature(payload, signature, settings.SUPABASE_JWT_SECRET):
                logger.warning("Invalid webhook signature")
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid webhook signature"
                )
        
        # Parse JSON payload
        webhook_data = await request.json()
        
        event_type = webhook_data.get('type')
        table = webhook_data.get('table')
        record = webhook_data.get('record', {})
        old_record = webhook_data.get('old_record', {})
        
        logger.info("Received Supabase database webhook", 
                   event_type=event_type, 
                   table=table,
                   record_id=record.get('id'))
        
        # Handle profile table changes
        if table == 'user_profiles':
            await handle_profile_change(event_type, record, old_record, background_tasks)
        else:
            logger.info("Ignoring webhook for table", table=table)
        
        return {"status": "success", "message": "Database webhook processed"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to process database webhook", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process database webhook"
        )


async def handle_profile_change(
    event_type: str,
    record: Dict[str, Any],
    old_record: Dict[str, Any],
    background_tasks: BackgroundTasks
):
    """
    Handle changes to user_profiles table in Supabase
    
    Args:
        event_type: Type of change (INSERT, UPDATE, DELETE)
        record: Current record data
        old_record: Previous record data (for updates)
        background_tasks: FastAPI background tasks
    """
    try:
        user_id = record.get('id')
        if not user_id:
            logger.error("No user ID in profile change webhook")
            return
        
        logger.info("Processing profile change", 
                   event_type=event_type, 
                   user_id=user_id)
        
        if event_type in ['INSERT', 'UPDATE']:
            # Trigger sync to update PostgreSQL with Supabase profile changes
            background_tasks.add_task(
                data_sync_task_manager.trigger_user_sync,
                user_id
            )
        elif event_type == 'DELETE':
            logger.warning("Profile deleted in Supabase", user_id=user_id)
            # Could trigger cleanup if needed
        
    except Exception as e:
        logger.error("Failed to handle profile change", error=str(e))


@router.get("/webhooks/health")
async def webhook_health_check():
    """Health check endpoint for webhook service"""
    return {
        "status": "healthy",
        "service": "supabase-webhooks",
        "timestamp": "2024-01-01T00:00:00Z"
    }