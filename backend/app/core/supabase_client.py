"""
Supabase client configuration for backend services
"""
from typing import Optional, Dict, Any, List
import asyncio
from datetime import datetime

import structlog
from supabase import create_client, Client
from postgrest.exceptions import APIError

from app.core.config import settings
from app.core.exceptions import DataSyncError

logger = structlog.get_logger()


class SupabaseClient:
    """Supabase client wrapper for backend operations"""
    
    def __init__(self):
        self.logger = structlog.get_logger(__name__)
        self._client: Optional[Client] = None
        self._service_client: Optional[Client] = None
        
        if settings.SUPABASE_URL and settings.SUPABASE_ANON_KEY:
            self._client = create_client(
                settings.SUPABASE_URL,
                settings.SUPABASE_ANON_KEY
            )
            
        if settings.SUPABASE_URL and settings.SUPABASE_SERVICE_ROLE_KEY:
            self._service_client = create_client(
                settings.SUPABASE_URL,
                settings.SUPABASE_SERVICE_ROLE_KEY
            )
    
    @property
    def client(self) -> Client:
        """Get the regular Supabase client"""
        if not self._client:
            raise DataSyncError("Supabase client not configured. Check SUPABASE_URL and SUPABASE_ANON_KEY.")
        return self._client
    
    @property
    def service_client(self) -> Client:
        """Get the service role Supabase client (for admin operations)"""
        if not self._service_client:
            raise DataSyncError("Supabase service client not configured. Check SUPABASE_SERVICE_ROLE_KEY.")
        return self._service_client
    
    async def get_user_by_id(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Get user data from Supabase by ID
        
        Args:
            user_id: User ID
            
        Returns:
            Dict containing user data or None if not found
        """
        try:
            response = self.service_client.auth.admin.get_user_by_id(user_id)
            if response.user:
                return {
                    "id": response.user.id,
                    "email": response.user.email,
                    "email_confirmed_at": response.user.email_confirmed_at,
                    "phone": response.user.phone,
                    "phone_confirmed_at": response.user.phone_confirmed_at,
                    "created_at": response.user.created_at,
                    "updated_at": response.user.updated_at,
                    "last_sign_in_at": response.user.last_sign_in_at,
                    "raw_user_meta_data": response.user.raw_user_meta_data or {},
                    "user_metadata": response.user.user_metadata or {},
                    "is_anonymous": response.user.is_anonymous or False,
                    "role": response.user.role or "authenticated"
                }
            return None
            
        except Exception as e:
            self.logger.error("Failed to get user from Supabase", user_id=user_id, error=str(e))
            return None
    
    async def get_users_batch(self, page: int = 1, per_page: int = 100) -> List[Dict[str, Any]]:
        """
        Get batch of users from Supabase
        
        Args:
            page: Page number (1-based)
            per_page: Number of users per page
            
        Returns:
            List of user data dictionaries
        """
        try:
            response = self.service_client.auth.admin.list_users(
                page=page,
                per_page=per_page
            )
            
            users = []
            for user in response:
                users.append({
                    "id": user.id,
                    "email": user.email,
                    "email_confirmed_at": user.email_confirmed_at,
                    "phone": user.phone,
                    "phone_confirmed_at": user.phone_confirmed_at,
                    "created_at": user.created_at,
                    "updated_at": user.updated_at,
                    "last_sign_in_at": user.last_sign_in_at,
                    "raw_user_meta_data": user.raw_user_meta_data or {},
                    "user_metadata": user.user_metadata or {},
                    "is_anonymous": user.is_anonymous or False,
                    "role": user.role or "authenticated"
                })
            
            return users
            
        except Exception as e:
            self.logger.error("Failed to get users batch from Supabase", 
                            page=page, per_page=per_page, error=str(e))
            return []
    
    async def get_user_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Get user profile from Supabase public.user_profiles table
        
        Args:
            user_id: User ID
            
        Returns:
            Dict containing profile data or None if not found
        """
        try:
            response = (
                self.service_client
                .table("user_profiles")
                .select("*")
                .eq("id", user_id)
                .execute()
            )
            
            if response.data:
                return response.data[0]
            return None
            
        except APIError as e:
            self.logger.error("API error getting user profile from Supabase", 
                            user_id=user_id, error=str(e))
            return None
        except Exception as e:
            self.logger.error("Failed to get user profile from Supabase", 
                            user_id=user_id, error=str(e))
            return None
    
    async def update_user_profile(self, user_id: str, updates: Dict[str, Any]) -> bool:
        """
        Update user profile in Supabase
        
        Args:
            user_id: User ID
            updates: Profile updates
            
        Returns:
            bool: True if successful
        """
        try:
            response = (
                self.service_client
                .table("user_profiles")
                .update(updates)
                .eq("id", user_id)
                .execute()
            )
            
            return len(response.data) > 0
            
        except APIError as e:
            self.logger.error("API error updating user profile in Supabase", 
                            user_id=user_id, error=str(e))
            return False
        except Exception as e:
            self.logger.error("Failed to update user profile in Supabase", 
                            user_id=user_id, error=str(e))
            return False
    
    async def listen_to_auth_changes(self, callback):
        """
        Listen to authentication changes in Supabase
        
        Args:
            callback: Function to call when auth changes occur
        """
        try:
            def auth_callback(event, session):
                asyncio.create_task(callback(event, session))
            
            self.client.auth.on_auth_state_change(auth_callback)
            self.logger.info("Started listening to Supabase auth changes")
            
        except Exception as e:
            self.logger.error("Failed to set up auth change listener", error=str(e))
    
    async def verify_jwt_token(self, token: str) -> Optional[Dict[str, Any]]:
        """
        Verify JWT token from Supabase
        
        Args:
            token: JWT token to verify
            
        Returns:
            Dict containing user data if valid, None otherwise
        """
        try:
            response = self.client.auth.get_user(token)
            if response.user:
                return {
                    "id": response.user.id,
                    "email": response.user.email,
                    "user_metadata": response.user.user_metadata or {},
                    "role": response.user.role or "authenticated"
                }
            return None
            
        except Exception as e:
            self.logger.error("Failed to verify JWT token", error=str(e))
            return None
    
    async def get_users_created_since(self, since: datetime) -> List[Dict[str, Any]]:
        """
        Get users created since a specific datetime
        
        Args:
            since: Datetime to filter from
            
        Returns:
            List of user data dictionaries
        """
        try:
            # Note: This would require a custom function in Supabase
            # For now, we'll get all users and filter client-side
            all_users = []
            page = 1
            per_page = 100
            
            while True:
                batch = await self.get_users_batch(page, per_page)
                if not batch:
                    break
                
                # Filter users created since the specified time
                filtered_batch = [
                    user for user in batch
                    if user.get("created_at") and 
                    datetime.fromisoformat(user["created_at"].replace("Z", "+00:00")) > since
                ]
                
                all_users.extend(filtered_batch)
                
                if len(batch) < per_page:
                    break
                
                page += 1
            
            return all_users
            
        except Exception as e:
            self.logger.error("Failed to get users created since", since=since, error=str(e))
            return []
    
    async def health_check(self) -> bool:
        """
        Check if Supabase connection is healthy
        
        Returns:
            bool: True if healthy
        """
        try:
            # Try to get a user count or similar lightweight operation
            response = (
                self.service_client
                .table("user_profiles")
                .select("id", count="exact")
                .limit(1)
                .execute()
            )
            
            return True
            
        except Exception as e:
            self.logger.error("Supabase health check failed", error=str(e))
            return False


# Global instance
supabase_client = SupabaseClient()