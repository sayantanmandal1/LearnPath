"""
Data synchronization service between Supabase and PostgreSQL
"""
import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from uuid import UUID

import structlog
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, delete
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from pydantic import BaseModel, Field

from app.core.database import get_db
from app.models.user import User
from app.models.profile import UserProfile
from app.core.exceptions import DataSyncError, ConflictResolutionError

logger = structlog.get_logger()


class SupabaseUser(BaseModel):
    """Supabase user data model"""
    id: str
    email: str
    email_confirmed_at: Optional[datetime] = None
    phone: Optional[str] = None
    phone_confirmed_at: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime
    last_sign_in_at: Optional[datetime] = None
    raw_user_meta_data: Dict[str, Any] = Field(default_factory=dict)
    user_metadata: Dict[str, Any] = Field(default_factory=dict)
    is_anonymous: bool = False
    role: str = "authenticated"


class PostgreSQLUser(BaseModel):
    """PostgreSQL user data model"""
    id: str
    email: str
    full_name: Optional[str] = None
    is_active: bool = True
    is_verified: bool = False
    created_at: datetime
    updated_at: datetime
    last_login: Optional[datetime] = None


class ProfileUpdates(BaseModel):
    """Profile update data model"""
    user_id: str
    updates: Dict[str, Any]
    source: str = "manual"  # manual, supabase, external_api
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class DataConflict(BaseModel):
    """Data conflict model"""
    field: str
    supabase_value: Any
    postgresql_value: Any
    last_modified_supabase: Optional[datetime] = None
    last_modified_postgresql: Optional[datetime] = None
    conflict_type: str  # "value_mismatch", "missing_field", "type_mismatch"


class DataConflicts(BaseModel):
    """Collection of data conflicts"""
    user_id: str
    conflicts: List[DataConflict]
    detected_at: datetime = Field(default_factory=datetime.utcnow)


class Resolution(BaseModel):
    """Conflict resolution result"""
    user_id: str
    resolved_conflicts: List[str]
    failed_resolutions: List[str]
    resolution_strategy: str
    resolved_at: datetime = Field(default_factory=datetime.utcnow)


class IntegrityReport(BaseModel):
    """Data integrity validation report"""
    user_id: str
    is_valid: bool
    issues: List[str]
    warnings: List[str]
    checked_at: datetime = Field(default_factory=datetime.utcnow)


class DataSyncService:
    """Service for synchronizing data between Supabase and PostgreSQL"""
    
    def __init__(self):
        self.logger = structlog.get_logger(__name__)
    
    async def sync_new_user(self, supabase_user: SupabaseUser) -> PostgreSQLUser:
        """
        Synchronize a new user from Supabase to PostgreSQL
        
        Args:
            supabase_user: User data from Supabase
            
        Returns:
            PostgreSQLUser: Created PostgreSQL user
            
        Raises:
            DataSyncError: If synchronization fails
        """
        try:
            self.logger.info("Syncing new user from Supabase", user_id=supabase_user.id)
            
            async for db in get_db():
                # Check if user already exists
                existing_user = await self._get_postgresql_user(db, supabase_user.id)
                if existing_user:
                    self.logger.warning("User already exists in PostgreSQL", user_id=supabase_user.id)
                    return existing_user
                
                # Extract user metadata
                full_name = (
                    supabase_user.user_metadata.get("full_name") or
                    supabase_user.raw_user_meta_data.get("full_name") or
                    f"{supabase_user.raw_user_meta_data.get('first_name', '')} {supabase_user.raw_user_meta_data.get('last_name', '')}".strip()
                )
                
                # Create PostgreSQL user
                postgresql_user = User(
                    id=supabase_user.id,
                    email=supabase_user.email,
                    hashed_password="",  # Password managed by Supabase
                    full_name=full_name or None,
                    is_active=True,
                    is_verified=supabase_user.email_confirmed_at is not None,
                    created_at=supabase_user.created_at,
                    updated_at=supabase_user.updated_at,
                    last_login=supabase_user.last_sign_in_at
                )
                
                db.add(postgresql_user)
                
                # Create user profile
                await self._create_user_profile(db, supabase_user)
                
                await db.commit()
                
                result = PostgreSQLUser(
                    id=postgresql_user.id,
                    email=postgresql_user.email,
                    full_name=postgresql_user.full_name,
                    is_active=postgresql_user.is_active,
                    is_verified=postgresql_user.is_verified,
                    created_at=postgresql_user.created_at,
                    updated_at=postgresql_user.updated_at,
                    last_login=postgresql_user.last_login
                )
                
                self.logger.info("Successfully synced new user", user_id=supabase_user.id)
                return result
                
        except Exception as e:
            self.logger.error("Failed to sync new user", user_id=supabase_user.id, error=str(e))
            raise DataSyncError(f"Failed to sync new user {supabase_user.id}: {str(e)}")
    
    async def sync_profile_updates(self, user_id: str, updates: ProfileUpdates) -> bool:
        """
        Synchronize profile updates between systems
        
        Args:
            user_id: User ID
            updates: Profile update data
            
        Returns:
            bool: True if sync successful
            
        Raises:
            DataSyncError: If synchronization fails
        """
        try:
            self.logger.info("Syncing profile updates", user_id=user_id, source=updates.source)
            
            async for db in get_db():
                # Get existing profile
                profile = await self._get_user_profile(db, user_id)
                if not profile:
                    self.logger.warning("Profile not found for user", user_id=user_id)
                    return False
                
                # Apply updates based on source
                if updates.source == "supabase":
                    await self._apply_supabase_updates(db, profile, updates.updates)
                elif updates.source == "external_api":
                    await self._apply_external_api_updates(db, profile, updates.updates)
                else:
                    await self._apply_manual_updates(db, profile, updates.updates)
                
                # Update timestamp
                profile.updated_at = datetime.utcnow()
                
                await db.commit()
                
                self.logger.info("Successfully synced profile updates", user_id=user_id)
                return True
                
        except Exception as e:
            self.logger.error("Failed to sync profile updates", user_id=user_id, error=str(e))
            raise DataSyncError(f"Failed to sync profile updates for user {user_id}: {str(e)}")
    
    async def resolve_data_conflicts(self, conflicts: DataConflicts) -> Resolution:
        """
        Resolve data conflicts between Supabase and PostgreSQL
        
        Args:
            conflicts: Detected data conflicts
            
        Returns:
            Resolution: Conflict resolution result
            
        Raises:
            ConflictResolutionError: If resolution fails
        """
        try:
            self.logger.info("Resolving data conflicts", user_id=conflicts.user_id, 
                           conflict_count=len(conflicts.conflicts))
            
            resolved_conflicts = []
            failed_resolutions = []
            
            async for db in get_db():
                user = await self._get_postgresql_user(db, conflicts.user_id)
                profile = await self._get_user_profile(db, conflicts.user_id)
                
                if not user or not profile:
                    raise ConflictResolutionError(f"User or profile not found: {conflicts.user_id}")
                
                for conflict in conflicts.conflicts:
                    try:
                        resolution_strategy = self._determine_resolution_strategy(conflict)
                        
                        if resolution_strategy == "use_latest":
                            await self._resolve_with_latest_timestamp(db, user, profile, conflict)
                        elif resolution_strategy == "use_supabase":
                            await self._resolve_with_supabase_value(db, user, profile, conflict)
                        elif resolution_strategy == "use_postgresql":
                            await self._resolve_with_postgresql_value(db, user, profile, conflict)
                        elif resolution_strategy == "merge":
                            await self._resolve_with_merge(db, user, profile, conflict)
                        else:
                            # Default to Supabase as source of truth for auth data
                            await self._resolve_with_supabase_value(db, user, profile, conflict)
                        
                        resolved_conflicts.append(conflict.field)
                        
                    except Exception as e:
                        self.logger.error("Failed to resolve conflict", 
                                        field=conflict.field, error=str(e))
                        failed_resolutions.append(conflict.field)
                
                await db.commit()
            
            resolution = Resolution(
                user_id=conflicts.user_id,
                resolved_conflicts=resolved_conflicts,
                failed_resolutions=failed_resolutions,
                resolution_strategy="hybrid"
            )
            
            self.logger.info("Completed conflict resolution", 
                           user_id=conflicts.user_id,
                           resolved=len(resolved_conflicts),
                           failed=len(failed_resolutions))
            
            return resolution
            
        except Exception as e:
            self.logger.error("Failed to resolve data conflicts", 
                            user_id=conflicts.user_id, error=str(e))
            raise ConflictResolutionError(f"Failed to resolve conflicts for user {conflicts.user_id}: {str(e)}")
    
    async def validate_data_integrity(self, user_id: str) -> IntegrityReport:
        """
        Validate data integrity for a user across systems
        
        Args:
            user_id: User ID to validate
            
        Returns:
            IntegrityReport: Validation report
        """
        try:
            self.logger.info("Validating data integrity", user_id=user_id)
            
            issues = []
            warnings = []
            
            async for db in get_db():
                # Check user exists
                user = await self._get_postgresql_user(db, user_id)
                if not user:
                    issues.append("User not found in PostgreSQL")
                    return IntegrityReport(
                        user_id=user_id,
                        is_valid=False,
                        issues=issues,
                        warnings=warnings
                    )
                
                # Check profile exists
                profile = await self._get_user_profile(db, user_id)
                if not profile:
                    issues.append("User profile not found")
                
                # Validate email format
                if not self._is_valid_email(user.email):
                    issues.append("Invalid email format")
                
                # Check for required fields
                if not user.email:
                    issues.append("Missing email")
                
                # Check data consistency
                if profile:
                    if profile.user_id != user.id:
                        issues.append("Profile user_id mismatch")
                    
                    # Check for orphaned data
                    if profile.skills and not isinstance(profile.skills, dict):
                        warnings.append("Skills data format inconsistent")
                    
                    if profile.platform_data and not isinstance(profile.platform_data, dict):
                        warnings.append("Platform data format inconsistent")
                
                # Check timestamps
                if user.updated_at < user.created_at:
                    issues.append("Invalid timestamp: updated_at before created_at")
                
                if profile and profile.updated_at < profile.created_at:
                    issues.append("Invalid profile timestamp: updated_at before created_at")
            
            is_valid = len(issues) == 0
            
            report = IntegrityReport(
                user_id=user_id,
                is_valid=is_valid,
                issues=issues,
                warnings=warnings
            )
            
            self.logger.info("Data integrity validation completed", 
                           user_id=user_id, is_valid=is_valid,
                           issues_count=len(issues), warnings_count=len(warnings))
            
            return report
            
        except Exception as e:
            self.logger.error("Failed to validate data integrity", user_id=user_id, error=str(e))
            return IntegrityReport(
                user_id=user_id,
                is_valid=False,
                issues=[f"Validation failed: {str(e)}"],
                warnings=[]
            )
    
    async def detect_conflicts(self, user_id: str, supabase_user: SupabaseUser) -> Optional[DataConflicts]:
        """
        Detect conflicts between Supabase and PostgreSQL data
        
        Args:
            user_id: User ID
            supabase_user: Current Supabase user data
            
        Returns:
            DataConflicts: Detected conflicts, None if no conflicts
        """
        try:
            conflicts = []
            
            async for db in get_db():
                postgresql_user = await self._get_postgresql_user(db, user_id)
                if not postgresql_user:
                    return None
                
                # Check email conflicts
                if postgresql_user.email != supabase_user.email:
                    conflicts.append(DataConflict(
                        field="email",
                        supabase_value=supabase_user.email,
                        postgresql_value=postgresql_user.email,
                        last_modified_supabase=supabase_user.updated_at,
                        last_modified_postgresql=postgresql_user.updated_at,
                        conflict_type="value_mismatch"
                    ))
                
                # Check full name conflicts
                supabase_full_name = (
                    supabase_user.user_metadata.get("full_name") or
                    supabase_user.raw_user_meta_data.get("full_name")
                )
                if postgresql_user.full_name != supabase_full_name:
                    conflicts.append(DataConflict(
                        field="full_name",
                        supabase_value=supabase_full_name,
                        postgresql_value=postgresql_user.full_name,
                        last_modified_supabase=supabase_user.updated_at,
                        last_modified_postgresql=postgresql_user.updated_at,
                        conflict_type="value_mismatch"
                    ))
                
                # Check verification status
                supabase_verified = supabase_user.email_confirmed_at is not None
                if postgresql_user.is_verified != supabase_verified:
                    conflicts.append(DataConflict(
                        field="is_verified",
                        supabase_value=supabase_verified,
                        postgresql_value=postgresql_user.is_verified,
                        last_modified_supabase=supabase_user.updated_at,
                        last_modified_postgresql=postgresql_user.updated_at,
                        conflict_type="value_mismatch"
                    ))
            
            if conflicts:
                return DataConflicts(user_id=user_id, conflicts=conflicts)
            
            return None
            
        except Exception as e:
            self.logger.error("Failed to detect conflicts", user_id=user_id, error=str(e))
            return None
    
    # Private helper methods
    
    async def _get_postgresql_user(self, db: AsyncSession, user_id: str) -> Optional[User]:
        """Get user from PostgreSQL"""
        try:
            result = await db.execute(select(User).where(User.id == user_id))
            return result.scalar_one_or_none()
        except Exception as e:
            self.logger.error("Failed to get PostgreSQL user", user_id=user_id, error=str(e))
            return None
    
    async def _get_user_profile(self, db: AsyncSession, user_id: str) -> Optional[UserProfile]:
        """Get user profile from PostgreSQL"""
        try:
            result = await db.execute(select(UserProfile).where(UserProfile.user_id == user_id))
            return result.scalar_one_or_none()
        except Exception as e:
            self.logger.error("Failed to get user profile", user_id=user_id, error=str(e))
            return None
    
    async def _create_user_profile(self, db: AsyncSession, supabase_user: SupabaseUser) -> UserProfile:
        """Create user profile from Supabase data"""
        profile = UserProfile(
            user_id=supabase_user.id,
            dream_job=supabase_user.raw_user_meta_data.get("career_goal"),
            current_role=supabase_user.raw_user_meta_data.get("current_role"),
            location=supabase_user.raw_user_meta_data.get("location"),
            industry=supabase_user.raw_user_meta_data.get("industry"),
            experience_years=supabase_user.raw_user_meta_data.get("experience_years"),
            created_at=supabase_user.created_at,
            updated_at=supabase_user.updated_at
        )
        
        db.add(profile)
        return profile
    
    async def _apply_supabase_updates(self, db: AsyncSession, profile: UserProfile, updates: Dict[str, Any]):
        """Apply updates from Supabase"""
        for field, value in updates.items():
            if hasattr(profile, field):
                setattr(profile, field, value)
    
    async def _apply_external_api_updates(self, db: AsyncSession, profile: UserProfile, updates: Dict[str, Any]):
        """Apply updates from external APIs"""
        # Merge with existing data rather than overwrite
        if "skills" in updates and profile.skills:
            merged_skills = {**profile.skills, **updates["skills"]}
            profile.skills = merged_skills
            updates.pop("skills")
        
        if "platform_data" in updates and profile.platform_data:
            merged_platform_data = {**profile.platform_data, **updates["platform_data"]}
            profile.platform_data = merged_platform_data
            updates.pop("platform_data")
        
        # Apply remaining updates
        for field, value in updates.items():
            if hasattr(profile, field):
                setattr(profile, field, value)
    
    async def _apply_manual_updates(self, db: AsyncSession, profile: UserProfile, updates: Dict[str, Any]):
        """Apply manual updates from user"""
        for field, value in updates.items():
            if hasattr(profile, field):
                setattr(profile, field, value)
    
    def _determine_resolution_strategy(self, conflict: DataConflict) -> str:
        """Determine the best strategy to resolve a conflict"""
        # For auth-related fields, prefer Supabase
        if conflict.field in ["email", "is_verified"]:
            return "use_supabase"
        
        # For profile fields, use latest timestamp
        if conflict.last_modified_supabase and conflict.last_modified_postgresql:
            if conflict.last_modified_supabase > conflict.last_modified_postgresql:
                return "use_supabase"
            else:
                return "use_postgresql"
        
        # Default to latest timestamp strategy
        return "use_latest"
    
    async def _resolve_with_latest_timestamp(self, db: AsyncSession, user: User, profile: UserProfile, conflict: DataConflict):
        """Resolve conflict using latest timestamp"""
        if conflict.last_modified_supabase and conflict.last_modified_postgresql:
            if conflict.last_modified_supabase > conflict.last_modified_postgresql:
                await self._apply_supabase_value(user, profile, conflict)
            else:
                await self._apply_postgresql_value(user, profile, conflict)
    
    async def _resolve_with_supabase_value(self, db: AsyncSession, user: User, profile: UserProfile, conflict: DataConflict):
        """Resolve conflict using Supabase value"""
        await self._apply_supabase_value(user, profile, conflict)
    
    async def _resolve_with_postgresql_value(self, db: AsyncSession, user: User, profile: UserProfile, conflict: DataConflict):
        """Resolve conflict using PostgreSQL value"""
        # PostgreSQL value is already in place, no action needed
        pass
    
    async def _resolve_with_merge(self, db: AsyncSession, user: User, profile: UserProfile, conflict: DataConflict):
        """Resolve conflict by merging values"""
        # For now, default to Supabase value
        # In the future, implement smart merging logic
        await self._apply_supabase_value(user, profile, conflict)
    
    async def _apply_supabase_value(self, user: User, profile: UserProfile, conflict: DataConflict):
        """Apply Supabase value to resolve conflict"""
        if conflict.field in ["email", "full_name", "is_verified"]:
            setattr(user, conflict.field, conflict.supabase_value)
        else:
            setattr(profile, conflict.field, conflict.supabase_value)
    
    async def _apply_postgresql_value(self, user: User, profile: UserProfile, conflict: DataConflict):
        """Apply PostgreSQL value to resolve conflict"""
        # PostgreSQL value is already applied, no action needed
        pass
    
    def _is_valid_email(self, email: str) -> bool:
        """Validate email format"""
        import re
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return re.match(pattern, email) is not None


# Global instance
data_sync_service = DataSyncService()