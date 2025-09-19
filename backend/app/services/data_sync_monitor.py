"""
Monitoring service for data synchronization between Supabase and PostgreSQL
"""
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

import structlog
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, text

from app.core.database import AsyncSessionLocal
from app.core.supabase_client import supabase_client
from app.core.config import settings
from app.models.user import User
from app.models.profile import UserProfile
from app.core.alerting import alerting_system

logger = structlog.get_logger()


@dataclass
class SyncMetrics:
    """Data synchronization metrics"""
    total_users_postgresql: int
    total_users_supabase: int
    users_in_sync: int
    users_with_conflicts: int
    users_missing_in_postgresql: int
    users_missing_in_supabase: int
    last_sync_check: datetime
    sync_success_rate: float
    average_sync_time: float


@dataclass
class SyncAlert:
    """Data synchronization alert"""
    alert_type: str
    severity: str  # low, medium, high, critical
    message: str
    details: Dict[str, Any]
    timestamp: datetime
    user_id: Optional[str] = None


class DataSyncMonitor:
    """Monitor data synchronization health and performance"""
    
    def __init__(self):
        self.logger = structlog.get_logger(__name__)
        self.metrics_history: List[SyncMetrics] = []
        self.alerts: List[SyncAlert] = []
        self.max_history_size = 1000
    
    async def collect_sync_metrics(self) -> SyncMetrics:
        """
        Collect comprehensive synchronization metrics
        
        Returns:
            SyncMetrics: Current sync metrics
        """
        try:
            self.logger.info("Collecting sync metrics")
            
            start_time = datetime.utcnow()
            
            # Get PostgreSQL user count
            async with AsyncSessionLocal() as db:
                result = await db.execute(select(func.count(User.id)))
                total_users_postgresql = result.scalar() or 0
                
                # Get users with profiles
                result = await db.execute(
                    select(func.count(UserProfile.user_id))
                )
                users_with_profiles = result.scalar() or 0
            
            # Get Supabase user count (approximate)
            total_users_supabase = await self._get_supabase_user_count()
            
            # Check sync status for a sample of users
            sync_check_results = await self._check_sync_status_sample()
            
            # Calculate metrics
            users_in_sync = sync_check_results.get('in_sync', 0)
            users_with_conflicts = sync_check_results.get('conflicts', 0)
            users_missing_in_postgresql = max(0, total_users_supabase - total_users_postgresql)
            users_missing_in_supabase = max(0, total_users_postgresql - total_users_supabase)
            
            # Calculate success rate
            total_checked = sync_check_results.get('total_checked', 1)
            sync_success_rate = (users_in_sync / total_checked) * 100 if total_checked > 0 else 0
            
            # Calculate average sync time
            end_time = datetime.utcnow()
            average_sync_time = (end_time - start_time).total_seconds()
            
            metrics = SyncMetrics(
                total_users_postgresql=total_users_postgresql,
                total_users_supabase=total_users_supabase,
                users_in_sync=users_in_sync,
                users_with_conflicts=users_with_conflicts,
                users_missing_in_postgresql=users_missing_in_postgresql,
                users_missing_in_supabase=users_missing_in_supabase,
                last_sync_check=datetime.utcnow(),
                sync_success_rate=sync_success_rate,
                average_sync_time=average_sync_time
            )
            
            # Store metrics
            self._store_metrics(metrics)
            
            # Check for alerts
            await self._check_for_alerts(metrics)
            
            self.logger.info("Sync metrics collected", 
                           postgresql_users=total_users_postgresql,
                           supabase_users=total_users_supabase,
                           sync_rate=sync_success_rate)
            
            return metrics
            
        except Exception as e:
            self.logger.error("Failed to collect sync metrics", error=str(e))
            raise
    
    async def get_sync_health_status(self) -> Dict[str, Any]:
        """
        Get overall synchronization health status
        
        Returns:
            Dict containing health status information
        """
        try:
            # Get latest metrics
            if not self.metrics_history:
                await self.collect_sync_metrics()
            
            latest_metrics = self.metrics_history[-1] if self.metrics_history else None
            
            if not latest_metrics:
                return {
                    "status": "unknown",
                    "message": "No metrics available",
                    "last_check": None
                }
            
            # Determine health status
            status = "healthy"
            issues = []
            
            # Check sync success rate
            if latest_metrics.sync_success_rate < 95:
                status = "degraded"
                issues.append(f"Low sync success rate: {latest_metrics.sync_success_rate:.1f}%")
            
            if latest_metrics.sync_success_rate < 80:
                status = "unhealthy"
            
            # Check for missing users
            if latest_metrics.users_missing_in_postgresql > 10:
                status = "degraded"
                issues.append(f"{latest_metrics.users_missing_in_postgresql} users missing in PostgreSQL")
            
            if latest_metrics.users_missing_in_supabase > 10:
                status = "degraded"
                issues.append(f"{latest_metrics.users_missing_in_supabase} users missing in Supabase")
            
            # Check for conflicts
            if latest_metrics.users_with_conflicts > 5:
                status = "degraded"
                issues.append(f"{latest_metrics.users_with_conflicts} users have data conflicts")
            
            # Check last sync time
            time_since_check = datetime.utcnow() - latest_metrics.last_sync_check
            if time_since_check > timedelta(hours=2):
                status = "stale"
                issues.append(f"Last sync check was {time_since_check} ago")
            
            return {
                "status": status,
                "message": "Sync system is " + status,
                "issues": issues,
                "metrics": {
                    "total_users_postgresql": latest_metrics.total_users_postgresql,
                    "total_users_supabase": latest_metrics.total_users_supabase,
                    "sync_success_rate": latest_metrics.sync_success_rate,
                    "users_with_conflicts": latest_metrics.users_with_conflicts,
                    "last_check": latest_metrics.last_sync_check.isoformat()
                },
                "alerts": len([a for a in self.alerts if a.timestamp > datetime.utcnow() - timedelta(hours=24)])
            }
            
        except Exception as e:
            self.logger.error("Failed to get sync health status", error=str(e))
            return {
                "status": "error",
                "message": f"Failed to get health status: {str(e)}",
                "last_check": None
            }
    
    async def get_sync_metrics_history(self, hours: int = 24) -> List[SyncMetrics]:
        """
        Get synchronization metrics history
        
        Args:
            hours: Number of hours of history to return
            
        Returns:
            List of SyncMetrics from the specified time period
        """
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        return [
            metrics for metrics in self.metrics_history
            if metrics.last_sync_check > cutoff_time
        ]
    
    async def get_recent_alerts(self, hours: int = 24) -> List[SyncAlert]:
        """
        Get recent synchronization alerts
        
        Args:
            hours: Number of hours of alerts to return
            
        Returns:
            List of recent SyncAlert objects
        """
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        return [
            alert for alert in self.alerts
            if alert.timestamp > cutoff_time
        ]
    
    async def check_user_sync_status(self, user_id: str) -> Dict[str, Any]:
        """
        Check synchronization status for a specific user
        
        Args:
            user_id: User ID to check
            
        Returns:
            Dict containing sync status information
        """
        try:
            self.logger.info("Checking user sync status", user_id=user_id)
            
            # Get user from PostgreSQL
            async with AsyncSessionLocal() as db:
                result = await db.execute(select(User).where(User.id == user_id))
                postgresql_user = result.scalar_one_or_none()
                
                result = await db.execute(select(UserProfile).where(UserProfile.user_id == user_id))
                postgresql_profile = result.scalar_one_or_none()
            
            # Get user from Supabase
            supabase_user_data = await supabase_client.get_user_by_id(user_id)
            supabase_profile_data = await supabase_client.get_user_profile(user_id)
            
            # Analyze sync status
            status = {
                "user_id": user_id,
                "postgresql": {
                    "user_exists": postgresql_user is not None,
                    "profile_exists": postgresql_profile is not None,
                    "last_updated": postgresql_user.updated_at.isoformat() if postgresql_user else None
                },
                "supabase": {
                    "user_exists": supabase_user_data is not None,
                    "profile_exists": supabase_profile_data is not None,
                    "last_updated": supabase_user_data.get('updated_at') if supabase_user_data else None
                },
                "sync_status": "unknown",
                "conflicts": [],
                "issues": []
            }
            
            # Determine sync status
            if not postgresql_user and not supabase_user_data:
                status["sync_status"] = "not_found"
                status["issues"].append("User not found in either system")
            elif not postgresql_user:
                status["sync_status"] = "missing_postgresql"
                status["issues"].append("User missing in PostgreSQL")
            elif not supabase_user_data:
                status["sync_status"] = "missing_supabase"
                status["issues"].append("User missing in Supabase")
            else:
                # Both exist, check for conflicts
                conflicts = await self._detect_user_conflicts(postgresql_user, supabase_user_data)
                if conflicts:
                    status["sync_status"] = "conflicts"
                    status["conflicts"] = conflicts
                else:
                    status["sync_status"] = "in_sync"
            
            return status
            
        except Exception as e:
            self.logger.error("Failed to check user sync status", user_id=user_id, error=str(e))
            return {
                "user_id": user_id,
                "sync_status": "error",
                "error": str(e)
            }
    
    # Private helper methods
    
    async def _get_supabase_user_count(self) -> int:
        """Get approximate user count from Supabase"""
        try:
            # This is a simplified approach - in production you might want to use
            # Supabase admin API or a custom function
            response = (
                supabase_client.service_client
                .table("user_profiles")
                .select("id", count="exact")
                .limit(1)
                .execute()
            )
            
            return response.count or 0
            
        except Exception as e:
            self.logger.error("Failed to get Supabase user count", error=str(e))
            return 0
    
    async def _check_sync_status_sample(self, sample_size: int = 100) -> Dict[str, int]:
        """Check sync status for a sample of users"""
        try:
            async with AsyncSessionLocal() as db:
                # Get a sample of users
                result = await db.execute(
                    select(User.id).limit(sample_size)
                )
                user_ids = [row[0] for row in result.fetchall()]
            
            in_sync = 0
            conflicts = 0
            total_checked = len(user_ids)
            
            for user_id in user_ids:
                try:
                    status = await self.check_user_sync_status(user_id)
                    if status["sync_status"] == "in_sync":
                        in_sync += 1
                    elif status["sync_status"] == "conflicts":
                        conflicts += 1
                except Exception as e:
                    self.logger.error("Failed to check user in sample", user_id=user_id, error=str(e))
            
            return {
                "total_checked": total_checked,
                "in_sync": in_sync,
                "conflicts": conflicts
            }
            
        except Exception as e:
            self.logger.error("Failed to check sync status sample", error=str(e))
            return {"total_checked": 0, "in_sync": 0, "conflicts": 0}
    
    async def _detect_user_conflicts(self, postgresql_user: User, supabase_user_data: Dict[str, Any]) -> List[str]:
        """Detect conflicts for a specific user"""
        conflicts = []
        
        try:
            # Check email
            if postgresql_user.email != supabase_user_data.get('email'):
                conflicts.append("email_mismatch")
            
            # Check verification status
            supabase_verified = supabase_user_data.get('email_confirmed_at') is not None
            if postgresql_user.is_verified != supabase_verified:
                conflicts.append("verification_status_mismatch")
            
            # Check full name
            supabase_full_name = (
                supabase_user_data.get('user_metadata', {}).get('full_name') or
                supabase_user_data.get('raw_user_meta_data', {}).get('full_name')
            )
            if postgresql_user.full_name != supabase_full_name:
                conflicts.append("full_name_mismatch")
            
        except Exception as e:
            self.logger.error("Failed to detect user conflicts", error=str(e))
            conflicts.append("detection_error")
        
        return conflicts
    
    def _store_metrics(self, metrics: SyncMetrics):
        """Store metrics in memory (could be extended to persist to database)"""
        self.metrics_history.append(metrics)
        
        # Keep only recent metrics
        if len(self.metrics_history) > self.max_history_size:
            self.metrics_history = self.metrics_history[-self.max_history_size:]
    
    async def _check_for_alerts(self, metrics: SyncMetrics):
        """Check metrics and generate alerts if needed"""
        try:
            # Check for low sync success rate
            if metrics.sync_success_rate < 90:
                alert = SyncAlert(
                    alert_type="low_sync_rate",
                    severity="high" if metrics.sync_success_rate < 80 else "medium",
                    message=f"Sync success rate is {metrics.sync_success_rate:.1f}%",
                    details={"sync_rate": metrics.sync_success_rate},
                    timestamp=datetime.utcnow()
                )
                await self._add_alert(alert)
            
            # Check for missing users
            if metrics.users_missing_in_postgresql > 20:
                alert = SyncAlert(
                    alert_type="missing_users_postgresql",
                    severity="high",
                    message=f"{metrics.users_missing_in_postgresql} users missing in PostgreSQL",
                    details={"missing_count": metrics.users_missing_in_postgresql},
                    timestamp=datetime.utcnow()
                )
                await self._add_alert(alert)
            
            # Check for conflicts
            if metrics.users_with_conflicts > 10:
                alert = SyncAlert(
                    alert_type="high_conflict_count",
                    severity="medium",
                    message=f"{metrics.users_with_conflicts} users have data conflicts",
                    details={"conflict_count": metrics.users_with_conflicts},
                    timestamp=datetime.utcnow()
                )
                await self._add_alert(alert)
            
        except Exception as e:
            self.logger.error("Failed to check for alerts", error=str(e))
    
    async def _add_alert(self, alert: SyncAlert):
        """Add an alert and optionally send notifications"""
        self.alerts.append(alert)
        
        # Keep only recent alerts
        cutoff_time = datetime.utcnow() - timedelta(days=7)
        self.alerts = [a for a in self.alerts if a.timestamp > cutoff_time]
        
        # Send alert notification
        try:
            from app.core.monitoring import Alert, AlertType, AlertSeverity
            import uuid
            
            # Map severity string to AlertSeverity enum
            severity_mapping = {
                "low": AlertSeverity.LOW,
                "medium": AlertSeverity.MEDIUM,
                "high": AlertSeverity.HIGH,
                "critical": AlertSeverity.CRITICAL
            }
            
            # Create Alert object
            monitoring_alert = Alert(
                id=str(uuid.uuid4()),
                type=AlertType.DATA_INTEGRITY,  # Use appropriate alert type
                severity=severity_mapping.get(alert.severity.lower(), AlertSeverity.MEDIUM),
                title=f"Data Sync Alert: {alert.alert_type}",
                description=alert.message,
                timestamp=alert.timestamp,
                service="data_sync",
                metadata=alert.details or {}
            )
            
            # Send alert through alerting system
            alerting_system.handle_alert(monitoring_alert)
            
        except Exception as e:
            self.logger.error("Failed to send alert notification", error=str(e))


# Global instance
data_sync_monitor = DataSyncMonitor()