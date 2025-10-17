"""
Comprehensive audit logging for security monitoring
"""
import json
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from enum import Enum
import structlog
from fastapi import Request
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import Column, String, DateTime, Text, Integer, Boolean, JSON
from sqlalchemy.dialects.postgresql import UUID

from app.core.database import Base
from app.core.config import settings

logger = structlog.get_logger()


class AuditEventType(str, Enum):
    """Types of audit events"""
    # Authentication events
    LOGIN_SUCCESS = "login_success"
    LOGIN_FAILURE = "login_failure"
    LOGOUT = "logout"
    PASSWORD_CHANGE = "password_change"
    PASSWORD_RESET = "password_reset"
    TOKEN_REFRESH = "token_refresh"
    
    # User management events
    USER_CREATED = "user_created"
    USER_UPDATED = "user_updated"
    USER_DELETED = "user_deleted"
    USER_ACTIVATED = "user_activated"
    USER_DEACTIVATED = "user_deactivated"
    
    # Profile events
    PROFILE_CREATED = "profile_created"
    PROFILE_UPDATED = "profile_updated"
    PROFILE_VIEWED = "profile_viewed"
    PROFILE_DELETED = "profile_deleted"
    
    # Data access events
    DATA_EXPORT = "data_export"
    DATA_IMPORT = "data_import"
    FILE_UPLOAD = "file_upload"
    FILE_DOWNLOAD = "file_download"
    
    # Security events
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    IP_BLOCKED = "ip_blocked"
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    
    # System events
    SYSTEM_ERROR = "system_error"
    CONFIGURATION_CHANGE = "configuration_change"
    BACKUP_CREATED = "backup_created"
    BACKUP_RESTORED = "backup_restored"
    
    # Privacy events
    CONSENT_GIVEN = "consent_given"
    CONSENT_WITHDRAWN = "consent_withdrawn"
    DATA_DELETION_REQUEST = "data_deletion_request"
    DATA_EXPORT_REQUEST = "data_export_request"


class AuditSeverity(str, Enum):
    """Severity levels for audit events"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AuditLog(Base):
    """Audit log database model"""
    __tablename__ = "audit_logs"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    event_type = Column(String(50), nullable=False, index=True)
    severity = Column(String(20), nullable=False, index=True)
    user_id = Column(UUID(as_uuid=True), nullable=True, index=True)
    session_id = Column(String(100), nullable=True, index=True)
    ip_address = Column(String(45), nullable=True, index=True)
    user_agent = Column(Text, nullable=True)
    endpoint = Column(String(200), nullable=True, index=True)
    method = Column(String(10), nullable=True)
    status_code = Column(Integer, nullable=True)
    message = Column(Text, nullable=False)
    details = Column(JSON, nullable=True)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    
    # Additional security fields
    request_id = Column(String(100), nullable=True, index=True)
    correlation_id = Column(String(100), nullable=True, index=True)
    source_system = Column(String(50), nullable=True)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert audit log to dictionary"""
        return {
            "id": str(self.id),
            "event_type": self.event_type,
            "severity": self.severity,
            "user_id": str(self.user_id) if self.user_id else None,
            "session_id": self.session_id,
            "ip_address": self.ip_address,
            "user_agent": self.user_agent,
            "endpoint": self.endpoint,
            "method": self.method,
            "status_code": self.status_code,
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp.isoformat(),
            "request_id": self.request_id,
            "correlation_id": self.correlation_id,
            "source_system": self.source_system
        }


class AuditLogger:
    """Comprehensive audit logging service"""
    
    def __init__(self):
        self.logger = structlog.get_logger("audit")
    
    async def log_event(
        self,
        db: AsyncSession,
        event_type: AuditEventType,
        message: str,
        severity: AuditSeverity = AuditSeverity.LOW,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        endpoint: Optional[str] = None,
        method: Optional[str] = None,
        status_code: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
        correlation_id: Optional[str] = None,
        source_system: str = "api"
    ) -> AuditLog:
        """Log an audit event"""
        
        # Create audit log entry
        audit_log = AuditLog(
            event_type=event_type.value,
            severity=severity.value,
            user_id=uuid.UUID(user_id) if user_id else None,
            session_id=session_id,
            ip_address=ip_address,
            user_agent=user_agent,
            endpoint=endpoint,
            method=method,
            status_code=status_code,
            message=message,
            details=details or {},
            request_id=request_id,
            correlation_id=correlation_id,
            source_system=source_system
        )
        
        try:
            # Save to database
            db.add(audit_log)
            await db.commit()
            await db.refresh(audit_log)
            
            # Also log to structured logger
            self.logger.info(
                "Audit event logged",
                event_type=event_type.value,
                severity=severity.value,
                user_id=user_id,
                session_id=session_id,
                ip_address=ip_address,
                endpoint=endpoint,
                message=message,
                details=details,
                audit_id=str(audit_log.id)
            )
            
            return audit_log
        
        except Exception as e:
            # If database logging fails, at least log to structured logger
            self.logger.error(
                "Failed to save audit log to database",
                event_type=event_type.value,
                message=message,
                error=str(e)
            )
            raise
    
    async def log_authentication_event(
        self,
        db: AsyncSession,
        event_type: AuditEventType,
        user_id: Optional[str],
        ip_address: str,
        user_agent: str,
        success: bool,
        details: Optional[Dict[str, Any]] = None
    ):
        """Log authentication-related events"""
        severity = AuditSeverity.LOW if success else AuditSeverity.MEDIUM
        message = f"Authentication event: {event_type.value}"
        
        if not success:
            message += " (FAILED)"
        
        await self.log_event(
            db=db,
            event_type=event_type,
            message=message,
            severity=severity,
            user_id=user_id,
            ip_address=ip_address,
            user_agent=user_agent,
            details=details
        )
    
    async def log_data_access_event(
        self,
        db: AsyncSession,
        event_type: AuditEventType,
        user_id: str,
        resource_type: str,
        resource_id: str,
        action: str,
        ip_address: str,
        details: Optional[Dict[str, Any]] = None
    ):
        """Log data access events"""
        message = f"Data access: {action} {resource_type} {resource_id}"
        
        await self.log_event(
            db=db,
            event_type=event_type,
            message=message,
            severity=AuditSeverity.LOW,
            user_id=user_id,
            ip_address=ip_address,
            details={
                "resource_type": resource_type,
                "resource_id": resource_id,
                "action": action,
                **(details or {})
            }
        )
    
    async def log_security_event(
        self,
        db: AsyncSession,
        event_type: AuditEventType,
        message: str,
        ip_address: str,
        severity: AuditSeverity = AuditSeverity.HIGH,
        user_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """Log security-related events"""
        await self.log_event(
            db=db,
            event_type=event_type,
            message=f"Security event: {message}",
            severity=severity,
            user_id=user_id,
            ip_address=ip_address,
            details=details
        )
    
    async def log_privacy_event(
        self,
        db: AsyncSession,
        event_type: AuditEventType,
        user_id: str,
        action: str,
        details: Optional[Dict[str, Any]] = None
    ):
        """Log privacy-related events"""
        message = f"Privacy event: {action}"
        
        await self.log_event(
            db=db,
            event_type=event_type,
            message=message,
            severity=AuditSeverity.MEDIUM,
            user_id=user_id,
            details=details
        )
    
    async def search_audit_logs(
        self,
        db: AsyncSession,
        event_types: Optional[List[AuditEventType]] = None,
        user_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        severity: Optional[AuditSeverity] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[AuditLog]:
        """Search audit logs with filters"""
        from sqlalchemy import and_, or_
        
        query = db.query(AuditLog)
        
        conditions = []
        
        if event_types:
            conditions.append(AuditLog.event_type.in_([et.value for et in event_types]))
        
        if user_id:
            conditions.append(AuditLog.user_id == uuid.UUID(user_id))
        
        if ip_address:
            conditions.append(AuditLog.ip_address == ip_address)
        
        if start_date:
            conditions.append(AuditLog.timestamp >= start_date)
        
        if end_date:
            conditions.append(AuditLog.timestamp <= end_date)
        
        if severity:
            conditions.append(AuditLog.severity == severity.value)
        
        if conditions:
            query = query.filter(and_(*conditions))
        
        query = query.order_by(AuditLog.timestamp.desc())
        query = query.offset(offset).limit(limit)
        
        result = await query.all()
        return result


class AuditMiddleware:
    """Middleware to automatically log HTTP requests"""
    
    def __init__(self):
        self.audit_logger = AuditLogger()
    
    async def log_request(
        self,
        request: Request,
        response_status: int,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None
    ):
        """Log HTTP request for audit purposes"""
        from app.core.database import get_db
        
        # Skip logging for health checks and static content
        if request.url.path in ["/health", "/metrics", "/favicon.ico"]:
            return
        
        # Determine event type based on endpoint
        event_type = self._determine_event_type(request.url.path, request.method)
        
        # Determine severity based on status code
        severity = self._determine_severity(response_status)
        
        # Get client IP
        ip_address = self._get_client_ip(request)
        
        # Create message
        message = f"{request.method} {request.url.path} - {response_status}"
        
        # Prepare details
        details = {
            "query_params": dict(request.query_params),
            "headers": dict(request.headers),
            "response_status": response_status
        }
        
        # Remove sensitive headers
        sensitive_headers = ["authorization", "cookie", "x-api-key"]
        for header in sensitive_headers:
            details["headers"].pop(header, None)
        
        try:
            async for db in get_db():
                await self.audit_logger.log_event(
                    db=db,
                    event_type=event_type,
                    message=message,
                    severity=severity,
                    user_id=user_id,
                    session_id=session_id,
                    ip_address=ip_address,
                    user_agent=request.headers.get("user-agent"),
                    endpoint=request.url.path,
                    method=request.method,
                    status_code=response_status,
                    details=details,
                    request_id=request.headers.get("x-request-id")
                )
                break
        except Exception as e:
            logger.error("Failed to log audit event", error=str(e))
    
    def _determine_event_type(self, path: str, method: str) -> AuditEventType:
        """Determine audit event type based on endpoint"""
        if "/auth/login" in path:
            return AuditEventType.LOGIN_SUCCESS
        elif "/auth/logout" in path:
            return AuditEventType.LOGOUT
        elif "/auth/register" in path:
            return AuditEventType.USER_CREATED
        elif "/profiles" in path and method == "POST":
            return AuditEventType.PROFILE_CREATED
        elif "/profiles" in path and method in ["PUT", "PATCH"]:
            return AuditEventType.PROFILE_UPDATED
        elif "/profiles" in path and method == "GET":
            return AuditEventType.PROFILE_VIEWED
        elif "/upload" in path:
            return AuditEventType.FILE_UPLOAD
        elif "/export" in path:
            return AuditEventType.DATA_EXPORT
        else:
            return AuditEventType.DATA_EXPORT  # Default for API access
    
    def _determine_severity(self, status_code: int) -> AuditSeverity:
        """Determine severity based on HTTP status code"""
        if status_code >= 500:
            return AuditSeverity.HIGH
        elif status_code >= 400:
            return AuditSeverity.MEDIUM
        else:
            return AuditSeverity.LOW
    
    def _get_client_ip(self, request: Request) -> str:
        """Get client IP address"""
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        return request.client.host if request.client else "unknown"


# Global audit logger instance
audit_logger = AuditLogger()
audit_middleware = AuditMiddleware()