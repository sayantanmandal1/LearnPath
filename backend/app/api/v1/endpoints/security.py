"""
Security monitoring and audit API endpoints
"""
from typing import List, Optional
from datetime import datetime, timedelta
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel, Field

from app.core.database import get_db
from app.api.dependencies import get_current_user, get_current_admin_user
from app.core.audit_logging import (
    audit_logger, AuditEventType, AuditSeverity, AuditLog
)
from app.models.user import User

router = APIRouter()


# Response Models
class AuditLogResponse(BaseModel):
    id: str
    event_type: str
    severity: str
    user_id: Optional[str]
    session_id: Optional[str]
    ip_address: Optional[str]
    user_agent: Optional[str]
    endpoint: Optional[str]
    method: Optional[str]
    status_code: Optional[int]
    message: str
    details: Optional[dict]
    timestamp: str
    request_id: Optional[str]


class SecuritySummaryResponse(BaseModel):
    total_events: int
    events_by_severity: dict
    events_by_type: dict
    recent_suspicious_activity: List[AuditLogResponse]
    top_source_ips: List[dict]


class UserSecurityResponse(BaseModel):
    user_id: str
    recent_logins: List[dict]
    failed_login_attempts: int
    last_password_change: Optional[str]
    active_sessions: int
    security_score: int
    recommendations: List[str]


@router.get("/audit-logs", response_model=List[AuditLogResponse])
async def get_audit_logs(
    event_types: Optional[List[str]] = Query(None, description="Filter by event types"),
    severity: Optional[str] = Query(None, description="Filter by severity"),
    user_id: Optional[str] = Query(None, description="Filter by user ID"),
    ip_address: Optional[str] = Query(None, description="Filter by IP address"),
    start_date: Optional[datetime] = Query(None, description="Start date filter"),
    end_date: Optional[datetime] = Query(None, description="End date filter"),
    limit: int = Query(100, le=1000, description="Maximum number of results"),
    offset: int = Query(0, description="Offset for pagination"),
    current_user: User = Depends(get_current_admin_user),
    db: AsyncSession = Depends(get_db)
):
    """Get audit logs (admin only)"""
    
    try:
        # Convert string event types to enum
        event_type_enums = None
        if event_types:
            event_type_enums = []
            for et in event_types:
                try:
                    event_type_enums.append(AuditEventType(et))
                except ValueError:
                    continue
        
        # Convert severity string to enum
        severity_enum = None
        if severity:
            try:
                severity_enum = AuditSeverity(severity)
            except ValueError:
                pass
        
        audit_logs = await audit_logger.search_audit_logs(
            db=db,
            event_types=event_type_enums,
            user_id=user_id,
            ip_address=ip_address,
            start_date=start_date,
            end_date=end_date,
            severity=severity_enum,
            limit=limit,
            offset=offset
        )
        
        return [AuditLogResponse(**log.to_dict()) for log in audit_logs]
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve audit logs"
        )


@router.get("/audit-logs/user", response_model=List[AuditLogResponse])
async def get_user_audit_logs(
    limit: int = Query(50, le=100, description="Maximum number of results"),
    offset: int = Query(0, description="Offset for pagination"),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get audit logs for the current user"""
    
    try:
        # Users can only see their own audit logs
        audit_logs = await audit_logger.search_audit_logs(
            db=db,
            user_id=str(current_user.id),
            limit=limit,
            offset=offset
        )
        
        # Filter out sensitive information for regular users
        filtered_logs = []
        for log in audit_logs:
            log_dict = log.to_dict()
            # Remove sensitive details
            if log_dict.get("details"):
                sensitive_keys = ["password", "token", "secret", "key"]
                log_dict["details"] = {
                    k: v for k, v in log_dict["details"].items()
                    if not any(sensitive in k.lower() for sensitive in sensitive_keys)
                }
            filtered_logs.append(AuditLogResponse(**log_dict))
        
        return filtered_logs
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve your audit logs"
        )


@router.get("/summary", response_model=SecuritySummaryResponse)
async def get_security_summary(
    days: int = Query(7, le=30, description="Number of days to analyze"),
    current_user: User = Depends(get_current_admin_user),
    db: AsyncSession = Depends(get_db)
):
    """Get security summary and statistics (admin only)"""
    
    try:
        start_date = datetime.utcnow() - timedelta(days=days)
        
        # Get all audit logs for the period
        all_logs = await audit_logger.search_audit_logs(
            db=db,
            start_date=start_date,
            limit=10000  # Large limit to get all records
        )
        
        # Calculate statistics
        total_events = len(all_logs)
        
        events_by_severity = {}
        events_by_type = {}
        ip_counts = {}
        suspicious_events = []
        
        for log in all_logs:
            # Count by severity
            severity = log.severity
            events_by_severity[severity] = events_by_severity.get(severity, 0) + 1
            
            # Count by type
            event_type = log.event_type
            events_by_type[event_type] = events_by_type.get(event_type, 0) + 1
            
            # Count by IP
            if log.ip_address:
                ip_counts[log.ip_address] = ip_counts.get(log.ip_address, 0) + 1
            
            # Collect suspicious events
            if log.severity in [AuditSeverity.HIGH.value, AuditSeverity.CRITICAL.value]:
                suspicious_events.append(AuditLogResponse(**log.to_dict()))
        
        # Get top source IPs
        top_source_ips = [
            {"ip_address": ip, "request_count": count}
            for ip, count in sorted(ip_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        ]
        
        # Sort suspicious events by timestamp (most recent first)
        suspicious_events.sort(key=lambda x: x.timestamp, reverse=True)
        
        return SecuritySummaryResponse(
            total_events=total_events,
            events_by_severity=events_by_severity,
            events_by_type=events_by_type,
            recent_suspicious_activity=suspicious_events[:20],
            top_source_ips=top_source_ips
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate security summary"
        )


@router.get("/user-security", response_model=UserSecurityResponse)
async def get_user_security_info(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get security information for the current user"""
    
    try:
        # Get recent login events
        login_events = await audit_logger.search_audit_logs(
            db=db,
            event_types=[AuditEventType.LOGIN_SUCCESS],
            user_id=str(current_user.id),
            start_date=datetime.utcnow() - timedelta(days=30),
            limit=10
        )
        
        recent_logins = []
        for event in login_events:
            recent_logins.append({
                "timestamp": event.timestamp.isoformat(),
                "ip_address": event.ip_address,
                "user_agent": event.user_agent,
                "location": event.details.get("location") if event.details else None
            })
        
        # Get failed login attempts
        failed_logins = await audit_logger.search_audit_logs(
            db=db,
            event_types=[AuditEventType.LOGIN_FAILURE],
            user_id=str(current_user.id),
            start_date=datetime.utcnow() - timedelta(days=7),
            limit=100
        )
        
        # Get last password change
        password_changes = await audit_logger.search_audit_logs(
            db=db,
            event_types=[AuditEventType.PASSWORD_CHANGE],
            user_id=str(current_user.id),
            limit=1
        )
        
        last_password_change = None
        if password_changes:
            last_password_change = password_changes[0].timestamp.isoformat()
        
        # Calculate security score
        security_score = 100
        recommendations = []
        
        # Deduct points for failed logins
        if len(failed_logins) > 5:
            security_score -= 20
            recommendations.append("Multiple failed login attempts detected. Consider enabling 2FA.")
        
        # Deduct points for old password
        if not last_password_change:
            security_score -= 30
            recommendations.append("Password has never been changed. Consider updating your password.")
        elif password_changes:
            days_since_change = (datetime.utcnow() - password_changes[0].timestamp).days
            if days_since_change > 90:
                security_score -= 15
                recommendations.append("Password is over 90 days old. Consider updating it.")
        
        # Check for suspicious activity
        suspicious_events = await audit_logger.search_audit_logs(
            db=db,
            event_types=[AuditEventType.SUSPICIOUS_ACTIVITY, AuditEventType.UNAUTHORIZED_ACCESS],
            user_id=str(current_user.id),
            start_date=datetime.utcnow() - timedelta(days=30),
            limit=10
        )
        
        if suspicious_events:
            security_score -= 25
            recommendations.append("Suspicious activity detected on your account. Review recent activity.")
        
        # Add positive recommendations
        if security_score >= 80:
            recommendations.append("Your account security looks good!")
        
        if not recommendations:
            recommendations.append("Consider enabling two-factor authentication for enhanced security.")
        
        return UserSecurityResponse(
            user_id=str(current_user.id),
            recent_logins=recent_logins,
            failed_login_attempts=len(failed_logins),
            last_password_change=last_password_change,
            active_sessions=1,  # TODO: Implement session tracking
            security_score=max(0, security_score),
            recommendations=recommendations
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve security information"
        )


class SecurityIncidentReport(BaseModel):
    incident_type: str = Field(..., description="Type of security incident")
    description: str = Field(..., description="Description of the incident")

@router.post("/report-incident")
async def report_security_incident(
    request: SecurityIncidentReport,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Report a security incident"""
    
    try:
        # Log the security incident
        await audit_logger.log_security_event(
            db=db,
            event_type=AuditEventType.SUSPICIOUS_ACTIVITY,
            message=f"User reported security incident: {request.incident_type}",
            ip_address="user_reported",
            severity=AuditSeverity.MEDIUM,
            user_id=str(current_user.id),
            details={
                "incident_type": request.incident_type,
                "description": request.description,
                "reported_by": str(current_user.id),
                "reported_at": datetime.utcnow().isoformat()
            }
        )
        
        return {
            "message": "Security incident reported successfully",
            "incident_id": f"INC-{datetime.utcnow().strftime('%Y%m%d')}-{str(current_user.id)[:8]}",
            "status": "under_review"
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to report security incident"
        )


@router.get("/blocked-ips")
async def get_blocked_ips(
    current_user: User = Depends(get_current_admin_user),
    db: AsyncSession = Depends(get_db)
):
    """Get list of blocked IP addresses (admin only)"""
    
    try:
        # Get IP blocking events from audit logs
        blocked_events = await audit_logger.search_audit_logs(
            db=db,
            event_types=[AuditEventType.IP_BLOCKED],
            start_date=datetime.utcnow() - timedelta(days=7),
            limit=100
        )
        
        blocked_ips = []
        for event in blocked_events:
            if event.details:
                blocked_ips.append({
                    "ip_address": event.details.get("ip"),
                    "blocked_at": event.timestamp.isoformat(),
                    "duration": event.details.get("duration"),
                    "reason": event.message,
                    "block_until": event.details.get("block_until")
                })
        
        return {
            "blocked_ips": blocked_ips,
            "total_count": len(blocked_ips)
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve blocked IPs"
        )