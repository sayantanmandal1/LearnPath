"""
Error handling and system health endpoints for enhanced profile analysis

These endpoints provide error reporting, system health monitoring,
and user-friendly error information.
"""
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from app.core.database import get_db
from app.core.graceful_degradation import degradation_manager, ServiceStatus
from app.core.user_friendly_errors import user_friendly_error_generator, ErrorCategory, ErrorSeverity
from app.services.error_recovery_service import error_recovery_service
from app.services.external_apis.circuit_breaker import circuit_breaker_manager
from app.models.analysis_result import AnalysisResult
from app.core.exceptions import APIException
from app.api.dependencies import get_current_user

router = APIRouter()


@router.get("/health", summary="Get system health status")
async def get_system_health():
    """
    Get comprehensive system health status including all services
    
    Returns:
        Dict containing health status of all monitored services
    """
    try:
        # Get service health from degradation manager
        services_health = degradation_manager.get_all_services_health()
        
        # Get circuit breaker stats
        circuit_breaker_stats = circuit_breaker_manager.get_all_stats()
        
        # Calculate overall system health
        healthy_services = sum(1 for service in services_health.values() if service.is_healthy)
        total_services = len(services_health)
        overall_health_percentage = (healthy_services / total_services * 100) if total_services > 0 else 100
        
        # Determine overall status
        if overall_health_percentage >= 90:
            overall_status = "healthy"
        elif overall_health_percentage >= 70:
            overall_status = "degraded"
        else:
            overall_status = "unhealthy"
        
        return {
            "status": overall_status,
            "timestamp": datetime.utcnow().isoformat(),
            "overall_health_percentage": overall_health_percentage,
            "services": {
                name: {
                    "status": service.status.value,
                    "last_check": datetime.fromtimestamp(service.last_check).isoformat(),
                    "error_count": service.error_count,
                    "success_count": service.success_count,
                    "success_rate": service.success_rate,
                    "response_time": service.response_time,
                    "last_error": service.last_error,
                    "fallback_available": service.fallback_available
                }
                for name, service in services_health.items()
            },
            "circuit_breakers": {
                name: {
                    "state": stats.state.value,
                    "failure_count": stats.failure_count,
                    "success_count": stats.success_count,
                    "total_requests": stats.total_requests,
                    "total_failures": stats.total_failures,
                    "total_successes": stats.total_successes,
                    "last_failure_time": stats.last_failure_time.isoformat() if stats.last_failure_time else None,
                    "last_success_time": stats.last_success_time.isoformat() if stats.last_success_time else None
                }
                for name, stats in circuit_breaker_stats.items()
            }
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get system health: {str(e)}"
        )


@router.get("/health/{service_name}", summary="Get specific service health")
async def get_service_health(service_name: str):
    """
    Get health status for a specific service
    
    Args:
        service_name: Name of the service to check
        
    Returns:
        Dict containing detailed health information for the service
    """
    service_health = degradation_manager.get_service_health(service_name)
    
    if not service_health:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Service '{service_name}' not found"
        )
    
    circuit_breaker_stats = circuit_breaker_manager.get_all_stats().get(service_name)
    
    return {
        "service_name": service_name,
        "status": service_health.status.value,
        "last_check": datetime.fromtimestamp(service_health.last_check).isoformat(),
        "error_count": service_health.error_count,
        "success_count": service_health.success_count,
        "success_rate": service_health.success_rate,
        "response_time": service_health.response_time,
        "last_error": service_health.last_error,
        "fallback_available": service_health.fallback_available,
        "circuit_breaker": {
            "state": circuit_breaker_stats.state.value if circuit_breaker_stats else "unknown",
            "failure_count": circuit_breaker_stats.failure_count if circuit_breaker_stats else 0,
            "success_count": circuit_breaker_stats.success_count if circuit_breaker_stats else 0,
            "total_requests": circuit_breaker_stats.total_requests if circuit_breaker_stats else 0
        } if circuit_breaker_stats else None
    }


@router.get("/errors/patterns", summary="Get error patterns and user-friendly messages")
async def get_error_patterns(
    category: Optional[ErrorCategory] = Query(None, description="Filter by error category"),
    severity: Optional[ErrorSeverity] = Query(None, description="Filter by error severity")
):
    """
    Get available error patterns with user-friendly messages
    
    Args:
        category: Optional category filter
        severity: Optional severity filter
        
    Returns:
        Dict containing error patterns and their user-friendly representations
    """
    try:
        # Get all error patterns from the generator
        all_patterns = user_friendly_error_generator.error_patterns
        
        # Filter patterns if requested
        filtered_patterns = {}
        for key, pattern in all_patterns.items():
            # Apply category filter
            if category and pattern["category"] != category:
                continue
            
            # Apply severity filter  
            if severity and pattern["severity"] != severity:
                continue
            
            filtered_patterns[key] = {
                "title": pattern["title"],
                "message": pattern["message"],
                "category": pattern["category"].value,
                "severity": pattern["severity"].value,
                "recovery_suggestions": pattern["recovery_suggestions"],
                "estimated_resolution_time": pattern.get("estimated_resolution_time"),
                "contact_support": pattern.get("contact_support", False),
                "retry_available": pattern.get("retry_available", True),
                "fallback_available": pattern.get("fallback_available", False),
                "user_action_required": pattern.get("user_action_required", False)
            }
        
        return {
            "patterns": filtered_patterns,
            "total_count": len(filtered_patterns),
            "categories": [cat.value for cat in ErrorCategory],
            "severities": [sev.value for sev in ErrorSeverity]
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get error patterns: {str(e)}"
        )


@router.post("/errors/report", summary="Report an error for analysis")
async def report_error(
    error_data: Dict[str, Any],
    current_user: Dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Report an error for analysis and get user-friendly response
    
    Args:
        error_data: Error information including type, message, context
        current_user: Current authenticated user
        db: Database session
        
    Returns:
        Dict containing user-friendly error information and recovery suggestions
    """
    try:
        # Extract error information
        error_type = error_data.get("error_type", "UnknownError")
        error_message = error_data.get("error_message", "")
        operation = error_data.get("operation", "unknown")
        service_name = error_data.get("service_name", "unknown")
        
        # Create mock exception for error generation
        class MockException(Exception):
            pass
        
        MockException.__name__ = error_type
        mock_exception = MockException(error_message)
        
        # Generate user-friendly error
        context = {
            "user_id": current_user.get("user_id"),
            "operation": operation,
            "service_name": service_name,
            **error_data.get("context", {})
        }
        
        user_friendly_error = user_friendly_error_generator.get_error_by_exception_type(
            mock_exception, context
        )
        
        # Format for API response
        error_response = user_friendly_error_generator.format_error_for_api_response(
            user_friendly_error
        )
        
        # Log the error report
        import structlog
        logger = structlog.get_logger()
        logger.info(
            "Error reported by user",
            user_id=current_user.get("user_id"),
            error_type=error_type,
            error_message=error_message,
            operation=operation,
            service_name=service_name
        )
        
        return error_response
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process error report: {str(e)}"
        )


@router.get("/errors/user-stats", summary="Get user error statistics")
async def get_user_error_stats(
    current_user: Dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    days: int = Query(7, description="Number of days to look back")
):
    """
    Get error statistics for the current user
    
    Args:
        current_user: Current authenticated user
        db: Database session
        days: Number of days to look back for statistics
        
    Returns:
        Dict containing user error statistics and trends
    """
    try:
        user_id = current_user.get("user_id")
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User ID not found"
            )
        
        # Calculate date range
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)
        
        # Query analysis results to get error patterns
        result = await db.execute(
            select(
                AnalysisResult.analysis_type,
                func.count(AnalysisResult.id).label("total_attempts"),
                func.avg(AnalysisResult.confidence_score).label("avg_confidence")
            )
            .where(
                AnalysisResult.user_id == user_id,
                AnalysisResult.created_at >= start_date
            )
            .group_by(AnalysisResult.analysis_type)
        )
        
        analysis_stats = result.fetchall()
        
        # Get recent failed operations (low confidence scores)
        failed_result = await db.execute(
            select(AnalysisResult)
            .where(
                AnalysisResult.user_id == user_id,
                AnalysisResult.created_at >= start_date,
                AnalysisResult.confidence_score < 0.5
            )
            .order_by(AnalysisResult.created_at.desc())
            .limit(10)
        )
        
        failed_operations = failed_result.scalars().all()
        
        return {
            "user_id": user_id,
            "period_days": days,
            "analysis_statistics": [
                {
                    "analysis_type": stat.analysis_type.value if hasattr(stat.analysis_type, 'value') else str(stat.analysis_type),
                    "total_attempts": stat.total_attempts,
                    "average_confidence": float(stat.avg_confidence) if stat.avg_confidence else 0.0
                }
                for stat in analysis_stats
            ],
            "recent_issues": [
                {
                    "analysis_type": op.analysis_type.value if hasattr(op.analysis_type, 'value') else str(op.analysis_type),
                    "confidence_score": op.confidence_score,
                    "created_at": op.created_at.isoformat(),
                    "has_fallback_data": bool(op.result_data)
                }
                for op in failed_operations
            ],
            "recommendations": [
                "Consider updating your profile with more detailed information",
                "Try uploading a higher quality resume if you're having parsing issues",
                "Check your platform account connections for accuracy"
            ]
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get user error statistics: {str(e)}"
        )


@router.post("/recovery/reset-circuit-breakers", summary="Reset circuit breakers")
async def reset_circuit_breakers(
    service_name: Optional[str] = Query(None, description="Specific service to reset, or all if not provided")
):
    """
    Reset circuit breakers for recovery
    
    Args:
        service_name: Optional specific service name to reset
        
    Returns:
        Dict containing reset status
    """
    try:
        if service_name:
            await circuit_breaker_manager.reset_breaker(service_name)
            return {
                "success": True,
                "message": f"Circuit breaker for {service_name} has been reset",
                "service": service_name
            }
        else:
            await circuit_breaker_manager.reset_all()
            return {
                "success": True,
                "message": "All circuit breakers have been reset",
                "services_reset": list(circuit_breaker_manager._breakers.keys())
            }
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to reset circuit breakers: {str(e)}"
        )


@router.get("/recovery/suggestions", summary="Get recovery suggestions for current issues")
async def get_recovery_suggestions():
    """
    Get recovery suggestions based on current system status
    
    Returns:
        Dict containing recovery suggestions and system recommendations
    """
    try:
        # Get current service health
        services_health = degradation_manager.get_all_services_health()
        
        suggestions = []
        affected_services = []
        
        for service_name, health in services_health.items():
            if health.status != ServiceStatus.HEALTHY:
                affected_services.append(service_name)
                
                if health.status == ServiceStatus.UNAVAILABLE:
                    if "gemini" in service_name.lower():
                        suggestions.extend([
                            "AI analysis is temporarily unavailable - using rule-based fallbacks",
                            "Try again in 15-30 minutes for full AI functionality",
                            "Basic profile features remain available"
                        ])
                    elif "github" in service_name.lower():
                        suggestions.extend([
                            "GitHub data collection is paused - you can manually add projects",
                            "Check your GitHub username and privacy settings",
                            "Analysis will continue with available data"
                        ])
                    elif "job" in service_name.lower():
                        suggestions.extend([
                            "Job recommendations may be limited - using cached data",
                            "Try searching job sites directly for latest postings",
                            "Premium features include additional job sources"
                        ])
        
        if not affected_services:
            suggestions.append("All systems are operating normally")
        
        return {
            "system_status": "healthy" if not affected_services else "degraded",
            "affected_services": affected_services,
            "recovery_suggestions": suggestions,
            "estimated_resolution": "Most issues resolve within 30 minutes",
            "alternative_actions": [
                "Use manual data entry for unavailable automated features",
                "Try again during off-peak hours for better performance",
                "Contact support for persistent issues"
            ]
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get recovery suggestions: {str(e)}"
        )