"""
Custom exception classes for the application
"""
import traceback
from typing import Any, Dict, List, Optional, Union
from datetime import datetime


class APIException(Exception):
    """Base API exception class"""
    
    def __init__(
        self,
        status_code: int,
        detail: str,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        recovery_suggestions: Optional[List[str]] = None,
        user_friendly_message: Optional[str] = None,
    ):
        self.status_code = status_code
        self.detail = detail
        self.error_code = error_code
        self.details = details or {}
        self.recovery_suggestions = recovery_suggestions or []
        self.user_friendly_message = user_friendly_message or detail
        self.timestamp = datetime.utcnow()
        super().__init__(detail)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for JSON response"""
        return {
            "error": {
                "code": self.error_code or "API_ERROR",
                "message": self.user_friendly_message,
                "technical_detail": self.detail,
                "timestamp": self.timestamp.isoformat(),
                "details": self.details,
                "recovery_suggestions": self.recovery_suggestions,
            }
        }


class AuthenticationError(APIException):
    """Authentication related errors"""
    
    def __init__(self, detail: str = "Authentication failed"):
        super().__init__(
            status_code=401,
            detail=detail,
            error_code="AUTHENTICATION_ERROR"
        )


class AuthorizationError(APIException):
    """Authorization related errors"""
    
    def __init__(self, detail: str = "Access denied"):
        super().__init__(
            status_code=403,
            detail=detail,
            error_code="AUTHORIZATION_ERROR"
        )


class ValidationError(APIException):
    """Validation related errors"""
    
    def __init__(self, detail: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            status_code=400,
            detail=detail,
            error_code="VALIDATION_ERROR",
            details=details
        )


class NotFoundError(APIException):
    """Resource not found errors"""
    
    def __init__(self, detail: str = "Resource not found"):
        super().__init__(
            status_code=404,
            detail=detail,
            error_code="NOT_FOUND"
        )


class ConflictError(APIException):
    """Resource conflict errors"""
    
    def __init__(self, detail: str = "Resource conflict"):
        super().__init__(
            status_code=409,
            detail=detail,
            error_code="CONFLICT"
        )


class RateLimitError(APIException):
    """Rate limiting errors"""
    
    def __init__(self, detail: str = "Rate limit exceeded"):
        super().__init__(
            status_code=429,
            detail=detail,
            error_code="RATE_LIMIT_EXCEEDED"
        )


class ExternalServiceError(APIException):
    """External service integration errors"""
    
    def __init__(self, service: str, detail: str = "External service error"):
        super().__init__(
            status_code=502,
            detail=f"{service}: {detail}",
            error_code="EXTERNAL_SERVICE_ERROR"
        )


class ServiceException(Exception):
    """General service layer exception"""
    
    def __init__(self, message: str, error_code: Optional[str] = None):
        self.message = message
        self.error_code = error_code
        super().__init__(message)


class AnalyticsError(ServiceException):
    """Analytics service specific errors"""
    
    def __init__(self, message: str):
        super().__init__(message, "ANALYTICS_ERROR")


class DataNotFoundError(ServiceException):
    """Data not found errors"""
    
    def __init__(self, message: str):
        super().__init__(message, "DATA_NOT_FOUND")


class VisualizationError(ServiceException):
    """Visualization service specific errors"""
    
    def __init__(self, message: str):
        super().__init__(message, "VISUALIZATION_ERROR")


class PDFGenerationError(ServiceException):
    """PDF generation specific errors"""
    
    def __init__(self, message: str):
        super().__init__(message, "PDF_GENERATION_ERROR")


# External API specific exceptions
class ExternalAPIException(APIException):
    """Base class for external API errors"""
    
    def __init__(
        self,
        service: str,
        detail: str,
        status_code: int = 502,
        original_error: Optional[Exception] = None,
        retry_after: Optional[int] = None,
    ):
        recovery_suggestions = [
            f"The {service} service is temporarily unavailable",
            "Please try again in a few minutes",
            "If the problem persists, contact support"
        ]
        
        if retry_after:
            recovery_suggestions.insert(1, f"Retry after {retry_after} seconds")
        
        super().__init__(
            status_code=status_code,
            detail=f"{service} API error: {detail}",
            error_code="EXTERNAL_API_ERROR",
            details={
                "service": service,
                "original_error": str(original_error) if original_error else None,
                "retry_after": retry_after,
            },
            recovery_suggestions=recovery_suggestions,
            user_friendly_message=f"We're having trouble connecting to {service}. Please try again later."
        )


class GitHubAPIError(ExternalAPIException):
    """GitHub API specific errors"""
    
    def __init__(self, detail: str, original_error: Optional[Exception] = None):
        super().__init__(
            service="GitHub",
            detail=detail,
            original_error=original_error
        )


class LeetCodeAPIError(ExternalAPIException):
    """LeetCode API specific errors"""
    
    def __init__(self, detail: str, original_error: Optional[Exception] = None):
        super().__init__(
            service="LeetCode",
            detail=detail,
            original_error=original_error
        )


class LinkedInAPIError(ExternalAPIException):
    """LinkedIn API specific errors"""
    
    def __init__(self, detail: str, original_error: Optional[Exception] = None):
        super().__init__(
            service="LinkedIn",
            detail=detail,
            original_error=original_error
        )


class JobScrapingError(ExternalAPIException):
    """Job scraping specific errors"""
    
    def __init__(self, platform: str, detail: str, original_error: Optional[Exception] = None):
        super().__init__(
            service=f"{platform} Job Scraper",
            detail=detail,
            original_error=original_error
        )


# ML Model specific exceptions
class MLModelError(APIException):
    """Machine learning model errors"""
    
    def __init__(
        self,
        model_name: str,
        detail: str,
        fallback_available: bool = False,
        original_error: Optional[Exception] = None,
    ):
        recovery_suggestions = [
            "The AI model is temporarily unavailable",
            "Please try again in a few minutes"
        ]
        
        if fallback_available:
            recovery_suggestions.append("A simplified version of the feature is being used")
        
        super().__init__(
            status_code=503,
            detail=f"ML Model '{model_name}' error: {detail}",
            error_code="ML_MODEL_ERROR",
            details={
                "model_name": model_name,
                "fallback_available": fallback_available,
                "original_error": str(original_error) if original_error else None,
            },
            recovery_suggestions=recovery_suggestions,
            user_friendly_message="Our AI system is temporarily unavailable. Please try again later."
        )


class RecommendationEngineError(MLModelError):
    """Recommendation engine specific errors"""
    
    def __init__(self, detail: str, original_error: Optional[Exception] = None):
        super().__init__(
            model_name="Recommendation Engine",
            detail=detail,
            fallback_available=True,
            original_error=original_error
        )


class NLPProcessingError(MLModelError):
    """NLP processing specific errors"""
    
    def __init__(self, detail: str, original_error: Optional[Exception] = None):
        super().__init__(
            model_name="NLP Engine",
            detail=detail,
            fallback_available=False,
            original_error=original_error
        )


# Database specific exceptions
class DatabaseError(APIException):
    """Database operation errors"""
    
    def __init__(
        self,
        detail: str,
        operation: Optional[str] = None,
        table: Optional[str] = None,
        original_error: Optional[Exception] = None,
    ):
        recovery_suggestions = [
            "Database is temporarily unavailable",
            "Please try again in a few minutes",
            "If the problem persists, contact support"
        ]
        
        super().__init__(
            status_code=503,
            detail=f"Database error: {detail}",
            error_code="DATABASE_ERROR",
            details={
                "operation": operation,
                "table": table,
                "original_error": str(original_error) if original_error else None,
            },
            recovery_suggestions=recovery_suggestions,
            user_friendly_message="We're experiencing database issues. Please try again later."
        )


class CacheError(APIException):
    """Cache operation errors"""
    
    def __init__(self, detail: str, original_error: Optional[Exception] = None):
        super().__init__(
            status_code=503,
            detail=f"Cache error: {detail}",
            error_code="CACHE_ERROR",
            details={
                "original_error": str(original_error) if original_error else None,
            },
            recovery_suggestions=[
                "Cache service is temporarily unavailable",
                "The system will work but may be slower",
                "Please try again if you experience issues"
            ],
            user_friendly_message="System performance may be temporarily affected."
        )


# Business logic exceptions
class InsufficientDataError(APIException):
    """Insufficient data for processing"""
    
    def __init__(self, required_data: List[str], missing_data: List[str]):
        detail = f"Insufficient data for processing. Missing: {', '.join(missing_data)}"
        
        recovery_suggestions = [
            "Please provide the following information:",
            *[f"- {item}" for item in missing_data],
            "You can update your profile to add this information"
        ]
        
        super().__init__(
            status_code=400,
            detail=detail,
            error_code="INSUFFICIENT_DATA",
            details={
                "required_data": required_data,
                "missing_data": missing_data,
            },
            recovery_suggestions=recovery_suggestions,
            user_friendly_message="We need more information to provide accurate recommendations."
        )


class ProcessingTimeoutError(APIException):
    """Processing timeout errors"""
    
    def __init__(self, operation: str, timeout_seconds: int):
        super().__init__(
            status_code=408,
            detail=f"Operation '{operation}' timed out after {timeout_seconds} seconds",
            error_code="PROCESSING_TIMEOUT",
            details={
                "operation": operation,
                "timeout_seconds": timeout_seconds,
            },
            recovery_suggestions=[
                "The operation is taking longer than expected",
                "Please try again with a smaller dataset",
                "Consider breaking the request into smaller parts"
            ],
            user_friendly_message="The request is taking too long to process. Please try again."
        )


# System health exceptions
class SystemHealthError(APIException):
    """System health check errors"""
    
    def __init__(self, component: str, status: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            status_code=503,
            detail=f"System component '{component}' is {status}",
            error_code="SYSTEM_HEALTH_ERROR",
            details={
                "component": component,
                "status": status,
                **(details or {})
            },
            recovery_suggestions=[
                f"The {component} service is currently unavailable",
                "Please try again in a few minutes",
                "Check system status page for updates"
            ],
            user_friendly_message="Some system components are temporarily unavailable."
        )


class ProcessingError(APIException):
    """File processing errors"""
    
    def __init__(
        self,
        detail: str,
        processing_type: Optional[str] = None,
        original_error: Optional[Exception] = None,
        fallback_available: bool = False,
    ):
        recovery_suggestions = [
            "File processing failed",
            "Please check the file format and try again"
        ]
        
        if fallback_available:
            recovery_suggestions.append("You can try manual data entry as an alternative")
        
        super().__init__(
            status_code=422,
            detail=f"Processing error: {detail}",
            error_code="PROCESSING_ERROR",
            details={
                "processing_type": processing_type,
                "original_error": str(original_error) if original_error else None,
                "fallback_available": fallback_available,
            },
            recovery_suggestions=recovery_suggestions,
            user_friendly_message="We couldn't process your file. Please try a different format or enter the information manually."
        )

# Additional exceptions for enhanced dashboard functionality
class ScrapingError(ExternalAPIException):
    """General scraping errors"""
    
    def __init__(self, detail: str, original_error: Optional[Exception] = None):
        super().__init__(
            service="Web Scraper",
            detail=detail,
            original_error=original_error
        )


class MatchingError(ServiceException):
    """Job matching service errors"""
    
    def __init__(self, message: str):
        super().__init__(message, "MATCHING_ERROR")


class DataSyncError(APIException):
    """Data synchronization errors"""
    
    def __init__(
        self,
        detail: str,
        sync_operation: Optional[str] = None,
        original_error: Optional[Exception] = None,
    ):
        recovery_suggestions = [
            "Data synchronization failed",
            "Please try again in a few minutes",
            "Check your network connection"
        ]
        
        super().__init__(
            status_code=503,
            detail=f"Data sync error: {detail}",
            error_code="DATA_SYNC_ERROR",
            details={
                "sync_operation": sync_operation,
                "original_error": str(original_error) if original_error else None,
            },
            recovery_suggestions=recovery_suggestions,
            user_friendly_message="We're having trouble syncing your data. Please try again later."
        )


class ConflictResolutionError(APIException):
    """Data conflict resolution errors"""
    
    def __init__(
        self,
        detail: str,
        conflict_type: Optional[str] = None,
        conflicting_fields: Optional[List[str]] = None,
    ):
        recovery_suggestions = [
            "Data conflict detected",
            "Please review and resolve the conflicts manually",
            "Choose which version of the data to keep"
        ]
        
        if conflicting_fields:
            recovery_suggestions.append(f"Conflicting fields: {', '.join(conflicting_fields)}")
        
        super().__init__(
            status_code=409,
            detail=f"Conflict resolution error: {detail}",
            error_code="CONFLICT_RESOLUTION_ERROR",
            details={
                "conflict_type": conflict_type,
                "conflicting_fields": conflicting_fields,
            },
            recovery_suggestions=recovery_suggestions,
            user_friendly_message="We found conflicting data that needs your attention to resolve."
        )