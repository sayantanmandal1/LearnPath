"""
Global error handling middleware with user-friendly messages and recovery suggestions
"""
import traceback
import uuid
from typing import Any, Dict, List

import structlog
from fastapi import Request, Response, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
from sqlalchemy.exc import SQLAlchemyError
from redis.exceptions import RedisError
from httpx import HTTPError, TimeoutException

from app.core.exceptions import APIException
from app.core.monitoring import system_monitor, AlertType, AlertSeverity

logger = structlog.get_logger()


class ErrorHandlerMiddleware(BaseHTTPMiddleware):
    """Enhanced global error handling middleware"""
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.error_patterns = self._setup_error_patterns()
    
    def _setup_error_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Setup error patterns for user-friendly messages"""
        return {
            "database_connection": {
                "keywords": ["connection", "database", "postgresql", "sqlalchemy"],
                "user_message": "We're experiencing database connectivity issues. Please try again in a few minutes.",
                "recovery_suggestions": [
                    "Wait a few minutes and try again",
                    "Check if you have a stable internet connection",
                    "Contact support if the problem persists"
                ],
                "severity": AlertSeverity.HIGH
            },
            "redis_connection": {
                "keywords": ["redis", "cache", "connection refused"],
                "user_message": "Our caching system is temporarily unavailable. The service may be slower than usual.",
                "recovery_suggestions": [
                    "The system will continue to work but may be slower",
                    "Try refreshing the page if you experience issues",
                    "Contact support if problems persist"
                ],
                "severity": AlertSeverity.MEDIUM
            },
            "external_api_timeout": {
                "keywords": ["timeout", "external", "api", "httpx"],
                "user_message": "We're having trouble connecting to external services. Please try again later.",
                "recovery_suggestions": [
                    "Wait a few minutes and try again",
                    "Some features may be temporarily unavailable",
                    "Check our status page for updates"
                ],
                "severity": AlertSeverity.MEDIUM
            },
            "ml_model_error": {
                "keywords": ["model", "prediction", "torch", "tensorflow", "sklearn"],
                "user_message": "Our AI system is temporarily unavailable. We're working to restore full functionality.",
                "recovery_suggestions": [
                    "Try using basic features instead of AI-powered ones",
                    "Check back in a few minutes for full functionality",
                    "Contact support if you need immediate assistance"
                ],
                "severity": AlertSeverity.HIGH
            },
            "file_processing": {
                "keywords": ["file", "upload", "pdf", "parsing", "resume"],
                "user_message": "We couldn't process your file. Please check the format and try again.",
                "recovery_suggestions": [
                    "Ensure your file is in PDF, DOC, or DOCX format",
                    "Check that the file size is under 10MB",
                    "Try uploading a different version of the file",
                    "Contact support if the file should be valid"
                ],
                "severity": AlertSeverity.LOW
            },
            "rate_limit": {
                "keywords": ["rate", "limit", "too many", "requests"],
                "user_message": "You're making requests too quickly. Please slow down and try again.",
                "recovery_suggestions": [
                    "Wait a minute before making another request",
                    "Reduce the frequency of your requests",
                    "Contact support if you need higher rate limits"
                ],
                "severity": AlertSeverity.LOW
            },
            "authentication": {
                "keywords": ["authentication", "token", "unauthorized", "login"],
                "user_message": "Your session has expired. Please log in again.",
                "recovery_suggestions": [
                    "Log out and log back in",
                    "Clear your browser cache and cookies",
                    "Reset your password if you can't log in"
                ],
                "severity": AlertSeverity.LOW
            },
            "validation": {
                "keywords": ["validation", "invalid", "required", "format"],
                "user_message": "The information you provided is invalid. Please check and try again.",
                "recovery_suggestions": [
                    "Check that all required fields are filled",
                    "Ensure data is in the correct format",
                    "Review the error details for specific issues"
                ],
                "severity": AlertSeverity.LOW
            }
        }
    
    def _classify_error(self, error_message: str, error_type: str) -> Dict[str, Any]:
        """Classify error and return user-friendly information"""
        error_message_lower = error_message.lower()
        error_type_lower = error_type.lower()
        
        for pattern_name, pattern_info in self.error_patterns.items():
            keywords = pattern_info["keywords"]
            
            # Check if any keyword matches
            if any(keyword in error_message_lower or keyword in error_type_lower 
                   for keyword in keywords):
                return pattern_info
        
        # Default fallback
        return {
            "user_message": "An unexpected error occurred. Please try again later.",
            "recovery_suggestions": [
                "Try refreshing the page",
                "Wait a few minutes and try again",
                "Contact support if the problem persists"
            ],
            "severity": AlertSeverity.MEDIUM
        }
    
    def _generate_error_id(self) -> str:
        """Generate unique error ID for tracking"""
        return str(uuid.uuid4())[:8]
    
    async def dispatch(self, request: Request, call_next) -> Response:
        """Handle requests and catch exceptions"""
        error_id = None
        
        try:
            response = await call_next(request)
            return response
            
        except APIException as exc:
            # Handle custom API exceptions (already have user-friendly messages)
            logger.warning(
                "API exception occurred",
                path=request.url.path,
                method=request.method,
                status_code=exc.status_code,
                detail=exc.detail,
                error_code=exc.error_code,
                user_agent=request.headers.get("user-agent"),
                client_ip=request.client.host if request.client else None,
            )
            
            return JSONResponse(
                status_code=exc.status_code,
                content=exc.to_dict()
            )
            
        except SQLAlchemyError as exc:
            # Handle database errors
            error_id = self._generate_error_id()
            error_info = self._classify_error(str(exc), "SQLAlchemyError")
            
            logger.error(
                "Database error occurred",
                error_id=error_id,
                path=request.url.path,
                method=request.method,
                error=str(exc),
                user_agent=request.headers.get("user-agent"),
                client_ip=request.client.host if request.client else None,
            )
            
            # Create alert for database issues
            system_monitor.create_alert(
                alert_type=AlertType.SERVICE_DOWN,
                severity=error_info["severity"],
                title="Database Error",
                description=f"Database error occurred: {str(exc)[:200]}",
                service="database",
                metadata={"error_id": error_id, "path": request.url.path}
            )
            
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content={
                    "error": {
                        "code": "DATABASE_ERROR",
                        "message": error_info["user_message"],
                        "error_id": error_id,
                        "recovery_suggestions": error_info["recovery_suggestions"],
                        "timestamp": str(exc.timestamp) if hasattr(exc, 'timestamp') else None,
                    }
                }
            )
            
        except RedisError as exc:
            # Handle Redis/cache errors
            error_id = self._generate_error_id()
            error_info = self._classify_error(str(exc), "RedisError")
            
            logger.warning(
                "Redis error occurred",
                error_id=error_id,
                path=request.url.path,
                method=request.method,
                error=str(exc),
            )
            
            # Create alert for cache issues
            system_monitor.create_alert(
                alert_type=AlertType.SERVICE_DOWN,
                severity=error_info["severity"],
                title="Cache Service Error",
                description=f"Redis error occurred: {str(exc)[:200]}",
                service="redis",
                metadata={"error_id": error_id, "path": request.url.path}
            )
            
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content={
                    "error": {
                        "code": "CACHE_ERROR",
                        "message": error_info["user_message"],
                        "error_id": error_id,
                        "recovery_suggestions": error_info["recovery_suggestions"],
                    }
                }
            )
            
        except (HTTPError, TimeoutException) as exc:
            # Handle external API errors
            error_id = self._generate_error_id()
            error_info = self._classify_error(str(exc), "HTTPError")
            
            logger.warning(
                "External API error occurred",
                error_id=error_id,
                path=request.url.path,
                method=request.method,
                error=str(exc),
            )
            
            return JSONResponse(
                status_code=status.HTTP_502_BAD_GATEWAY,
                content={
                    "error": {
                        "code": "EXTERNAL_API_ERROR",
                        "message": error_info["user_message"],
                        "error_id": error_id,
                        "recovery_suggestions": error_info["recovery_suggestions"],
                    }
                }
            )
            
        except ValueError as exc:
            # Handle validation errors
            error_id = self._generate_error_id()
            error_info = self._classify_error(str(exc), "ValueError")
            
            logger.warning(
                "Validation error occurred",
                error_id=error_id,
                path=request.url.path,
                method=request.method,
                error=str(exc),
            )
            
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={
                    "error": {
                        "code": "VALIDATION_ERROR",
                        "message": error_info["user_message"],
                        "technical_detail": str(exc),
                        "error_id": error_id,
                        "recovery_suggestions": error_info["recovery_suggestions"],
                    }
                }
            )
            
        except Exception as exc:
            # Handle unexpected errors
            error_id = self._generate_error_id()
            error_info = self._classify_error(str(exc), type(exc).__name__)
            
            logger.error(
                "Unexpected error occurred",
                error_id=error_id,
                path=request.url.path,
                method=request.method,
                error=str(exc),
                error_type=type(exc).__name__,
                traceback=traceback.format_exc(),
                user_agent=request.headers.get("user-agent"),
                client_ip=request.client.host if request.client else None,
            )
            
            # Create critical alert for unexpected errors
            system_monitor.create_alert(
                alert_type=AlertType.SECURITY_INCIDENT,
                severity=AlertSeverity.CRITICAL,
                title="Unexpected System Error",
                description=f"Unexpected error: {str(exc)[:200]}",
                metadata={
                    "error_id": error_id,
                    "path": request.url.path,
                    "error_type": type(exc).__name__,
                    "traceback": traceback.format_exc()[:1000]
                }
            )
            
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={
                    "error": {
                        "code": "INTERNAL_SERVER_ERROR",
                        "message": error_info["user_message"],
                        "error_id": error_id,
                        "recovery_suggestions": error_info["recovery_suggestions"],
                        "support_info": {
                            "message": "Please provide this error ID when contacting support",
                            "error_id": error_id,
                            "timestamp": traceback.format_exc().split('\n')[-2] if traceback.format_exc() else None
                        }
                    }
                }
            )