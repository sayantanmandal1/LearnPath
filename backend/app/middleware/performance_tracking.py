"""
Performance Tracking Middleware
Automatically tracks API performance metrics and user engagement
"""
import time
import asyncio
from datetime import datetime
from typing import Callable, Optional
import structlog

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from app.services.system_performance_analytics import (
    system_performance_analytics,
    APIPerformanceMetric,
    UserEngagementMetric
)

logger = structlog.get_logger()


class PerformanceTrackingMiddleware(BaseHTTPMiddleware):
    """Middleware to automatically track API performance and user engagement"""
    
    def __init__(
        self,
        app: ASGIApp,
        track_all_endpoints: bool = True,
        excluded_paths: Optional[list] = None,
        track_user_engagement: bool = True
    ):
        super().__init__(app)
        self.track_all_endpoints = track_all_endpoints
        self.excluded_paths = excluded_paths or [
            "/health",
            "/docs",
            "/openapi.json",
            "/favicon.ico",
            "/static"
        ]
        self.track_user_engagement = track_user_engagement
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request and track performance metrics"""
        start_time = time.time()
        
        # Skip tracking for excluded paths
        if any(request.url.path.startswith(path) for path in self.excluded_paths):
            return await call_next(request)
        
        # Extract request information
        method = request.method
        endpoint = request.url.path
        user_id = await self._extract_user_id(request)
        request_size = await self._get_request_size(request)
        
        # Process request
        response = await call_next(request)
        
        # Calculate metrics
        end_time = time.time()
        response_time_ms = (end_time - start_time) * 1000
        status_code = response.status_code
        response_size = self._get_response_size(response)
        
        # Track API performance asynchronously
        if self.track_all_endpoints:
            asyncio.create_task(self._record_api_performance(
                endpoint=endpoint,
                method=method,
                response_time_ms=response_time_ms,
                status_code=status_code,
                user_id=user_id,
                request_size_bytes=request_size,
                response_size_bytes=response_size,
                error_message=await self._extract_error_message(response) if status_code >= 400 else None
            ))
        
        # Track user engagement for specific endpoints
        if self.track_user_engagement and user_id:
            asyncio.create_task(self._record_user_engagement(
                user_id=user_id,
                endpoint=endpoint,
                method=method,
                response_time_ms=response_time_ms,
                status_code=status_code,
                request=request
            ))
        
        return response
    
    async def _extract_user_id(self, request: Request) -> Optional[str]:
        """Extract user ID from request"""
        try:
            # Try to get user ID from JWT token
            auth_header = request.headers.get("authorization")
            if auth_header and auth_header.startswith("Bearer "):
                # This would typically decode the JWT token
                # For now, we'll check if there's a user in the request state
                if hasattr(request.state, "user") and request.state.user:
                    return str(request.state.user.id)
            
            # Try to get from query parameters (for testing)
            user_id = request.query_params.get("user_id")
            if user_id:
                return user_id
            
            # Try to get from headers (for internal services)
            user_id = request.headers.get("x-user-id")
            if user_id:
                return user_id
                
        except Exception as e:
            logger.debug(f"Failed to extract user ID: {e}")
        
        return None
    
    async def _get_request_size(self, request: Request) -> Optional[int]:
        """Get request body size"""
        try:
            content_length = request.headers.get("content-length")
            if content_length:
                return int(content_length)
        except Exception as e:
            logger.debug(f"Failed to get request size: {e}")
        
        return None
    
    def _get_response_size(self, response: Response) -> Optional[int]:
        """Get response body size"""
        try:
            content_length = response.headers.get("content-length")
            if content_length:
                return int(content_length)
        except Exception as e:
            logger.debug(f"Failed to get response size: {e}")
        
        return None
    
    async def _extract_error_message(self, response: Response) -> Optional[str]:
        """Extract error message from response"""
        try:
            if response.status_code >= 400:
                # This is a simplified approach - in practice, you might want to
                # read the response body to extract the actual error message
                return f"HTTP {response.status_code} Error"
        except Exception as e:
            logger.debug(f"Failed to extract error message: {e}")
        
        return None
    
    async def _record_api_performance(
        self,
        endpoint: str,
        method: str,
        response_time_ms: float,
        status_code: int,
        user_id: Optional[str] = None,
        request_size_bytes: Optional[int] = None,
        response_size_bytes: Optional[int] = None,
        error_message: Optional[str] = None
    ):
        """Record API performance metric"""
        try:
            metric = APIPerformanceMetric(
                endpoint=endpoint,
                method=method,
                response_time_ms=response_time_ms,
                status_code=status_code,
                timestamp=datetime.utcnow(),
                user_id=user_id,
                error_message=error_message,
                request_size_bytes=request_size_bytes,
                response_size_bytes=response_size_bytes
            )
            
            await system_performance_analytics.record_api_performance(metric)
            
        except Exception as e:
            logger.error(f"Failed to record API performance metric: {e}")
    
    async def _record_user_engagement(
        self,
        user_id: str,
        endpoint: str,
        method: str,
        response_time_ms: float,
        status_code: int,
        request: Request
    ):
        """Record user engagement metric"""
        try:
            # Determine action and feature from endpoint
            action, feature = self._categorize_endpoint(endpoint, method)
            
            # Skip if not a user-facing action
            if not action or not feature:
                return
            
            # Extract session ID if available
            session_id = request.headers.get("x-session-id") or request.cookies.get("session_id")
            
            # Create engagement metric
            metric = UserEngagementMetric(
                user_id=user_id,
                action=action,
                feature=feature,
                timestamp=datetime.utcnow(),
                session_id=session_id,
                duration_seconds=response_time_ms / 1000,  # Convert to seconds
                metadata={
                    "endpoint": endpoint,
                    "method": method,
                    "status_code": status_code,
                    "user_agent": request.headers.get("user-agent"),
                    "referer": request.headers.get("referer")
                }
            )
            
            await system_performance_analytics.record_user_engagement(metric)
            
        except Exception as e:
            logger.error(f"Failed to record user engagement metric: {e}")
    
    def _categorize_endpoint(self, endpoint: str, method: str) -> tuple[Optional[str], Optional[str]]:
        """Categorize endpoint into action and feature"""
        try:
            # Remove API version prefix
            clean_endpoint = endpoint.replace("/api/v1", "").strip("/")
            
            # Define endpoint mappings
            endpoint_mappings = {
                # Profile management
                "profiles": ("view_profile", "profile_management"),
                "profiles/create": ("create_profile", "profile_management"),
                "profiles/update": ("update_profile", "profile_management"),
                
                # Resume handling
                "resume/upload": ("upload_resume", "resume_processing"),
                "resume/analyze": ("analyze_resume", "resume_processing"),
                
                # Platform connections
                "platform-accounts": ("connect_platform", "platform_integration"),
                "external-profiles": ("sync_external", "platform_integration"),
                
                # AI Analysis
                "ai-analysis": ("request_analysis", "ai_insights"),
                "career-analysis": ("analyze_career", "ai_insights"),
                
                # Job recommendations
                "job-recommendations": ("view_jobs", "job_matching"),
                "job-applications": ("apply_job", "job_matching"),
                
                # Learning paths
                "learning-paths": ("view_learning", "skill_development"),
                "recommendations": ("get_recommendations", "skill_development"),
                
                # Dashboard
                "dashboard": ("view_dashboard", "dashboard"),
                "analytics": ("view_analytics", "dashboard"),
                
                # Career guidance
                "career-guidance": ("get_guidance", "career_planning"),
                "career-trajectory": ("view_trajectory", "career_planning"),
            }
            
            # Find matching endpoint
            for pattern, (action, feature) in endpoint_mappings.items():
                if pattern in clean_endpoint:
                    # Modify action based on HTTP method
                    if method == "POST":
                        if "create" not in action and "upload" not in action:
                            action = f"create_{action.split('_')[-1]}"
                    elif method == "PUT" or method == "PATCH":
                        action = f"update_{action.split('_')[-1]}"
                    elif method == "DELETE":
                        action = f"delete_{action.split('_')[-1]}"
                    
                    return action, feature
            
            # Default categorization for unknown endpoints
            if method == "GET":
                return "view_data", "general"
            elif method == "POST":
                return "create_data", "general"
            elif method in ["PUT", "PATCH"]:
                return "update_data", "general"
            elif method == "DELETE":
                return "delete_data", "general"
            
        except Exception as e:
            logger.debug(f"Failed to categorize endpoint {endpoint}: {e}")
        
        return None, None


class ExternalAPITrackingMixin:
    """Mixin for tracking external API calls"""
    
    @staticmethod
    async def track_external_api_call(
        service_name: str,
        endpoint: str,
        response_time_ms: float,
        status_code: int,
        success: bool,
        error_message: Optional[str] = None,
        retry_count: int = 0
    ):
        """Track external API call performance"""
        try:
            from app.services.system_performance_analytics import ExternalAPIMetric
            
            metric = ExternalAPIMetric(
                service_name=service_name,
                endpoint=endpoint,
                response_time_ms=response_time_ms,
                status_code=status_code,
                timestamp=datetime.utcnow(),
                success=success,
                error_message=error_message,
                retry_count=retry_count
            )
            
            await system_performance_analytics.record_external_api_performance(metric)
            
        except Exception as e:
            logger.error(f"Failed to track external API call: {e}")


def track_external_api(service_name: str, endpoint: str = ""):
    """Decorator for tracking external API calls"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            success = False
            status_code = 0
            error_message = None
            retry_count = 0
            
            try:
                result = await func(*args, **kwargs)
                success = True
                status_code = 200  # Assume success
                return result
                
            except Exception as e:
                error_message = str(e)
                # Try to extract status code from exception
                if hasattr(e, 'status_code'):
                    status_code = e.status_code
                elif hasattr(e, 'response') and hasattr(e.response, 'status_code'):
                    status_code = e.response.status_code
                else:
                    status_code = 500
                
                # Try to extract retry count
                if hasattr(e, 'retry_count'):
                    retry_count = e.retry_count
                
                raise
                
            finally:
                end_time = time.time()
                response_time_ms = (end_time - start_time) * 1000
                
                # Track the API call
                asyncio.create_task(
                    ExternalAPITrackingMixin.track_external_api_call(
                        service_name=service_name,
                        endpoint=endpoint or func.__name__,
                        response_time_ms=response_time_ms,
                        status_code=status_code,
                        success=success,
                        error_message=error_message,
                        retry_count=retry_count
                    )
                )
        
        return wrapper
    return decorator