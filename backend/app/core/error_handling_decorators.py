"""
Error handling decorators for enhanced profile analysis system

These decorators provide comprehensive error handling with graceful degradation,
user-friendly messages, and automatic recovery mechanisms.
"""
import asyncio
import functools
import logging
from typing import Any, Callable, Dict, List, Optional, Type, Union
from datetime import datetime

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import (
    ExternalAPIException, ProcessingError, ValidationError,
    GitHubAPIError, LeetCodeAPIError, LinkedInAPIError, JobScrapingError,
    MLModelError, DatabaseError
)
from app.services.error_recovery_service import (
    error_recovery_service, ErrorContext, RecoveryAction, RecoveryStrategy
)
from app.core.graceful_degradation import degradation_manager

logger = structlog.get_logger()


def with_error_recovery(
    service_name: str,
    operation: str,
    max_retries: int = 3,
    fallback_data: Optional[Any] = None,
    critical: bool = False,
    user_friendly_message: Optional[str] = None
):
    """
    Decorator for comprehensive error handling with recovery
    
    Args:
        service_name: Name of the service being called
        operation: Operation being performed
        max_retries: Maximum number of retry attempts
        fallback_data: Data to return if all recovery attempts fail
        critical: Whether this operation is critical to the user flow
        user_friendly_message: Custom user-friendly error message
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            retry_count = 0
            last_error = None
            
            # Extract user_id and db session from args/kwargs if available
            user_id = kwargs.get('user_id') or (args[0] if args and isinstance(args[0], str) else None)
            db = kwargs.get('db') or next((arg for arg in args if isinstance(arg, AsyncSession)), None)
            
            while retry_count <= max_retries:
                try:
                    # Execute the original function
                    if asyncio.iscoroutinefunction(func):
                        result = await func(*args, **kwargs)
                    else:
                        result = func(*args, **kwargs)
                    
                    # Record success if this was a retry
                    if retry_count > 0:
                        degradation_manager.update_service_health(
                            service_name,
                            degradation_manager.ServiceStatus.HEALTHY
                        )
                        logger.info(
                            f"Service {service_name} recovered after {retry_count} retries",
                            service=service_name,
                            operation=operation,
                            retry_count=retry_count
                        )
                    
                    return result
                
                except Exception as error:
                    last_error = error
                    retry_count += 1
                    
                    # Create error context
                    context = ErrorContext(
                        service_name=service_name,
                        operation=operation,
                        user_id=user_id,
                        error_type=type(error).__name__,
                        error_message=str(error),
                        retry_count=retry_count - 1,
                        original_request=kwargs
                    )
                    
                    # Get recovery action
                    recovery_action = await error_recovery_service.handle_error(
                        error, context, db
                    )
                    
                    # If we should retry, continue the loop
                    if (recovery_action.strategy == RecoveryStrategy.RETRY_WITH_BACKOFF and 
                        retry_count <= max_retries):
                        if recovery_action.retry_after:
                            await asyncio.sleep(recovery_action.retry_after)
                        continue
                    
                    # Execute recovery action
                    try:
                        recovery_result = await error_recovery_service.execute_recovery_action(
                            recovery_action, func, *args, **kwargs
                        )
                        
                        # Log successful recovery
                        logger.info(
                            f"Successfully recovered from {service_name} error",
                            service=service_name,
                            operation=operation,
                            recovery_strategy=recovery_action.strategy.value,
                            error_type=type(error).__name__
                        )
                        
                        return recovery_result
                    
                    except Exception as recovery_error:
                        # Recovery failed, continue to final error handling
                        logger.error(
                            f"Recovery failed for {service_name}",
                            service=service_name,
                            operation=operation,
                            original_error=str(error),
                            recovery_error=str(recovery_error)
                        )
                        break
            
            # All retries and recovery attempts failed
            if critical:
                # For critical operations, raise the error with user-friendly message
                if isinstance(last_error, (ExternalAPIException, ProcessingError)):
                    raise last_error
                else:
                    raise ProcessingError(
                        user_friendly_message or f"{service_name} is temporarily unavailable",
                        processing_type=operation,
                        original_error=last_error,
                        fallback_available=fallback_data is not None
                    )
            else:
                # For non-critical operations, return fallback data
                logger.warning(
                    f"Non-critical operation {operation} failed, returning fallback",
                    service=service_name,
                    operation=operation,
                    error=str(last_error)
                )
                return fallback_data
        
        return wrapper
    return decorator


def with_platform_error_handling(platform: str):
    """
    Decorator specifically for platform scraping operations
    
    Args:
        platform: Platform name (github, leetcode, linkedin, etc.)
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            try:
                return await func(*args, **kwargs)
            
            except Exception as error:
                # Create platform-specific error
                if platform.lower() == "github":
                    platform_error = GitHubAPIError(str(error), original_error=error)
                elif platform.lower() == "leetcode":
                    platform_error = LeetCodeAPIError(str(error), original_error=error)
                elif platform.lower() == "linkedin":
                    platform_error = LinkedInAPIError(str(error), original_error=error)
                else:
                    platform_error = ExternalAPIException(
                        service=platform,
                        detail=str(error),
                        original_error=error
                    )
                
                # Use error recovery service
                context = ErrorContext(
                    service_name=f"{platform}_scraper",
                    operation="scrape_profile",
                    error_type=type(error).__name__,
                    error_message=str(error)
                )
                
                recovery_action = await error_recovery_service.handle_error(
                    platform_error, context
                )
                
                # For platform scraping, usually skip component on failure
                if recovery_action.strategy == RecoveryStrategy.SKIP_COMPONENT:
                    logger.warning(
                        f"Skipping {platform} data due to error",
                        platform=platform,
                        error=str(error)
                    )
                    return None
                
                # Return fallback data if available
                return recovery_action.fallback_data
        
        return wrapper
    return decorator


def with_gemini_error_handling(operation: str = "ai_analysis"):
    """
    Decorator specifically for Gemini API operations
    
    Args:
        operation: Type of AI operation being performed
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            try:
                return await func(*args, **kwargs)
            
            except Exception as error:
                # Create Gemini-specific error
                gemini_error = MLModelError(
                    model_name="Gemini API",
                    detail=str(error),
                    fallback_available=True,
                    original_error=error
                )
                
                # Extract user_id from args/kwargs
                user_id = kwargs.get('user_id') or (args[0] if args and isinstance(args[0], str) else None)
                db = kwargs.get('db') or next((arg for arg in args if isinstance(arg, AsyncSession)), None)
                
                context = ErrorContext(
                    service_name="gemini_api",
                    operation=operation,
                    user_id=user_id,
                    error_type=type(error).__name__,
                    error_message=str(error)
                )
                
                recovery_action = await error_recovery_service.handle_error(
                    gemini_error, context, db
                )
                
                # Execute recovery action
                try:
                    return await error_recovery_service.execute_recovery_action(
                        recovery_action, func, *args, **kwargs
                    )
                except Exception:
                    # If recovery fails, return basic fallback
                    logger.error(
                        f"Gemini API recovery failed for {operation}",
                        operation=operation,
                        error=str(error)
                    )
                    return await _get_gemini_fallback(operation, user_id, db)
        
        return wrapper
    return decorator


def with_job_scraping_error_handling(job_platform: str):
    """
    Decorator for job scraping operations
    
    Args:
        job_platform: Job platform name (linkedin_jobs, naukri, etc.)
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            try:
                return await func(*args, **kwargs)
            
            except Exception as error:
                job_error = JobScrapingError(
                    platform=job_platform,
                    detail=str(error),
                    original_error=error
                )
                
                context = ErrorContext(
                    service_name="job_scraper",
                    operation=f"scrape_{job_platform}",
                    error_type=type(error).__name__,
                    error_message=str(error)
                )
                
                recovery_action = await error_recovery_service.handle_error(
                    job_error, context
                )
                
                # Return cached or generic job data
                if recovery_action.fallback_data:
                    return recovery_action.fallback_data
                
                # Return empty list if no fallback available
                logger.warning(
                    f"Job scraping failed for {job_platform}, returning empty results",
                    platform=job_platform,
                    error=str(error)
                )
                return []
        
        return wrapper
    return decorator


def with_resume_processing_error_handling():
    """Decorator for resume processing operations"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            try:
                return await func(*args, **kwargs)
            
            except Exception as error:
                context = ErrorContext(
                    service_name="resume_processing",
                    operation="process_resume",
                    error_type=type(error).__name__,
                    error_message=str(error)
                )
                
                recovery_action = await error_recovery_service.handle_error(
                    error, context
                )
                
                # For resume processing, usually require user action
                if recovery_action.requires_user_action:
                    raise ProcessingError(
                        recovery_action.user_message,
                        processing_type="resume_processing",
                        original_error=error,
                        fallback_available=True
                    )
                
                return recovery_action.fallback_data
        
        return wrapper
    return decorator


def with_database_error_handling(operation: str):
    """
    Decorator for database operations
    
    Args:
        operation: Database operation being performed
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            try:
                return await func(*args, **kwargs)
            
            except Exception as error:
                db_error = DatabaseError(
                    detail=str(error),
                    operation=operation,
                    original_error=error
                )
                
                context = ErrorContext(
                    service_name="database",
                    operation=operation,
                    error_type=type(error).__name__,
                    error_message=str(error)
                )
                
                recovery_action = await error_recovery_service.handle_error(
                    db_error, context
                )
                
                # For database errors, usually retry or fail
                if recovery_action.strategy == RecoveryStrategy.RETRY_WITH_BACKOFF:
                    if recovery_action.retry_after:
                        await asyncio.sleep(recovery_action.retry_after)
                    return await func(*args, **kwargs)
                
                # If retry not possible, raise the error
                raise db_error
        
        return wrapper
    return decorator


async def _get_gemini_fallback(
    operation: str,
    user_id: Optional[str],
    db: Optional[AsyncSession]
) -> Dict[str, Any]:
    """Get fallback data for Gemini API operations"""
    fallback_data = {
        "analysis_type": "fallback",
        "note": "AI analysis temporarily unavailable"
    }
    
    if operation == "skill_assessment":
        fallback_data.update({
            "technical_skills": {},
            "soft_skills": {},
            "skill_strengths": [],
            "skill_gaps": [],
            "improvement_areas": [],
            "market_relevance_score": 0.0,
            "confidence_score": 0.0
        })
    elif operation == "career_recommendation":
        fallback_data.update({
            "recommendations": [
                {
                    "recommended_role": "Software Developer",
                    "match_score": 0.5,
                    "reasoning": "Basic recommendation while AI system recovers",
                    "required_skills": [],
                    "skill_gaps": [],
                    "preparation_timeline": "Variable",
                    "salary_range": "Competitive",
                    "market_demand": "Good"
                }
            ]
        })
    elif operation == "learning_path":
        fallback_data.update({
            "learning_paths": [
                {
                    "title": "General Skill Development",
                    "description": "Continue developing your technical and professional skills",
                    "target_skills": [],
                    "learning_modules": [],
                    "estimated_duration": "Ongoing",
                    "difficulty_level": "Intermediate",
                    "resources": []
                }
            ]
        })
    
    return fallback_data


# Utility functions for error handling

def create_user_friendly_error_response(
    error: Exception,
    operation: str,
    recovery_suggestions: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Create a user-friendly error response"""
    error_response = {
        "success": False,
        "error": {
            "message": "An error occurred while processing your request",
            "operation": operation,
            "timestamp": datetime.utcnow().isoformat(),
            "recovery_suggestions": recovery_suggestions or [
                "Please try again in a few minutes",
                "Contact support if the problem persists"
            ]
        }
    }
    
    # Add specific messages for known error types
    if isinstance(error, ValidationError):
        error_response["error"]["message"] = "The information provided is invalid"
        error_response["error"]["recovery_suggestions"] = [
            "Please check your input and try again",
            "Ensure all required fields are filled correctly"
        ]
    elif isinstance(error, ExternalAPIException):
        error_response["error"]["message"] = f"External service temporarily unavailable"
        error_response["error"]["recovery_suggestions"] = [
            "We're experiencing issues with external services",
            "Please try again in a few minutes",
            "Some features may be temporarily limited"
        ]
    elif isinstance(error, ProcessingError):
        error_response["error"]["message"] = "Processing failed"
        if hasattr(error, 'fallback_available') and error.fallback_available:
            error_response["error"]["recovery_suggestions"] = [
                "You can try manual data entry as an alternative",
                "Contact support if you need assistance"
            ]
    
    return error_response


def log_error_with_context(
    error: Exception,
    context: Dict[str, Any],
    level: str = "error"
) -> None:
    """Log error with comprehensive context"""
    log_data = {
        "error_type": type(error).__name__,
        "error_message": str(error),
        "timestamp": datetime.utcnow().isoformat(),
        **context
    }
    
    if level == "error":
        logger.error("Error occurred with context", **log_data)
    elif level == "warning":
        logger.warning("Warning with context", **log_data)
    else:
        logger.info("Info with context", **log_data)