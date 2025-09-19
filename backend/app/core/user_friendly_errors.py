"""
User-friendly error messages and recovery options for enhanced profile analysis

This module provides comprehensive user-friendly error messages with specific
recovery suggestions for different types of failures in the profile analysis system.
"""
from typing import Dict, List, Optional, Any
from enum import Enum
from dataclasses import dataclass
from datetime import datetime

import structlog

logger = structlog.get_logger()


class ErrorCategory(Enum):
    """Categories of errors for user-friendly messaging"""
    EXTERNAL_API = "external_api"
    FILE_PROCESSING = "file_processing"
    DATA_VALIDATION = "data_validation"
    SYSTEM_UNAVAILABLE = "system_unavailable"
    USER_INPUT = "user_input"
    AUTHENTICATION = "authentication"
    RATE_LIMITING = "rate_limiting"
    NETWORK_CONNECTIVITY = "network_connectivity"


class ErrorSeverity(Enum):
    """Severity levels for errors"""
    LOW = "low"          # Minor issues, workarounds available
    MEDIUM = "medium"    # Moderate impact, some features affected
    HIGH = "high"        # Significant impact, major features unavailable
    CRITICAL = "critical"  # System-wide issues, service largely unusable


@dataclass
class UserFriendlyError:
    """User-friendly error representation"""
    title: str
    message: str
    category: ErrorCategory
    severity: ErrorSeverity
    recovery_suggestions: List[str]
    technical_details: Optional[str] = None
    error_code: Optional[str] = None
    estimated_resolution_time: Optional[str] = None
    contact_support: bool = False
    retry_available: bool = True
    fallback_available: bool = False
    user_action_required: bool = False


class UserFriendlyErrorGenerator:
    """Generator for user-friendly error messages and recovery suggestions"""
    
    def __init__(self):
        self.error_patterns = self._initialize_error_patterns()
    
    def _initialize_error_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Initialize error patterns with user-friendly messages"""
        return {
            # Gemini API Errors
            "gemini_api_unavailable": {
                "title": "AI Analysis Temporarily Unavailable",
                "message": "Our AI-powered analysis system is currently experiencing issues. We're working to restore full functionality.",
                "category": ErrorCategory.EXTERNAL_API,
                "severity": ErrorSeverity.HIGH,
                "recovery_suggestions": [
                    "We're using alternative analysis methods to provide basic insights",
                    "Full AI analysis will be restored as soon as possible",
                    "You can continue using other features of the platform",
                    "Check back in 15-30 minutes for restored AI functionality"
                ],
                "estimated_resolution_time": "15-30 minutes",
                "fallback_available": True
            },
            
            "gemini_rate_limit": {
                "title": "AI Analysis Rate Limit Reached",
                "message": "We've reached the usage limit for our AI analysis service. Please wait a moment before trying again.",
                "category": ErrorCategory.RATE_LIMITING,
                "severity": ErrorSeverity.MEDIUM,
                "recovery_suggestions": [
                    "Please wait 5-10 minutes before requesting another analysis",
                    "You can continue using other features while waiting",
                    "Consider upgrading to premium for higher limits",
                    "We'll automatically retry your request shortly"
                ],
                "estimated_resolution_time": "5-10 minutes",
                "retry_available": True
            },
            
            "gemini_authentication": {
                "title": "AI Service Configuration Issue",
                "message": "There's a configuration issue with our AI analysis service. Our technical team has been notified.",
                "category": ErrorCategory.SYSTEM_UNAVAILABLE,
                "severity": ErrorSeverity.CRITICAL,
                "recovery_suggestions": [
                    "Our technical team is working on resolving this issue",
                    "You can use basic profile features while we fix this",
                    "We'll notify you once AI analysis is restored",
                    "Contact support if you need immediate assistance"
                ],
                "contact_support": True,
                "retry_available": False,
                "fallback_available": True
            },
            
            # Platform Scraping Errors
            "github_not_found": {
                "title": "GitHub Profile Not Found",
                "message": "We couldn't find your GitHub profile. Please check your username and privacy settings.",
                "category": ErrorCategory.USER_INPUT,
                "severity": ErrorSeverity.LOW,
                "recovery_suggestions": [
                    "Double-check that your GitHub username is spelled correctly",
                    "Ensure your GitHub profile is set to public",
                    "You can update your GitHub username in your profile settings",
                    "Your analysis will continue without GitHub data for now"
                ],
                "user_action_required": True,
                "fallback_available": True
            },
            
            "github_rate_limit": {
                "title": "GitHub Data Collection Paused",
                "message": "GitHub has temporarily limited our data collection. We'll automatically retry later.",
                "category": ErrorCategory.RATE_LIMITING,
                "severity": ErrorSeverity.LOW,
                "recovery_suggestions": [
                    "GitHub data collection will resume automatically in 1 hour",
                    "Your profile analysis will continue without GitHub data for now",
                    "You can manually add your GitHub projects to your profile",
                    "Check back later to see your GitHub data included"
                ],
                "estimated_resolution_time": "1 hour",
                "retry_available": True,
                "fallback_available": True
            },
            
            "leetcode_blocked": {
                "title": "LeetCode Data Temporarily Unavailable",
                "message": "LeetCode has blocked our data collection due to anti-bot measures. This is temporary.",
                "category": ErrorCategory.EXTERNAL_API,
                "severity": ErrorSeverity.LOW,
                "recovery_suggestions": [
                    "LeetCode data collection may be restored later",
                    "You can manually enter your LeetCode statistics",
                    "Your analysis will continue without LeetCode data",
                    "Consider connecting other coding platforms instead"
                ],
                "user_action_required": True,
                "fallback_available": True
            },
            
            "linkedin_blocked": {
                "title": "LinkedIn Data Collection Restricted",
                "message": "LinkedIn has restricted our data collection. This is common due to their anti-scraping policies.",
                "category": ErrorCategory.EXTERNAL_API,
                "severity": ErrorSeverity.LOW,
                "recovery_suggestions": [
                    "LinkedIn data collection is restricted by their policies",
                    "You can manually enter your LinkedIn experience",
                    "Upload your resume to include professional experience",
                    "Your analysis will continue with available data"
                ],
                "user_action_required": True,
                "fallback_available": True
            },
            
            # Resume Processing Errors
            "resume_format_unsupported": {
                "title": "Resume Format Not Supported",
                "message": "The file format you uploaded is not supported. Please use PDF, DOC, or DOCX format.",
                "category": ErrorCategory.FILE_PROCESSING,
                "severity": ErrorSeverity.LOW,
                "recovery_suggestions": [
                    "Convert your resume to PDF format (recommended)",
                    "Ensure your file is in DOC or DOCX format",
                    "Try saving your resume from a different application",
                    "You can manually enter your information instead"
                ],
                "user_action_required": True,
                "fallback_available": True
            },
            
            "resume_file_too_large": {
                "title": "Resume File Too Large",
                "message": "Your resume file is larger than our 10MB limit. Please reduce the file size.",
                "category": ErrorCategory.FILE_PROCESSING,
                "severity": ErrorSeverity.LOW,
                "recovery_suggestions": [
                    "Compress images in your resume to reduce file size",
                    "Save your resume as a PDF to reduce size",
                    "Remove unnecessary graphics or formatting",
                    "Try uploading a text-only version of your resume"
                ],
                "user_action_required": True
            },
            
            "resume_parsing_failed": {
                "title": "Resume Processing Failed",
                "message": "We couldn't automatically extract information from your resume. You can enter the details manually.",
                "category": ErrorCategory.FILE_PROCESSING,
                "severity": ErrorSeverity.MEDIUM,
                "recovery_suggestions": [
                    "Use our manual data entry form to add your information",
                    "Try uploading a different version of your resume",
                    "Ensure your resume has clear formatting and readable text",
                    "Contact support if you continue having issues"
                ],
                "user_action_required": True,
                "fallback_available": True
            },
            
            "resume_corrupted": {
                "title": "Resume File Corrupted",
                "message": "The resume file appears to be corrupted or damaged. Please try uploading a different file.",
                "category": ErrorCategory.FILE_PROCESSING,
                "severity": ErrorSeverity.LOW,
                "recovery_suggestions": [
                    "Try re-saving your resume and uploading again",
                    "Create a new PDF version of your resume",
                    "Check if the file opens correctly on your computer",
                    "Use manual data entry if the file continues to have issues"
                ],
                "user_action_required": True,
                "fallback_available": True
            },
            
            # Job Scraping Errors
            "job_scraping_failed": {
                "title": "Job Recommendations Temporarily Limited",
                "message": "We're having trouble collecting the latest job postings. We'll show you cached recommendations for now.",
                "category": ErrorCategory.EXTERNAL_API,
                "severity": ErrorSeverity.MEDIUM,
                "recovery_suggestions": [
                    "We're showing cached job recommendations from recent data",
                    "Live job data will be restored as soon as possible",
                    "You can manually search job portals in the meantime",
                    "Check back later for updated job recommendations"
                ],
                "estimated_resolution_time": "30 minutes",
                "fallback_available": True
            },
            
            "job_sites_blocked": {
                "title": "Job Site Access Restricted",
                "message": "Job sites have temporarily restricted our access. We're working on alternative data sources.",
                "category": ErrorCategory.EXTERNAL_API,
                "severity": ErrorSeverity.MEDIUM,
                "recovery_suggestions": [
                    "We're working on alternative job data sources",
                    "You can search job sites directly for now",
                    "We'll provide general job market insights instead",
                    "Premium users get access to additional job sources"
                ],
                "estimated_resolution_time": "2-4 hours",
                "fallback_available": True
            },
            
            # Database Errors
            "database_unavailable": {
                "title": "Service Temporarily Unavailable",
                "message": "We're experiencing database connectivity issues. Please try again in a few minutes.",
                "category": ErrorCategory.SYSTEM_UNAVAILABLE,
                "severity": ErrorSeverity.HIGH,
                "recovery_suggestions": [
                    "Please wait 2-3 minutes and try again",
                    "Your data is safe and will be available once we're back online",
                    "We're working to restore service as quickly as possible",
                    "Contact support if the issue persists for more than 10 minutes"
                ],
                "estimated_resolution_time": "5-10 minutes",
                "retry_available": True
            },
            
            "database_timeout": {
                "title": "Request Taking Too Long",
                "message": "Your request is taking longer than expected. Please try again with a smaller dataset.",
                "category": ErrorCategory.SYSTEM_UNAVAILABLE,
                "severity": ErrorSeverity.MEDIUM,
                "recovery_suggestions": [
                    "Try breaking your request into smaller parts",
                    "Reduce the amount of data you're processing at once",
                    "Wait a few minutes and try again",
                    "Contact support if you consistently see this error"
                ],
                "retry_available": True
            },
            
            # Network and Connectivity Errors
            "network_timeout": {
                "title": "Connection Timeout",
                "message": "We couldn't connect to external services due to network issues. Please try again.",
                "category": ErrorCategory.NETWORK_CONNECTIVITY,
                "severity": ErrorSeverity.MEDIUM,
                "recovery_suggestions": [
                    "Check your internet connection",
                    "Try refreshing the page and attempting again",
                    "Wait a few minutes for network issues to resolve",
                    "Some features may work while others are affected"
                ],
                "retry_available": True,
                "fallback_available": True
            },
            
            # Authentication Errors
            "session_expired": {
                "title": "Session Expired",
                "message": "Your session has expired for security reasons. Please log in again.",
                "category": ErrorCategory.AUTHENTICATION,
                "severity": ErrorSeverity.LOW,
                "recovery_suggestions": [
                    "Click the login button to sign in again",
                    "Your data has been saved and will be available after login",
                    "Consider enabling 'Remember Me' for longer sessions",
                    "Clear your browser cache if you continue having issues"
                ],
                "user_action_required": True
            },
            
            # Generic System Errors
            "system_overloaded": {
                "title": "System Temporarily Overloaded",
                "message": "We're experiencing high traffic. Please wait a moment and try again.",
                "category": ErrorCategory.SYSTEM_UNAVAILABLE,
                "severity": ErrorSeverity.MEDIUM,
                "recovery_suggestions": [
                    "Please wait 2-3 minutes before trying again",
                    "Try using the service during off-peak hours",
                    "Some features may be slower than usual",
                    "We're working to increase capacity"
                ],
                "estimated_resolution_time": "5-15 minutes",
                "retry_available": True
            },
            
            "maintenance_mode": {
                "title": "Scheduled Maintenance",
                "message": "We're performing scheduled maintenance to improve our services. Please check back soon.",
                "category": ErrorCategory.SYSTEM_UNAVAILABLE,
                "severity": ErrorSeverity.HIGH,
                "recovery_suggestions": [
                    "Maintenance is expected to complete within the scheduled window",
                    "Your data is safe and will be available after maintenance",
                    "Check our status page for updates",
                    "We apologize for any inconvenience"
                ],
                "retry_available": False
            }
        }
    
    def generate_user_friendly_error(
        self,
        error_key: str,
        custom_message: Optional[str] = None,
        additional_suggestions: Optional[List[str]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> UserFriendlyError:
        """
        Generate a user-friendly error message
        
        Args:
            error_key: Key identifying the error pattern
            custom_message: Custom message to override default
            additional_suggestions: Additional recovery suggestions
            context: Additional context for error customization
            
        Returns:
            UserFriendlyError: User-friendly error representation
        """
        pattern = self.error_patterns.get(error_key)
        if not pattern:
            # Return generic error if pattern not found
            return self._generate_generic_error(error_key, custom_message)
        
        # Create base error from pattern
        error = UserFriendlyError(
            title=pattern["title"],
            message=custom_message or pattern["message"],
            category=pattern["category"],
            severity=pattern["severity"],
            recovery_suggestions=pattern["recovery_suggestions"].copy(),
            technical_details=pattern.get("technical_details"),
            error_code=error_key,
            estimated_resolution_time=pattern.get("estimated_resolution_time"),
            contact_support=pattern.get("contact_support", False),
            retry_available=pattern.get("retry_available", True),
            fallback_available=pattern.get("fallback_available", False),
            user_action_required=pattern.get("user_action_required", False)
        )
        
        # Add additional suggestions if provided
        if additional_suggestions:
            error.recovery_suggestions.extend(additional_suggestions)
        
        # Customize based on context
        if context:
            error = self._customize_error_with_context(error, context)
        
        return error
    
    def _generate_generic_error(
        self,
        error_key: str,
        custom_message: Optional[str] = None
    ) -> UserFriendlyError:
        """Generate a generic error when specific pattern is not found"""
        return UserFriendlyError(
            title="Unexpected Error",
            message=custom_message or "An unexpected error occurred. Please try again.",
            category=ErrorCategory.SYSTEM_UNAVAILABLE,
            severity=ErrorSeverity.MEDIUM,
            recovery_suggestions=[
                "Please try again in a few minutes",
                "Refresh the page and attempt your action again",
                "Contact support if the problem persists",
                "Check our status page for any known issues"
            ],
            error_code=error_key,
            contact_support=True,
            retry_available=True
        )
    
    def _customize_error_with_context(
        self,
        error: UserFriendlyError,
        context: Dict[str, Any]
    ) -> UserFriendlyError:
        """Customize error message based on context"""
        # Add user-specific information
        if context.get("user_id"):
            error.technical_details = f"User ID: {context['user_id']}"
        
        # Add operation-specific information
        if context.get("operation"):
            operation = context["operation"]
            if operation == "profile_analysis":
                error.recovery_suggestions.append(
                    "You can continue building your profile while we resolve this issue"
                )
            elif operation == "resume_upload":
                error.recovery_suggestions.append(
                    "Try uploading a different version of your resume"
                )
        
        # Add platform-specific suggestions
        if context.get("platform"):
            platform = context["platform"]
            error.recovery_suggestions.append(
                f"You can try connecting other platforms while {platform} is unavailable"
            )
        
        # Add retry count information
        if context.get("retry_count", 0) > 0:
            error.recovery_suggestions.insert(0,
                f"We've already tried {context['retry_count']} times - this may be a persistent issue"
            )
            error.contact_support = True
        
        return error
    
    def get_error_by_exception_type(
        self,
        exception: Exception,
        context: Optional[Dict[str, Any]] = None
    ) -> UserFriendlyError:
        """
        Get user-friendly error based on exception type
        
        Args:
            exception: The exception that occurred
            context: Additional context information
            
        Returns:
            UserFriendlyError: User-friendly error representation
        """
        exception_name = type(exception).__name__
        error_message = str(exception)
        
        # Map exception types to error keys
        error_key_mapping = {
            "GitHubAPIError": self._determine_github_error_key(error_message),
            "LeetCodeAPIError": "leetcode_blocked",
            "LinkedInAPIError": "linkedin_blocked",
            "JobScrapingError": "job_scraping_failed",
            "MLModelError": self._determine_ml_error_key(error_message),
            "ProcessingError": self._determine_processing_error_key(error_message),
            "DatabaseError": "database_unavailable",
            "ValidationError": "resume_format_unsupported",
            "TimeoutError": "network_timeout",
            "ConnectionError": "network_timeout"
        }
        
        error_key = error_key_mapping.get(exception_name, "system_overloaded")
        
        return self.generate_user_friendly_error(
            error_key,
            context=context
        )
    
    def _determine_github_error_key(self, error_message: str) -> str:
        """Determine specific GitHub error key based on message"""
        error_lower = error_message.lower()
        if "not found" in error_lower or "404" in error_lower:
            return "github_not_found"
        elif "rate limit" in error_lower:
            return "github_rate_limit"
        else:
            return "github_not_found"  # Default
    
    def _determine_ml_error_key(self, error_message: str) -> str:
        """Determine specific ML/AI error key based on message"""
        error_lower = error_message.lower()
        if "rate limit" in error_lower or "quota" in error_lower:
            return "gemini_rate_limit"
        elif "authentication" in error_lower or "api key" in error_lower:
            return "gemini_authentication"
        else:
            return "gemini_api_unavailable"
    
    def _determine_processing_error_key(self, error_message: str) -> str:
        """Determine specific processing error key based on message"""
        error_lower = error_message.lower()
        if "file format" in error_lower or "unsupported" in error_lower:
            return "resume_format_unsupported"
        elif "file size" in error_lower or "too large" in error_lower:
            return "resume_file_too_large"
        elif "parsing" in error_lower or "extraction" in error_lower:
            return "resume_parsing_failed"
        elif "corrupted" in error_lower:
            return "resume_corrupted"
        else:
            return "resume_parsing_failed"  # Default
    
    def format_error_for_api_response(self, error: UserFriendlyError) -> Dict[str, Any]:
        """Format error for API response"""
        return {
            "success": False,
            "error": {
                "code": error.error_code,
                "title": error.title,
                "message": error.message,
                "category": error.category.value,
                "severity": error.severity.value,
                "recovery_suggestions": error.recovery_suggestions,
                "estimated_resolution_time": error.estimated_resolution_time,
                "retry_available": error.retry_available,
                "fallback_available": error.fallback_available,
                "user_action_required": error.user_action_required,
                "contact_support": error.contact_support,
                "timestamp": datetime.utcnow().isoformat()
            }
        }
    
    def format_error_for_frontend(self, error: UserFriendlyError) -> Dict[str, Any]:
        """Format error for frontend display"""
        return {
            "type": "error",
            "title": error.title,
            "message": error.message,
            "severity": error.severity.value,
            "suggestions": error.recovery_suggestions,
            "actions": {
                "retry": error.retry_available,
                "contact_support": error.contact_support,
                "user_action_required": error.user_action_required
            },
            "metadata": {
                "error_code": error.error_code,
                "category": error.category.value,
                "estimated_resolution": error.estimated_resolution_time,
                "fallback_available": error.fallback_available
            }
        }


# Global instance
user_friendly_error_generator = UserFriendlyErrorGenerator()