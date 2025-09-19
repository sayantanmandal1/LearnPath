"""
Enhanced Error Recovery Service for Profile Analysis System

This service provides comprehensive error handling and graceful degradation
for all external API integrations and system components.
"""
import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Union
from enum import Enum
from dataclasses import dataclass, field

import structlog
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update

from app.core.exceptions import (
    ExternalAPIException, GitHubAPIError, LeetCodeAPIError, LinkedInAPIError,
    JobScrapingError, MLModelError, ProcessingError, DatabaseError
)
from app.core.graceful_degradation import degradation_manager, ServiceStatus
from app.services.external_apis.circuit_breaker import circuit_breaker_manager
from app.models.analysis_result import AnalysisResult, AnalysisType

logger = structlog.get_logger()


class RecoveryStrategy(Enum):
    """Recovery strategies for different types of failures"""
    RETRY_WITH_BACKOFF = "retry_with_backoff"
    USE_CACHED_DATA = "use_cached_data"
    FALLBACK_SERVICE = "fallback_service"
    GRACEFUL_DEGRADATION = "graceful_degradation"
    MANUAL_INTERVENTION = "manual_intervention"
    SKIP_COMPONENT = "skip_component"


@dataclass
class RecoveryAction:
    """Recovery action to be taken for a specific error"""
    strategy: RecoveryStrategy
    fallback_data: Optional[Any] = None
    retry_after: Optional[int] = None
    user_message: Optional[str] = None
    recovery_suggestions: List[str] = field(default_factory=list)
    requires_user_action: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ErrorContext:
    """Context information for error handling"""
    service_name: str
    operation: str
    user_id: Optional[str] = None
    error_type: str = ""
    error_message: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    retry_count: int = 0
    original_request: Optional[Dict[str, Any]] = None


class ErrorRecoveryService:
    """Enhanced error recovery service for profile analysis system"""
    
    def __init__(self):
        self.max_retry_attempts = 3
        self.base_retry_delay = 1.0  # seconds
        self.max_retry_delay = 60.0  # seconds
        self.cache_ttl = 3600  # 1 hour
        
        # Service-specific configurations
        self.service_configs = {
            "gemini_api": {
                "max_retries": 3,
                "retry_delay": 2.0,
                "fallback_available": True,
                "critical": True
            },
            "github_scraper": {
                "max_retries": 2,
                "retry_delay": 5.0,
                "fallback_available": True,
                "critical": False
            },
            "leetcode_scraper": {
                "max_retries": 2,
                "retry_delay": 10.0,
                "fallback_available": True,
                "critical": False
            },
            "linkedin_scraper": {
                "max_retries": 1,
                "retry_delay": 30.0,
                "fallback_available": True,
                "critical": False
            },
            "job_scraper": {
                "max_retries": 2,
                "retry_delay": 5.0,
                "fallback_available": True,
                "critical": False
            },
            "resume_processing": {
                "max_retries": 2,
                "retry_delay": 1.0,
                "fallback_available": True,
                "critical": True
            }
        }
    
    async def handle_error(
        self,
        error: Exception,
        context: ErrorContext,
        db: Optional[AsyncSession] = None
    ) -> RecoveryAction:
        """
        Handle error and determine recovery action
        
        Args:
            error: The exception that occurred
            context: Error context information
            db: Database session for caching/logging
            
        Returns:
            RecoveryAction: Action to take for recovery
        """
        logger.error(
            "Handling error in error recovery service",
            service=context.service_name,
            operation=context.operation,
            error_type=type(error).__name__,
            error_message=str(error),
            retry_count=context.retry_count,
            user_id=context.user_id
        )
        
        # Update service health
        degradation_manager.update_service_health(
            context.service_name,
            ServiceStatus.UNAVAILABLE,
            error=str(error)
        )
        
        # Determine recovery strategy based on error type and context
        if isinstance(error, ExternalAPIException):
            return await self._handle_external_api_error(error, context, db)
        elif isinstance(error, MLModelError):
            return await self._handle_ml_model_error(error, context, db)
        elif isinstance(error, ProcessingError):
            return await self._handle_processing_error(error, context, db)
        elif isinstance(error, DatabaseError):
            return await self._handle_database_error(error, context, db)
        else:
            return await self._handle_generic_error(error, context, db)
    
    async def _handle_external_api_error(
        self,
        error: ExternalAPIException,
        context: ErrorContext,
        db: Optional[AsyncSession]
    ) -> RecoveryAction:
        """Handle external API errors with specific strategies"""
        service_config = self.service_configs.get(context.service_name, {})
        
        # Check if we should retry
        if context.retry_count < service_config.get("max_retries", self.max_retry_attempts):
            retry_delay = min(
                service_config.get("retry_delay", self.base_retry_delay) * (2 ** context.retry_count),
                self.max_retry_delay
            )
            
            return RecoveryAction(
                strategy=RecoveryStrategy.RETRY_WITH_BACKOFF,
                retry_after=int(retry_delay),
                user_message=f"Retrying {context.service_name} in {retry_delay:.0f} seconds...",
                recovery_suggestions=[
                    f"The {context.service_name} service is temporarily unavailable",
                    "We're automatically retrying the request",
                    "Please wait while we attempt to recover"
                ]
            )
        
        # Try to use cached data
        if db and service_config.get("fallback_available", False):
            cached_data = await self._get_cached_data(context, db)
            if cached_data:
                return RecoveryAction(
                    strategy=RecoveryStrategy.USE_CACHED_DATA,
                    fallback_data=cached_data,
                    user_message="Using previously cached data while service recovers",
                    recovery_suggestions=[
                        "We're using cached data from your previous analysis",
                        "Some information may be slightly outdated",
                        "Try again later for the most current data"
                    ]
                )
        
        # Use fallback service or graceful degradation
        if isinstance(error, GitHubAPIError):
            return await self._handle_github_error(error, context)
        elif isinstance(error, LeetCodeAPIError):
            return await self._handle_leetcode_error(error, context)
        elif isinstance(error, LinkedInAPIError):
            return await self._handle_linkedin_error(error, context)
        elif isinstance(error, JobScrapingError):
            return await self._handle_job_scraping_error(error, context)
        
        # Default graceful degradation
        return RecoveryAction(
            strategy=RecoveryStrategy.GRACEFUL_DEGRADATION,
            fallback_data=await self._get_fallback_data(context),
            user_message=f"The {context.service_name} service is temporarily unavailable",
            recovery_suggestions=[
                "We're providing basic functionality while the service recovers",
                "Some features may be limited temporarily",
                "Please try again in a few minutes for full functionality"
            ]
        )
    
    async def _handle_ml_model_error(
        self,
        error: MLModelError,
        context: ErrorContext,
        db: Optional[AsyncSession]
    ) -> RecoveryAction:
        """Handle ML model errors with fallback strategies"""
        # For Gemini API errors, provide comprehensive fallback
        if "gemini" in context.service_name.lower():
            return await self._handle_gemini_api_error(error, context, db)
        
        # For other ML models, use cached results or basic analysis
        cached_data = None
        if db:
            cached_data = await self._get_cached_analysis(context.user_id, context.operation, db)
        
        if cached_data:
            return RecoveryAction(
                strategy=RecoveryStrategy.USE_CACHED_DATA,
                fallback_data=cached_data,
                user_message="Using previous AI analysis while the system recovers",
                recovery_suggestions=[
                    "We're showing your previous AI analysis results",
                    "The analysis may be from an earlier session",
                    "Try again later for updated insights"
                ]
            )
        
        # Provide basic analysis without AI
        return RecoveryAction(
            strategy=RecoveryStrategy.GRACEFUL_DEGRADATION,
            fallback_data=await self._generate_basic_analysis(context),
            user_message="AI analysis is temporarily unavailable - showing basic insights",
            recovery_suggestions=[
                "Our AI system is temporarily down",
                "We're providing basic analysis based on your profile data",
                "Full AI insights will be available once the system recovers"
            ]
        )
    
    async def _handle_gemini_api_error(
        self,
        error: MLModelError,
        context: ErrorContext,
        db: Optional[AsyncSession]
    ) -> RecoveryAction:
        """Handle Gemini API specific errors with comprehensive fallbacks"""
        # Check for rate limiting
        if "rate limit" in str(error).lower() or "quota" in str(error).lower():
            return RecoveryAction(
                strategy=RecoveryStrategy.RETRY_WITH_BACKOFF,
                retry_after=300,  # 5 minutes for rate limits
                user_message="AI service is temporarily rate-limited",
                recovery_suggestions=[
                    "The AI service has reached its usage limit",
                    "We'll automatically retry in 5 minutes",
                    "You can continue using other features in the meantime"
                ]
            )
        
        # Check for authentication errors
        if "authentication" in str(error).lower() or "api key" in str(error).lower():
            return RecoveryAction(
                strategy=RecoveryStrategy.MANUAL_INTERVENTION,
                user_message="AI service configuration issue",
                recovery_suggestions=[
                    "There's a configuration issue with our AI service",
                    "Our team has been notified and is working on a fix",
                    "Please try again later or contact support"
                ],
                requires_user_action=False
            )
        
        # For other Gemini errors, use rule-based fallback
        fallback_data = await self._generate_rule_based_analysis(context, db)
        
        return RecoveryAction(
            strategy=RecoveryStrategy.FALLBACK_SERVICE,
            fallback_data=fallback_data,
            user_message="Using alternative analysis while AI service recovers",
            recovery_suggestions=[
                "Our AI service is temporarily unavailable",
                "We're using rule-based analysis as a backup",
                "Results may be less detailed than usual"
            ]
        )
    
    async def _handle_processing_error(
        self,
        error: ProcessingError,
        context: ErrorContext,
        db: Optional[AsyncSession]
    ) -> RecoveryAction:
        """Handle file processing and data processing errors"""
        if "resume" in context.operation.lower():
            return await self._handle_resume_processing_error(error, context)
        elif "platform" in context.operation.lower():
            return await self._handle_platform_processing_error(error, context)
        
        # Generic processing error
        return RecoveryAction(
            strategy=RecoveryStrategy.GRACEFUL_DEGRADATION,
            user_message="Processing temporarily unavailable",
            recovery_suggestions=[
                "There was an issue processing your request",
                "Please try again with different data",
                "Contact support if the problem persists"
            ],
            requires_user_action=True
        )
    
    async def _handle_resume_processing_error(
        self,
        error: ProcessingError,
        context: ErrorContext
    ) -> RecoveryAction:
        """Handle resume processing specific errors"""
        error_msg = str(error).lower()
        
        if "file format" in error_msg or "unsupported" in error_msg:
            return RecoveryAction(
                strategy=RecoveryStrategy.MANUAL_INTERVENTION,
                user_message="Resume file format not supported",
                recovery_suggestions=[
                    "Please ensure your resume is in PDF, DOC, or DOCX format",
                    "Try converting your file to PDF and uploading again",
                    "You can also enter your information manually"
                ],
                requires_user_action=True
            )
        
        if "file size" in error_msg:
            return RecoveryAction(
                strategy=RecoveryStrategy.MANUAL_INTERVENTION,
                user_message="Resume file is too large",
                recovery_suggestions=[
                    "Please reduce your file size to under 10MB",
                    "Try compressing images in your resume",
                    "Consider uploading a simpler version"
                ],
                requires_user_action=True
            )
        
        # For parsing errors, offer manual entry
        return RecoveryAction(
            strategy=RecoveryStrategy.GRACEFUL_DEGRADATION,
            user_message="Resume parsing failed - manual entry available",
            recovery_suggestions=[
                "We couldn't automatically extract information from your resume",
                "You can enter your information manually instead",
                "Try uploading a different version of your resume"
            ],
            requires_user_action=True
        )
    
    async def _handle_github_error(
        self,
        error: GitHubAPIError,
        context: ErrorContext
    ) -> RecoveryAction:
        """Handle GitHub API specific errors"""
        error_msg = str(error).lower()
        
        if "not found" in error_msg or "404" in error_msg:
            return RecoveryAction(
                strategy=RecoveryStrategy.SKIP_COMPONENT,
                user_message="GitHub profile not found",
                recovery_suggestions=[
                    "Please check that your GitHub username is correct",
                    "Ensure your GitHub profile is public",
                    "You can update your GitHub username in settings"
                ],
                requires_user_action=True
            )
        
        if "rate limit" in error_msg:
            return RecoveryAction(
                strategy=RecoveryStrategy.RETRY_WITH_BACKOFF,
                retry_after=3600,  # 1 hour for GitHub rate limits
                user_message="GitHub API rate limit reached",
                recovery_suggestions=[
                    "GitHub has rate-limited our requests",
                    "We'll automatically retry in 1 hour",
                    "Your profile analysis will continue without GitHub data for now"
                ]
            )
        
        # Default GitHub fallback
        return RecoveryAction(
            strategy=RecoveryStrategy.SKIP_COMPONENT,
            user_message="GitHub data temporarily unavailable",
            recovery_suggestions=[
                "We couldn't fetch your GitHub data right now",
                "Your analysis will continue without GitHub information",
                "Try reconnecting your GitHub account later"
            ]
        )
    
    async def _handle_leetcode_error(
        self,
        error: LeetCodeAPIError,
        context: ErrorContext
    ) -> RecoveryAction:
        """Handle LeetCode scraping specific errors"""
        return RecoveryAction(
            strategy=RecoveryStrategy.SKIP_COMPONENT,
            user_message="LeetCode data temporarily unavailable",
            recovery_suggestions=[
                "We couldn't fetch your LeetCode statistics",
                "This might be due to anti-bot measures",
                "Your analysis will continue without LeetCode data",
                "Try again later or verify your LeetCode username"
            ]
        )
    
    async def _handle_linkedin_error(
        self,
        error: LinkedInAPIError,
        context: ErrorContext
    ) -> RecoveryAction:
        """Handle LinkedIn scraping specific errors"""
        return RecoveryAction(
            strategy=RecoveryStrategy.SKIP_COMPONENT,
            user_message="LinkedIn data temporarily unavailable",
            recovery_suggestions=[
                "LinkedIn has blocked our data collection",
                "This is common due to LinkedIn's anti-scraping measures",
                "Your analysis will continue without LinkedIn data",
                "Consider manually entering your LinkedIn experience"
            ]
        )
    
    async def _handle_job_scraping_error(
        self,
        error: JobScrapingError,
        context: ErrorContext
    ) -> RecoveryAction:
        """Handle job scraping specific errors"""
        # Use cached job data if available
        cached_jobs = await self._get_cached_job_data(context)
        
        if cached_jobs:
            return RecoveryAction(
                strategy=RecoveryStrategy.USE_CACHED_DATA,
                fallback_data=cached_jobs,
                user_message="Using cached job recommendations",
                recovery_suggestions=[
                    "Live job data is temporarily unavailable",
                    "We're showing cached job recommendations",
                    "Job data will be updated once scraping resumes"
                ]
            )
        
        # Provide generic job recommendations
        return RecoveryAction(
            strategy=RecoveryStrategy.GRACEFUL_DEGRADATION,
            fallback_data=await self._get_generic_job_recommendations(context),
            user_message="Job recommendations temporarily limited",
            recovery_suggestions=[
                "Live job scraping is temporarily unavailable",
                "We're showing general job categories for your profile",
                "Check back later for personalized job recommendations"
            ]
        )
    
    async def _handle_database_error(
        self,
        error: DatabaseError,
        context: ErrorContext,
        db: Optional[AsyncSession]
    ) -> RecoveryAction:
        """Handle database connectivity and operation errors"""
        return RecoveryAction(
            strategy=RecoveryStrategy.RETRY_WITH_BACKOFF,
            retry_after=30,
            user_message="Database temporarily unavailable",
            recovery_suggestions=[
                "We're experiencing database connectivity issues",
                "Your request will be retried automatically",
                "Please wait while we restore the connection"
            ]
        )
    
    async def _handle_generic_error(
        self,
        error: Exception,
        context: ErrorContext,
        db: Optional[AsyncSession]
    ) -> RecoveryAction:
        """Handle generic/unknown errors"""
        return RecoveryAction(
            strategy=RecoveryStrategy.GRACEFUL_DEGRADATION,
            user_message="An unexpected error occurred",
            recovery_suggestions=[
                "We encountered an unexpected issue",
                "Please try again in a few minutes",
                "Contact support if the problem persists"
            ]
        )
    
    async def _get_cached_data(
        self,
        context: ErrorContext,
        db: AsyncSession
    ) -> Optional[Dict[str, Any]]:
        """Retrieve cached data for the given context"""
        try:
            if not context.user_id:
                return None
            
            # Look for recent analysis results
            result = await db.execute(
                select(AnalysisResult)
                .where(
                    AnalysisResult.user_id == context.user_id,
                    AnalysisResult.created_at > datetime.utcnow() - timedelta(hours=24)
                )
                .order_by(AnalysisResult.created_at.desc())
                .limit(1)
            )
            
            cached_result = result.scalar_one_or_none()
            if cached_result:
                return cached_result.result_data
            
        except Exception as e:
            logger.warning(f"Failed to retrieve cached data: {str(e)}")
        
        return None
    
    async def _get_cached_analysis(
        self,
        user_id: Optional[str],
        operation: str,
        db: AsyncSession
    ) -> Optional[Dict[str, Any]]:
        """Get cached analysis results"""
        if not user_id:
            return None
        
        try:
            # Map operation to analysis type
            analysis_type_map = {
                "skill_assessment": AnalysisType.SKILL_ASSESSMENT,
                "career_recommendation": AnalysisType.CAREER_RECOMMENDATION,
                "learning_path": AnalysisType.LEARNING_PATH
            }
            
            analysis_type = analysis_type_map.get(operation)
            if not analysis_type:
                return None
            
            result = await db.execute(
                select(AnalysisResult)
                .where(
                    AnalysisResult.user_id == user_id,
                    AnalysisResult.analysis_type == analysis_type,
                    AnalysisResult.created_at > datetime.utcnow() - timedelta(hours=6)
                )
                .order_by(AnalysisResult.created_at.desc())
                .limit(1)
            )
            
            cached_analysis = result.scalar_one_or_none()
            if cached_analysis:
                return cached_analysis.result_data
            
        except Exception as e:
            logger.warning(f"Failed to retrieve cached analysis: {str(e)}")
        
        return None
    
    async def _get_fallback_data(self, context: ErrorContext) -> Dict[str, Any]:
        """Generate fallback data based on context"""
        if "github" in context.service_name:
            return {
                "repositories": [],
                "languages": [],
                "contributions": 0,
                "note": "GitHub data temporarily unavailable"
            }
        elif "leetcode" in context.service_name:
            return {
                "problems_solved": 0,
                "contest_rating": 0,
                "note": "LeetCode data temporarily unavailable"
            }
        elif "linkedin" in context.service_name:
            return {
                "connections": 0,
                "experience": [],
                "note": "LinkedIn data temporarily unavailable"
            }
        
        return {"note": f"{context.service_name} temporarily unavailable"}
    
    async def _generate_basic_analysis(self, context: ErrorContext) -> Dict[str, Any]:
        """Generate basic analysis without AI"""
        return {
            "analysis_type": "basic",
            "message": "Basic analysis provided while AI system recovers",
            "recommendations": [
                "Continue building your technical skills",
                "Expand your professional network",
                "Keep your resume updated with recent projects"
            ],
            "note": "Full AI analysis will be available once the system recovers"
        }
    
    async def _generate_rule_based_analysis(
        self,
        context: ErrorContext,
        db: Optional[AsyncSession]
    ) -> Dict[str, Any]:
        """Generate rule-based analysis as Gemini fallback"""
        # This would implement rule-based career analysis
        # For now, return a structured fallback
        return {
            "analysis_type": "rule_based",
            "skill_assessment": {
                "technical_skills": {},
                "soft_skills": {},
                "note": "Detailed skill assessment requires AI analysis"
            },
            "career_recommendations": [
                {
                    "role": "Software Developer",
                    "match_score": 0.7,
                    "reasoning": "Based on your profile data and common career paths"
                }
            ],
            "learning_paths": [
                {
                    "title": "General Skill Development",
                    "description": "Continue developing your technical and soft skills",
                    "duration": "Ongoing"
                }
            ],
            "note": "This is a simplified analysis. Full AI insights will be available once the system recovers."
        }
    
    async def _get_cached_job_data(self, context: ErrorContext) -> Optional[List[Dict[str, Any]]]:
        """Get cached job data"""
        # This would retrieve cached job postings
        # For now, return None to trigger generic recommendations
        return None
    
    async def _get_generic_job_recommendations(self, context: ErrorContext) -> List[Dict[str, Any]]:
        """Generate generic job recommendations"""
        return [
            {
                "title": "Software Developer",
                "company": "Various Companies",
                "location": "India",
                "match_score": 0.6,
                "note": "Generic recommendation - live job data temporarily unavailable"
            },
            {
                "title": "Data Analyst",
                "company": "Various Companies", 
                "location": "India",
                "match_score": 0.5,
                "note": "Generic recommendation - live job data temporarily unavailable"
            }
        ]
    
    async def execute_recovery_action(
        self,
        action: RecoveryAction,
        original_function: Callable,
        *args,
        **kwargs
    ) -> Any:
        """Execute the determined recovery action"""
        if action.strategy == RecoveryStrategy.RETRY_WITH_BACKOFF:
            if action.retry_after:
                await asyncio.sleep(action.retry_after)
            return await original_function(*args, **kwargs)
        
        elif action.strategy == RecoveryStrategy.USE_CACHED_DATA:
            return action.fallback_data
        
        elif action.strategy == RecoveryStrategy.FALLBACK_SERVICE:
            return action.fallback_data
        
        elif action.strategy == RecoveryStrategy.GRACEFUL_DEGRADATION:
            return action.fallback_data
        
        elif action.strategy == RecoveryStrategy.SKIP_COMPONENT:
            return None
        
        elif action.strategy == RecoveryStrategy.MANUAL_INTERVENTION:
            # Return error information for user action
            raise ProcessingError(
                action.user_message,
                fallback_available=action.fallback_data is not None
            )
        
        return action.fallback_data


# Global error recovery service instance
error_recovery_service = ErrorRecoveryService()