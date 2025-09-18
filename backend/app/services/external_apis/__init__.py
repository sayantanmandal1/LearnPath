"""External API integration services."""

from .github_client import GitHubClient
from .leetcode_scraper import LeetCodeScraper
from .linkedin_scraper import LinkedInScraper
from .codeforces_scraper import CodeforcesScraper
from .atcoder_scraper import AtCoderScraper
from .hackerrank_scraper import HackerRankScraper
from .kaggle_scraper import KaggleScraper
from .multi_platform_scraper import MultiPlatformScraper, PlatformType, PlatformAccount, ScrapingResult, MultiPlatformData
from .base_client import BaseAPIClient, APIError, RateLimitError
from .data_validator import DataValidator, DataQuality, ValidationResult
from .integration_service import ExternalAPIIntegrationService, ProfileExtractionRequest, ProfileExtractionResult
from .profile_merger import ProfileMerger, MergedProfile
from .circuit_breaker import CircuitBreaker, CircuitBreakerManager, CircuitBreakerError, CircuitBreakerConfig, circuit_breaker_manager

__all__ = [
    "GitHubClient",
    "LeetCodeScraper", 
    "LinkedInScraper",
    "CodeforcesScraper",
    "AtCoderScraper",
    "HackerRankScraper",
    "KaggleScraper",
    "MultiPlatformScraper",
    "PlatformType",
    "PlatformAccount",
    "ScrapingResult",
    "MultiPlatformData",
    "BaseAPIClient",
    "APIError",
    "RateLimitError",
    "DataValidator",
    "DataQuality",
    "ValidationResult",
    "ExternalAPIIntegrationService",
    "ProfileExtractionRequest",
    "ProfileExtractionResult",
    "ProfileMerger",
    "MergedProfile",
    "CircuitBreaker",
    "CircuitBreakerManager",
    "CircuitBreakerError",
    "CircuitBreakerConfig",
    "circuit_breaker_manager"
]