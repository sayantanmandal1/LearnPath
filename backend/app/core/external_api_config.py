"""Configuration for external API integrations."""

import os
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class ExternalAPIConfig(BaseSettings):
    """Configuration for external API services."""
    
    # GitHub API Configuration
    github_token: Optional[str] = Field(None, env="GITHUB_TOKEN")
    github_api_url: str = Field("https://api.github.com", env="GITHUB_API_URL")
    github_timeout: float = Field(30.0, env="GITHUB_TIMEOUT")
    github_max_repos: int = Field(100, env="GITHUB_MAX_REPOS")
    
    # LeetCode Configuration
    leetcode_api_url: str = Field("https://leetcode.com", env="LEETCODE_API_URL")
    leetcode_timeout: float = Field(30.0, env="LEETCODE_TIMEOUT")
    leetcode_max_problems: int = Field(100, env="LEETCODE_MAX_PROBLEMS")
    leetcode_max_contests: int = Field(10, env="LEETCODE_MAX_CONTESTS")
    
    # LinkedIn Configuration
    linkedin_api_url: str = Field("https://www.linkedin.com", env="LINKEDIN_API_URL")
    linkedin_timeout: float = Field(30.0, env="LINKEDIN_TIMEOUT")
    linkedin_min_request_interval: float = Field(3.0, env="LINKEDIN_MIN_REQUEST_INTERVAL")
    
    # Rate Limiting Configuration
    rate_limit_max_retries: int = Field(3, env="RATE_LIMIT_MAX_RETRIES")
    rate_limit_base_delay: float = Field(1.0, env="RATE_LIMIT_BASE_DELAY")
    rate_limit_max_delay: float = Field(60.0, env="RATE_LIMIT_MAX_DELAY")
    rate_limit_exponential_base: float = Field(2.0, env="RATE_LIMIT_EXPONENTIAL_BASE")
    
    # Caching Configuration
    enable_caching: bool = Field(True, env="ENABLE_API_CACHING")
    cache_ttl_seconds: int = Field(3600, env="API_CACHE_TTL_SECONDS")  # 1 hour
    
    # Data Validation Configuration
    enable_data_validation: bool = Field(True, env="ENABLE_DATA_VALIDATION")
    enable_graceful_degradation: bool = Field(True, env="ENABLE_GRACEFUL_DEGRADATION")
    
    # Extraction Timeouts
    profile_extraction_timeout: int = Field(120, env="PROFILE_EXTRACTION_TIMEOUT")  # 2 minutes
    
    # Security Configuration
    respect_robots_txt: bool = Field(True, env="RESPECT_ROBOTS_TXT")
    user_agent: str = Field("AI-Career-Recommender/1.0", env="USER_AGENT")
    
    model_config = {
        "env_file": ".env",
        "case_sensitive": False,
        "extra": "ignore"
    }


# Global configuration instance
external_api_config = ExternalAPIConfig()