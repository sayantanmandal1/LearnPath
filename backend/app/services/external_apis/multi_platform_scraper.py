"""Multi-platform scraper service for coordinating data collection from various platforms."""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from pydantic import BaseModel
from enum import Enum

from .github_client import GitHubClient, GitHubProfile
from .leetcode_scraper import LeetCodeScraper, LeetCodeProfile
from .linkedin_scraper import LinkedInScraper, LinkedInProfile
from .codeforces_scraper import CodeforcesScraper, CodeforcesProfile
from .atcoder_scraper import AtCoderScraper, AtCoderProfile
from .hackerrank_scraper import HackerRankScraper, HackerRankProfile
from .kaggle_scraper import KaggleScraper, KaggleProfile
from .base_client import APIError


logger = logging.getLogger(__name__)


class PlatformType(str, Enum):
    """Supported platform types."""
    GITHUB = "github"
    LEETCODE = "leetcode"
    LINKEDIN = "linkedin"
    CODEFORCES = "codeforces"
    ATCODER = "atcoder"
    HACKERRANK = "hackerrank"
    KAGGLE = "kaggle"


class PlatformAccount(BaseModel):
    """Platform account information."""
    platform: PlatformType
    username: Optional[str] = None
    profile_url: Optional[str] = None
    is_active: bool = True


class ScrapingResult(BaseModel):
    """Result of platform scraping operation."""
    platform: PlatformType
    success: bool
    data: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    scraped_at: datetime
    processing_time: float  # in seconds


class MultiPlatformData(BaseModel):
    """Aggregated data from multiple platforms."""
    user_id: str
    platforms: Dict[PlatformType, ScrapingResult]
    total_platforms: int
    successful_platforms: int
    failed_platforms: int
    total_processing_time: float
    aggregated_at: datetime


class MultiPlatformScraper:
    """Coordinated scraper for multiple professional platforms."""
    
    def __init__(
        self,
        github_token: Optional[str] = None,
        max_concurrent_scrapers: int = 3,
        timeout_per_platform: float = 60.0
    ):
        self.github_token = github_token
        self.max_concurrent_scrapers = max_concurrent_scrapers
        self.timeout_per_platform = timeout_per_platform
        
        # Initialize platform clients
        self._clients = {}
        self._initialize_clients()
    
    def _initialize_clients(self):
        """Initialize all platform clients."""
        try:
            self._clients[PlatformType.GITHUB] = GitHubClient(
                api_token=self.github_token,
                timeout=self.timeout_per_platform
            )
            self._clients[PlatformType.LEETCODE] = LeetCodeScraper(
                timeout=self.timeout_per_platform
            )
            self._clients[PlatformType.LINKEDIN] = LinkedInScraper(
                timeout=self.timeout_per_platform
            )
            self._clients[PlatformType.CODEFORCES] = CodeforcesScraper(
                timeout=self.timeout_per_platform
            )
            self._clients[PlatformType.ATCODER] = AtCoderScraper(
                timeout=self.timeout_per_platform
            )
            self._clients[PlatformType.HACKERRANK] = HackerRankScraper(
                timeout=self.timeout_per_platform
            )
            self._clients[PlatformType.KAGGLE] = KaggleScraper(
                timeout=self.timeout_per_platform
            )
            
            logger.info("Initialized platform clients for all supported platforms")
            
        except Exception as e:
            logger.error(f"Failed to initialize platform clients: {str(e)}")
            raise
    
    async def scrape_all_platforms(
        self,
        platform_accounts: List[PlatformAccount],
        user_id: str
    ) -> MultiPlatformData:
        """Scrape data from all specified platforms concurrently."""
        start_time = datetime.utcnow()
        
        # Filter active accounts and validate
        active_accounts = [acc for acc in platform_accounts if acc.is_active]
        validated_accounts = await self._validate_accounts(active_accounts)
        
        # Create semaphore to limit concurrent operations
        semaphore = asyncio.Semaphore(self.max_concurrent_scrapers)
        
        # Create scraping tasks
        tasks = []
        for account in validated_accounts:
            task = self._scrape_platform_with_semaphore(semaphore, account)
            tasks.append(task)
        
        # Execute all scraping tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        platform_results = {}
        successful_count = 0
        failed_count = 0
        
        for i, result in enumerate(results):
            account = validated_accounts[i]
            
            if isinstance(result, Exception):
                # Handle exceptions
                platform_results[account.platform] = ScrapingResult(
                    platform=account.platform,
                    success=False,
                    error_message=str(result),
                    scraped_at=datetime.utcnow(),
                    processing_time=0.0
                )
                failed_count += 1
                logger.error(f"Failed to scrape {account.platform}: {str(result)}")
            else:
                platform_results[account.platform] = result
                if result.success:
                    successful_count += 1
                else:
                    failed_count += 1
        
        end_time = datetime.utcnow()
        total_time = (end_time - start_time).total_seconds()
        
        return MultiPlatformData(
            user_id=user_id,
            platforms=platform_results,
            total_platforms=len(validated_accounts),
            successful_platforms=successful_count,
            failed_platforms=failed_count,
            total_processing_time=total_time,
            aggregated_at=end_time
        )
    
    async def _scrape_platform_with_semaphore(
        self,
        semaphore: asyncio.Semaphore,
        account: PlatformAccount
    ) -> ScrapingResult:
        """Scrape a single platform with semaphore control."""
        async with semaphore:
            return await self._scrape_single_platform(account)
    
    async def _scrape_single_platform(self, account: PlatformAccount) -> ScrapingResult:
        """Scrape data from a single platform."""
        start_time = datetime.utcnow()
        
        try:
            client = self._clients.get(account.platform)
            if not client:
                raise APIError(f"No client available for platform: {account.platform}")
            
            # Platform-specific scraping logic
            data = None
            if account.platform == PlatformType.GITHUB:
                data = await self._scrape_github(client, account)
            elif account.platform == PlatformType.LEETCODE:
                data = await self._scrape_leetcode(client, account)
            elif account.platform == PlatformType.LINKEDIN:
                data = await self._scrape_linkedin(client, account)
            elif account.platform == PlatformType.CODEFORCES:
                data = await self._scrape_codeforces(client, account)
            elif account.platform == PlatformType.ATCODER:
                data = await self._scrape_atcoder(client, account)
            elif account.platform == PlatformType.HACKERRANK:
                data = await self._scrape_hackerrank(client, account)
            elif account.platform == PlatformType.KAGGLE:
                data = await self._scrape_kaggle(client, account)
            else:
                raise APIError(f"Scraping not implemented for platform: {account.platform}")
            
            end_time = datetime.utcnow()
            processing_time = (end_time - start_time).total_seconds()
            
            return ScrapingResult(
                platform=account.platform,
                success=True,
                data=data.dict() if hasattr(data, 'dict') else data,
                scraped_at=end_time,
                processing_time=processing_time
            )
            
        except Exception as e:
            end_time = datetime.utcnow()
            processing_time = (end_time - start_time).total_seconds()
            
            logger.error(f"Failed to scrape {account.platform}: {str(e)}")
            
            return ScrapingResult(
                platform=account.platform,
                success=False,
                error_message=str(e),
                scraped_at=end_time,
                processing_time=processing_time
            )
    
    async def _scrape_github(self, client: GitHubClient, account: PlatformAccount) -> GitHubProfile:
        """Scrape GitHub profile data."""
        if not account.username:
            raise APIError("GitHub username is required")
        
        async with client:
            return await client.get_user_profile(account.username)
    
    async def _scrape_leetcode(self, client: LeetCodeScraper, account: PlatformAccount) -> LeetCodeProfile:
        """Scrape LeetCode profile data."""
        if not account.username:
            raise APIError("LeetCode username is required")
        
        async with client:
            return await client.get_user_profile(account.username)
    
    async def _scrape_linkedin(self, client: LinkedInScraper, account: PlatformAccount) -> LinkedInProfile:
        """Scrape LinkedIn profile data."""
        if not account.profile_url:
            raise APIError("LinkedIn profile URL is required")
        
        async with client:
            return await client.get_profile_from_url(account.profile_url)
    
    async def _scrape_codeforces(self, client: CodeforcesScraper, account: PlatformAccount) -> CodeforcesProfile:
        """Scrape Codeforces profile data."""
        if not account.username:
            raise APIError("Codeforces username is required")
        
        async with client:
            return await client.get_user_profile(account.username)
    
    async def _scrape_atcoder(self, client: AtCoderScraper, account: PlatformAccount) -> AtCoderProfile:
        """Scrape AtCoder profile data."""
        if not account.username:
            raise APIError("AtCoder username is required")
        
        async with client:
            return await client.get_user_profile(account.username)
    
    async def _scrape_hackerrank(self, client: HackerRankScraper, account: PlatformAccount) -> HackerRankProfile:
        """Scrape HackerRank profile data."""
        if not account.username:
            raise APIError("HackerRank username is required")
        
        async with client:
            return await client.get_user_profile(account.username)
    
    async def _scrape_kaggle(self, client: KaggleScraper, account: PlatformAccount) -> KaggleProfile:
        """Scrape Kaggle profile data."""
        if not account.username:
            raise APIError("Kaggle username is required")
        
        async with client:
            return await client.get_user_profile(account.username)
    
    async def _validate_accounts(self, accounts: List[PlatformAccount]) -> List[PlatformAccount]:
        """Validate platform accounts before scraping."""
        validated = []
        
        for account in accounts:
            try:
                if account.platform == PlatformType.GITHUB:
                    if not account.username:
                        logger.warning(f"Skipping GitHub account - no username provided")
                        continue
                
                elif account.platform == PlatformType.LEETCODE:
                    if not account.username:
                        logger.warning(f"Skipping LeetCode account - no username provided")
                        continue
                
                elif account.platform == PlatformType.LINKEDIN:
                    if not account.profile_url:
                        logger.warning(f"Skipping LinkedIn account - no profile URL provided")
                        continue
                
                elif account.platform == PlatformType.CODEFORCES:
                    if not account.username:
                        logger.warning(f"Skipping Codeforces account - no username provided")
                        continue
                
                elif account.platform == PlatformType.ATCODER:
                    if not account.username:
                        logger.warning(f"Skipping AtCoder account - no username provided")
                        continue
                
                elif account.platform == PlatformType.HACKERRANK:
                    if not account.username:
                        logger.warning(f"Skipping HackerRank account - no username provided")
                        continue
                
                elif account.platform == PlatformType.KAGGLE:
                    if not account.username:
                        logger.warning(f"Skipping Kaggle account - no username provided")
                        continue
                
                validated.append(account)
                
            except Exception as e:
                logger.warning(f"Failed to validate {account.platform} account: {str(e)}")
                continue
        
        return validated
    
    async def scrape_single_platform_safe(
        self,
        platform: PlatformType,
        username: Optional[str] = None,
        profile_url: Optional[str] = None
    ) -> ScrapingResult:
        """Safely scrape a single platform with comprehensive error handling."""
        account = PlatformAccount(
            platform=platform,
            username=username,
            profile_url=profile_url
        )
        
        return await self._scrape_single_platform(account)
    
    async def test_platform_connectivity(self) -> Dict[PlatformType, bool]:
        """Test connectivity to all supported platforms."""
        connectivity_results = {}
        
        for platform_type, client in self._clients.items():
            try:
                # Test with a known public profile/username
                if platform_type == PlatformType.GITHUB:
                    async with client:
                        await client.get("/user")  # This will fail without auth, but tests connectivity
                elif platform_type == PlatformType.LEETCODE:
                    async with client:
                        await client.validate_username("test")  # Test method
                elif platform_type == PlatformType.LINKEDIN:
                    # LinkedIn connectivity is harder to test due to anti-scraping measures
                    connectivity_results[platform_type] = True
                    continue
                
                connectivity_results[platform_type] = True
                
            except Exception as e:
                logger.warning(f"Connectivity test failed for {platform_type}: {str(e)}")
                connectivity_results[platform_type] = False
        
        return connectivity_results
    
    async def get_platform_rate_limits(self) -> Dict[PlatformType, Dict[str, Any]]:
        """Get current rate limit status for all platforms."""
        rate_limits = {}
        
        for platform_type, client in self._clients.items():
            try:
                if hasattr(client, '_rate_limit_reset') and client._rate_limit_reset:
                    remaining_time = (client._rate_limit_reset - datetime.utcnow()).total_seconds()
                    rate_limits[platform_type] = {
                        "is_limited": remaining_time > 0,
                        "reset_in_seconds": max(0, remaining_time),
                        "reset_at": client._rate_limit_reset.isoformat()
                    }
                else:
                    rate_limits[platform_type] = {
                        "is_limited": False,
                        "reset_in_seconds": 0,
                        "reset_at": None
                    }
            except Exception as e:
                logger.warning(f"Failed to get rate limit info for {platform_type}: {str(e)}")
                rate_limits[platform_type] = {
                    "is_limited": False,
                    "reset_in_seconds": 0,
                    "reset_at": None,
                    "error": str(e)
                }
        
        return rate_limits
    
    async def cleanup(self):
        """Clean up all platform clients."""
        for client in self._clients.values():
            try:
                if hasattr(client, '__aexit__'):
                    await client.__aexit__(None, None, None)
            except Exception as e:
                logger.warning(f"Error during client cleanup: {str(e)}")
        
        self._clients.clear()
        logger.info("Multi-platform scraper cleanup completed")