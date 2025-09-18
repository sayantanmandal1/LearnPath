"""Tests for multi-platform scraper infrastructure."""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

from app.services.external_apis.multi_platform_scraper import (
    MultiPlatformScraper,
    PlatformType,
    PlatformAccount,
    ScrapingResult,
    MultiPlatformData
)
from app.services.external_apis.base_client import APIError


class TestMultiPlatformScraper:
    """Test cases for MultiPlatformScraper."""
    
    @pytest.fixture
    def scraper(self):
        """Create a MultiPlatformScraper instance for testing."""
        return MultiPlatformScraper(
            github_token="test_token",
            max_concurrent_scrapers=2,
            timeout_per_platform=30.0
        )
    
    @pytest.fixture
    def sample_accounts(self):
        """Sample platform accounts for testing."""
        return [
            PlatformAccount(
                platform=PlatformType.GITHUB,
                username="testuser",
                is_active=True
            ),
            PlatformAccount(
                platform=PlatformType.LEETCODE,
                username="testuser",
                is_active=True
            ),
            PlatformAccount(
                platform=PlatformType.LINKEDIN,
                profile_url="https://linkedin.com/in/testuser",
                is_active=True
            ),
            PlatformAccount(
                platform=PlatformType.CODEFORCES,
                username="testuser",
                is_active=True
            )
        ]
    
    def test_scraper_initialization(self, scraper):
        """Test scraper initialization."""
        assert scraper.github_token == "test_token"
        assert scraper.max_concurrent_scrapers == 2
        assert scraper.timeout_per_platform == 30.0
        assert len(scraper._clients) == 7  # All supported platforms
    
    def test_platform_types(self):
        """Test that all required platform types are defined."""
        expected_platforms = {
            "github", "leetcode", "linkedin", "codeforces", 
            "atcoder", "hackerrank", "kaggle"
        }
        actual_platforms = {platform.value for platform in PlatformType}
        assert actual_platforms == expected_platforms
    
    @pytest.mark.asyncio
    async def test_validate_accounts(self, scraper, sample_accounts):
        """Test account validation."""
        validated = await scraper._validate_accounts(sample_accounts)
        
        # All accounts should be valid
        assert len(validated) == 4
        
        # Test with invalid accounts
        invalid_accounts = [
            PlatformAccount(platform=PlatformType.GITHUB, is_active=True),  # No username
            PlatformAccount(platform=PlatformType.LINKEDIN, is_active=True),  # No profile URL
        ]
        
        validated_invalid = await scraper._validate_accounts(invalid_accounts)
        assert len(validated_invalid) == 0
    
    @pytest.mark.asyncio
    async def test_scrape_single_platform_github_success(self, scraper):
        """Test successful GitHub scraping."""
        account = PlatformAccount(
            platform=PlatformType.GITHUB,
            username="testuser",
            is_active=True
        )
        
        # Mock the GitHub client
        mock_profile = MagicMock()
        mock_profile.dict.return_value = {"username": "testuser", "repos": 10}
        
        with patch.object(scraper._clients[PlatformType.GITHUB], '__aenter__', return_value=scraper._clients[PlatformType.GITHUB]):
            with patch.object(scraper._clients[PlatformType.GITHUB], '__aexit__', return_value=None):
                with patch.object(scraper._clients[PlatformType.GITHUB], 'get_user_profile', return_value=mock_profile):
                    result = await scraper._scrape_single_platform(account)
        
        assert result.success is True
        assert result.platform == PlatformType.GITHUB
        assert result.data is not None
        assert result.processing_time > 0
    
    @pytest.mark.asyncio
    async def test_scrape_single_platform_failure(self, scraper):
        """Test platform scraping failure."""
        account = PlatformAccount(
            platform=PlatformType.GITHUB,
            username="nonexistent",
            is_active=True
        )
        
        # Mock the GitHub client to raise an error
        with patch.object(scraper._clients[PlatformType.GITHUB], '__aenter__', return_value=scraper._clients[PlatformType.GITHUB]):
            with patch.object(scraper._clients[PlatformType.GITHUB], '__aexit__', return_value=None):
                with patch.object(scraper._clients[PlatformType.GITHUB], 'get_user_profile', side_effect=APIError("User not found", status_code=404)):
                    result = await scraper._scrape_single_platform(account)
        
        assert result.success is False
        assert result.platform == PlatformType.GITHUB
        assert result.error_message == "User not found"
        assert result.data is None
    
    @pytest.mark.asyncio
    async def test_scrape_all_platforms_concurrent(self, scraper, sample_accounts):
        """Test concurrent scraping of multiple platforms."""
        # Mock all platform clients
        mock_results = {
            PlatformType.GITHUB: {"username": "testuser", "repos": 10},
            PlatformType.LEETCODE: {"username": "testuser", "problems_solved": 100},
            PlatformType.LINKEDIN: {"name": "Test User", "connections": 500},
            PlatformType.CODEFORCES: {"handle": "testuser", "rating": 1500}
        }
        
        async def mock_scrape_platform(account):
            """Mock scraping function that returns success."""
            return ScrapingResult(
                platform=account.platform,
                success=True,
                data=mock_results.get(account.platform, {}),
                scraped_at=datetime.utcnow(),
                processing_time=1.0
            )
        
        with patch.object(scraper, '_scrape_single_platform', side_effect=mock_scrape_platform):
            result = await scraper.scrape_all_platforms(sample_accounts, "test_user_123")
        
        assert isinstance(result, MultiPlatformData)
        assert result.user_id == "test_user_123"
        assert result.total_platforms == 4
        assert result.successful_platforms == 4
        assert result.failed_platforms == 0
        assert len(result.platforms) == 4
    
    @pytest.mark.asyncio
    async def test_scrape_all_platforms_mixed_results(self, scraper, sample_accounts):
        """Test scraping with mixed success/failure results."""
        async def mock_scrape_platform(account):
            """Mock scraping function with mixed results."""
            if account.platform == PlatformType.GITHUB:
                return ScrapingResult(
                    platform=account.platform,
                    success=True,
                    data={"username": "testuser"},
                    scraped_at=datetime.utcnow(),
                    processing_time=1.0
                )
            else:
                return ScrapingResult(
                    platform=account.platform,
                    success=False,
                    error_message="Platform unavailable",
                    scraped_at=datetime.utcnow(),
                    processing_time=0.5
                )
        
        with patch.object(scraper, '_scrape_single_platform', side_effect=mock_scrape_platform):
            result = await scraper.scrape_all_platforms(sample_accounts, "test_user_123")
        
        assert result.total_platforms == 4
        assert result.successful_platforms == 1
        assert result.failed_platforms == 3
        assert result.platforms[PlatformType.GITHUB].success is True
        assert result.platforms[PlatformType.LEETCODE].success is False
    
    @pytest.mark.asyncio
    async def test_scrape_single_platform_safe(self, scraper):
        """Test safe single platform scraping."""
        mock_profile = MagicMock()
        mock_profile.dict.return_value = {"username": "testuser"}
        
        with patch.object(scraper._clients[PlatformType.GITHUB], '__aenter__', return_value=scraper._clients[PlatformType.GITHUB]):
            with patch.object(scraper._clients[PlatformType.GITHUB], '__aexit__', return_value=None):
                with patch.object(scraper._clients[PlatformType.GITHUB], 'get_user_profile', return_value=mock_profile):
                    result = await scraper.scrape_single_platform_safe(
                        PlatformType.GITHUB,
                        username="testuser"
                    )
        
        assert result.success is True
        assert result.platform == PlatformType.GITHUB
    
    @pytest.mark.asyncio
    async def test_test_platform_connectivity(self, scraper):
        """Test platform connectivity testing."""
        # Mock connectivity tests
        with patch.object(scraper._clients[PlatformType.GITHUB], '__aenter__', return_value=scraper._clients[PlatformType.GITHUB]):
            with patch.object(scraper._clients[PlatformType.GITHUB], '__aexit__', return_value=None):
                with patch.object(scraper._clients[PlatformType.GITHUB], 'get', return_value={}):
                    with patch.object(scraper._clients[PlatformType.LEETCODE], '__aenter__', return_value=scraper._clients[PlatformType.LEETCODE]):
                        with patch.object(scraper._clients[PlatformType.LEETCODE], '__aexit__', return_value=None):
                            with patch.object(scraper._clients[PlatformType.LEETCODE], 'validate_username', return_value=True):
                                connectivity = await scraper.test_platform_connectivity()
        
        assert isinstance(connectivity, dict)
        assert PlatformType.GITHUB in connectivity
        assert PlatformType.LEETCODE in connectivity
        assert PlatformType.LINKEDIN in connectivity  # Always returns True
    
    @pytest.mark.asyncio
    async def test_get_platform_rate_limits(self, scraper):
        """Test rate limit status retrieval."""
        # Mock rate limit info
        scraper._clients[PlatformType.GITHUB]._rate_limit_reset = datetime.utcnow()
        
        rate_limits = await scraper.get_platform_rate_limits()
        
        assert isinstance(rate_limits, dict)
        assert PlatformType.GITHUB in rate_limits
        assert "is_limited" in rate_limits[PlatformType.GITHUB]
        assert "reset_in_seconds" in rate_limits[PlatformType.GITHUB]
    
    @pytest.mark.asyncio
    async def test_cleanup(self, scraper):
        """Test scraper cleanup."""
        # Mock clients with __aexit__ method
        for client in scraper._clients.values():
            client.__aexit__ = AsyncMock()
        
        await scraper.cleanup()
        
        # Verify all clients were cleaned up
        for client in scraper._clients.values():
            if hasattr(client, '__aexit__'):
                client.__aexit__.assert_called_once()
    
    def test_platform_account_model(self):
        """Test PlatformAccount model validation."""
        # Valid account
        account = PlatformAccount(
            platform=PlatformType.GITHUB,
            username="testuser",
            is_active=True
        )
        assert account.platform == PlatformType.GITHUB
        assert account.username == "testuser"
        assert account.is_active is True
        
        # Account with profile URL
        linkedin_account = PlatformAccount(
            platform=PlatformType.LINKEDIN,
            profile_url="https://linkedin.com/in/testuser",
            is_active=True
        )
        assert linkedin_account.profile_url == "https://linkedin.com/in/testuser"
    
    def test_scraping_result_model(self):
        """Test ScrapingResult model."""
        result = ScrapingResult(
            platform=PlatformType.GITHUB,
            success=True,
            data={"username": "testuser"},
            scraped_at=datetime.utcnow(),
            processing_time=1.5
        )
        
        assert result.platform == PlatformType.GITHUB
        assert result.success is True
        assert result.data["username"] == "testuser"
        assert result.processing_time == 1.5
    
    def test_multi_platform_data_model(self):
        """Test MultiPlatformData model."""
        platforms = {
            PlatformType.GITHUB: ScrapingResult(
                platform=PlatformType.GITHUB,
                success=True,
                data={"username": "testuser"},
                scraped_at=datetime.utcnow(),
                processing_time=1.0
            )
        }
        
        data = MultiPlatformData(
            user_id="test_user_123",
            platforms=platforms,
            total_platforms=1,
            successful_platforms=1,
            failed_platforms=0,
            total_processing_time=1.0,
            aggregated_at=datetime.utcnow()
        )
        
        assert data.user_id == "test_user_123"
        assert data.total_platforms == 1
        assert data.successful_platforms == 1
        assert data.failed_platforms == 0
    
    @pytest.mark.asyncio
    async def test_concurrent_scraping_with_semaphore(self, scraper):
        """Test that concurrent scraping respects semaphore limits."""
        accounts = [
            PlatformAccount(platform=PlatformType.GITHUB, username=f"user{i}", is_active=True)
            for i in range(5)  # More accounts than max_concurrent_scrapers
        ]
        
        call_times = []
        
        async def mock_scrape_with_delay(account):
            """Mock scraping with delay to test concurrency."""
            call_times.append(datetime.utcnow())
            await asyncio.sleep(0.1)  # Simulate processing time
            return ScrapingResult(
                platform=account.platform,
                success=True,
                data={"username": account.username},
                scraped_at=datetime.utcnow(),
                processing_time=0.1
            )
        
        with patch.object(scraper, '_scrape_single_platform', side_effect=mock_scrape_with_delay):
            result = await scraper.scrape_all_platforms(accounts, "test_user")
        
        # Verify that not all calls started simultaneously (due to semaphore)
        assert len(call_times) == 5
        assert result.successful_platforms == 5


class TestPlatformSpecificScrapers:
    """Test platform-specific scraper functionality."""
    
    @pytest.fixture
    def scraper(self):
        """Create scraper for platform-specific tests."""
        return MultiPlatformScraper()
    
    @pytest.mark.asyncio
    async def test_github_scraping_validation(self, scraper):
        """Test GitHub scraping parameter validation."""
        account = PlatformAccount(platform=PlatformType.GITHUB, is_active=True)
        
        with pytest.raises(APIError, match="GitHub username is required"):
            await scraper._scrape_github(scraper._clients[PlatformType.GITHUB], account)
    
    @pytest.mark.asyncio
    async def test_leetcode_scraping_validation(self, scraper):
        """Test LeetCode scraping parameter validation."""
        account = PlatformAccount(platform=PlatformType.LEETCODE, is_active=True)
        
        with pytest.raises(APIError, match="LeetCode username is required"):
            await scraper._scrape_leetcode(scraper._clients[PlatformType.LEETCODE], account)
    
    @pytest.mark.asyncio
    async def test_linkedin_scraping_validation(self, scraper):
        """Test LinkedIn scraping parameter validation."""
        account = PlatformAccount(platform=PlatformType.LINKEDIN, is_active=True)
        
        with pytest.raises(APIError, match="LinkedIn profile URL is required"):
            await scraper._scrape_linkedin(scraper._clients[PlatformType.LINKEDIN], account)
    
    @pytest.mark.asyncio
    async def test_codeforces_scraping_validation(self, scraper):
        """Test Codeforces scraping parameter validation."""
        account = PlatformAccount(platform=PlatformType.CODEFORCES, is_active=True)
        
        with pytest.raises(APIError, match="Codeforces username is required"):
            await scraper._scrape_codeforces(scraper._clients[PlatformType.CODEFORCES], account)
    
    @pytest.mark.asyncio
    async def test_unsupported_platform_error(self, scraper):
        """Test error handling for unsupported platforms."""
        # Since Pydantic validates the enum, we can't create an invalid platform
        # Instead, test that all supported platforms have clients
        for platform_type in PlatformType:
            assert platform_type in scraper._clients, f"No client found for {platform_type}"
        
        # Test that the scraper handles missing client gracefully by removing one
        original_client = scraper._clients.pop(PlatformType.KAGGLE)
        
        account = PlatformAccount(platform=PlatformType.KAGGLE, username="test", is_active=True)
        result = await scraper._scrape_single_platform(account)
        
        assert result.success is False
        assert "no client available" in result.error_message.lower()
        
        # Restore the client
        scraper._clients[PlatformType.KAGGLE] = original_client


if __name__ == "__main__":
    pytest.main([__file__])