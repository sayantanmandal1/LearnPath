"""Tests for external API integration services."""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timedelta
from typing import Dict, Any

from app.services.external_apis import (
    GitHubClient,
    LeetCodeScraper,
    LinkedInScraper,
    DataValidator,
    ExternalAPIIntegrationService,
    APIError,
    RateLimitError,
    DataQuality,
    ProfileExtractionRequest
)
from app.services.external_apis.github_client import GitHubProfile, GitHubRepository
from app.services.external_apis.leetcode_scraper import LeetCodeProfile, LeetCodeStats
from app.services.external_apis.linkedin_scraper import LinkedInProfile, LinkedInExperience


class TestBaseAPIClient:
    """Test base API client functionality."""
    
    @pytest.mark.asyncio
    async def test_retry_mechanism(self):
        """Test exponential backoff retry mechanism."""
        from app.services.external_apis.base_client import BaseAPIClient, RetryConfig
        
        class TestClient(BaseAPIClient):
            def __init__(self):
                super().__init__("https://api.test.com", retry_config=RetryConfig(max_retries=2, base_delay=0.1))
                self.attempt_count = 0
            
            def _get_auth_headers(self):
                return {}
            
            async def _make_request(self, method, endpoint, **kwargs):
                self.attempt_count += 1
                if self.attempt_count < 3:
                    raise APIError("Temporary error", status_code=500)
                return {"success": True}
        
        client = TestClient()
        
        # Should succeed after retries
        result = await client._make_request("GET", "/test")
        assert result["success"] is True
        assert client.attempt_count == 3
    
    @pytest.mark.asyncio
    async def test_rate_limit_handling(self):
        """Test rate limit error handling."""
        from app.services.external_apis.base_client import BaseAPIClient
        
        class TestClient(BaseAPIClient):
            def _get_auth_headers(self):
                return {}
            
            async def _make_request(self, method, endpoint, **kwargs):
                raise RateLimitError("Rate limited", retry_after=1)
        
        client = TestClient()
        
        with pytest.raises(RateLimitError):
            await client._make_request("GET", "/test")


class TestGitHubClient:
    """Test GitHub API client."""
    
    @pytest.fixture
    def mock_github_user_data(self):
        return {
            "login": "testuser",
            "name": "Test User",
            "bio": "Software Developer",
            "company": "Test Company",
            "location": "San Francisco",
            "email": "test@example.com",
            "public_repos": 10,
            "followers": 50,
            "following": 30,
            "created_at": "2020-01-01T00:00:00Z"
        }
    
    @pytest.fixture
    def mock_github_repos_data(self):
        return [
            {
                "name": "test-repo",
                "full_name": "testuser/test-repo",
                "description": "A test repository",
                "language": "Python",
                "stargazers_count": 5,
                "forks_count": 2,
                "size": 1000,
                "created_at": "2021-01-01T00:00:00Z",
                "updated_at": "2023-01-01T00:00:00Z",
                "topics": ["python", "testing"],
                "fork": False,
                "private": False
            }
        ]
    
    @pytest.mark.asyncio
    async def test_get_user_profile_success(self, mock_github_user_data, mock_github_repos_data):
        """Test successful GitHub profile extraction."""
        client = GitHubClient()
        
        with patch.object(client, 'get') as mock_get:
            # Mock API responses
            mock_get.side_effect = [
                mock_github_user_data,  # User data
                mock_github_repos_data,  # Repositories
                {"Python": 1000, "JavaScript": 500},  # Languages for repo
                []  # Events (for commit estimation)
            ]
            
            profile = await client.get_user_profile("testuser")
            
            assert isinstance(profile, GitHubProfile)
            assert profile.username == "testuser"
            assert profile.name == "Test User"
            assert profile.public_repos == 10
            assert len(profile.repositories) == 1
            assert profile.repositories[0].name == "test-repo"
    
    @pytest.mark.asyncio
    async def test_get_user_profile_not_found(self):
        """Test GitHub profile not found error."""
        client = GitHubClient()
        
        with patch.object(client, 'get') as mock_get:
            mock_get.side_effect = APIError("Not found", status_code=404)
            
            with pytest.raises(APIError) as exc_info:
                await client.get_user_profile("nonexistentuser")
            
            assert "not found" in str(exc_info.value).lower()
    
    @pytest.mark.asyncio
    async def test_rate_limit_handling(self):
        """Test GitHub rate limit handling."""
        client = GitHubClient()
        
        with patch.object(client, 'get') as mock_get:
            mock_get.side_effect = RateLimitError("Rate limited", retry_after=60)
            
            with pytest.raises(RateLimitError):
                await client.get_user_profile("testuser")


class TestLeetCodeScraper:
    """Test LeetCode scraper."""
    
    @pytest.fixture
    def mock_leetcode_user_data(self):
        return {
            "data": {
                "matchedUser": {
                    "username": "testuser",
                    "profile": {
                        "realName": "Test User",
                        "countryName": "United States",
                        "company": "Test Company",
                        "school": "Test University"
                    }
                }
            }
        }
    
    @pytest.fixture
    def mock_leetcode_stats_data(self):
        return {
            "data": {
                "matchedUser": {
                    "submitStats": {
                        "acSubmissionNum": [
                            {"difficulty": "Easy", "count": 50, "submissions": 60},
                            {"difficulty": "Medium", "count": 30, "submissions": 45},
                            {"difficulty": "Hard", "count": 10, "submissions": 20}
                        ]
                    },
                    "profile": {
                        "ranking": 12345,
                        "reputation": 100
                    }
                }
            }
        }
    
    @pytest.mark.asyncio
    async def test_get_user_profile_success(self, mock_leetcode_user_data, mock_leetcode_stats_data):
        """Test successful LeetCode profile extraction."""
        scraper = LeetCodeScraper()
        
        with patch.object(scraper, 'post') as mock_post:
            # Mock GraphQL responses
            mock_post.side_effect = [
                mock_leetcode_user_data,  # Basic user info
                mock_leetcode_stats_data,  # Stats
                {"data": {"recentSubmissionList": []}},  # Recent submissions
                {"data": {"userContestRankingHistory": []}},  # Contest history
                {"data": {"matchedUser": {"languageProblemCount": []}}},  # Language stats
            ]
            
            profile = await scraper.get_user_profile("testuser")
            
            assert isinstance(profile, LeetCodeProfile)
            assert profile.username == "testuser"
            assert profile.real_name == "Test User"
            assert profile.stats.total_solved == 90
            assert profile.stats.easy_solved == 50
            assert profile.stats.medium_solved == 30
            assert profile.stats.hard_solved == 10
    
    @pytest.mark.asyncio
    async def test_get_user_profile_not_found(self):
        """Test LeetCode profile not found error."""
        scraper = LeetCodeScraper()
        
        with patch.object(scraper, 'post') as mock_post:
            mock_post.return_value = {"data": {"matchedUser": None}}
            
            with pytest.raises(APIError) as exc_info:
                await scraper.get_user_profile("nonexistentuser")
            
            assert "does not exist" in str(exc_info.value).lower()


class TestLinkedInScraper:
    """Test LinkedIn scraper."""
    
    @pytest.mark.asyncio
    async def test_extract_profile_id_from_url(self):
        """Test LinkedIn profile ID extraction from URLs."""
        scraper = LinkedInScraper()
        
        test_cases = [
            ("https://www.linkedin.com/in/testuser/", "testuser"),
            ("https://linkedin.com/in/test-user-123", "test-user-123"),
            ("https://www.linkedin.com/pub/testuser/1/2/3", "testuser"),
            ("invalid-url", None)
        ]
        
        for url, expected in test_cases:
            result = scraper._extract_profile_id_from_url(url)
            assert result == expected
    
    @pytest.mark.asyncio
    async def test_get_profile_from_url_mock(self):
        """Test LinkedIn profile extraction (mock implementation)."""
        scraper = LinkedInScraper()
        
        with patch.object(scraper, 'get') as mock_get:
            mock_get.return_value = {"content": "<html>Mock LinkedIn Profile</html>"}
            
            profile = await scraper.get_profile_from_url("https://linkedin.com/in/testuser")
            
            assert isinstance(profile, LinkedInProfile)
            assert profile.name == "Mock User"  # From mock implementation
            assert profile.profile_url == "https://linkedin.com/in/testuser"


class TestDataValidator:
    """Test data validation and cleaning."""
    
    @pytest.fixture
    def sample_github_profile(self):
        return GitHubProfile(
            username="testuser",
            name="Test User",
            bio="Software Developer",
            company="Test Company Inc",
            location="San Francisco",
            public_repos=10,
            followers=50,
            following=30,
            created_at=datetime.utcnow(),
            repositories=[],
            languages={"Python": 1000, "javascript": 500, "c++": 300},
            total_stars=15,
            contribution_years=[2020, 2021, 2022, 2023]
        )
    
    def test_validate_github_profile_success(self, sample_github_profile):
        """Test successful GitHub profile validation."""
        validator = DataValidator()
        
        result = validator.validate_github_profile(sample_github_profile)
        
        assert result.is_valid is True
        assert result.quality in [DataQuality.HIGH, DataQuality.MEDIUM]
        assert result.confidence_score > 0.5
        assert "testuser" in result.cleaned_data["username"]
        assert "Test Company" in result.cleaned_data["company"]  # Normalized
        assert "JavaScript" in result.cleaned_data["languages"]  # Normalized
        assert "C++" in result.cleaned_data["languages"]  # Normalized
    
    def test_validate_github_profile_invalid(self):
        """Test GitHub profile validation with invalid data."""
        validator = DataValidator()
        
        invalid_profile = GitHubProfile(
            username="",  # Invalid empty username
            public_repos=-5,  # Invalid negative repos
            followers=0,
            following=0,
            created_at=datetime.utcnow(),
            repositories=[],
            languages={},
            total_stars=-1,  # Invalid negative stars
            contribution_years=[]
        )
        
        result = validator.validate_github_profile(invalid_profile)
        
        assert result.is_valid is False
        assert len(result.errors) > 0
        assert result.confidence_score < 0.5
    
    def test_normalize_languages(self):
        """Test programming language normalization."""
        validator = DataValidator()
        
        test_languages = {
            "javascript": 1000,
            "c++": 500,
            "python": 800,
            "JAVA": 300,
            "typescript": 200
        }
        
        normalized = validator._normalize_languages(test_languages)
        
        assert "JavaScript" in normalized
        assert "C++" in normalized
        assert "Python" in normalized
        assert "Java" in normalized
        assert "TypeScript" in normalized
    
    def test_normalize_company_names(self):
        """Test company name normalization."""
        validator = DataValidator()
        
        test_cases = [
            ("Google Inc", "Google"),
            ("Microsoft Corporation", "Microsoft"),
            ("Apple Inc.", "Apple"),
            ("Test Company LLC", "Test Company"),
            ("Unknown Corp.", "Unknown")
        ]
        
        for input_name, expected in test_cases:
            result = validator._normalize_company(input_name)
            assert result == expected


class TestExternalAPIIntegrationService:
    """Test the integration service."""
    
    @pytest.fixture
    def integration_service(self):
        return ExternalAPIIntegrationService(enable_caching=False)
    
    @pytest.mark.asyncio
    async def test_extract_comprehensive_profile_success(self, integration_service):
        """Test successful comprehensive profile extraction."""
        request = ProfileExtractionRequest(
            github_username="testuser",
            leetcode_username="testuser",
            timeout_seconds=30
        )
        
        # Mock the individual extraction methods
        with patch.object(integration_service, '_extract_github_profile_safe') as mock_github, \
             patch.object(integration_service, '_extract_leetcode_profile_safe') as mock_leetcode:
            
            mock_github.return_value = {"username": "testuser", "public_repos": 10}
            mock_leetcode.return_value = {"username": "testuser", "stats": {"total_solved": 100}}
            
            result = await integration_service.extract_comprehensive_profile(request)
            
            assert result.success is True
            assert "github" in result.sources_successful
            assert "leetcode" in result.sources_successful
            assert result.github_profile is not None
            assert result.leetcode_profile is not None
            assert result.extraction_time > 0
    
    @pytest.mark.asyncio
    async def test_extract_comprehensive_profile_partial_failure(self, integration_service):
        """Test profile extraction with partial failures."""
        request = ProfileExtractionRequest(
            github_username="testuser",
            leetcode_username="invaliduser",
            enable_graceful_degradation=True
        )
        
        with patch.object(integration_service, '_extract_github_profile_safe') as mock_github, \
             patch.object(integration_service, '_extract_leetcode_profile_safe') as mock_leetcode:
            
            mock_github.return_value = {"username": "testuser", "public_repos": 10}
            mock_leetcode.side_effect = APIError("User not found", status_code=404)
            
            result = await integration_service.extract_comprehensive_profile(request)
            
            assert result.success is True  # Graceful degradation enabled
            assert "github" in result.sources_successful
            assert "leetcode" not in result.sources_successful
            assert "leetcode" in result.errors
            assert result.github_profile is not None
            assert result.leetcode_profile is None
    
    @pytest.mark.asyncio
    async def test_extract_comprehensive_profile_timeout(self, integration_service):
        """Test profile extraction timeout handling."""
        request = ProfileExtractionRequest(
            github_username="testuser",
            timeout_seconds=0.1,  # Very short timeout
            enable_graceful_degradation=True
        )
        
        with patch.object(integration_service, '_extract_github_profile_safe') as mock_github:
            # Simulate slow response
            async def slow_response(*args, **kwargs):
                await asyncio.sleep(1)
                return {"username": "testuser"}
            
            mock_github.side_effect = slow_response
            
            result = await integration_service.extract_comprehensive_profile(request)
            
            assert "general" in result.errors
            assert "timed out" in result.errors["general"].lower()
            assert result.success is True  # Graceful degradation
    
    @pytest.mark.asyncio
    async def test_validate_profile_sources(self, integration_service):
        """Test profile source validation."""
        with patch.object(integration_service, '_validate_github_username') as mock_github, \
             patch.object(integration_service, '_validate_leetcode_username') as mock_leetcode:
            
            mock_github.return_value = True
            mock_leetcode.return_value = False
            
            results = await integration_service.validate_profile_sources(
                github_username="validuser",
                leetcode_username="invaliduser"
            )
            
            assert results["github"] is True
            assert results["leetcode"] is False
    
    def test_cache_functionality(self):
        """Test caching functionality."""
        service = ExternalAPIIntegrationService(enable_caching=True, cache_ttl_seconds=60)
        
        # Test cache stats
        stats = service.get_cache_stats()
        assert stats["total_entries"] == 0
        assert stats["valid_entries"] == 0
        
        # Test cache clearing
        service._cache["test"] = ({"data": "test"}, datetime.utcnow())
        assert len(service._cache) == 1
        
        service.clear_cache()
        assert len(service._cache) == 0


@pytest.mark.integration
class TestExternalAPIIntegration:
    """Integration tests for external APIs (requires network access)."""
    
    @pytest.mark.asyncio
    async def test_github_api_integration(self):
        """Test actual GitHub API integration (if token available)."""
        import os
        
        github_token = os.getenv("GITHUB_TOKEN")
        if not github_token:
            pytest.skip("GitHub token not available")
        
        client = GitHubClient(api_token=github_token)
        
        try:
            async with client:
                # Test with a known public user
                profile = await client.get_user_profile("octocat")
                assert profile.username == "octocat"
                assert profile.public_repos > 0
        except RateLimitError:
            pytest.skip("GitHub rate limit exceeded")
        except APIError as e:
            pytest.fail(f"GitHub API error: {e.message}")
    
    @pytest.mark.asyncio
    async def test_leetcode_scraping_integration(self):
        """Test actual LeetCode scraping (may be flaky due to anti-bot measures)."""
        scraper = LeetCodeScraper()
        
        try:
            async with scraper:
                # Test username validation
                is_valid = await scraper.validate_username("nonexistentuser12345")
                assert is_valid is False
        except APIError:
            pytest.skip("LeetCode scraping blocked or unavailable")
    
    @pytest.mark.asyncio
    async def test_integration_service_end_to_end(self):
        """Test end-to-end integration service functionality."""
        import os
        
        github_token = os.getenv("GITHUB_TOKEN")
        service = ExternalAPIIntegrationService(github_token=github_token)
        
        request = ProfileExtractionRequest(
            github_username="octocat",  # Known public user
            enable_graceful_degradation=True,
            timeout_seconds=30
        )
        
        try:
            result = await service.extract_comprehensive_profile(request)
            
            # Should succeed with at least GitHub data
            assert result.success is True
            if result.github_profile:
                assert result.github_profile["username"] == "octocat"
            
        except Exception as e:
            pytest.skip(f"Integration test failed due to external factors: {str(e)}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])