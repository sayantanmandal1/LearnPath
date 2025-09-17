"""Tests for enhanced external API integration features."""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timedelta
from typing import Dict, Any

from app.services.external_apis import (
    ExternalAPIIntegrationService,
    ProfileExtractionRequest,
    ProfileMerger,
    MergedProfile,
    CircuitBreaker,
    CircuitBreakerError,
    CircuitBreakerConfig,
    circuit_breaker_manager,
    APIError,
    RateLimitError
)


class TestProfileMerger:
    """Test profile merging functionality."""
    
    @pytest.fixture
    def sample_github_data(self):
        return {
            "username": "testuser",
            "name": "Test User",
            "bio": "Software Developer",
            "company": "Tech Company",
            "location": "San Francisco",
            "email": "test@example.com",
            "public_repos": 25,
            "followers": 100,
            "following": 50,
            "languages": {"Python": 5000, "JavaScript": 3000, "Go": 1000},
            "total_stars": 150,
            "total_commits": 500,
            "created_at": "2020-01-01T00:00:00Z"
        }
    
    @pytest.fixture
    def sample_leetcode_data(self):
        return {
            "username": "testuser",
            "real_name": "Test User",
            "country": "United States",
            "company": "Tech Company",
            "stats": {
                "total_solved": 200,
                "easy_solved": 100,
                "medium_solved": 80,
                "hard_solved": 20,
                "acceptance_rate": 85.5,
                "ranking": 5000
            },
            "languages_used": {"Python": 120, "Java": 50, "C++": 30},
            "skill_tags": {"Array": 50, "Dynamic Programming": 30, "Tree": 25}
        }
    
    @pytest.fixture
    def sample_linkedin_data(self):
        return {
            "name": "Test User",
            "headline": "Senior Software Engineer",
            "location": "San Francisco, CA",
            "industry": "Technology",
            "summary": "Experienced software engineer with 5+ years in web development",
            "current_company": "Tech Company",
            "current_position": "Senior Software Engineer",
            "skills": [
                {"name": "Python", "endorsements": 25, "is_top_skill": True},
                {"name": "Machine Learning", "endorsements": 15, "is_top_skill": True},
                {"name": "React", "endorsements": 20, "is_top_skill": False}
            ],
            "experience": [
                {
                    "company": "Tech Company",
                    "position": "Senior Software Engineer",
                    "duration": "2 years 6 months",
                    "is_current": True
                },
                {
                    "company": "Previous Company",
                    "position": "Software Engineer",
                    "duration": "3 years",
                    "is_current": False
                }
            ]
        }
    
    def test_merge_basic_info(self, sample_github_data, sample_leetcode_data, sample_linkedin_data):
        """Test merging of basic profile information."""
        merger = ProfileMerger()
        
        merged = merger.merge_profiles(
            github_profile=sample_github_data,
            leetcode_profile=sample_leetcode_data,
            linkedin_profile=sample_linkedin_data
        )
        
        assert isinstance(merged, MergedProfile)
        assert merged.name == "Test User"  # From LinkedIn (priority)
        assert merged.username == "testuser"  # From GitHub (priority)
        assert merged.email == "test@example.com"  # From GitHub
        assert merged.location == "San Francisco, CA"  # From LinkedIn (priority)
        assert merged.current_company == "Tech Company"  # From LinkedIn (priority)
        assert merged.current_position == "Senior Software Engineer"  # From LinkedIn
        assert len(merged.data_sources) == 3
    
    def test_merge_skills(self, sample_github_data, sample_leetcode_data, sample_linkedin_data):
        """Test merging of skills and technologies."""
        merger = ProfileMerger()
        
        merged = merger.merge_profiles(
            github_profile=sample_github_data,
            leetcode_profile=sample_leetcode_data,
            linkedin_profile=sample_linkedin_data
        )
        
        # Check programming languages are merged
        assert "Python" in merged.programming_languages
        assert "JavaScript" in merged.programming_languages
        assert "Java" in merged.programming_languages
        
        # Check technical skills
        assert "Machine Learning" in merged.technical_skills
        
        # Python should have scores from all sources
        python_score = merged.programming_languages.get("Python", 0)
        assert python_score > 0  # Should be weighted combination
    
    def test_merge_coding_stats(self, sample_github_data, sample_leetcode_data):
        """Test merging of coding statistics."""
        merger = ProfileMerger()
        
        merged = merger.merge_profiles(
            github_profile=sample_github_data,
            leetcode_profile=sample_leetcode_data
        )
        
        assert merged.total_repositories == 25
        assert merged.total_stars == 150
        assert merged.total_problems_solved == 200
        assert merged.github_stats is not None
        assert merged.leetcode_stats is not None
        assert merged.github_stats["public_repos"] == 25
        assert merged.leetcode_stats["total_solved"] == 200
    
    def test_quality_calculation(self, sample_github_data, sample_linkedin_data):
        """Test data quality calculation."""
        merger = ProfileMerger()
        
        # Test with complete data
        merged_complete = merger.merge_profiles(
            github_profile=sample_github_data,
            linkedin_profile=sample_linkedin_data
        )
        
        assert merged_complete.data_quality_score > 0.5
        assert merged_complete.confidence_level in ["low", "medium", "high"]
        
        # Test with minimal data
        minimal_github = {"username": "testuser", "public_repos": 1}
        merged_minimal = merger.merge_profiles(github_profile=minimal_github)
        
        assert merged_minimal.data_quality_score < merged_complete.data_quality_score
        assert merged_minimal.confidence_level == "low"
    
    def test_experience_estimation(self):
        """Test experience years estimation."""
        merger = ProfileMerger()
        
        experience_data = [
            {"duration": "2 years 6 months"},
            {"duration": "3 years"},
            {"duration": "1 year 2 months"}
        ]
        
        years = merger._estimate_experience_years(experience_data)
        assert years == 6  # 2.5 + 3 + 1.17 ≈ 6.67 → 6


class TestCircuitBreaker:
    """Test circuit breaker functionality."""
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_normal_operation(self):
        """Test circuit breaker in normal operation."""
        breaker = CircuitBreaker("test", CircuitBreakerConfig(failure_threshold=3))
        
        async def successful_call():
            return "success"
        
        result = await breaker.call(successful_call)
        assert result == "success"
        
        stats = breaker.get_stats()
        assert stats.total_requests == 1
        assert stats.total_successes == 1
        assert stats.total_failures == 0
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_failure_handling(self):
        """Test circuit breaker failure handling."""
        breaker = CircuitBreaker("test", CircuitBreakerConfig(failure_threshold=2, recovery_timeout=1))
        
        async def failing_call():
            raise Exception("Test failure")
        
        # First failure
        with pytest.raises(Exception):
            await breaker.call(failing_call)
        
        # Second failure - should open circuit
        with pytest.raises(Exception):
            await breaker.call(failing_call)
        
        # Third call should be blocked by circuit breaker
        with pytest.raises(CircuitBreakerError):
            await breaker.call(failing_call)
        
        stats = breaker.get_stats()
        assert stats.total_failures == 2
        assert stats.total_requests == 3  # Including the blocked call
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_recovery(self):
        """Test circuit breaker recovery mechanism."""
        breaker = CircuitBreaker("test", CircuitBreakerConfig(
            failure_threshold=2,
            recovery_timeout=0.1,  # Very short for testing
            success_threshold=1
        ))
        
        async def failing_call():
            raise Exception("Test failure")
        
        async def successful_call():
            return "success"
        
        # Trigger circuit opening
        with pytest.raises(Exception):
            await breaker.call(failing_call)
        with pytest.raises(Exception):
            await breaker.call(failing_call)
        
        # Wait for recovery timeout
        await asyncio.sleep(0.2)
        
        # Should allow one call in half-open state
        result = await breaker.call(successful_call)
        assert result == "success"
        
        # Circuit should be closed now
        stats = breaker.get_stats()
        assert stats.state.value == "closed"
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_timeout(self):
        """Test circuit breaker timeout handling."""
        breaker = CircuitBreaker("test", CircuitBreakerConfig(timeout=0.1))
        
        async def slow_call():
            await asyncio.sleep(0.2)  # Longer than timeout
            return "success"
        
        with pytest.raises(CircuitBreakerError) as exc_info:
            await breaker.call(slow_call)
        
        assert "timed out" in str(exc_info.value)


class TestEnhancedIntegrationService:
    """Test enhanced integration service functionality."""
    
    @pytest.fixture
    def integration_service(self):
        return ExternalAPIIntegrationService(enable_caching=False)
    
    @pytest.mark.asyncio
    async def test_profile_extraction_with_merging(self, integration_service):
        """Test profile extraction with automatic merging."""
        request = ProfileExtractionRequest(
            github_username="testuser",
            leetcode_username="testuser",
            enable_validation=True
        )
        
        # Mock successful extractions
        with patch.object(integration_service, '_extract_github_profile_safe') as mock_github, \
             patch.object(integration_service, '_extract_leetcode_profile_safe') as mock_leetcode:
            
            mock_github.return_value = {
                "username": "testuser",
                "name": "Test User",
                "public_repos": 10,
                "languages": {"Python": 1000}
            }
            
            mock_leetcode.return_value = {
                "username": "testuser",
                "stats": {
                    "total_solved": 100,
                    "easy_solved": 60,
                    "medium_solved": 30,
                    "hard_solved": 10,
                    "acceptance_rate": 85.0
                }
            }
            
            result = await integration_service.extract_comprehensive_profile(request)
            
            assert result.success is True
            assert result.merged_profile is not None
            assert result.merged_profile["name"] == "Test User"
            assert result.merged_profile["total_problems_solved"] == 100
            assert len(result.sources_successful) == 2
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_integration(self, integration_service):
        """Test circuit breaker integration in service."""
        request = ProfileExtractionRequest(github_username="testuser")
        
        # Mock circuit breaker error
        with patch.object(integration_service, '_extract_github_profile_safe') as mock_github:
            mock_github.side_effect = CircuitBreakerError("Circuit breaker open")
            
            result = await integration_service.extract_comprehensive_profile(request)
            
            # Should handle circuit breaker gracefully
            assert "github" in result.errors
            assert "circuit breaker" in result.errors["github"].lower() or "temporarily unavailable" in result.errors["github"].lower()
    
    @pytest.mark.asyncio
    async def test_enhanced_error_handling(self, integration_service):
        """Test enhanced error handling with retries."""
        request = ProfileExtractionRequest(github_username="testuser")
        
        # Mock intermittent failures
        call_count = 0
        
        async def mock_github_call():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise APIError("Temporary error", status_code=500)
            return {"username": "testuser", "public_repos": 5}
        
        with patch.object(integration_service.github_client, 'get_user_profile', side_effect=mock_github_call):
            # Mock the circuit breaker to allow retries
            with patch('app.services.external_apis.integration_service.circuit_breaker_manager') as mock_manager:
                mock_breaker = AsyncMock()
                mock_breaker.call = AsyncMock(side_effect=mock_github_call)
                mock_manager.get_breaker.return_value = mock_breaker
                
                result = await integration_service.extract_comprehensive_profile(request)
                
                # Should eventually succeed after retries
                assert result.success is True
                assert call_count == 3  # Should have retried
    
    def test_circuit_breaker_stats(self, integration_service):
        """Test circuit breaker statistics retrieval."""
        stats = integration_service.get_circuit_breaker_stats()
        
        assert isinstance(stats, dict)
        # Should have stats for all configured services
        expected_services = ["github", "leetcode", "linkedin"]
        for service in expected_services:
            if service in stats:
                assert "state" in stats[service]
                assert "total_requests" in stats[service]
                assert "total_failures" in stats[service]
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_reset(self, integration_service):
        """Test circuit breaker reset functionality."""
        # Should not raise any exceptions
        await integration_service.reset_circuit_breakers()
        await integration_service.reset_circuit_breaker("github")


class TestRateLimitHandling:
    """Test enhanced rate limit handling."""
    
    @pytest.mark.asyncio
    async def test_github_rate_limit_with_backoff(self):
        """Test GitHub rate limit handling with exponential backoff."""
        service = ExternalAPIIntegrationService(enable_caching=False)
        
        call_count = 0
        
        async def mock_rate_limited_call():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise RateLimitError("Rate limited", retry_after=1)
            return {"username": "testuser", "public_repos": 1}
        
        with patch.object(service.github_client, 'get_user_profile', side_effect=mock_rate_limited_call):
            # Mock circuit breaker
            with patch('app.services.external_apis.integration_service.circuit_breaker_manager') as mock_manager:
                mock_breaker = AsyncMock()
                mock_breaker.call = AsyncMock(side_effect=mock_rate_limited_call)
                mock_manager.get_breaker.return_value = mock_breaker
                
                result = await service._extract_github_profile_safe("testuser")
                
                assert result is not None
                assert result["username"] == "testuser"
                assert call_count == 3  # Should have retried
    
    @pytest.mark.asyncio
    async def test_multiple_service_rate_limits(self):
        """Test handling rate limits across multiple services."""
        service = ExternalAPIIntegrationService(enable_caching=False)
        
        request = ProfileExtractionRequest(
            github_username="testuser",
            leetcode_username="testuser",
            enable_graceful_degradation=True
        )
        
        # Mock both services being rate limited
        with patch.object(service, '_extract_github_profile_safe') as mock_github, \
             patch.object(service, '_extract_leetcode_profile_safe') as mock_leetcode:
            
            mock_github.side_effect = APIError("GitHub rate limit exceeded")
            mock_leetcode.side_effect = APIError("LeetCode rate limit exceeded")
            
            result = await service.extract_comprehensive_profile(request)
            
            # Should handle gracefully
            assert "github" in result.errors
            assert "leetcode" in result.errors
            assert result.success is True  # Due to graceful degradation


if __name__ == "__main__":
    pytest.main([__file__, "-v"])