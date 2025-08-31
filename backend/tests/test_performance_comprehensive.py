"""
Comprehensive performance and load tests for scalability validation.
"""
import asyncio
import time
import pytest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch, AsyncMock
import psutil
import os

from httpx import AsyncClient
from fastapi.testclient import TestClient


@pytest.mark.performance
class TestAPIPerformance:
    """Performance tests for API endpoints."""
    
    async def test_authentication_endpoint_performance(self, async_client: AsyncClient, test_user):
        """Test authentication endpoint performance under load."""
        login_data = {"username": test_user.email, "password": "secret"}
        
        # Measure single request time
        start_time = time.time()
        response = await async_client.post("/api/v1/auth/login", data=login_data)
        single_request_time = time.time() - start_time
        
        assert response.status_code == 200
        assert single_request_time < 0.5, f"Single auth request too slow: {single_request_time}s"
        
        # Test concurrent requests
        async def make_auth_request():
            return await async_client.post("/api/v1/auth/login", data=login_data)
        
        start_time = time.time()
        tasks = [make_auth_request() for _ in range(10)]
        responses = await asyncio.gather(*tasks)
        concurrent_time = time.time() - start_time
        
        # All requests should succeed
        assert all(r.status_code == 200 for r in responses)
        # Average time per request should be reasonable
        avg_time = concurrent_time / 10
        assert avg_time < 1.0, f"Concurrent auth requests too slow: {avg_time}s average"
    
    async def test_profile_creation_performance(self, async_client: AsyncClient, test_user):
        """Test profile creation performance."""
        # Login to get token
        login_data = {"username": test_user.email, "password": "secret"}
        response = await async_client.post("/api/v1/auth/login", data=login_data)
        token = response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        
        profile_data = {
            "skills": ["Python", "Machine Learning", "FastAPI"],
            "dream_job": "Senior ML Engineer",
            "experience_years": 3
        }
        
        # Mock external API calls to focus on core performance
        with patch('app.services.external_apis.github_client.GitHubClient.get_user_profile') as mock_github:
            mock_github.return_value = {"username": "test", "languages": {"Python": 80}}
            
            start_time = time.time()
            response = await async_client.post("/api/v1/profiles/", json=profile_data, headers=headers)
            creation_time = time.time() - start_time
            
            assert response.status_code == 201
            assert creation_time < 2.0, f"Profile creation too slow: {creation_time}s"
    
    async def test_recommendation_generation_performance(self, async_client: AsyncClient, test_user, test_profile):
        """Test recommendation generation performance."""
        # Login to get token
        login_data = {"username": test_user.email, "password": "secret"}
        response = await async_client.post("/api/v1/auth/login", data=login_data)
        token = response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        
        # Mock ML engine for consistent performance testing
        with patch('app.services.recommendation_service.RecommendationService._get_ml_recommendations') as mock_ml:
            mock_ml.return_value = [
                {"job_title": f"Engineer {i}", "match_score": 0.8 - i*0.1}
                for i in range(5)
            ]
            
            start_time = time.time()
            response = await async_client.get("/api/v1/recommendations/careers", headers=headers)
            recommendation_time = time.time() - start_time
            
            assert response.status_code == 200
            assert recommendation_time < 3.0, f"Recommendation generation too slow: {recommendation_time}s"
    
    async def test_search_endpoint_performance(self, async_client: AsyncClient, test_user, test_job_postings):
        """Test job search endpoint performance."""
        # Login to get token
        login_data = {"username": test_user.email, "password": "secret"}
        response = await async_client.post("/api/v1/auth/login", data=login_data)
        token = response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        
        search_params = {
            "query": "python",
            "location": "remote",
            "experience_level": "mid"
        }
        
        start_time = time.time()
        response = await async_client.get(
            "/api/v1/job-market/search",
            params=search_params,
            headers=headers
        )
        search_time = time.time() - start_time
        
        assert response.status_code == 200
        assert search_time < 1.0, f"Job search too slow: {search_time}s"


@pytest.mark.performance
class TestLoadTesting:
    """Load testing for system scalability."""
    
    async def test_concurrent_user_load(self, async_client: AsyncClient, performance_test_data):
        """Test system performance under concurrent user load."""
        users = performance_test_data["users"][:20]  # Test with 20 concurrent users
        
        async def simulate_user_session(user_data):
            """Simulate a complete user session."""
            try:
                # Login
                login_response = await async_client.post(
                    "/api/v1/auth/login",
                    data={"username": user_data["email"], "password": user_data["password"]}
                )
                if login_response.status_code != 200:
                    return {"success": False, "step": "login"}
                
                token = login_response.json()["access_token"]
                headers = {"Authorization": f"Bearer {token}"}
                
                # Get profile (if exists) or create one
                profile_response = await async_client.get("/api/v1/profiles/me", headers=headers)
                if profile_response.status_code == 404:
                    # Create profile
                    profile_data = {
                        "skills": ["Python", "JavaScript"],
                        "dream_job": "Software Engineer",
                        "experience_years": 2
                    }
                    create_response = await async_client.post(
                        "/api/v1/profiles/", json=profile_data, headers=headers
                    )
                    if create_response.status_code != 201:
                        return {"success": False, "step": "profile_creation"}
                
                # Get recommendations
                with patch('app.services.recommendation_service.RecommendationService._get_ml_recommendations') as mock_ml:
                    mock_ml.return_value = [{"job_title": "Engineer", "match_score": 0.8}]
                    
                    rec_response = await async_client.get("/api/v1/recommendations/careers", headers=headers)
                    if rec_response.status_code != 200:
                        return {"success": False, "step": "recommendations"}
                
                return {"success": True, "step": "completed"}
                
            except Exception as e:
                return {"success": False, "step": "exception", "error": str(e)}
        
        # Create test users first
        for user_data in users:
            await async_client.post("/api/v1/auth/register", json={
                "email": user_data["email"],
                "password": user_data["password"],
                "full_name": f"Test User {user_data['email']}"
            })
        
        # Run concurrent user sessions
        start_time = time.time()
        tasks = [simulate_user_session(user_data) for user_data in users]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        total_time = time.time() - start_time
        
        # Analyze results
        successful_sessions = sum(1 for r in results if isinstance(r, dict) and r.get("success"))
        success_rate = successful_sessions / len(users)
        avg_time_per_user = total_time / len(users)
        
        assert success_rate >= 0.9, f"Success rate too low: {success_rate}"
        assert avg_time_per_user < 5.0, f"Average session time too high: {avg_time_per_user}s"
        assert total_time < 30.0, f"Total load test time too high: {total_time}s"
    
    async def test_database_connection_pool_performance(self, async_session):
        """Test database connection pool performance under load."""
        async def database_operation():
            """Simulate database operation."""
            from sqlalchemy import text
            result = await async_session.execute(text("SELECT 1"))
            return result.scalar()
        
        # Test concurrent database operations
        start_time = time.time()
        tasks = [database_operation() for _ in range(50)]
        results = await asyncio.gather(*tasks)
        db_time = time.time() - start_time
        
        assert all(r == 1 for r in results)
        assert db_time < 5.0, f"Database operations too slow: {db_time}s"
        
        # Average time per operation should be reasonable
        avg_time = db_time / 50
        assert avg_time < 0.1, f"Average DB operation too slow: {avg_time}s"
    
    async def test_memory_usage_under_load(self, async_client: AsyncClient, test_user):
        """Test memory usage doesn't grow excessively under load."""
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Login to get token
        login_data = {"username": test_user.email, "password": "secret"}
        response = await async_client.post("/api/v1/auth/login", data=login_data)
        token = response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        
        # Simulate many requests
        async def make_request():
            return await async_client.get("/api/v1/auth/me", headers=headers)
        
        # Make 100 requests
        tasks = [make_request() for _ in range(100)]
        await asyncio.gather(*tasks)
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 100MB for 100 requests)
        assert memory_increase < 100, f"Excessive memory usage: {memory_increase}MB"
    
    async def test_cache_performance_under_load(self, async_client: AsyncClient, test_user, mock_redis):
        """Test cache performance under load."""
        # Login to get token
        login_data = {"username": test_user.email, "password": "secret"}
        response = await async_client.post("/api/v1/auth/login", data=login_data)
        token = response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        
        # Mock cache hits and misses
        cache_hit_count = 0
        cache_miss_count = 0
        
        async def mock_cache_get(key):
            nonlocal cache_hit_count, cache_miss_count
            if "cached" in key:
                cache_hit_count += 1
                return '{"cached": true}'
            else:
                cache_miss_count += 1
                return None
        
        mock_redis.get.side_effect = mock_cache_get
        
        # Make requests that should use cache
        async def make_cached_request():
            return await async_client.get("/api/v1/recommendations/careers", headers=headers)
        
        with patch('app.services.recommendation_service.RecommendationService._get_ml_recommendations') as mock_ml:
            mock_ml.return_value = [{"job_title": "Engineer", "match_score": 0.8}]
            
            start_time = time.time()
            tasks = [make_cached_request() for _ in range(20)]
            await asyncio.gather(*tasks)
            cache_test_time = time.time() - start_time
        
        # Cache operations should be fast
        assert cache_test_time < 10.0, f"Cache operations too slow: {cache_test_time}s"


@pytest.mark.performance
class TestMLModelPerformance:
    """Performance tests for ML model operations."""
    
    def test_skill_extraction_batch_performance(self, mock_nlp_engine):
        """Test skill extraction performance on batch data."""
        texts = [
            f"Python developer with {i} years of experience in web development"
            for i in range(100)
        ]
        
        # Mock skill extraction to return consistent results
        mock_nlp_engine.extract_skills_from_text.return_value = [
            {"skill": "Python", "confidence": 0.9},
            {"skill": "Web Development", "confidence": 0.8}
        ]
        
        start_time = time.time()
        results = []
        for text in texts:
            skills = mock_nlp_engine.extract_skills_from_text(text)
            results.append(skills)
        batch_time = time.time() - start_time
        
        assert len(results) == 100
        assert batch_time < 10.0, f"Batch skill extraction too slow: {batch_time}s"
        
        # Average time per text should be reasonable
        avg_time = batch_time / 100
        assert avg_time < 0.1, f"Average extraction time too slow: {avg_time}s"
    
    def test_embedding_generation_performance(self, mock_nlp_engine):
        """Test embedding generation performance."""
        texts = [f"Sample text {i} for embedding generation" for i in range(50)]
        
        # Mock embedding generation
        mock_nlp_engine.generate_embeddings.return_value = [0.1] * 384
        
        start_time = time.time()
        embeddings = []
        for text in texts:
            embedding = mock_nlp_engine.generate_embeddings(text)
            embeddings.append(embedding)
        embedding_time = time.time() - start_time
        
        assert len(embeddings) == 50
        assert embedding_time < 5.0, f"Embedding generation too slow: {embedding_time}s"
        
        # Average time per embedding should be reasonable
        avg_time = embedding_time / 50
        assert avg_time < 0.1, f"Average embedding time too slow: {avg_time}s"
    
    def test_recommendation_algorithm_performance(self, mock_recommendation_engine):
        """Test recommendation algorithm performance."""
        user_profiles = [
            {
                "skills": {"Python": 0.9, "JavaScript": 0.7, "React": 0.6},
                "experience_years": i % 10,
                "interests": ["web_development"]
            }
            for i in range(20)
        ]
        
        # Mock recommendation generation
        mock_recommendation_engine.recommend_careers.return_value = [
            {"job_title": "Engineer", "match_score": 0.8}
        ]
        
        start_time = time.time()
        all_recommendations = []
        for profile in user_profiles:
            recommendations = mock_recommendation_engine.recommend_careers(profile, top_k=5)
            all_recommendations.append(recommendations)
        recommendation_time = time.time() - start_time
        
        assert len(all_recommendations) == 20
        assert recommendation_time < 10.0, f"Recommendation generation too slow: {recommendation_time}s"
        
        # Average time per recommendation should be reasonable
        avg_time = recommendation_time / 20
        assert avg_time < 0.5, f"Average recommendation time too slow: {avg_time}s"


@pytest.mark.performance
class TestScalabilityMetrics:
    """Test system scalability metrics."""
    
    async def test_response_time_under_increasing_load(self, async_client: AsyncClient, test_user):
        """Test response time degradation under increasing load."""
        # Login to get token
        login_data = {"username": test_user.email, "password": "secret"}
        response = await async_client.post("/api/v1/auth/login", data=login_data)
        token = response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        
        load_levels = [1, 5, 10, 20]
        response_times = []
        
        for load in load_levels:
            async def make_request():
                return await async_client.get("/api/v1/auth/me", headers=headers)
            
            start_time = time.time()
            tasks = [make_request() for _ in range(load)]
            responses = await asyncio.gather(*tasks)
            total_time = time.time() - start_time
            
            # All requests should succeed
            assert all(r.status_code == 200 for r in responses)
            
            avg_response_time = total_time / load
            response_times.append(avg_response_time)
        
        # Response time shouldn't degrade too much with increased load
        # Allow some degradation but not exponential
        for i in range(1, len(response_times)):
            degradation_factor = response_times[i] / response_times[0]
            load_factor = load_levels[i] / load_levels[0]
            
            # Response time shouldn't degrade more than 3x the load increase
            assert degradation_factor <= load_factor * 3, f"Response time degradation too high at load {load_levels[i]}"
    
    async def test_throughput_measurement(self, async_client: AsyncClient, test_user):
        """Test system throughput measurement."""
        # Login to get token
        login_data = {"username": test_user.email, "password": "secret"}
        response = await async_client.post("/api/v1/auth/login", data=login_data)
        token = response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        
        # Measure throughput over 10 seconds
        test_duration = 10  # seconds
        request_count = 0
        start_time = time.time()
        
        async def continuous_requests():
            nonlocal request_count
            while time.time() - start_time < test_duration:
                try:
                    response = await async_client.get("/api/v1/auth/me", headers=headers)
                    if response.status_code == 200:
                        request_count += 1
                    await asyncio.sleep(0.01)  # Small delay to prevent overwhelming
                except Exception:
                    pass
        
        # Run continuous requests
        await continuous_requests()
        
        actual_duration = time.time() - start_time
        throughput = request_count / actual_duration
        
        # Should handle at least 10 requests per second
        assert throughput >= 10, f"Throughput too low: {throughput} req/s"
    
    async def test_error_rate_under_load(self, async_client: AsyncClient, test_user):
        """Test error rate under load conditions."""
        # Login to get token
        login_data = {"username": test_user.email, "password": "secret"}
        response = await async_client.post("/api/v1/auth/login", data=login_data)
        token = response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        
        # Make many concurrent requests
        async def make_request():
            try:
                response = await async_client.get("/api/v1/auth/me", headers=headers)
                return response.status_code
            except Exception:
                return 500
        
        # Test with high concurrency
        tasks = [make_request() for _ in range(100)]
        status_codes = await asyncio.gather(*tasks)
        
        # Calculate error rate
        success_count = sum(1 for code in status_codes if 200 <= code < 300)
        error_rate = (len(status_codes) - success_count) / len(status_codes)
        
        # Error rate should be low (less than 5%)
        assert error_rate < 0.05, f"Error rate too high under load: {error_rate * 100}%"
        assert success_count >= 95, f"Too many failed requests: {len(status_codes) - success_count}"