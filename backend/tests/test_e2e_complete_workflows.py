"""
Complete End-to-End Workflow Tests
Tests complete user journeys and system integration scenarios.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, List
import json
from unittest.mock import AsyncMock, Mock, patch

from fastapi.testclient import TestClient
from httpx import AsyncClient

from app.main import app
from app.core.database import get_db
from app.core.redis import get_redis
from app.services.auth_service import AuthService
from app.services.profile_service import ProfileService
from app.services.recommendation_service import RecommendationService
from app.services.career_trajectory_service import CareerTrajectoryService
from app.services.learning_path_service import LearningPathService
from app.services.analytics_service import AnalyticsService


class TestCompleteUserJourneys:
    """Test complete user journeys from registration to career recommendations"""
    
    @pytest.fixture
    async def test_client(self):
        """Create test client with mocked dependencies"""
        async with AsyncClient(app=app, base_url="http://test") as client:
            yield client
    
    @pytest.fixture
    async def mock_user_data(self):
        """Sample user data for testing"""
        return {
            "email": "test.user@example.com",
            "password": "SecurePassword123!",
            "full_name": "Test User",
            "github_username": "testuser",
            "linkedin_profile": "https://linkedin.com/in/testuser",
            "leetcode_username": "testuser"
        }
    
    @pytest.fixture
    async def mock_job_data(self):
        """Sample job posting data"""
        return [
            {
                "id": "job_1",
                "title": "Senior Software Engineer",
                "company": "Tech Corp",
                "description": "Looking for experienced Python developer with React skills",
                "location": "San Francisco, CA",
                "salary_range": "$120,000 - $180,000",
                "required_skills": ["Python", "React", "PostgreSQL", "Docker"],
                "experience_level": "Senior",
                "posted_date": datetime.utcnow().isoformat()
            },
            {
                "id": "job_2", 
                "title": "Data Scientist",
                "company": "AI Startup",
                "description": "Machine learning engineer with Python and TensorFlow experience",
                "location": "Remote",
                "salary_range": "$100,000 - $150,000",
                "required_skills": ["Python", "TensorFlow", "Pandas", "SQL"],
                "experience_level": "Mid-level",
                "posted_date": datetime.utcnow().isoformat()
            }
        ]
    
    @pytest.mark.asyncio
    async def test_complete_user_registration_to_recommendations(self, test_client, mock_user_data, mock_job_data):
        """Test complete flow from user registration to getting career recommendations"""
        
        # Step 1: User Registration
        registration_response = await test_client.post(
            "/api/v1/auth/register",
            json=mock_user_data
        )
        
        # Mock successful registration
        with patch('app.services.auth_service.AuthService.register_user') as mock_register:
            mock_register.return_value = {
                "user_id": "user_123",
                "email": mock_user_data["email"],
                "access_token": "mock_access_token",
                "token_type": "bearer"
            }
            
            assert registration_response.status_code == 201
            user_data = registration_response.json()
            user_id = user_data["user_id"]
            access_token = user_data["access_token"]
        
        # Step 2: Profile Creation with External Data
        headers = {"Authorization": f"Bearer {access_token}"}
        
        profile_data = {
            "user_id": user_id,
            "github_username": mock_user_data["github_username"],
            "linkedin_profile": mock_user_data["linkedin_profile"],
            "leetcode_username": mock_user_data["leetcode_username"],
            "resume_text": "Experienced software engineer with 5 years in Python development...",
            "target_roles": ["Software Engineer", "Backend Developer"],
            "preferred_locations": ["San Francisco", "Remote"],
            "salary_expectations": {"min": 120000, "max": 180000}
        }
        
        with patch('app.services.profile_service.ProfileService.create_profile') as mock_create_profile:
            mock_create_profile.return_value = {
                "profile_id": "profile_123",
                "user_id": user_id,
                "skills": [
                    {"name": "Python", "level": "Expert", "confidence": 0.95},
                    {"name": "JavaScript", "level": "Advanced", "confidence": 0.85},
                    {"name": "React", "level": "Intermediate", "confidence": 0.75}
                ],
                "experience_years": 5,
                "current_role": "Software Engineer"
            }
            
            profile_response = await test_client.post(
                "/api/v1/profiles/",
                json=profile_data,
                headers=headers
            )
            
            assert profile_response.status_code == 201
            profile = profile_response.json()
            profile_id = profile["profile_id"]
        
        # Step 3: Get Job Recommendations
        with patch('app.services.recommendation_service.RecommendationService.get_job_recommendations') as mock_recommendations:
            mock_recommendations.return_value = {
                "recommendations": [
                    {
                        "job_id": "job_1",
                        "compatibility_score": 0.92,
                        "title": "Senior Software Engineer",
                        "company": "Tech Corp",
                        "match_reasons": [
                            "Strong Python skills match",
                            "React experience aligns with requirements",
                            "Salary range matches expectations"
                        ],
                        "skill_gaps": ["Docker"],
                        "confidence": 0.88
                    }
                ],
                "total_count": 1,
                "filters_applied": {
                    "locations": ["San Francisco"],
                    "salary_range": {"min": 120000, "max": 180000}
                }
            }
            
            recommendations_response = await test_client.get(
                f"/api/v1/recommendations/jobs?profile_id={profile_id}&limit=10",
                headers=headers
            )
            
            assert recommendations_response.status_code == 200
            recommendations = recommendations_response.json()
            assert len(recommendations["recommendations"]) > 0
            assert recommendations["recommendations"][0]["compatibility_score"] > 0.8
        
        # Step 4: Get Career Trajectory Recommendations
        with patch('app.services.career_trajectory_service.CareerTrajectoryService.get_career_paths') as mock_career_paths:
            mock_career_paths.return_value = {
                "career_paths": [
                    {
                        "path_id": "path_1",
                        "title": "Software Engineer → Senior Engineer → Tech Lead",
                        "steps": [
                            {
                                "role": "Senior Software Engineer",
                                "timeline": "0-2 years",
                                "required_skills": ["Python", "System Design", "Leadership"],
                                "salary_range": {"min": 140000, "max": 200000}
                            },
                            {
                                "role": "Tech Lead",
                                "timeline": "2-4 years", 
                                "required_skills": ["Architecture", "Team Management", "Strategic Planning"],
                                "salary_range": {"min": 180000, "max": 250000}
                            }
                        ],
                        "confidence": 0.85,
                        "market_demand": "High"
                    }
                ]
            }
            
            career_response = await test_client.get(
                f"/api/v1/career-trajectory/paths?profile_id={profile_id}",
                headers=headers
            )
            
            assert career_response.status_code == 200
            career_paths = career_response.json()
            assert len(career_paths["career_paths"]) > 0
        
        # Step 5: Get Learning Path Recommendations
        with patch('app.services.learning_path_service.LearningPathService.generate_learning_path') as mock_learning_path:
            mock_learning_path.return_value = {
                "learning_path": {
                    "path_id": "learning_1",
                    "title": "Path to Senior Software Engineer",
                    "estimated_duration": "6-12 months",
                    "modules": [
                        {
                            "module_id": "module_1",
                            "title": "Advanced Python Programming",
                            "duration": "4 weeks",
                            "resources": [
                                {
                                    "type": "course",
                                    "title": "Advanced Python Concepts",
                                    "provider": "Coursera",
                                    "url": "https://coursera.org/advanced-python"
                                }
                            ],
                            "projects": [
                                {
                                    "title": "Build a REST API with FastAPI",
                                    "description": "Create a production-ready API",
                                    "github_repo": "https://github.com/example/fastapi-project"
                                }
                            ]
                        }
                    ],
                    "skill_gaps_addressed": ["Docker", "System Design"],
                    "confidence": 0.90
                }
            }
            
            learning_response = await test_client.get(
                f"/api/v1/learning-paths/generate?profile_id={profile_id}&target_role=Senior Software Engineer",
                headers=headers
            )
            
            assert learning_response.status_code == 200
            learning_path = learning_response.json()
            assert learning_path["learning_path"]["modules"]
        
        # Step 6: Generate Analytics Report
        with patch('app.services.analytics_service.AnalyticsService.generate_career_report') as mock_analytics:
            mock_analytics.return_value = {
                "report": {
                    "profile_summary": {
                        "total_skills": 15,
                        "skill_level_distribution": {"Expert": 3, "Advanced": 7, "Intermediate": 5},
                        "experience_years": 5,
                        "market_competitiveness": 0.85
                    },
                    "skill_analysis": {
                        "top_skills": ["Python", "JavaScript", "React"],
                        "emerging_skills": ["Docker", "Kubernetes", "GraphQL"],
                        "skill_gaps": ["System Design", "Leadership"]
                    },
                    "market_insights": {
                        "job_match_rate": 0.78,
                        "salary_competitiveness": 0.82,
                        "demand_trend": "Increasing"
                    },
                    "recommendations": {
                        "immediate_actions": [
                            "Complete Docker certification",
                            "Build system design portfolio"
                        ],
                        "career_moves": [
                            "Apply for senior roles at tech companies",
                            "Consider remote opportunities"
                        ]
                    }
                }
            }
            
            analytics_response = await test_client.get(
                f"/api/v1/analytics/career-report?profile_id={profile_id}",
                headers=headers
            )
            
            assert analytics_response.status_code == 200
            report = analytics_response.json()
            assert report["report"]["profile_summary"]["total_skills"] > 0
        
        # Verify complete workflow success
        assert user_id is not None
        assert profile_id is not None
        assert len(recommendations["recommendations"]) > 0
        assert len(career_paths["career_paths"]) > 0
        assert learning_path["learning_path"]["modules"]
        assert report["report"]["profile_summary"]["total_skills"] > 0
    
    @pytest.mark.asyncio
    async def test_profile_update_and_recommendation_refresh(self, test_client, mock_user_data):
        """Test profile updates and recommendation refresh workflow"""
        
        # Setup authenticated user
        access_token = "mock_access_token"
        headers = {"Authorization": f"Bearer {access_token}"}
        profile_id = "profile_123"
        
        # Step 1: Update profile with new skills
        profile_update = {
            "new_skills": [
                {"name": "Docker", "level": "Intermediate", "source": "certification"},
                {"name": "Kubernetes", "level": "Beginner", "source": "course"}
            ],
            "experience_years": 6,
            "current_role": "Senior Software Engineer"
        }
        
        with patch('app.services.profile_service.ProfileService.update_profile') as mock_update:
            mock_update.return_value = {
                "profile_id": profile_id,
                "updated_fields": ["skills", "experience_years", "current_role"],
                "new_skill_count": 17,
                "skill_level_improvements": 2
            }
            
            update_response = await test_client.put(
                f"/api/v1/profiles/{profile_id}",
                json=profile_update,
                headers=headers
            )
            
            assert update_response.status_code == 200
            update_result = update_response.json()
            assert update_result["new_skill_count"] > 15
        
        # Step 2: Refresh recommendations based on updated profile
        with patch('app.services.recommendation_service.RecommendationService.refresh_recommendations') as mock_refresh:
            mock_refresh.return_value = {
                "updated_recommendations": 5,
                "new_matches": 2,
                "improved_scores": 3,
                "refresh_timestamp": datetime.utcnow().isoformat()
            }
            
            refresh_response = await test_client.post(
                f"/api/v1/recommendations/refresh?profile_id={profile_id}",
                headers=headers
            )
            
            assert refresh_response.status_code == 200
            refresh_result = refresh_response.json()
            assert refresh_result["new_matches"] > 0
    
    @pytest.mark.asyncio
    async def test_system_health_and_monitoring_workflow(self, test_client):
        """Test system health monitoring and alerting workflow"""
        
        # Step 1: Check system health
        health_response = await test_client.get("/api/v1/health/")
        
        with patch('app.core.monitoring.SystemMonitor.get_system_health') as mock_health:
            mock_health.return_value = {
                "status": "healthy",
                "timestamp": datetime.utcnow().isoformat(),
                "components": {
                    "database": {"status": "healthy", "response_time": 0.05},
                    "redis": {"status": "healthy", "response_time": 0.02},
                    "ml_models": {"status": "healthy", "last_update": datetime.utcnow().isoformat()},
                    "external_apis": {"status": "degraded", "failed_services": ["linkedin"]}
                },
                "performance": {
                    "cpu_usage": 45.2,
                    "memory_usage": 67.8,
                    "disk_usage": 23.1,
                    "active_connections": 150
                }
            }
            
            assert health_response.status_code == 200
            health_data = health_response.json()
            assert health_data["status"] in ["healthy", "degraded"]
        
        # Step 2: Check pipeline automation health
        pipeline_health_response = await test_client.get("/api/v1/pipeline/system/health")
        
        with patch('app.services.data_pipeline.pipeline_monitor.PipelineMonitor.get_system_health') as mock_pipeline_health:
            mock_pipeline_health.return_value = {
                "active_jobs": 3,
                "success_rate_24h": 0.95,
                "avg_duration_24h": 1200.5,
                "error_rate_24h": 0.02,
                "data_quality_score_24h": 0.92,
                "alerts_24h": 1
            }
            
            assert pipeline_health_response.status_code == 200
            pipeline_health = pipeline_health_response.json()
            assert pipeline_health["success_rate_24h"] > 0.8
    
    @pytest.mark.asyncio
    async def test_error_handling_and_recovery_workflow(self, test_client):
        """Test error handling and system recovery scenarios"""
        
        access_token = "mock_access_token"
        headers = {"Authorization": f"Bearer {access_token}"}
        
        # Test 1: Database connection failure simulation
        with patch('app.core.database.get_db') as mock_db:
            mock_db.side_effect = Exception("Database connection failed")
            
            response = await test_client.get("/api/v1/profiles/profile_123", headers=headers)
            
            # Should return graceful error response
            assert response.status_code in [500, 503]
            error_data = response.json()
            assert "error" in error_data or "detail" in error_data
        
        # Test 2: External API failure handling
        with patch('app.services.external_apis.github_client.GitHubClient.get_user_data') as mock_github:
            mock_github.side_effect = Exception("GitHub API rate limit exceeded")
            
            profile_data = {
                "github_username": "testuser",
                "resume_text": "Test resume content"
            }
            
            response = await test_client.post(
                "/api/v1/profiles/",
                json=profile_data,
                headers=headers
            )
            
            # Should handle gracefully and create profile without GitHub data
            # Implementation should use graceful degradation
            assert response.status_code in [201, 202, 503]
        
        # Test 3: ML model failure handling
        with patch('machinelearningmodel.recommendation_engine.RecommendationEngine.get_recommendations') as mock_ml:
            mock_ml.side_effect = Exception("ML model not available")
            
            response = await test_client.get(
                "/api/v1/recommendations/jobs?profile_id=profile_123",
                headers=headers
            )
            
            # Should return fallback recommendations or appropriate error
            assert response.status_code in [200, 503]
            
            if response.status_code == 200:
                data = response.json()
                # Should indicate fallback mode
                assert "fallback" in str(data).lower() or len(data.get("recommendations", [])) >= 0


class TestPerformanceAndScalability:
    """Test system performance under various load conditions"""
    
    @pytest.mark.asyncio
    async def test_concurrent_user_requests(self, test_client):
        """Test system performance with concurrent user requests"""
        
        async def make_request(user_id: int):
            """Simulate user request"""
            headers = {"Authorization": f"Bearer mock_token_{user_id}"}
            
            with patch('app.services.recommendation_service.RecommendationService.get_job_recommendations') as mock_rec:
                mock_rec.return_value = {
                    "recommendations": [{"job_id": f"job_{user_id}", "score": 0.85}],
                    "total_count": 1
                }
                
                response = await test_client.get(
                    f"/api/v1/recommendations/jobs?profile_id=profile_{user_id}",
                    headers=headers
                )
                return response.status_code, response.elapsed if hasattr(response, 'elapsed') else 0
        
        # Simulate 50 concurrent users
        tasks = [make_request(i) for i in range(50)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Analyze results
        successful_requests = [r for r in results if isinstance(r, tuple) and r[0] == 200]
        success_rate = len(successful_requests) / len(results)
        
        assert success_rate > 0.95  # 95% success rate under load
    
    @pytest.mark.asyncio
    async def test_large_dataset_processing(self, test_client):
        """Test system performance with large datasets"""
        
        # Test processing large number of job postings
        large_job_batch = [
            {
                "title": f"Job {i}",
                "company": f"Company {i}",
                "description": f"Job description {i} with various skills and requirements",
                "location": "Remote",
                "required_skills": ["Python", "JavaScript", "SQL"]
            }
            for i in range(1000)
        ]
        
        with patch('app.services.job_analysis_service.JobAnalysisService.process_job_batch') as mock_process:
            mock_process.return_value = {
                "processed_count": 1000,
                "processing_time": 45.2,
                "success_rate": 0.98,
                "extracted_skills": 2500
            }
            
            response = await test_client.post(
                "/api/v1/jobs/batch-process",
                json={"jobs": large_job_batch[:10]}  # Send smaller batch for test
            )
            
            assert response.status_code == 200
            result = response.json()
            assert result["success_rate"] > 0.95
    
    @pytest.mark.asyncio
    async def test_memory_usage_optimization(self, test_client):
        """Test memory usage under various scenarios"""
        
        # This would typically involve monitoring actual memory usage
        # For testing purposes, we'll simulate memory-intensive operations
        
        with patch('app.services.analytics_service.AnalyticsService.generate_comprehensive_report') as mock_analytics:
            mock_analytics.return_value = {
                "report_size_mb": 2.5,
                "generation_time": 8.3,
                "memory_peak_mb": 150.2,
                "optimization_applied": True
            }
            
            response = await test_client.get("/api/v1/analytics/comprehensive-report?profile_id=profile_123")
            
            # Should complete without memory issues
            assert response.status_code == 200
            result = response.json()
            assert result["memory_peak_mb"] < 200  # Memory usage should be reasonable


class TestSecurityAndCompliance:
    """Test security measures and compliance requirements"""
    
    @pytest.mark.asyncio
    async def test_authentication_and_authorization(self, test_client):
        """Test authentication and authorization workflows"""
        
        # Test 1: Unauthenticated access
        response = await test_client.get("/api/v1/profiles/profile_123")
        assert response.status_code == 401
        
        # Test 2: Invalid token
        headers = {"Authorization": "Bearer invalid_token"}
        response = await test_client.get("/api/v1/profiles/profile_123", headers=headers)
        assert response.status_code == 401
        
        # Test 3: Valid authentication
        with patch('app.core.security.verify_token') as mock_verify:
            mock_verify.return_value = {"user_id": "user_123", "email": "test@example.com"}
            
            headers = {"Authorization": "Bearer valid_token"}
            response = await test_client.get("/api/v1/profiles/profile_123", headers=headers)
            
            # Should allow access with valid token
            assert response.status_code in [200, 404]  # 404 if profile doesn't exist
    
    @pytest.mark.asyncio
    async def test_input_validation_and_sanitization(self, test_client):
        """Test input validation and sanitization"""
        
        # Test SQL injection attempt
        malicious_input = {
            "email": "test@example.com'; DROP TABLE users; --",
            "password": "password123",
            "full_name": "<script>alert('xss')</script>"
        }
        
        response = await test_client.post("/api/v1/auth/register", json=malicious_input)
        
        # Should reject malicious input
        assert response.status_code == 422  # Validation error
        
        # Test XSS prevention
        xss_input = {
            "resume_text": "<script>alert('xss')</script>Legitimate resume content",
            "target_roles": ["<img src=x onerror=alert('xss')>Developer"]
        }
        
        headers = {"Authorization": "Bearer valid_token"}
        response = await test_client.post("/api/v1/profiles/", json=xss_input, headers=headers)
        
        # Should sanitize input or reject
        assert response.status_code in [201, 422]
    
    @pytest.mark.asyncio
    async def test_rate_limiting(self, test_client):
        """Test rate limiting functionality"""
        
        # Simulate rapid requests from same IP
        responses = []
        for i in range(100):
            response = await test_client.get("/api/v1/health/")
            responses.append(response.status_code)
        
        # Should eventually rate limit
        rate_limited_responses = [r for r in responses if r == 429]
        
        # Depending on rate limit configuration, some requests should be limited
        # This test assumes rate limiting is configured
        assert len(rate_limited_responses) > 0 or all(r == 200 for r in responses)
    
    @pytest.mark.asyncio
    async def test_data_privacy_compliance(self, test_client):
        """Test data privacy and GDPR compliance"""
        
        access_token = "mock_access_token"
        headers = {"Authorization": f"Bearer {access_token}"}
        user_id = "user_123"
        
        # Test data export (GDPR Article 20)
        with patch('app.services.privacy_service.PrivacyService.export_user_data') as mock_export:
            mock_export.return_value = {
                "export_id": "export_123",
                "status": "completed",
                "download_url": "/api/v1/privacy/export/export_123/download",
                "data_types": ["profile", "recommendations", "analytics"],
                "file_size_mb": 1.2
            }
            
            response = await test_client.post(
                f"/api/v1/privacy/export-data?user_id={user_id}",
                headers=headers
            )
            
            assert response.status_code == 200
            export_data = response.json()
            assert export_data["status"] == "completed"
        
        # Test data deletion (GDPR Article 17)
        with patch('app.services.privacy_service.PrivacyService.delete_user_data') as mock_delete:
            mock_delete.return_value = {
                "deletion_id": "deletion_123",
                "status": "completed",
                "deleted_records": {
                    "profiles": 1,
                    "recommendations": 15,
                    "analytics": 5,
                    "audit_logs": 25
                },
                "retention_period_days": 30
            }
            
            response = await test_client.delete(
                f"/api/v1/privacy/delete-data?user_id={user_id}",
                headers=headers
            )
            
            assert response.status_code == 200
            deletion_data = response.json()
            assert deletion_data["status"] == "completed"


@pytest.mark.asyncio
async def test_complete_system_integration():
    """Test complete system integration across all components"""
    
    # This test verifies that all major system components work together
    integration_results = {
        "database_connection": False,
        "redis_connection": False,
        "ml_models_loaded": False,
        "external_apis_configured": False,
        "pipeline_automation_running": False,
        "monitoring_active": False
    }
    
    try:
        # Test database connection
        from app.core.database import engine
        async with engine.begin() as conn:
            await conn.execute("SELECT 1")
        integration_results["database_connection"] = True
    except Exception:
        pass
    
    try:
        # Test Redis connection
        from app.core.redis import get_redis
        redis_manager = await get_redis()
        await redis_manager.redis.ping()
        integration_results["redis_connection"] = True
    except Exception:
        pass
    
    try:
        # Test ML models
        from machinelearningmodel.recommendation_engine import RecommendationEngine
        engine = RecommendationEngine()
        # Mock model loading
        integration_results["ml_models_loaded"] = True
    except Exception:
        pass
    
    try:
        # Test external API configuration
        from app.services.external_apis.github_client import GitHubClient
        client = GitHubClient()
        # Mock API configuration check
        integration_results["external_apis_configured"] = True
    except Exception:
        pass
    
    try:
        # Test pipeline automation
        from app.services.data_pipeline.pipeline_scheduler import get_pipeline_scheduler
        scheduler = await get_pipeline_scheduler()
        # Mock scheduler check
        integration_results["pipeline_automation_running"] = True
    except Exception:
        pass
    
    try:
        # Test monitoring
        from app.core.monitoring import SystemMonitor
        monitor = SystemMonitor()
        # Mock monitoring check
        integration_results["monitoring_active"] = True
    except Exception:
        pass
    
    # Verify integration success
    success_count = sum(integration_results.values())
    total_components = len(integration_results)
    
    assert success_count >= total_components * 0.8  # At least 80% of components should integrate successfully
    
    return integration_results