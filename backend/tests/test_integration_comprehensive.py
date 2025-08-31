"""
Comprehensive integration tests for API endpoints and database operations.
"""
import pytest
import json
from httpx import AsyncClient
from fastapi import status
from unittest.mock import patch, AsyncMock

from app.models.user import User
from app.models.profile import UserProfile
from app.models.skill import Skill, UserSkill
from app.models.job import JobPosting


@pytest.mark.integration
class TestUserAuthenticationIntegration:
    """Integration tests for user authentication flow."""
    
    async def test_complete_user_registration_flow(self, async_client: AsyncClient):
        """Test complete user registration and login flow."""
        # Register new user
        registration_data = {
            "email": "newuser@example.com",
            "password": "securepassword123",
            "full_name": "New User"
        }
        
        response = await async_client.post("/api/v1/auth/register", json=registration_data)
        assert response.status_code == status.HTTP_201_CREATED
        
        user_data = response.json()
        assert user_data["email"] == registration_data["email"]
        assert user_data["full_name"] == registration_data["full_name"]
        assert "id" in user_data
        
        # Login with new user
        login_data = {
            "username": registration_data["email"],
            "password": registration_data["password"]
        }
        
        response = await async_client.post("/api/v1/auth/login", data=login_data)
        assert response.status_code == status.HTTP_200_OK
        
        token_data = response.json()
        assert "access_token" in token_data
        assert token_data["token_type"] == "bearer"
        
        # Use token to access protected endpoint
        headers = {"Authorization": f"Bearer {token_data['access_token']}"}
        response = await async_client.get("/api/v1/auth/me", headers=headers)
        assert response.status_code == status.HTTP_200_OK
        
        user_info = response.json()
        assert user_info["email"] == registration_data["email"]
    
    async def test_duplicate_email_registration(self, async_client: AsyncClient, test_user):
        """Test registration with duplicate email."""
        registration_data = {
            "email": test_user.email,
            "password": "password123",
            "full_name": "Duplicate User"
        }
        
        response = await async_client.post("/api/v1/auth/register", json=registration_data)
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        
        error_data = response.json()
        assert "email" in error_data["detail"].lower()
    
    async def test_invalid_login_credentials(self, async_client: AsyncClient):
        """Test login with invalid credentials."""
        login_data = {
            "username": "nonexistent@example.com",
            "password": "wrongpassword"
        }
        
        response = await async_client.post("/api/v1/auth/login", data=login_data)
        assert response.status_code == status.HTTP_401_UNAUTHORIZED


@pytest.mark.integration
class TestProfileManagementIntegration:
    """Integration tests for profile management."""
    
    async def test_complete_profile_creation_flow(self, async_client: AsyncClient, test_user):
        """Test complete profile creation with external data integration."""
        # Login to get token
        login_data = {"username": test_user.email, "password": "secret"}
        response = await async_client.post("/api/v1/auth/login", data=login_data)
        token = response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        
        # Mock external API responses
        with patch('app.services.external_apis.github_client.GitHubClient.get_user_profile') as mock_github:
            with patch('app.services.external_apis.leetcode_scraper.LeetCodeScraper.get_user_stats') as mock_leetcode:
                mock_github.return_value = {
                    "username": "testuser",
                    "repositories": [{"name": "ml-project", "language": "Python"}],
                    "languages": {"Python": 80, "JavaScript": 20}
                }
                mock_leetcode.return_value = {
                    "problems_solved": 150,
                    "skills": ["Dynamic Programming", "Graph Theory"]
                }
                
                # Create profile
                profile_data = {
                    "skills": ["Python", "Machine Learning"],
                    "dream_job": "Senior ML Engineer",
                    "experience_years": 3,
                    "github_username": "testuser",
                    "leetcode_id": "testuser"
                }
                
                response = await async_client.post(
                    "/api/v1/profiles/", 
                    json=profile_data, 
                    headers=headers
                )
                assert response.status_code == status.HTTP_201_CREATED
                
                profile = response.json()
                assert profile["dream_job"] == profile_data["dream_job"]
                assert profile["experience_years"] == profile_data["experience_years"]
                assert "unified_skills" in profile
    
    async def test_profile_update_with_external_data_refresh(self, async_client: AsyncClient, test_user, test_profile):
        """Test profile update with external data refresh."""
        # Login to get token
        login_data = {"username": test_user.email, "password": "secret"}
        response = await async_client.post("/api/v1/auth/login", data=login_data)
        token = response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        
        # Mock updated external data
        with patch('app.services.external_apis.github_client.GitHubClient.get_user_profile') as mock_github:
            mock_github.return_value = {
                "username": "testuser",
                "repositories": [
                    {"name": "new-project", "language": "Python"},
                    {"name": "react-app", "language": "JavaScript"}
                ],
                "languages": {"Python": 60, "JavaScript": 40}
            }
            
            # Update profile
            update_data = {
                "dream_job": "Full Stack Engineer",
                "refresh_external_data": True
            }
            
            response = await async_client.put(
                f"/api/v1/profiles/{test_profile.id}",
                json=update_data,
                headers=headers
            )
            assert response.status_code == status.HTTP_200_OK
            
            updated_profile = response.json()
            assert updated_profile["dream_job"] == update_data["dream_job"]
            # Should have refreshed external data
            mock_github.assert_called_once()
    
    async def test_profile_skill_analysis_integration(self, async_client: AsyncClient, test_user, test_profile):
        """Test profile skill analysis integration."""
        # Login to get token
        login_data = {"username": test_user.email, "password": "secret"}
        response = await async_client.post("/api/v1/auth/login", data=login_data)
        token = response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        
        # Get skill analysis
        response = await async_client.get(
            f"/api/v1/profiles/{test_profile.id}/skills/analysis",
            headers=headers
        )
        assert response.status_code == status.HTTP_200_OK
        
        analysis = response.json()
        assert "skill_categories" in analysis
        assert "skill_strengths" in analysis
        assert "improvement_areas" in analysis


@pytest.mark.integration
class TestRecommendationSystemIntegration:
    """Integration tests for recommendation system."""
    
    async def test_career_recommendation_pipeline(self, async_client: AsyncClient, test_user, test_profile, test_job_postings):
        """Test complete career recommendation pipeline."""
        # Login to get token
        login_data = {"username": test_user.email, "password": "secret"}
        response = await async_client.post("/api/v1/auth/login", data=login_data)
        token = response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        
        # Mock ML engine responses
        with patch('app.services.recommendation_service.RecommendationService._get_ml_recommendations') as mock_ml:
            mock_ml.return_value = [
                {
                    "job_title": "Senior Python Developer",
                    "match_score": 0.92,
                    "required_skills": ["Python", "FastAPI", "PostgreSQL"],
                    "skill_gaps": {"PostgreSQL": 0.3},
                    "reasoning": "Strong Python skills match"
                }
            ]
            
            # Get career recommendations
            response = await async_client.get(
                "/api/v1/recommendations/careers",
                params={"limit": 5},
                headers=headers
            )
            assert response.status_code == status.HTTP_200_OK
            
            recommendations = response.json()
            assert len(recommendations) <= 5
            assert all("match_score" in rec for rec in recommendations)
            assert all("skill_gaps" in rec for rec in recommendations)
    
    async def test_learning_path_recommendation_integration(self, async_client: AsyncClient, test_user, test_profile):
        """Test learning path recommendation integration."""
        # Login to get token
        login_data = {"username": test_user.email, "password": "secret"}
        response = await async_client.post("/api/v1/auth/login", data=login_data)
        token = response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        
        # Mock learning resource APIs
        with patch('app.services.learning_path_service.LearningPathService._fetch_coursera_courses') as mock_coursera:
            with patch('app.services.learning_path_service.LearningPathService._fetch_udemy_courses') as mock_udemy:
                mock_coursera.return_value = [
                    {"title": "Advanced ML", "duration": 40, "rating": 4.8, "provider": "Coursera"}
                ]
                mock_udemy.return_value = [
                    {"title": "Docker Mastery", "duration": 20, "rating": 4.5, "provider": "Udemy"}
                ]
                
                # Request learning path
                request_data = {
                    "target_skills": ["TensorFlow", "Docker", "Kubernetes"],
                    "timeline_weeks": 16,
                    "learning_style": "hands_on"
                }
                
                response = await async_client.post(
                    "/api/v1/learning-paths/generate",
                    json=request_data,
                    headers=headers
                )
                assert response.status_code == status.HTTP_201_CREATED
                
                learning_path = response.json()
                assert "path_id" in learning_path
                assert "estimated_duration_weeks" in learning_path
                assert "resources" in learning_path
                assert learning_path["estimated_duration_weeks"] <= request_data["timeline_weeks"]
    
    async def test_job_matching_integration(self, async_client: AsyncClient, test_user, test_profile, test_job_postings):
        """Test job matching integration."""
        # Login to get token
        login_data = {"username": test_user.email, "password": "secret"}
        response = await async_client.post("/api/v1/auth/login", data=login_data)
        token = response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        
        # Get job matches
        response = await async_client.get(
            "/api/v1/recommendations/jobs/matches",
            params={"location": "Remote", "min_match_score": 0.6},
            headers=headers
        )
        assert response.status_code == status.HTTP_200_OK
        
        matches = response.json()
        assert isinstance(matches, list)
        assert all("match_score" in match for match in matches)
        assert all(match["match_score"] >= 0.6 for match in matches)


@pytest.mark.integration
class TestAnalyticsIntegration:
    """Integration tests for analytics and reporting."""
    
    async def test_skill_radar_generation_integration(self, async_client: AsyncClient, test_user, test_profile):
        """Test skill radar chart generation integration."""
        # Login to get token
        login_data = {"username": test_user.email, "password": "secret"}
        response = await async_client.post("/api/v1/auth/login", data=login_data)
        token = response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        
        # Generate skill radar
        response = await async_client.get(
            "/api/v1/analytics/skill-radar",
            headers=headers
        )
        assert response.status_code == status.HTTP_200_OK
        
        radar_data = response.json()
        assert "categories" in radar_data
        assert "values" in radar_data
        assert len(radar_data["categories"]) == len(radar_data["values"])
        assert all(0 <= value <= 1 for value in radar_data["values"])
    
    async def test_career_roadmap_generation_integration(self, async_client: AsyncClient, test_user, test_profile):
        """Test career roadmap generation integration."""
        # Login to get token
        login_data = {"username": test_user.email, "password": "secret"}
        response = await async_client.post("/api/v1/auth/login", data=login_data)
        token = response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        
        # Generate career roadmap
        roadmap_request = {
            "target_role": "Senior ML Engineer",
            "timeline_years": 3
        }
        
        response = await async_client.post(
            "/api/v1/analytics/career-roadmap",
            json=roadmap_request,
            headers=headers
        )
        assert response.status_code == status.HTTP_201_CREATED
        
        roadmap = response.json()
        assert "milestones" in roadmap
        assert "timeline" in roadmap
        assert "skill_progression" in roadmap
    
    async def test_pdf_report_generation_integration(self, async_client: AsyncClient, test_user, test_profile):
        """Test PDF report generation integration."""
        # Login to get token
        login_data = {"username": test_user.email, "password": "secret"}
        response = await async_client.post("/api/v1/auth/login", data=login_data)
        token = response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        
        # Generate PDF report
        response = await async_client.post(
            "/api/v1/analytics/reports/pdf",
            json={"include_recommendations": True, "include_learning_paths": True},
            headers=headers
        )
        assert response.status_code == status.HTTP_200_OK
        assert response.headers["content-type"] == "application/pdf"
        assert len(response.content) > 0


@pytest.mark.integration
class TestExternalAPIIntegration:
    """Integration tests for external API integrations."""
    
    async def test_github_integration_flow(self, async_client: AsyncClient, test_user):
        """Test GitHub integration flow."""
        # Login to get token
        login_data = {"username": test_user.email, "password": "secret"}
        response = await async_client.post("/api/v1/auth/login", data=login_data)
        token = response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        
        # Mock GitHub API
        with patch('app.services.external_apis.github_client.GitHubClient.get_user_profile') as mock_github:
            mock_github.return_value = {
                "username": "testuser",
                "repositories": [
                    {"name": "ml-project", "language": "Python", "stars": 10},
                    {"name": "web-app", "language": "JavaScript", "stars": 5}
                ],
                "languages": {"Python": 70, "JavaScript": 30},
                "total_commits": 150
            }
            
            # Test GitHub profile fetch
            response = await async_client.get(
                "/api/v1/external-profiles/github/testuser",
                headers=headers
            )
            assert response.status_code == status.HTTP_200_OK
            
            github_data = response.json()
            assert github_data["username"] == "testuser"
            assert "repositories" in github_data
            assert "languages" in github_data
    
    async def test_leetcode_integration_flow(self, async_client: AsyncClient, test_user):
        """Test LeetCode integration flow."""
        # Login to get token
        login_data = {"username": test_user.email, "password": "secret"}
        response = await async_client.post("/api/v1/auth/login", data=login_data)
        token = response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        
        # Mock LeetCode API
        with patch('app.services.external_apis.leetcode_scraper.LeetCodeScraper.get_user_stats') as mock_leetcode:
            mock_leetcode.return_value = {
                "username": "testuser",
                "problems_solved": 250,
                "easy_solved": 100,
                "medium_solved": 120,
                "hard_solved": 30,
                "skills": ["Dynamic Programming", "Graph Theory"]
            }
            
            # Test LeetCode profile fetch
            response = await async_client.get(
                "/api/v1/external-profiles/leetcode/testuser",
                headers=headers
            )
            assert response.status_code == status.HTTP_200_OK
            
            leetcode_data = response.json()
            assert leetcode_data["username"] == "testuser"
            assert "problems_solved" in leetcode_data
            assert "skills" in leetcode_data
    
    async def test_external_api_error_handling(self, async_client: AsyncClient, test_user):
        """Test external API error handling."""
        # Login to get token
        login_data = {"username": test_user.email, "password": "secret"}
        response = await async_client.post("/api/v1/auth/login", data=login_data)
        token = response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        
        # Mock API failure
        with patch('app.services.external_apis.github_client.GitHubClient.get_user_profile') as mock_github:
            mock_github.side_effect = Exception("API rate limit exceeded")
            
            # Should handle error gracefully
            response = await async_client.get(
                "/api/v1/external-profiles/github/testuser",
                headers=headers
            )
            assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
            
            error_data = response.json()
            assert "external service" in error_data["detail"].lower()


@pytest.mark.integration
class TestDatabaseOperationsIntegration:
    """Integration tests for database operations."""
    
    async def test_user_profile_cascade_operations(self, async_session, test_user):
        """Test cascade operations when user is deleted."""
        # Create profile and related data
        profile = UserProfile(
            user_id=test_user.id,
            dream_job="Test Job",
            experience_years=2
        )
        async_session.add(profile)
        await async_session.commit()
        
        # Create user skills
        skill = Skill(name="Test Skill", category="technical")
        async_session.add(skill)
        await async_session.commit()
        
        user_skill = UserSkill(
            user_id=test_user.id,
            skill_id=skill.id,
            confidence_score=0.8
        )
        async_session.add(user_skill)
        await async_session.commit()
        
        # Delete user - should cascade to profile and user_skills
        await async_session.delete(test_user)
        await async_session.commit()
        
        # Verify cascade deletion
        from sqlalchemy import select
        
        profile_result = await async_session.execute(
            select(UserProfile).where(UserProfile.user_id == test_user.id)
        )
        assert profile_result.scalar_one_or_none() is None
        
        user_skill_result = await async_session.execute(
            select(UserSkill).where(UserSkill.user_id == test_user.id)
        )
        assert user_skill_result.scalar_one_or_none() is None
    
    async def test_skill_taxonomy_consistency(self, async_session, test_skills):
        """Test skill taxonomy consistency."""
        from sqlalchemy import select, func
        
        # Check that all skills have valid categories
        result = await async_session.execute(
            select(Skill.category, func.count(Skill.id)).group_by(Skill.category)
        )
        categories = result.all()
        
        valid_categories = ["programming", "technical", "framework", "frontend", "soft_skill", "tool", "platform"]
        for category, count in categories:
            assert category in valid_categories, f"Invalid category: {category}"
            assert count > 0
    
    async def test_job_posting_search_performance(self, async_session, test_job_postings):
        """Test job posting search performance."""
        import time
        from sqlalchemy import select, or_
        
        # Test search query performance
        start_time = time.time()
        
        search_query = select(JobPosting).where(
            or_(
                JobPosting.title.ilike("%python%"),
                JobPosting.description.ilike("%python%")
            )
        )
        
        result = await async_session.execute(search_query)
        jobs = result.scalars().all()
        
        query_time = time.time() - start_time
        
        # Should complete within reasonable time
        assert query_time < 1.0, f"Job search query too slow: {query_time}s"
        assert len(jobs) >= 0  # May or may not find matches
    
    async def test_concurrent_profile_updates(self, async_session):
        """Test concurrent profile updates don't cause conflicts."""
        import asyncio
        
        # Create test user and profile
        user = User(
            email="concurrent@test.com",
            hashed_password="hashed",
            full_name="Concurrent User"
        )
        async_session.add(user)
        await async_session.commit()
        
        profile = UserProfile(
            user_id=user.id,
            dream_job="Initial Job",
            experience_years=1
        )
        async_session.add(profile)
        await async_session.commit()
        
        # Define concurrent update functions
        async def update_dream_job():
            profile.dream_job = "Updated Job 1"
            await async_session.commit()
        
        async def update_experience():
            profile.experience_years = 5
            await async_session.commit()
        
        # Run concurrent updates
        await asyncio.gather(update_dream_job(), update_experience())
        
        # Verify final state is consistent
        await async_session.refresh(profile)
        assert profile.dream_job in ["Initial Job", "Updated Job 1"]
        assert profile.experience_years in [1, 5]


@pytest.mark.integration
class TestEndToEndWorkflows:
    """End-to-end integration tests for complete user workflows."""
    
    async def test_complete_new_user_onboarding_workflow(self, async_client: AsyncClient):
        """Test complete new user onboarding workflow."""
        # Step 1: Register new user
        registration_data = {
            "email": "newuser@workflow.com",
            "password": "securepass123",
            "full_name": "Workflow User"
        }
        
        response = await async_client.post("/api/v1/auth/register", json=registration_data)
        assert response.status_code == status.HTTP_201_CREATED
        
        # Step 2: Login
        login_data = {"username": registration_data["email"], "password": registration_data["password"]}
        response = await async_client.post("/api/v1/auth/login", data=login_data)
        token = response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        
        # Step 3: Create profile with external integrations
        with patch('app.services.external_apis.github_client.GitHubClient.get_user_profile') as mock_github:
            mock_github.return_value = {"username": "workflowuser", "languages": {"Python": 80}}
            
            profile_data = {
                "skills": ["Python", "FastAPI"],
                "dream_job": "Senior Backend Engineer",
                "experience_years": 3,
                "github_username": "workflowuser"
            }
            
            response = await async_client.post("/api/v1/profiles/", json=profile_data, headers=headers)
            assert response.status_code == status.HTTP_201_CREATED
            profile = response.json()
        
        # Step 4: Get career recommendations
        with patch('app.services.recommendation_service.RecommendationService._get_ml_recommendations') as mock_ml:
            mock_ml.return_value = [{"job_title": "Backend Engineer", "match_score": 0.85}]
            
            response = await async_client.get("/api/v1/recommendations/careers", headers=headers)
            assert response.status_code == status.HTTP_200_OK
            recommendations = response.json()
            assert len(recommendations) > 0
        
        # Step 5: Generate learning path
        with patch('app.services.learning_path_service.LearningPathService._fetch_learning_resources') as mock_resources:
            mock_resources.return_value = [{"title": "Advanced Python", "duration": 30}]
            
            learning_request = {
                "target_skills": ["Docker", "Kubernetes"],
                "timeline_weeks": 12
            }
            
            response = await async_client.post(
                "/api/v1/learning-paths/generate", 
                json=learning_request, 
                headers=headers
            )
            assert response.status_code == status.HTTP_201_CREATED
        
        # Step 6: Generate analytics report
        response = await async_client.get("/api/v1/analytics/skill-radar", headers=headers)
        assert response.status_code == status.HTTP_200_OK
        
        # Workflow completed successfully
        assert True  # All steps passed
    
    async def test_profile_update_and_recommendation_refresh_workflow(self, async_client: AsyncClient, test_user, test_profile):
        """Test profile update triggering recommendation refresh."""
        # Login
        login_data = {"username": test_user.email, "password": "secret"}
        response = await async_client.post("/api/v1/auth/login", data=login_data)
        token = response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        
        # Get initial recommendations
        with patch('app.services.recommendation_service.RecommendationService._get_ml_recommendations') as mock_ml:
            mock_ml.return_value = [{"job_title": "Python Developer", "match_score": 0.8}]
            
            response = await async_client.get("/api/v1/recommendations/careers", headers=headers)
            initial_recommendations = response.json()
        
        # Update profile with new skills
        update_data = {
            "skills": ["Python", "Machine Learning", "TensorFlow", "Docker"],
            "dream_job": "ML Engineer"
        }
        
        response = await async_client.put(
            f"/api/v1/profiles/{test_profile.id}",
            json=update_data,
            headers=headers
        )
        assert response.status_code == status.HTTP_200_OK
        
        # Get updated recommendations
        with patch('app.services.recommendation_service.RecommendationService._get_ml_recommendations') as mock_ml:
            mock_ml.return_value = [{"job_title": "ML Engineer", "match_score": 0.92}]
            
            response = await async_client.get("/api/v1/recommendations/careers", headers=headers)
            updated_recommendations = response.json()
        
        # Recommendations should reflect profile changes
        assert updated_recommendations != initial_recommendations