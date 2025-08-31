"""
End-to-end tests for complete user workflows.
"""
import pytest
import tempfile
import os
from httpx import AsyncClient
from fastapi import status
from unittest.mock import patch, AsyncMock


@pytest.mark.e2e
class TestCompleteUserJourney:
    """End-to-end tests for complete user journey from registration to recommendations."""
    
    async def test_complete_user_journey_success_path(self, async_client: AsyncClient):
        """Test complete successful user journey."""
        
        # Step 1: User Registration
        registration_data = {
            "email": "journey@example.com",
            "password": "securepassword123",
            "full_name": "Journey User"
        }
        
        response = await async_client.post("/api/v1/auth/register", json=registration_data)
        assert response.status_code == status.HTTP_201_CREATED
        user_data = response.json()
        user_id = user_data["id"]
        
        # Step 2: User Login
        login_data = {
            "username": registration_data["email"],
            "password": registration_data["password"]
        }
        
        response = await async_client.post("/api/v1/auth/login", data=login_data)
        assert response.status_code == status.HTTP_200_OK
        token_data = response.json()
        access_token = token_data["access_token"]
        headers = {"Authorization": f"Bearer {access_token}"}
        
        # Step 3: Profile Creation with External Data
        with patch('app.services.external_apis.github_client.GitHubClient.get_user_profile') as mock_github:
            with patch('app.services.external_apis.leetcode_scraper.LeetCodeScraper.get_user_stats') as mock_leetcode:
                with patch('app.services.external_apis.linkedin_scraper.LinkedInScraper.get_profile_data') as mock_linkedin:
                    
                    # Mock external API responses
                    mock_github.return_value = {
                        "username": "journeyuser",
                        "repositories": [
                            {"name": "ml-project", "language": "Python", "stars": 15},
                            {"name": "web-app", "language": "JavaScript", "stars": 8}
                        ],
                        "languages": {"Python": 70, "JavaScript": 30},
                        "total_commits": 200
                    }
                    
                    mock_leetcode.return_value = {
                        "username": "journeyuser",
                        "problems_solved": 180,
                        "easy_solved": 80,
                        "medium_solved": 80,
                        "hard_solved": 20,
                        "contest_rating": 1500,
                        "skills": ["Dynamic Programming", "Graph Theory", "Binary Search"]
                    }
                    
                    mock_linkedin.return_value = {
                        "name": "Journey User",
                        "headline": "Software Engineer",
                        "experience": [
                            {
                                "title": "Junior Developer",
                                "company": "Tech Startup",
                                "duration": "2 years",
                                "skills": ["Python", "React", "PostgreSQL"]
                            }
                        ],
                        "skills": ["Python", "JavaScript", "React", "Machine Learning"],
                        "connections": 300
                    }
                    
                    # Create profile with external integrations
                    profile_data = {
                        "skills": ["Python", "JavaScript", "Machine Learning"],
                        "dream_job": "Senior Full Stack Engineer",
                        "experience_years": 2,
                        "github_username": "journeyuser",
                        "leetcode_id": "journeyuser",
                        "linkedin_url": "https://linkedin.com/in/journeyuser"
                    }
                    
                    response = await async_client.post(
                        "/api/v1/profiles/",
                        json=profile_data,
                        headers=headers
                    )
                    assert response.status_code == status.HTTP_201_CREATED
                    profile = response.json()
                    profile_id = profile["id"]
                    
                    # Verify external data was integrated
                    assert "unified_skills" in profile
                    assert len(profile["unified_skills"]) > len(profile_data["skills"])
        
        # Step 4: Get Career Recommendations
        with patch('app.services.recommendation_service.RecommendationService._get_ml_recommendations') as mock_ml:
            mock_ml.return_value = [
                {
                    "job_title": "Senior Full Stack Engineer",
                    "match_score": 0.92,
                    "required_skills": ["Python", "JavaScript", "React", "Node.js"],
                    "skill_gaps": {"Node.js": 0.4},
                    "salary_range": (100000, 150000),
                    "reasoning": "Strong match based on current skills and experience"
                },
                {
                    "job_title": "ML Engineer",
                    "match_score": 0.85,
                    "required_skills": ["Python", "Machine Learning", "TensorFlow"],
                    "skill_gaps": {"TensorFlow": 0.6},
                    "salary_range": (110000, 160000),
                    "reasoning": "Good fit with some ML background"
                }
            ]
            
            response = await async_client.get(
                "/api/v1/recommendations/careers",
                params={"limit": 5},
                headers=headers
            )
            assert response.status_code == status.HTTP_200_OK
            recommendations = response.json()
            
            assert len(recommendations) >= 2
            assert recommendations[0]["job_title"] == "Senior Full Stack Engineer"
            assert recommendations[0]["match_score"] >= 0.9
        
        # Step 5: Generate Learning Path
        with patch('app.services.learning_path_service.LearningPathService._fetch_learning_resources') as mock_resources:
            mock_resources.return_value = [
                {
                    "title": "Node.js Complete Guide",
                    "type": "course",
                    "provider": "Udemy",
                    "duration_hours": 30,
                    "rating": 4.6,
                    "cost": 49.99,
                    "url": "https://udemy.com/nodejs-course"
                },
                {
                    "title": "Advanced React Patterns",
                    "type": "course",
                    "provider": "Coursera",
                    "duration_hours": 25,
                    "rating": 4.8,
                    "cost": 0,
                    "url": "https://coursera.org/react-advanced"
                }
            ]
            
            learning_request = {
                "target_skills": ["Node.js", "Advanced React", "System Design"],
                "timeline_weeks": 16,
                "learning_style": "hands_on",
                "budget_preference": "mixed"
            }
            
            response = await async_client.post(
                "/api/v1/learning-paths/generate",
                json=learning_request,
                headers=headers
            )
            assert response.status_code == status.HTTP_201_CREATED
            learning_path = response.json()
            
            assert "path_id" in learning_path
            assert learning_path["estimated_duration_weeks"] <= 16
            assert len(learning_path["resources"]) >= 2
        
        # Step 6: Get Analytics and Visualizations
        response = await async_client.get(
            "/api/v1/analytics/skill-radar",
            headers=headers
        )
        assert response.status_code == status.HTTP_200_OK
        skill_radar = response.json()
        
        assert "categories" in skill_radar
        assert "values" in skill_radar
        assert len(skill_radar["categories"]) == len(skill_radar["values"])
        
        # Step 7: Generate Career Roadmap
        roadmap_request = {
            "target_role": "Senior Full Stack Engineer",
            "timeline_years": 2
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
        assert len(roadmap["milestones"]) > 0
        
        # Step 8: Generate PDF Report
        response = await async_client.post(
            "/api/v1/analytics/reports/pdf",
            json={
                "include_recommendations": True,
                "include_learning_paths": True,
                "include_skill_analysis": True
            },
            headers=headers
        )
        assert response.status_code == status.HTTP_200_OK
        assert response.headers["content-type"] == "application/pdf"
        assert len(response.content) > 1000  # PDF should have substantial content
        
        # Step 9: Update Profile and Verify Recommendation Refresh
        update_data = {
            "skills": ["Python", "JavaScript", "React", "Node.js", "Machine Learning"],
            "experience_years": 3,
            "dream_job": "Tech Lead"
        }
        
        response = await async_client.put(
            f"/api/v1/profiles/{profile_id}",
            json=update_data,
            headers=headers
        )
        assert response.status_code == status.HTTP_200_OK
        
        # Get updated recommendations
        with patch('app.services.recommendation_service.RecommendationService._get_ml_recommendations') as mock_ml:
            mock_ml.return_value = [
                {
                    "job_title": "Tech Lead",
                    "match_score": 0.95,
                    "required_skills": ["Python", "JavaScript", "Leadership", "System Design"],
                    "skill_gaps": {"Leadership": 0.5, "System Design": 0.4},
                    "salary_range": (130000, 180000),
                    "reasoning": "Excellent technical skills, ready for leadership role"
                }
            ]
            
            response = await async_client.get(
                "/api/v1/recommendations/careers",
                headers=headers
            )
            updated_recommendations = response.json()
            
            assert updated_recommendations[0]["job_title"] == "Tech Lead"
            assert updated_recommendations[0]["match_score"] >= 0.95
        
        # Journey completed successfully
        print("[PASS] Complete user journey test passed!")
    
    async def test_user_journey_with_resume_upload(self, async_client: AsyncClient):
        """Test user journey with resume upload."""
        
        # Step 1: Register and login
        registration_data = {
            "email": "resume@example.com",
            "password": "password123",
            "full_name": "Resume User"
        }
        
        response = await async_client.post("/api/v1/auth/register", json=registration_data)
        assert response.status_code == status.HTTP_201_CREATED
        
        login_data = {"username": registration_data["email"], "password": registration_data["password"]}
        response = await async_client.post("/api/v1/auth/login", data=login_data)
        token = response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        
        # Step 2: Create resume file
        resume_content = """
        John Doe
        Senior Software Engineer
        
        EXPERIENCE:
        - 5 years of Python development
        - Expert in Django and Flask frameworks
        - Machine learning projects using scikit-learn and TensorFlow
        - AWS cloud infrastructure management
        - Team leadership and mentoring
        
        SKILLS:
        Python, JavaScript, React, Django, Flask, TensorFlow, AWS, Docker, Kubernetes
        
        EDUCATION:
        BS Computer Science, University of Technology
        """
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(resume_content)
            f.flush()
            resume_file_path = f.name
        
        try:
            # Step 3: Upload resume and create profile
            with patch('app.services.profile_service.ProfileService._extract_resume_skills') as mock_extract:
                mock_extract.return_value = [
                    "Python", "JavaScript", "React", "Django", "Flask", 
                    "TensorFlow", "AWS", "Docker", "Kubernetes", "Machine Learning"
                ]
                
                with open(resume_file_path, 'rb') as resume_file:
                    files = {"resume_file": ("resume.txt", resume_file, "text/plain")}
                    data = {
                        "dream_job": "Senior ML Engineer",
                        "experience_years": "5"
                    }
                    
                    response = await async_client.post(
                        "/api/v1/profiles/upload-resume",
                        files=files,
                        data=data,
                        headers=headers
                    )
                    assert response.status_code == status.HTTP_201_CREATED
                    profile = response.json()
                    
                    # Verify skills were extracted from resume
                    assert "unified_skills" in profile
                    extracted_skills = list(profile["unified_skills"].keys())
                    assert "Python" in extracted_skills
                    assert "Machine Learning" in extracted_skills
                    assert len(extracted_skills) >= 8
        
        finally:
            # Cleanup
            os.unlink(resume_file_path)
        
        # Step 4: Get recommendations based on resume
        with patch('app.services.recommendation_service.RecommendationService._get_ml_recommendations') as mock_ml:
            mock_ml.return_value = [
                {
                    "job_title": "Senior ML Engineer",
                    "match_score": 0.94,
                    "required_skills": ["Python", "TensorFlow", "Machine Learning", "AWS"],
                    "skill_gaps": {},
                    "salary_range": (140000, 200000),
                    "reasoning": "Perfect match for senior ML role"
                }
            ]
            
            response = await async_client.get("/api/v1/recommendations/careers", headers=headers)
            recommendations = response.json()
            
            assert recommendations[0]["job_title"] == "Senior ML Engineer"
            assert recommendations[0]["match_score"] >= 0.9
    
    async def test_user_journey_error_recovery(self, async_client: AsyncClient):
        """Test user journey with error scenarios and recovery."""
        
        # Step 1: Register and login
        registration_data = {
            "email": "error@example.com",
            "password": "password123",
            "full_name": "Error User"
        }
        
        response = await async_client.post("/api/v1/auth/register", json=registration_data)
        assert response.status_code == status.HTTP_201_CREATED
        
        login_data = {"username": registration_data["email"], "password": registration_data["password"]}
        response = await async_client.post("/api/v1/auth/login", data=login_data)
        token = response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        
        # Step 2: Try to create profile with external API failure
        with patch('app.services.external_apis.github_client.GitHubClient.get_user_profile') as mock_github:
            mock_github.side_effect = Exception("GitHub API rate limit exceeded")
            
            profile_data = {
                "skills": ["Python", "JavaScript"],
                "dream_job": "Software Engineer",
                "experience_years": 2,
                "github_username": "erroruser"
            }
            
            # Profile creation should still succeed with graceful degradation
            response = await async_client.post(
                "/api/v1/profiles/",
                json=profile_data,
                headers=headers
            )
            # Should succeed but with warning about external data
            assert response.status_code in [status.HTTP_201_CREATED, status.HTTP_207_MULTI_STATUS]
            profile = response.json()
            
            # Basic profile should be created even if external data fails
            assert profile["dream_job"] == profile_data["dream_job"]
        
        # Step 3: Get recommendations with ML service failure
        with patch('app.services.recommendation_service.RecommendationService._get_ml_recommendations') as mock_ml:
            mock_ml.side_effect = Exception("ML service temporarily unavailable")
            
            # Should fall back to rule-based recommendations
            response = await async_client.get("/api/v1/recommendations/careers", headers=headers)
            
            # Should either succeed with fallback or return appropriate error
            assert response.status_code in [status.HTTP_200_OK, status.HTTP_503_SERVICE_UNAVAILABLE]
            
            if response.status_code == status.HTTP_200_OK:
                recommendations = response.json()
                assert len(recommendations) >= 1  # Fallback recommendations
            else:
                error_data = response.json()
                assert "temporarily unavailable" in error_data["detail"].lower()
        
        # Step 4: Retry after service recovery
        with patch('app.services.recommendation_service.RecommendationService._get_ml_recommendations') as mock_ml:
            mock_ml.return_value = [
                {
                    "job_title": "Software Engineer",
                    "match_score": 0.8,
                    "required_skills": ["Python", "JavaScript"],
                    "skill_gaps": {},
                    "salary_range": (80000, 120000),
                    "reasoning": "Good match for current skills"
                }
            ]
            
            response = await async_client.get("/api/v1/recommendations/careers", headers=headers)
            assert response.status_code == status.HTTP_200_OK
            recommendations = response.json()
            assert len(recommendations) >= 1


@pytest.mark.e2e
class TestMultiUserScenarios:
    """End-to-end tests for multi-user scenarios."""
    
    async def test_concurrent_user_interactions(self, async_client: AsyncClient):
        """Test concurrent user interactions don't interfere."""
        
        # Create two users
        users_data = [
            {"email": "user1@concurrent.com", "password": "pass1", "full_name": "User One"},
            {"email": "user2@concurrent.com", "password": "pass2", "full_name": "User Two"}
        ]
        
        tokens = []
        
        # Register and login both users
        for user_data in users_data:
            response = await async_client.post("/api/v1/auth/register", json=user_data)
            assert response.status_code == status.HTTP_201_CREATED
            
            login_data = {"username": user_data["email"], "password": user_data["password"]}
            response = await async_client.post("/api/v1/auth/login", data=login_data)
            token = response.json()["access_token"]
            tokens.append(token)
        
        # Create different profiles for each user
        profiles_data = [
            {
                "skills": ["Python", "Django"],
                "dream_job": "Backend Engineer",
                "experience_years": 3
            },
            {
                "skills": ["JavaScript", "React"],
                "dream_job": "Frontend Engineer", 
                "experience_years": 2
            }
        ]
        
        # Create profiles concurrently
        import asyncio
        
        async def create_profile(token, profile_data):
            headers = {"Authorization": f"Bearer {token}"}
            return await async_client.post("/api/v1/profiles/", json=profile_data, headers=headers)
        
        profile_tasks = [
            create_profile(tokens[0], profiles_data[0]),
            create_profile(tokens[1], profiles_data[1])
        ]
        
        profile_responses = await asyncio.gather(*profile_tasks)
        
        # Both profiles should be created successfully
        assert all(r.status_code == status.HTTP_201_CREATED for r in profile_responses)
        
        profiles = [r.json() for r in profile_responses]
        assert profiles[0]["dream_job"] == "Backend Engineer"
        assert profiles[1]["dream_job"] == "Frontend Engineer"
        
        # Get recommendations for both users concurrently
        with patch('app.services.recommendation_service.RecommendationService._get_ml_recommendations') as mock_ml:
            def mock_recommendations(user_profile, **kwargs):
                if "Backend" in user_profile.get("dream_job", ""):
                    return [{"job_title": "Senior Backend Engineer", "match_score": 0.9}]
                else:
                    return [{"job_title": "Senior Frontend Engineer", "match_score": 0.85}]
            
            mock_ml.side_effect = mock_recommendations
            
            async def get_recommendations(token):
                headers = {"Authorization": f"Bearer {token}"}
                return await async_client.get("/api/v1/recommendations/careers", headers=headers)
            
            rec_tasks = [get_recommendations(tokens[0]), get_recommendations(tokens[1])]
            rec_responses = await asyncio.gather(*rec_tasks)
            
            # Both should get appropriate recommendations
            assert all(r.status_code == status.HTTP_200_OK for r in rec_responses)
            
            recommendations = [r.json() for r in rec_responses]
            assert "Backend" in recommendations[0][0]["job_title"]
            assert "Frontend" in recommendations[1][0]["job_title"]
    
    async def test_user_data_isolation(self, async_client: AsyncClient):
        """Test that user data is properly isolated."""
        
        # Create two users with similar profiles
        users_data = [
            {"email": "isolated1@test.com", "password": "pass1", "full_name": "Isolated One"},
            {"email": "isolated2@test.com", "password": "pass2", "full_name": "Isolated Two"}
        ]
        
        tokens = []
        profile_ids = []
        
        for user_data in users_data:
            # Register and login
            response = await async_client.post("/api/v1/auth/register", json=user_data)
            assert response.status_code == status.HTTP_201_CREATED
            
            login_data = {"username": user_data["email"], "password": user_data["password"]}
            response = await async_client.post("/api/v1/auth/login", data=login_data)
            token = response.json()["access_token"]
            tokens.append(token)
            
            # Create profile
            profile_data = {
                "skills": ["Python", "Machine Learning"],
                "dream_job": f"ML Engineer {len(tokens)}",  # Different dream jobs
                "experience_years": len(tokens) + 1
            }
            
            headers = {"Authorization": f"Bearer {token}"}
            response = await async_client.post("/api/v1/profiles/", json=profile_data, headers=headers)
            assert response.status_code == status.HTTP_201_CREATED
            profile = response.json()
            profile_ids.append(profile["id"])
        
        # User 1 should only see their own profile
        headers1 = {"Authorization": f"Bearer {tokens[0]}"}
        response = await async_client.get("/api/v1/profiles/me", headers=headers1)
        assert response.status_code == status.HTTP_200_OK
        user1_profile = response.json()
        assert user1_profile["dream_job"] == "ML Engineer 1"
        
        # User 2 should only see their own profile
        headers2 = {"Authorization": f"Bearer {tokens[1]}"}
        response = await async_client.get("/api/v1/profiles/me", headers=headers2)
        assert response.status_code == status.HTTP_200_OK
        user2_profile = response.json()
        assert user2_profile["dream_job"] == "ML Engineer 2"
        
        # User 1 should not be able to access User 2's profile
        response = await async_client.get(f"/api/v1/profiles/{profile_ids[1]}", headers=headers1)
        assert response.status_code == status.HTTP_403_FORBIDDEN
        
        # User 2 should not be able to access User 1's profile
        response = await async_client.get(f"/api/v1/profiles/{profile_ids[0]}", headers=headers2)
        assert response.status_code == status.HTTP_403_FORBIDDEN


@pytest.mark.e2e
class TestSystemIntegrationWorkflows:
    """End-to-end tests for system integration workflows."""
    
    async def test_data_pipeline_integration(self, async_client: AsyncClient):
        """Test complete data pipeline integration."""
        
        # Step 1: Create user and profile
        registration_data = {
            "email": "pipeline@test.com",
            "password": "password123",
            "full_name": "Pipeline User"
        }
        
        response = await async_client.post("/api/v1/auth/register", json=registration_data)
        assert response.status_code == status.HTTP_201_CREATED
        
        login_data = {"username": registration_data["email"], "password": registration_data["password"]}
        response = await async_client.post("/api/v1/auth/login", data=login_data)
        token = response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        
        # Step 2: Create profile that triggers data collection
        with patch('app.services.external_apis.github_client.GitHubClient.get_user_profile') as mock_github:
            with patch('app.services.job_scrapers.scraper_manager.ScraperManager.scrape_jobs') as mock_scraper:
                
                mock_github.return_value = {
                    "username": "pipelineuser",
                    "repositories": [{"name": "data-project", "language": "Python"}],
                    "languages": {"Python": 100}
                }
                
                mock_scraper.return_value = [
                    {
                        "title": "Data Engineer",
                        "company": "Data Corp",
                        "required_skills": ["Python", "SQL", "Apache Spark"],
                        "salary_range": (90000, 130000)
                    }
                ]
                
                profile_data = {
                    "skills": ["Python", "SQL"],
                    "dream_job": "Data Engineer",
                    "experience_years": 2,
                    "github_username": "pipelineuser"
                }
                
                response = await async_client.post("/api/v1/profiles/", json=profile_data, headers=headers)
                assert response.status_code == status.HTTP_201_CREATED
        
        # Step 3: Trigger market analysis
        response = await async_client.post("/api/v1/job-market/analyze", headers=headers)
        assert response.status_code in [status.HTTP_200_OK, status.HTTP_202_ACCEPTED]
        
        # Step 4: Get market-informed recommendations
        with patch('app.services.recommendation_service.RecommendationService._get_ml_recommendations') as mock_ml:
            mock_ml.return_value = [
                {
                    "job_title": "Data Engineer",
                    "match_score": 0.88,
                    "required_skills": ["Python", "SQL", "Apache Spark"],
                    "skill_gaps": {"Apache Spark": 0.7},
                    "market_demand": "high",
                    "salary_range": (90000, 130000)
                }
            ]
            
            response = await async_client.get("/api/v1/recommendations/careers", headers=headers)
            assert response.status_code == status.HTTP_200_OK
            recommendations = response.json()
            
            assert recommendations[0]["job_title"] == "Data Engineer"
            assert "market_demand" in recommendations[0]
    
    async def test_ml_model_pipeline_integration(self, async_client: AsyncClient):
        """Test ML model pipeline integration."""
        
        # Create user and profile
        registration_data = {
            "email": "mlpipeline@test.com",
            "password": "password123",
            "full_name": "ML Pipeline User"
        }
        
        response = await async_client.post("/api/v1/auth/register", json=registration_data)
        login_data = {"username": registration_data["email"], "password": registration_data["password"]}
        response = await async_client.post("/api/v1/auth/login", data=login_data)
        token = response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        
        # Create profile with skills that will be processed by ML pipeline
        profile_data = {
            "skills": ["Python", "Machine Learning", "Deep Learning"],
            "dream_job": "AI Research Scientist",
            "experience_years": 4
        }
        
        with patch('machinelearningmodel.nlp_engine.NLPEngine.extract_skills_from_text') as mock_nlp:
            with patch('machinelearningmodel.nlp_engine.NLPEngine.generate_embeddings') as mock_embeddings:
                
                mock_nlp.return_value = [
                    {"skill": "Python", "confidence": 0.95},
                    {"skill": "Machine Learning", "confidence": 0.90},
                    {"skill": "Deep Learning", "confidence": 0.85},
                    {"skill": "Neural Networks", "confidence": 0.80}
                ]
                
                mock_embeddings.return_value = [0.1] * 384  # Mock 384-dim embedding
                
                response = await async_client.post("/api/v1/profiles/", json=profile_data, headers=headers)
                assert response.status_code == status.HTTP_201_CREATED
                profile = response.json()
                
                # Verify ML processing enhanced the profile
                assert len(profile["unified_skills"]) >= len(profile_data["skills"])
        
        # Test recommendation pipeline with ML models
        with patch('machinelearningmodel.recommendation_engine.RecommendationEngine.recommend_careers') as mock_rec:
            mock_rec.return_value = [
                {
                    "job_title": "AI Research Scientist",
                    "match_score": 0.94,
                    "required_skills": ["Python", "Deep Learning", "Research", "Publications"],
                    "skill_gaps": {"Research": 0.3, "Publications": 0.5},
                    "reasoning": "Strong technical background, needs research experience"
                }
            ]
            
            response = await async_client.get("/api/v1/recommendations/careers", headers=headers)
            assert response.status_code == status.HTTP_200_OK
            recommendations = response.json()
            
            assert recommendations[0]["job_title"] == "AI Research Scientist"
            assert recommendations[0]["match_score"] >= 0.9
        
        # Test learning path optimization
        with patch('machinelearningmodel.learning_path_optimizer.LearningPathOptimizer.optimize_path') as mock_optimizer:
            mock_optimizer.return_value = {
                "optimized_sequence": ["Research Methods", "Academic Writing", "Paper Publication"],
                "estimated_duration": 24,  # weeks
                "priority_scores": {"Research Methods": 0.9, "Academic Writing": 0.8}
            }
            
            learning_request = {
                "target_skills": ["Research", "Publications", "Academic Writing"],
                "timeline_weeks": 24
            }
            
            response = await async_client.post(
                "/api/v1/learning-paths/optimize",
                json=learning_request,
                headers=headers
            )
            assert response.status_code == status.HTTP_201_CREATED
            optimized_path = response.json()
            
            assert "optimized_sequence" in optimized_path
            assert optimized_path["estimated_duration"] <= 24