"""
Simple unit tests to verify test infrastructure.
"""
import pytest
from unittest.mock import MagicMock, patch


@pytest.mark.unit
def test_basic_functionality():
    """Test basic functionality."""
    assert 1 + 1 == 2
    assert "hello".upper() == "HELLO"


@pytest.mark.unit
def test_test_data_generator(test_data_generator):
    """Test that test data generator works."""
    users = test_data_generator.generate_user_data(3)
    
    assert len(users) == 3
    assert all("email" in user for user in users)
    assert all("password" in user for user in users)
    assert all("full_name" in user for user in users)


@pytest.mark.unit
def test_profile_data_generation(test_data_generator):
    """Test profile data generation."""
    profiles = test_data_generator.generate_profile_data(2)
    
    assert len(profiles) == 2
    assert all("skills" in profile for profile in profiles)
    assert all("dream_job" in profile for profile in profiles)
    assert all("experience_years" in profile for profile in profiles)
    assert all(isinstance(profile["skills"], list) for profile in profiles)


@pytest.mark.unit
def test_job_posting_generation(test_data_generator):
    """Test job posting generation."""
    jobs = test_data_generator.generate_job_postings(5)
    
    assert len(jobs) == 5
    assert all("title" in job for job in jobs)
    assert all("company" in job for job in jobs)
    assert all("required_skills" in job for job in jobs)
    assert all(isinstance(job["required_skills"], list) for job in jobs)


@pytest.mark.unit
def test_mock_nlp_engine(mock_nlp_engine):
    """Test mock NLP engine."""
    text = "Python developer with machine learning experience"
    skills = mock_nlp_engine.extract_skills_from_text(text)
    
    assert len(skills) == 3
    assert skills[0]["skill"] == "Python"
    assert skills[0]["confidence"] == 0.95
    
    embedding = mock_nlp_engine.generate_embeddings(text)
    assert len(embedding) == 384
    assert all(isinstance(x, float) for x in embedding)


@pytest.mark.unit
def test_mock_recommendation_engine(mock_recommendation_engine):
    """Test mock recommendation engine."""
    user_profile = {
        "skills": {"Python": 0.9, "Machine Learning": 0.8},
        "experience_years": 3
    }
    
    recommendations = mock_recommendation_engine.recommend_careers(user_profile)
    
    assert len(recommendations) == 1
    assert recommendations[0]["job_title"] == "Senior ML Engineer"
    assert recommendations[0]["match_score"] == 0.92
    assert "skill_gaps" in recommendations[0]


@pytest.mark.unit
def test_resume_text_generation(test_data_generator, sample_profile_data):
    """Test resume text generation."""
    resume_text = test_data_generator.generate_resume_text(sample_profile_data)
    
    assert isinstance(resume_text, str)
    assert len(resume_text) > 100
    assert "PROFESSIONAL SUMMARY" in resume_text
    assert "TECHNICAL SKILLS" in resume_text
    assert sample_profile_data["dream_job"].lower() in resume_text.lower()


@pytest.mark.unit
def test_github_profile_generation(test_data_generator):
    """Test GitHub profile data generation."""
    github_data = test_data_generator.generate_github_profile_data("testuser")
    
    assert github_data["username"] == "testuser"
    assert "repositories" in github_data
    assert "languages" in github_data
    assert "total_commits" in github_data
    assert isinstance(github_data["repositories"], list)
    assert isinstance(github_data["languages"], dict)


@pytest.mark.unit
def test_leetcode_profile_generation(test_data_generator):
    """Test LeetCode profile data generation."""
    leetcode_data = test_data_generator.generate_leetcode_profile_data("testuser")
    
    assert leetcode_data["username"] == "testuser"
    assert "problems_solved" in leetcode_data
    assert "skills" in leetcode_data
    assert isinstance(leetcode_data["skills"], list)
    assert leetcode_data["problems_solved"] > 0


@pytest.mark.unit
def test_learning_resources_generation(test_data_generator):
    """Test learning resources generation."""
    skills = ["Python", "Machine Learning", "Docker"]
    resources = test_data_generator.generate_learning_resources(skills, count=5)
    
    assert len(resources) == 5
    assert all("title" in resource for resource in resources)
    assert all("provider" in resource for resource in resources)
    assert all("skill" in resource for resource in resources)
    assert all(resource["skill"] in skills for resource in resources)


@pytest.mark.unit
def test_performance_test_data_generation(test_data_generator):
    """Test performance test data generation."""
    perf_data = test_data_generator.generate_performance_test_data(num_users=10)
    
    assert "users" in perf_data
    assert "profiles" in perf_data
    assert "jobs" in perf_data
    assert "learning_resources" in perf_data
    
    assert len(perf_data["users"]) == 10
    assert len(perf_data["profiles"]) == 10
    assert len(perf_data["jobs"]) == 20  # 2x users
    assert len(perf_data["learning_resources"]) == 10


@pytest.mark.unit
def test_ml_test_datasets_generation(test_data_generator):
    """Test ML test datasets generation."""
    ml_data = test_data_generator.generate_ml_test_datasets()
    
    assert "resume_skill_extraction" in ml_data
    assert "job_candidate_matching" in ml_data
    assert "skill_embeddings" in ml_data
    
    assert len(ml_data["resume_skill_extraction"]) == 50
    assert len(ml_data["job_candidate_matching"]) > 0
    assert len(ml_data["skill_embeddings"]) > 0


@pytest.mark.unit
def test_mocking_capabilities():
    """Test that mocking works correctly."""
    with patch('builtins.open', create=True) as mock_open:
        mock_open.return_value.__enter__.return_value.read.return_value = "test content"
        
        # Simulate file reading
        with open("fake_file.txt", "r") as f:
            content = f.read()
        
        assert content == "test content"
        mock_open.assert_called_once_with("fake_file.txt", "r")


@pytest.mark.unit
def test_async_mock():
    """Test async mocking capabilities."""
    import asyncio
    
    async def async_function():
        mock_service = MagicMock()
        mock_service.async_method = MagicMock(return_value="async result")
        
        result = mock_service.async_method()
        return result
    
    result = asyncio.run(async_function())
    assert result == "async result"