"""
Test enhanced profile schemas for analyze page integration.
"""

import pytest
from pydantic import ValidationError

from app.schemas.profile import ProfileCreate, ProfileUpdate, ProfileResponse


class TestEnhancedProfileSchemas:
    """Test enhanced profile schemas with analyze page fields."""
    
    def test_profile_create_with_analyze_page_fields(self):
        """Test ProfileCreate schema with all analyze page fields."""
        data = {
            # Personal Information
            "current_role": "Software Developer",
            "experience_years": 3,
            "industry": "Technology",
            "location": "San Francisco, CA",
            
            # Career Goals
            "desired_role": "Senior Software Engineer",
            "career_goals": "I want to become a technical lead and work on scalable systems",
            "timeframe": "medium",
            "salary_expectation": "$120,000 - $150,000",
            
            # Skills & Education
            "skills": {"Python": 0.8, "JavaScript": 0.7, "React": 0.6},
            "education": "Bachelor's in Computer Science",
            "certifications": "AWS Certified Developer",
            "languages": "English (Native), Spanish (Conversational)",
            
            # Work Preferences
            "work_type": "hybrid",
            "company_size": "medium",
            "work_culture": "Collaborative environment with focus on innovation",
            "benefits": ["Health Insurance", "401(k) Matching", "Remote Work", "Professional Development"],
            
            # Platform IDs
            "github_username": "testuser",
            "leetcode_id": "testuser123",
            "linkedin_url": "https://linkedin.com/in/testuser"
        }
        
        profile = ProfileCreate(**data)
        
        # Verify all fields are properly set
        assert profile.current_role == "Software Developer"
        assert profile.experience_years == 3
        assert profile.industry == "Technology"
        assert profile.location == "San Francisco, CA"
        assert profile.desired_role == "Senior Software Engineer"
        assert profile.career_goals == "I want to become a technical lead and work on scalable systems"
        assert profile.timeframe == "medium"
        assert profile.salary_expectation == "$120,000 - $150,000"
        assert profile.education == "Bachelor's in Computer Science"
        assert profile.certifications == "AWS Certified Developer"
        assert profile.languages == "English (Native), Spanish (Conversational)"
        assert profile.work_type == "hybrid"
        assert profile.company_size == "medium"
        assert profile.work_culture == "Collaborative environment with focus on innovation"
        assert profile.benefits == ["Health Insurance", "401(k) Matching", "Remote Work", "Professional Development"]
        assert profile.skills == {"Python": 0.8, "JavaScript": 0.7, "React": 0.6}
        assert profile.github_username == "testuser"
        assert profile.leetcode_id == "testuser123"
        assert profile.linkedin_url == "https://linkedin.com/in/testuser"
    
    def test_profile_create_minimal_fields(self):
        """Test ProfileCreate with minimal required fields."""
        data = {
            "current_role": "Developer"
        }
        
        profile = ProfileCreate(**data)
        assert profile.current_role == "Developer"
        assert profile.experience_years is None
        assert profile.industry is None
        assert profile.skills is None
    
    def test_profile_create_validation_errors(self):
        """Test ProfileCreate validation errors."""
        # Test negative experience years
        with pytest.raises(ValidationError) as exc_info:
            ProfileCreate(experience_years=-1)
        
        assert "ensure this value is greater than or equal to 0" in str(exc_info.value)
        
        # Test too high experience years
        with pytest.raises(ValidationError) as exc_info:
            ProfileCreate(experience_years=100)
        
        assert "ensure this value is less than or equal to 50" in str(exc_info.value)
        
        # Test invalid GitHub username
        with pytest.raises(ValidationError) as exc_info:
            ProfileCreate(github_username="invalid@username")
        
        assert "Invalid GitHub username format" in str(exc_info.value)
        
        # Test invalid LinkedIn URL
        with pytest.raises(ValidationError) as exc_info:
            ProfileCreate(linkedin_url="https://facebook.com/user")
        
        assert "Invalid LinkedIn URL format" in str(exc_info.value)
    
    def test_profile_update_with_analyze_page_fields(self):
        """Test ProfileUpdate schema with analyze page fields."""
        data = {
            "industry": "Healthcare",
            "desired_role": "Lead Developer",
            "career_goals": "Updated career goals",
            "timeframe": "short",
            "salary_expectation": "$150,000+",
            "education": "Master's in Computer Science",
            "certifications": "AWS Solutions Architect",
            "languages": "English, French, Spanish",
            "work_type": "remote",
            "company_size": "large",
            "work_culture": "Fast-paced startup environment",
            "benefits": ["Stock Options", "Unlimited PTO"],
            "skills": {"Python": 0.9, "Go": 0.7}
        }
        
        profile_update = ProfileUpdate(**data)
        
        assert profile_update.industry == "Healthcare"
        assert profile_update.desired_role == "Lead Developer"
        assert profile_update.career_goals == "Updated career goals"
        assert profile_update.timeframe == "short"
        assert profile_update.salary_expectation == "$150,000+"
        assert profile_update.education == "Master's in Computer Science"
        assert profile_update.certifications == "AWS Solutions Architect"
        assert profile_update.languages == "English, French, Spanish"
        assert profile_update.work_type == "remote"
        assert profile_update.company_size == "large"
        assert profile_update.work_culture == "Fast-paced startup environment"
        assert profile_update.benefits == ["Stock Options", "Unlimited PTO"]
        assert profile_update.skills == {"Python": 0.9, "Go": 0.7}
    
    def test_profile_response_includes_new_fields(self):
        """Test that ProfileResponse includes all new fields."""
        from datetime import datetime
        
        data = {
            "id": "test-id",
            "user_id": "user-id",
            "current_role": "Developer",
            "experience_years": 2,
            "industry": "Technology",
            "location": "New York, NY",
            "desired_role": "Senior Developer",
            "career_goals": "Advance to senior level",
            "timeframe": "medium",
            "salary_expectation": "$100,000",
            "education": "Bachelor's Degree",
            "certifications": "None",
            "languages": "English",
            "work_type": "hybrid",
            "company_size": "medium",
            "work_culture": "Collaborative",
            "benefits": ["Health Insurance"],
            "skills": {"Python": 0.8},
            "profile_score": 85.5,
            "completeness_score": 90.0,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }
        
        response = ProfileResponse(**data)
        
        # Verify all new fields are included
        assert response.industry == "Technology"
        assert response.desired_role == "Senior Developer"
        assert response.career_goals == "Advance to senior level"
        assert response.timeframe == "medium"
        assert response.salary_expectation == "$100,000"
        assert response.education == "Bachelor's Degree"
        assert response.certifications == "None"
        assert response.languages == "English"
        assert response.work_type == "hybrid"
        assert response.company_size == "medium"
        assert response.work_culture == "Collaborative"
        assert response.benefits == ["Health Insurance"]
        assert response.profile_score == 85.5
        assert response.completeness_score == 90.0
    
    def test_benefits_field_validation(self):
        """Test benefits field accepts list of strings."""
        # Valid benefits list
        data = {"benefits": ["Health Insurance", "401(k)", "Remote Work"]}
        profile = ProfileCreate(**data)
        assert profile.benefits == ["Health Insurance", "401(k)", "Remote Work"]
        
        # Empty benefits list
        data = {"benefits": []}
        profile = ProfileCreate(**data)
        assert profile.benefits == []
        
        # None benefits
        data = {"benefits": None}
        profile = ProfileCreate(**data)
        assert profile.benefits is None
    
    def test_skills_field_validation(self):
        """Test skills field accepts dict with float values."""
        # Valid skills dict
        data = {"skills": {"Python": 0.8, "JavaScript": 0.6, "SQL": 1.0}}
        profile = ProfileCreate(**data)
        assert profile.skills == {"Python": 0.8, "JavaScript": 0.6, "SQL": 1.0}
        
        # Empty skills dict
        data = {"skills": {}}
        profile = ProfileCreate(**data)
        assert profile.skills == {}
        
        # None skills
        data = {"skills": None}
        profile = ProfileCreate(**data)
        assert profile.skills is None
    
    def test_text_field_length_limits(self):
        """Test text field length validation."""
        # Test career_goals max length (2000 chars)
        long_goals = "x" * 2001
        with pytest.raises(ValidationError) as exc_info:
            ProfileCreate(career_goals=long_goals)
        assert "ensure this value has at most 2000 characters" in str(exc_info.value)
        
        # Test work_culture max length (2000 chars)
        long_culture = "x" * 2001
        with pytest.raises(ValidationError) as exc_info:
            ProfileCreate(work_culture=long_culture)
        assert "ensure this value has at most 2000 characters" in str(exc_info.value)
        
        # Test certifications max length (1000 chars)
        long_certs = "x" * 1001
        with pytest.raises(ValidationError) as exc_info:
            ProfileCreate(certifications=long_certs)
        assert "ensure this value has at most 1000 characters" in str(exc_info.value)
    
    def test_enum_like_field_values(self):
        """Test fields that should accept specific values."""
        # Valid timeframe values
        valid_timeframes = ["immediate", "short", "medium", "long"]
        for timeframe in valid_timeframes:
            profile = ProfileCreate(timeframe=timeframe)
            assert profile.timeframe == timeframe
        
        # Valid work_type values
        valid_work_types = ["remote", "hybrid", "onsite", "flexible"]
        for work_type in valid_work_types:
            profile = ProfileCreate(work_type=work_type)
            assert profile.work_type == work_type
        
        # Valid company_size values
        valid_sizes = ["startup", "small", "medium", "large"]
        for size in valid_sizes:
            profile = ProfileCreate(company_size=size)
            assert profile.company_size == size


if __name__ == "__main__":
    pytest.main([__file__])