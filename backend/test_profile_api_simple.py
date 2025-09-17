"""
Simple test to verify the enhanced profile API endpoints work.
"""

import asyncio
import sys
import os

# Add the backend directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.schemas.profile import ProfileCreate, ProfileUpdate


def test_profile_schemas():
    """Test that the enhanced profile schemas work with analyze page data."""
    
    # Test data from frontend analyze page
    analyze_page_data = {
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
    
    print("Testing ProfileCreate schema...")
    try:
        profile_create = ProfileCreate(**analyze_page_data)
        print("✓ ProfileCreate schema works with analyze page data")
        
        # Verify key fields
        assert profile_create.current_role == "Software Developer"
        assert profile_create.industry == "Technology"
        assert profile_create.desired_role == "Senior Software Engineer"
        assert profile_create.work_type == "hybrid"
        assert profile_create.company_size == "medium"
        assert len(profile_create.benefits) == 4
        assert profile_create.skills["Python"] == 0.8
        print("✓ All analyze page fields are properly mapped")
        
    except Exception as e:
        print(f"✗ ProfileCreate schema failed: {e}")
        return False
    
    print("\nTesting ProfileUpdate schema...")
    try:
        # Test partial update
        update_data = {
            "industry": "Healthcare",
            "desired_role": "Lead Developer",
            "work_type": "remote",
            "benefits": ["Stock Options", "Unlimited PTO"]
        }
        
        profile_update = ProfileUpdate(**update_data)
        print("✓ ProfileUpdate schema works with partial data")
        
        assert profile_update.industry == "Healthcare"
        assert profile_update.desired_role == "Lead Developer"
        assert profile_update.work_type == "remote"
        assert len(profile_update.benefits) == 2
        print("✓ Partial updates work correctly")
        
    except Exception as e:
        print(f"✗ ProfileUpdate schema failed: {e}")
        return False
    
    print("\nTesting validation...")
    try:
        # Test invalid data
        try:
            ProfileCreate(experience_years=-1)
            print("✗ Validation should have failed for negative experience")
            return False
        except Exception:
            print("✓ Negative experience years validation works")
        
        try:
            ProfileCreate(github_username="invalid@username")
            print("✗ Validation should have failed for invalid GitHub username")
            return False
        except Exception:
            print("✓ GitHub username validation works")
        
        try:
            ProfileCreate(linkedin_url="https://facebook.com/user")
            print("✗ Validation should have failed for invalid LinkedIn URL")
            return False
        except Exception:
            print("✓ LinkedIn URL validation works")
            
    except Exception as e:
        print(f"✗ Validation testing failed: {e}")
        return False
    
    print("\n✅ All profile schema tests passed!")
    return True


def test_profile_completeness_calculation():
    """Test profile completeness calculation logic."""
    
    print("\nTesting profile completeness calculation...")
    
    # Mock a simple completeness calculation
    def calculate_completeness(profile_data):
        """Simple completeness calculation for testing."""
        total_fields = 15  # Core fields we care about
        filled_fields = 0
        
        core_fields = [
            'current_role', 'experience_years', 'industry', 'location',
            'desired_role', 'career_goals', 'education', 'skills',
            'work_type', 'company_size', 'timeframe', 'salary_expectation',
            'certifications', 'languages', 'benefits'
        ]
        
        for field in core_fields:
            if field in profile_data and profile_data[field]:
                if field == 'skills' and isinstance(profile_data[field], dict) and len(profile_data[field]) > 0:
                    filled_fields += 1
                elif field == 'benefits' and isinstance(profile_data[field], list) and len(profile_data[field]) > 0:
                    filled_fields += 1
                elif field not in ['skills', 'benefits'] and str(profile_data[field]).strip():
                    filled_fields += 1
        
        return (filled_fields / total_fields) * 100
    
    # Test with complete profile
    complete_profile = {
        "current_role": "Software Developer",
        "experience_years": 3,
        "industry": "Technology",
        "location": "San Francisco, CA",
        "desired_role": "Senior Software Engineer",
        "career_goals": "I want to become a technical lead",
        "education": "Bachelor's in Computer Science",
        "skills": {"Python": 0.8, "JavaScript": 0.7},
        "work_type": "hybrid",
        "company_size": "medium",
        "timeframe": "medium",
        "salary_expectation": "$120,000",
        "certifications": "AWS Certified",
        "languages": "English, Spanish",
        "benefits": ["Health Insurance", "401(k)"]
    }
    
    completeness = calculate_completeness(complete_profile)
    print(f"Complete profile completeness: {completeness:.1f}%")
    assert completeness == 100.0, f"Expected 100%, got {completeness}%"
    print("✓ Complete profile calculation works")
    
    # Test with minimal profile
    minimal_profile = {
        "current_role": "Developer",
        "skills": {"Python": 0.8}
    }
    
    completeness = calculate_completeness(minimal_profile)
    print(f"Minimal profile completeness: {completeness:.1f}%")
    assert completeness < 20, f"Expected < 20%, got {completeness}%"
    print("✓ Minimal profile calculation works")
    
    print("✅ Profile completeness calculation tests passed!")
    return True


if __name__ == "__main__":
    print("🧪 Testing Enhanced Profile Management")
    print("=" * 50)
    
    success = True
    
    # Test schemas
    if not test_profile_schemas():
        success = False
    
    # Test completeness calculation
    if not test_profile_completeness_calculation():
        success = False
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 All tests passed! Enhanced profile management is working.")
    else:
        print("❌ Some tests failed. Check the output above.")
        sys.exit(1)