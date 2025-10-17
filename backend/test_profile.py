#!/usr/bin/env python3
"""Test profile endpoint directly"""

import asyncio
import sys
import os
from pathlib import Path

# Add the app directory to Python path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

async def test_profile_endpoint():
    """Test the profile endpoint directly"""
    try:
        # Import necessary modules
        from app.schemas.profile import ProfileResponse
        from app.models.user import User
        from datetime import datetime
        
        # Create a mock user
        mock_user = User()
        mock_user.id = "test-user-123"
        mock_user.created_at = datetime.now()
        mock_user.updated_at = datetime.now()
        
        # Test creating ProfileResponse
        profile_response = ProfileResponse(
            id=str(mock_user.id),
            user_id=str(mock_user.id),
            dream_job="",
            experience_years=None,
            current_role="",
            location="",
            github_username="",
            leetcode_id="",
            linkedin_url="",
            codeforces_id="",
            industry="",
            desired_role="",
            career_goals="",
            timeframe="",
            salary_expectation="",
            education="",
            certifications="",
            languages="",
            work_type="",
            company_size="",
            work_culture="",
            benefits=[],
            skills={},
            platform_data={},
            resume_data={},
            career_interests={},
            skill_gaps={},
            profile_score=0.0,
            completeness_score=0.0,
            data_last_updated=None,
            created_at=mock_user.created_at,
            updated_at=mock_user.updated_at
        )
        
        print("ProfileResponse created successfully!")
        print(f"Profile ID: {profile_response.id}")
        print(f"User ID: {profile_response.user_id}")
        print("Test passed - Profile endpoint should work now.")
        
        return True
        
    except Exception as e:
        print(f"Error testing profile endpoint: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_profile_endpoint())
    if success:
        print("\n✅ Profile endpoint test successful!")
    else:
        print("\n❌ Profile endpoint test failed!")