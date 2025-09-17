"""
Demo script showing enhanced profile management functionality.
This demonstrates the integration with frontend analyze page data.
"""

import asyncio
import sys
import os
from datetime import datetime

# Add the backend directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.schemas.profile import ProfileCreate, ProfileUpdate, ProfileResponse


def demo_analyze_page_integration():
    """Demonstrate integration with frontend analyze page data."""
    
    print("🎯 Enhanced Profile Management Demo")
    print("=" * 60)
    print("Demonstrating integration with frontend analyze page data\n")
    
    # Sample data that would come from the frontend analyze page
    analyze_page_data = {
        # Personal Information (Step 1)
        "current_role": "Software Developer",
        "experience_years": 3,
        "industry": "Technology",
        "location": "San Francisco, CA",
        
        # Career Goals (Step 2)
        "desired_role": "Senior Software Engineer",
        "career_goals": "I want to become a technical lead and work on scalable distributed systems. My goal is to mentor junior developers while contributing to architectural decisions.",
        "timeframe": "medium",  # 6-12 months
        "salary_expectation": "$120,000 - $150,000",
        
        # Skills & Education (Step 3)
        "skills": {
            "Python": 0.85,
            "JavaScript": 0.75,
            "React": 0.70,
            "Node.js": 0.65,
            "Docker": 0.60,
            "AWS": 0.55,
            "PostgreSQL": 0.70
        },
        "education": "Bachelor's in Computer Science from UC Berkeley",
        "certifications": "AWS Certified Developer Associate, Google Cloud Professional Developer",
        "languages": "English (Native), Spanish (Conversational), Mandarin (Basic)",
        
        # Work Preferences (Step 4)
        "work_type": "hybrid",
        "company_size": "medium",  # 201-1000 employees
        "work_culture": "I thrive in collaborative environments that value innovation, continuous learning, and work-life balance. I prefer teams that embrace agile methodologies and encourage experimentation.",
        "benefits": [
            "Health Insurance",
            "401(k) Matching", 
            "Remote Work",
            "Professional Development",
            "Stock Options",
            "Flexible PTO"
        ],
        
        # Platform connections
        "github_username": "johndoe-dev",
        "leetcode_id": "johndoe123",
        "linkedin_url": "https://linkedin.com/in/johndoe-developer"
    }
    
    print("📝 Creating profile from analyze page data...")
    try:
        profile = ProfileCreate(**analyze_page_data)
        print("✅ Profile created successfully!")
        
        print(f"\n👤 Profile Summary:")
        print(f"   Current Role: {profile.current_role}")
        print(f"   Experience: {profile.experience_years} years")
        print(f"   Industry: {profile.industry}")
        print(f"   Location: {profile.location}")
        print(f"   Desired Role: {profile.desired_role}")
        print(f"   Timeframe: {profile.timeframe}")
        print(f"   Work Type: {profile.work_type}")
        print(f"   Company Size: {profile.company_size}")
        print(f"   Skills: {len(profile.skills)} skills listed")
        print(f"   Benefits: {len(profile.benefits)} preferred benefits")
        print(f"   Platforms: GitHub, LeetCode, LinkedIn connected")
        
    except Exception as e:
        print(f"❌ Error creating profile: {e}")
        return False
    
    print(f"\n🔧 Skills Analysis:")
    if profile.skills:
        sorted_skills = sorted(profile.skills.items(), key=lambda x: x[1], reverse=True)
        for skill, confidence in sorted_skills[:5]:  # Top 5 skills
            confidence_bar = "█" * int(confidence * 10) + "░" * (10 - int(confidence * 10))
            print(f"   {skill:<12} [{confidence_bar}] {confidence:.2f}")
    
    print(f"\n🎯 Career Goals Analysis:")
    print(f"   Current → Target: {profile.current_role} → {profile.desired_role}")
    print(f"   Timeline: {profile.timeframe} term")
    print(f"   Salary Target: {profile.salary_expectation}")
    
    print(f"\n💼 Work Preferences:")
    print(f"   Work Type: {profile.work_type}")
    print(f"   Company Size: {profile.company_size}")
    print(f"   Top Benefits: {', '.join(profile.benefits[:3])}")
    
    return True


def demo_profile_scoring():
    """Demonstrate profile scoring and analytics."""
    
    print("\n📊 Profile Scoring & Analytics Demo")
    print("=" * 60)
    
    # Mock profile scoring calculation
    def calculate_profile_metrics(profile_data):
        """Calculate profile metrics for demo purposes."""
        
        # Completeness Score (0-100)
        total_fields = 16
        filled_fields = 0
        
        key_fields = [
            'current_role', 'experience_years', 'industry', 'location',
            'desired_role', 'career_goals', 'education', 'skills',
            'work_type', 'company_size', 'timeframe', 'salary_expectation',
            'certifications', 'languages', 'benefits', 'github_username'
        ]
        
        for field in key_fields:
            if hasattr(profile_data, field):
                value = getattr(profile_data, field)
                if value:
                    if field == 'skills' and isinstance(value, dict) and len(value) > 0:
                        filled_fields += 1
                    elif field == 'benefits' and isinstance(value, list) and len(value) > 0:
                        filled_fields += 1
                    elif field not in ['skills', 'benefits'] and str(value).strip():
                        filled_fields += 1
        
        completeness_score = (filled_fields / total_fields) * 100
        
        # Skills Score (0-100)
        skills_score = 0
        if hasattr(profile_data, 'skills') and profile_data.skills:
            skill_count = len(profile_data.skills)
            avg_confidence = sum(profile_data.skills.values()) / skill_count
            skills_score = min(100, (skill_count * 8) + (avg_confidence * 40))
        
        # Platform Integration Score (0-100)
        platform_score = 0
        platforms = ['github_username', 'leetcode_id', 'linkedin_url']
        connected = sum(1 for p in platforms if hasattr(profile_data, p) and getattr(profile_data, p))
        platform_score = (connected / len(platforms)) * 100
        
        # Overall Score (weighted average)
        overall_score = (
            completeness_score * 0.4 +
            skills_score * 0.3 +
            platform_score * 0.2 +
            80 * 0.1  # Data freshness (mock)
        )
        
        return {
            'overall_score': overall_score,
            'completeness_score': completeness_score,
            'skills_score': skills_score,
            'platform_score': platform_score,
            'skill_count': len(profile_data.skills) if profile_data.skills else 0,
            'connected_platforms': connected
        }
    
    # Create a sample profile
    sample_data = {
        "current_role": "Software Developer",
        "experience_years": 3,
        "industry": "Technology",
        "location": "San Francisco, CA",
        "desired_role": "Senior Software Engineer",
        "career_goals": "Become a technical lead",
        "education": "Bachelor's in CS",
        "skills": {"Python": 0.8, "JavaScript": 0.7, "React": 0.6, "AWS": 0.5},
        "work_type": "hybrid",
        "company_size": "medium",
        "timeframe": "medium",
        "salary_expectation": "$120,000",
        "certifications": "AWS Certified",
        "languages": "English, Spanish",
        "benefits": ["Health Insurance", "401(k)"],
        "github_username": "testuser",
        "leetcode_id": "testuser",
        "linkedin_url": "https://linkedin.com/in/testuser"
    }
    
    profile = ProfileCreate(**sample_data)
    metrics = calculate_profile_metrics(profile)
    
    print("📈 Profile Metrics:")
    print(f"   Overall Score:     {metrics['overall_score']:.1f}/100")
    print(f"   Completeness:      {metrics['completeness_score']:.1f}/100")
    print(f"   Skills Quality:    {metrics['skills_score']:.1f}/100")
    print(f"   Platform Integration: {metrics['platform_score']:.1f}/100")
    
    print(f"\n📊 Profile Statistics:")
    print(f"   Total Skills:      {metrics['skill_count']}")
    print(f"   Connected Platforms: {metrics['connected_platforms']}/3")
    
    # Generate recommendations based on score
    print(f"\n💡 Recommendations:")
    if metrics['completeness_score'] < 80:
        print("   • Complete missing profile fields to improve your score")
    if metrics['skills_score'] < 70:
        print("   • Add more skills or improve confidence ratings")
    if metrics['platform_score'] < 100:
        print("   • Connect additional platforms (GitHub, LeetCode, LinkedIn)")
    if metrics['overall_score'] >= 85:
        print("   • Excellent profile! Consider refreshing your data regularly")
    
    return True


def demo_profile_updates():
    """Demonstrate profile update functionality."""
    
    print("\n🔄 Profile Update Demo")
    print("=" * 60)
    
    print("📝 Simulating profile updates from analyze page...")
    
    # Original profile data
    original_data = {
        "current_role": "Junior Developer",
        "experience_years": 1,
        "industry": "Technology",
        "desired_role": "Software Engineer",
        "skills": {"Python": 0.6, "JavaScript": 0.5},
        "work_type": "onsite",
        "company_size": "small"
    }
    
    original_profile = ProfileCreate(**original_data)
    print(f"Original: {original_profile.current_role} with {len(original_profile.skills)} skills")
    
    # Update data (user progressed in their career)
    update_data = {
        "current_role": "Software Developer",
        "experience_years": 3,
        "desired_role": "Senior Software Engineer",
        "skills": {
            "Python": 0.85,
            "JavaScript": 0.75,
            "React": 0.70,
            "Docker": 0.60,
            "AWS": 0.55
        },
        "work_type": "hybrid",
        "company_size": "medium",
        "certifications": "AWS Certified Developer",
        "salary_expectation": "$120,000 - $150,000"
    }
    
    profile_update = ProfileUpdate(**update_data)
    print(f"Updated: {profile_update.current_role} with {len(profile_update.skills)} skills")
    
    print(f"\n📈 Changes Detected:")
    print(f"   Role: {original_data['current_role']} → {update_data['current_role']}")
    print(f"   Experience: {original_data['experience_years']} → {update_data['experience_years']} years")
    print(f"   Skills: {len(original_data['skills'])} → {len(update_data['skills'])} skills")
    print(f"   Work Type: {original_data['work_type']} → {update_data['work_type']}")
    print(f"   Added: Certifications, Salary Expectation")
    
    return True


def main():
    """Run all demos."""
    
    success = True
    
    # Demo 1: Analyze page integration
    if not demo_analyze_page_integration():
        success = False
    
    # Demo 2: Profile scoring
    if not demo_profile_scoring():
        success = False
    
    # Demo 3: Profile updates
    if not demo_profile_updates():
        success = False
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 All demos completed successfully!")
        print("\n✨ Enhanced Profile Management Features:")
        print("   • ✅ Full analyze page field support")
        print("   • ✅ Profile completeness scoring")
        print("   • ✅ Skills analysis and categorization")
        print("   • ✅ Platform integration tracking")
        print("   • ✅ Career progression analytics")
        print("   • ✅ Personalized recommendations")
        print("   • ✅ Data validation and error handling")
    else:
        print("❌ Some demos failed. Check the output above.")
        return False
    
    return True


if __name__ == "__main__":
    main()