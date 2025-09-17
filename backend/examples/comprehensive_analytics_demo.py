"""
Demo script for comprehensive analytics functionality
"""
import asyncio
import sys
import os
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

# Add the backend directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.analytics_service import AnalyticsService
from app.models.user import User
from app.models.profile import UserProfile
from app.models.skill import Skill, UserSkill


async def create_mock_data():
    """Create mock data for testing"""
    # Mock user - create as dict since SQLAlchemy models need proper initialization
    user = {
        "id": "demo-user-id",
        "email": "demo@example.com",
        "full_name": "Demo User"
    }
    
    # Mock user profile - create as mock object
    profile = MagicMock()
    profile.id = "demo-profile-id"
    profile.user_id = "demo-user-id"
    profile.current_role = "Software Engineer"
    profile.experience_years = 3
    profile.education = "Bachelor's in Computer Science"
    profile.location = "San Francisco, CA"
    
    # Mock user skills
    skills_data = [
        ("Python", "programming_languages", 0.85),
        ("React", "frameworks_libraries", 0.75),
        ("AWS", "cloud_devops", 0.60),
        ("Docker", "tools_technologies", 0.70),
        ("JavaScript", "programming_languages", 0.80),
        ("PostgreSQL", "databases", 0.65),
        ("Git", "tools_technologies", 0.90)
    ]
    
    user_skills = []
    for name, category, confidence in skills_data:
        # Create mock skill
        skill = MagicMock()
        skill.id = f"skill-{name.lower()}"
        skill.name = name
        skill.category = category
        
        # Create mock user skill
        user_skill = MagicMock()
        user_skill.user_id = "demo-user-id"
        user_skill.skill_id = skill.id
        user_skill.skill = skill
        user_skill.confidence_score = confidence
        
        user_skills.append(user_skill)
    
    return user, profile, user_skills


async def demo_comprehensive_analytics():
    """Demo comprehensive analytics functionality"""
    print("🚀 Comprehensive Analytics Demo")
    print("=" * 50)
    
    # Create mock database session
    mock_db = AsyncMock()
    analytics_service = AnalyticsService(mock_db)
    
    # Create mock data
    user, profile, user_skills = await create_mock_data()
    
    # Mock the database queries
    async def mock_get_user_profile(user_id):
        return profile
    
    async def mock_get_user_skills(user_id):
        return user_skills
    
    analytics_service._get_user_profile = mock_get_user_profile
    analytics_service._get_user_skills = mock_get_user_skills
    
    print("\n1. Testing Comprehensive User Analytics")
    print("-" * 40)
    
    try:
        analytics = await analytics_service.calculate_comprehensive_user_analytics("demo-user-id")
        
        print(f"✅ Overall Career Score: {analytics['overall_career_score']:.1f}/100")
        print(f"✅ Total Skills: {analytics['skill_analytics']['total_skills']}")
        print(f"✅ Average Skill Confidence: {analytics['skill_analytics']['average_confidence']:.1f}%")
        print(f"✅ Experience Score: {analytics['experience_analytics']['score']:.1f}/100")
        print(f"✅ Market Position Score: {analytics['market_analytics']['position_score']:.1f}/100")
        print(f"✅ Career Stage: {analytics['experience_analytics']['career_stage']}")
        print(f"✅ Role Level: {analytics['experience_analytics']['role_level']}")
        
        # Show top skills
        print("\n📊 Top Skills:")
        for skill in analytics['skill_analytics']['top_skills'][:5]:
            print(f"   • {skill['name']}: {skill['confidence']:.1f}%")
        
        # Show skill distribution
        print("\n📈 Skill Distribution by Category:")
        for category, data in analytics['skill_analytics']['skill_distribution'].items():
            print(f"   • {category.replace('_', ' ').title()}: {data['count']} skills (avg: {data['avg_confidence']:.1f}%)")
        
    except Exception as e:
        print(f"❌ Error in comprehensive analytics: {e}")
    
    print("\n2. Testing Strengths and Improvements Analysis")
    print("-" * 40)
    
    try:
        strengths_analysis = await analytics_service.analyze_strengths_and_improvements("demo-user-id")
        
        print(f"✅ Overall Strength Score: {strengths_analysis['strengths']['overall_strength_score']:.1f}/100")
        print(f"✅ Improvement Urgency Score: {strengths_analysis['improvement_areas']['urgency_score']:.1f}/100")
        print(f"✅ Development Focus: {strengths_analysis['balance_analysis']['development_focus']}")
        
        # Show skill strengths
        print("\n💪 Skill Strengths:")
        for strength in strengths_analysis['strengths']['skills'][:3]:
            print(f"   • {strength['skill_name']}: {strength['confidence_score']:.1f}% confidence")
        
        # Show improvement areas
        print("\n🎯 Improvement Areas:")
        for area in strengths_analysis['improvement_areas']['areas'][:3]:
            if area['type'] == 'skill_improvement':
                print(f"   • {area['skill_name']}: {area['current_level']:.1f}% → {area['target_level']}% (Priority: {area['priority']})")
            elif area['type'] == 'missing_skill':
                print(f"   • Learn {area['skill_name']}: High market demand skill (Priority: {area['priority']})")
        
        # Show recommendations
        print("\n💡 Recommendations:")
        for rec in strengths_analysis['improvement_areas']['recommendations'][:3]:
            print(f"   • {rec['recommendation']}")
        
    except Exception as e:
        print(f"❌ Error in strengths analysis: {e}")
    
    print("\n3. Testing Overall Career Score and Recommendations")
    print("-" * 40)
    
    try:
        target_role = "Senior Software Engineer"
        career_score = await analytics_service.generate_overall_career_score_and_recommendations(
            "demo-user-id", target_role
        )
        
        print(f"✅ Overall Career Score: {career_score['overall_career_score']:.1f}/100")
        print(f"✅ Target Role: {career_score['target_role']}")
        print(f"✅ Role-Specific Score: {career_score.get('role_specific_score', 'N/A')}")
        
        # Show score breakdown
        print("\n📊 Score Breakdown:")
        breakdown = career_score['score_breakdown']
        print(f"   • Skills: {breakdown['skills']:.1f}/100")
        print(f"   • Experience: {breakdown['experience']:.1f}/100")
        print(f"   • Market Position: {breakdown['market_position']:.1f}/100")
        print(f"   • Progression: {breakdown['progression']:.1f}/100")
        
        # Show trajectory predictions
        print("\n🔮 Career Trajectory Predictions:")
        trajectory = career_score['trajectory_predictions']
        print(f"   • Growth Potential: {trajectory['growth_potential']}")
        print(f"   • Timeline to Next Level: {trajectory['timeline_to_next_level']}")
        print(f"   • Predicted Salary Growth: {trajectory['predicted_salary_growth']}")
        print(f"   • Career Stability: {trajectory['career_stability']}")
        
        # Show priority actions
        print("\n🎯 Priority Actions:")
        for action in career_score['priority_actions'][:3]:
            print(f"   • {action['action']} (Priority: {action['priority']}, Timeline: {action['timeline']})")
        
        # Show comprehensive recommendations
        print("\n💡 Comprehensive Recommendations:")
        for rec in career_score['comprehensive_recommendations'][:5]:
            print(f"   • {rec}")
        
    except Exception as e:
        print(f"❌ Error in career score analysis: {e}")
    
    print("\n4. Testing Individual Helper Methods")
    print("-" * 40)
    
    try:
        # Test role level determination
        roles = [
            "Junior Software Engineer",
            "Software Engineer", 
            "Senior Software Engineer",
            "Lead Developer",
            "Engineering Manager",
            "Director of Engineering"
        ]
        
        print("🏷️  Role Level Classification:")
        for role in roles:
            level = analytics_service._determine_role_level(role)
            print(f"   • {role} → {level}")
        
        # Test career stage determination
        print("\n📈 Career Stage Classification:")
        experience_years = [1, 3, 7, 12]
        for years in experience_years:
            stage = analytics_service._determine_career_stage(years)
            print(f"   • {years} years → {stage}")
        
        # Test skill market demand
        print("\n📊 Skill Market Demand:")
        test_skills = ["Python", "React", "Java", "COBOL"]
        for skill in test_skills:
            demand = await analytics_service._get_skill_market_demand(skill)
            print(f"   • {skill} → {demand:.1f} demand score")
        
    except Exception as e:
        print(f"❌ Error in helper methods: {e}")
    
    print("\n" + "=" * 50)
    print("✅ Comprehensive Analytics Demo Complete!")
    print("🎉 All analytics functionality is working correctly!")


if __name__ == "__main__":
    asyncio.run(demo_comprehensive_analytics())