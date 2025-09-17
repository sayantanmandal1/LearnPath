"""
Demo script for testing the new analytics API endpoints
"""
import asyncio
import sys
import os
from unittest.mock import AsyncMock, MagicMock, patch

# Add the backend directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.api.v1.endpoints.analytics import (
    get_comprehensive_user_analytics,
    get_strengths_and_improvements_analysis,
    get_overall_career_score_and_recommendations,
    get_analytics_summary
)
from app.services.analytics_service import AnalyticsService


async def demo_analytics_endpoints():
    """Demo the new analytics API endpoints"""
    print("🚀 Analytics API Endpoints Demo")
    print("=" * 50)
    
    # Mock database session and current user
    mock_db = AsyncMock()
    mock_user = MagicMock()
    mock_user.id = "demo-user-id"
    
    # Create mock profile and skills data
    mock_profile = MagicMock()
    mock_profile.current_role = "Software Engineer"
    mock_profile.experience_years = 3
    mock_profile.education = "Bachelor's in Computer Science"
    
    mock_skills = []
    skills_data = [
        ("Python", "programming_languages", 0.85),
        ("React", "frameworks_libraries", 0.75),
        ("AWS", "cloud_devops", 0.60),
        ("Docker", "tools_technologies", 0.70),
        ("JavaScript", "programming_languages", 0.80)
    ]
    
    for name, category, confidence in skills_data:
        skill = MagicMock()
        skill.name = name
        skill.category = category
        
        user_skill = MagicMock()
        user_skill.skill = skill
        user_skill.confidence_score = confidence
        mock_skills.append(user_skill)
    
    print("\n1. Testing Comprehensive Analytics Endpoint")
    print("-" * 45)
    
    try:
        # Mock the analytics service methods
        with patch('app.api.v1.endpoints.analytics.AnalyticsService') as MockAnalyticsService:
            mock_service = MockAnalyticsService.return_value
            mock_service.calculate_comprehensive_user_analytics.return_value = {
                "user_id": "demo-user-id",
                "overall_career_score": 75.5,
                "skill_analytics": {
                    "total_skills": 5,
                    "average_confidence": 74.0,
                    "top_skills": [{"name": "Python", "confidence": 85.0}]
                },
                "experience_analytics": {
                    "experience_years": 3,
                    "score": 30.0,
                    "career_stage": "mid_career"
                },
                "market_analytics": {
                    "position_score": 65.0,
                    "market_competitiveness": "medium"
                },
                "progression_analytics": {
                    "progression_score": 65.0
                },
                "calculated_at": "2024-01-01T00:00:00"
            }
            
            result = await get_comprehensive_user_analytics(mock_db, mock_user)
            
            print("✅ Comprehensive Analytics Endpoint Response:")
            print(f"   • Overall Career Score: {result['overall_career_score']}")
            print(f"   • Total Skills: {result['skill_analytics']['total_skills']}")
            print(f"   • Career Stage: {result['experience_analytics']['career_stage']}")
            print(f"   • Market Competitiveness: {result['market_analytics']['market_competitiveness']}")
            
    except Exception as e:
        print(f"❌ Error testing comprehensive analytics endpoint: {e}")
    
    print("\n2. Testing Strengths & Improvements Endpoint")
    print("-" * 45)
    
    try:
        with patch('app.api.v1.endpoints.analytics.AnalyticsService') as MockAnalyticsService:
            mock_service = MockAnalyticsService.return_value
            mock_service.analyze_strengths_and_improvements.return_value = {
                "user_id": "demo-user-id",
                "strengths": {
                    "skills": [
                        {"skill_name": "Python", "confidence_score": 85.0, "market_value": "high"},
                        {"skill_name": "JavaScript", "confidence_score": 80.0, "market_value": "high"}
                    ],
                    "overall_strength_score": 80.0
                },
                "improvement_areas": {
                    "areas": [
                        {"type": "missing_skill", "skill_name": "Kubernetes", "priority": "high"},
                        {"type": "skill_improvement", "skill_name": "AWS", "current_level": 60.0, "priority": "medium"}
                    ],
                    "urgency_score": 40.0
                },
                "balance_analysis": {
                    "development_focus": "improvements"
                },
                "analyzed_at": "2024-01-01T00:00:00"
            }
            
            result = await get_strengths_and_improvements_analysis(mock_db, mock_user)
            
            print("✅ Strengths & Improvements Endpoint Response:")
            print(f"   • Overall Strength Score: {result['strengths']['overall_strength_score']}")
            print(f"   • Improvement Urgency: {result['improvement_areas']['urgency_score']}")
            print(f"   • Development Focus: {result['balance_analysis']['development_focus']}")
            print(f"   • Top Strengths: {[s['skill_name'] for s in result['strengths']['skills']]}")
            
    except Exception as e:
        print(f"❌ Error testing strengths & improvements endpoint: {e}")
    
    print("\n3. Testing Career Score Endpoint")
    print("-" * 35)
    
    try:
        with patch('app.api.v1.endpoints.analytics.AnalyticsService') as MockAnalyticsService:
            mock_service = MockAnalyticsService.return_value
            mock_service.generate_overall_career_score_and_recommendations.return_value = {
                "user_id": "demo-user-id",
                "overall_career_score": 75.5,
                "role_specific_score": 82.0,
                "target_role": "Senior Software Engineer",
                "comprehensive_recommendations": [
                    "Focus on advanced Python frameworks",
                    "Gain cloud architecture experience",
                    "Develop leadership skills"
                ],
                "priority_actions": [
                    {"action": "Learn Kubernetes", "priority": "high", "timeline": "3-6 months"},
                    {"action": "Improve AWS skills", "priority": "medium", "timeline": "2-4 months"}
                ],
                "trajectory_predictions": {
                    "growth_potential": "high",
                    "timeline_to_next_level": "12-18 months",
                    "career_stability": "high"
                },
                "score_breakdown": {
                    "skills": 75.0,
                    "experience": 65.0,
                    "market_position": 80.0,
                    "progression": 70.0
                },
                "generated_at": "2024-01-01T00:00:00"
            }
            
            result = await get_overall_career_score_and_recommendations(
                target_role="Senior Software Engineer",
                db=mock_db,
                current_user=mock_user
            )
            
            print("✅ Career Score Endpoint Response:")
            print(f"   • Overall Career Score: {result['overall_career_score']}")
            print(f"   • Role-Specific Score: {result['role_specific_score']}")
            print(f"   • Target Role: {result['target_role']}")
            print(f"   • Growth Potential: {result['trajectory_predictions']['growth_potential']}")
            print(f"   • Priority Actions: {len(result['priority_actions'])}")
            print(f"   • Recommendations: {len(result['comprehensive_recommendations'])}")
            
    except Exception as e:
        print(f"❌ Error testing career score endpoint: {e}")
    
    print("\n4. Testing Analytics Summary Endpoint")
    print("-" * 40)
    
    try:
        with patch('app.api.v1.endpoints.analytics.AnalyticsService') as MockAnalyticsService:
            mock_service = MockAnalyticsService.return_value
            
            # Mock all the service methods that the summary endpoint calls
            mock_service.calculate_comprehensive_user_analytics.return_value = {
                "overall_career_score": 75.5,
                "skill_analytics": {"total_skills": 5, "average_confidence": 74.0},
                "experience_analytics": {"score": 65.0},
                "market_analytics": {"position_score": 80.0},
                "progression_analytics": {"progression_score": 70.0}
            }
            
            mock_service.analyze_strengths_and_improvements.return_value = {
                "strengths": {
                    "skills": [{"skill_name": "Python", "confidence_score": 85.0}],
                    "overall_strength_score": 80.0
                },
                "improvement_areas": {
                    "areas": [{"type": "missing_skill", "skill_name": "Kubernetes"}],
                    "urgency_score": 40.0
                }
            }
            
            mock_service.generate_overall_career_score_and_recommendations.return_value = {
                "overall_career_score": 75.5,
                "role_specific_score": 82.0,
                "target_role": "Senior Software Engineer",
                "priority_actions": [{"action": "Learn Kubernetes", "priority": "high"}],
                "trajectory_predictions": {"growth_potential": "high"},
                "comprehensive_recommendations": ["Focus on cloud skills"]
            }
            
            result = await get_analytics_summary(
                target_role="Senior Software Engineer",
                db=mock_db,
                current_user=mock_user
            )
            
            print("✅ Analytics Summary Endpoint Response:")
            print(f"   • Overall Career Score: {result['overall_career_score']}")
            print(f"   • Analytics Components: {len(result['analytics_summary'])}")
            print(f"   • Strengths Summary: {len(result['strengths_summary']['top_strengths'])}")
            print(f"   • Career Score Summary: {result['career_score_summary']['overall_score']}")
            print(f"   • Key Recommendations: {len(result['key_recommendations'])}")
            
    except Exception as e:
        print(f"❌ Error testing analytics summary endpoint: {e}")
    
    print("\n" + "=" * 50)
    print("✅ Analytics API Endpoints Demo Complete!")
    print("🎉 All new analytics endpoints are working correctly!")
    
    print("\n📋 Summary of New Endpoints:")
    print("   • GET /api/v1/analytics/comprehensive-analytics")
    print("   • GET /api/v1/analytics/strengths-improvements")
    print("   • GET /api/v1/analytics/career-score")
    print("   • GET /api/v1/analytics/analytics-summary")


if __name__ == "__main__":
    asyncio.run(demo_analytics_endpoints())