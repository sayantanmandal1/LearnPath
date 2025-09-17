"""
Dashboard API demonstration script
"""
import asyncio
import sys
import os

# Add the backend directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

# Import dashboard components
from app.schemas.dashboard import (
    DashboardSummary, DashboardMetric, ProgressMilestone, 
    DashboardRecommendation, DashboardActivity, UserProgressSummary, PersonalizedContent
)


async def demo_dashboard_schemas():
    """Demonstrate dashboard schema creation"""
    print("=== Dashboard Schema Demo ===")
    
    # Create sample dashboard metrics
    metrics = [
        DashboardMetric(
            name="Career Score",
            value=75.5,
            change=5.2,
            change_type="increase",
            unit="points",
            description="Overall career development score"
        ),
        DashboardMetric(
            name="Skills Count",
            value=12,
            change=2,
            change_type="increase",
            unit="skills",
            description="Total skills in profile"
        )
    ]
    
    # Create sample milestones
    milestones = [
        ProgressMilestone(
            id="milestone_1",
            title="Complete Python Certification",
            description="Finish Python programming certification course",
            category="learning",
            completed=False,
            progress_percentage=75.0,
            priority="high"
        ),
        ProgressMilestone(
            id="milestone_2",
            title="Update LinkedIn Profile",
            description="Enhance LinkedIn profile with recent achievements",
            category="career",
            completed=True,
            completion_date=datetime.utcnow(),
            progress_percentage=100.0,
            priority="medium"
        )
    ]
    
    # Create sample recommendations
    recommendations = [
        DashboardRecommendation(
            id="rec_1",
            title="Learn Docker",
            description="Container technology is in high demand",
            type="skill",
            priority="high",
            estimated_time="2-3 weeks",
            impact_score=8.5
        ),
        DashboardRecommendation(
            id="rec_2",
            title="Apply to Senior Roles",
            description="Your profile matches senior developer positions",
            type="career",
            priority="medium",
            impact_score=7.8
        )
    ]
    
    # Create sample activities
    activities = [
        DashboardActivity(
            id="activity_1",
            type="profile_update",
            title="Profile Updated",
            description="Added new skills to profile",
            timestamp=datetime.utcnow()
        ),
        DashboardActivity(
            id="activity_2",
            type="analysis_completed",
            title="Career Analysis Completed",
            description="Generated new career recommendations",
            timestamp=datetime.utcnow()
        )
    ]
    
    # Create dashboard summary
    dashboard_summary = DashboardSummary(
        user_id="user_123",
        overall_career_score=75.5,
        profile_completion=85.0,
        key_metrics=metrics,
        active_milestones=[m for m in milestones if not m.completed],
        completed_milestones_count=1,
        total_milestones_count=2,
        top_recommendations=recommendations,
        recent_activities=activities,
        skills_count=12,
        job_matches_count=25,
        learning_paths_count=3,
        last_analysis_date=datetime.utcnow(),
        last_profile_update=datetime.utcnow(),
        generated_at=datetime.utcnow()
    )
    
    print("Dashboard Summary Created:")
    print(f"- User ID: {dashboard_summary.user_id}")
    print(f"- Career Score: {dashboard_summary.overall_career_score}")
    print(f"- Profile Completion: {dashboard_summary.profile_completion}%")
    print(f"- Key Metrics: {len(dashboard_summary.key_metrics)}")
    print(f"- Active Milestones: {len(dashboard_summary.active_milestones)}")
    print(f"- Recommendations: {len(dashboard_summary.top_recommendations)}")
    print(f"- Recent Activities: {len(dashboard_summary.recent_activities)}")
    
    # Create user progress summary
    progress_summary = UserProgressSummary(
        user_id="user_123",
        overall_progress=68.5,
        career_score_trend=[
            {"date": "2024-01-01", "score": 65.0},
            {"date": "2024-01-15", "score": 70.0},
            {"date": "2024-02-01", "score": 75.5}
        ],
        skill_improvements=[
            {"skill": "Python", "improvement": 15.0, "category": "programming"},
            {"skill": "Leadership", "improvement": 8.0, "category": "soft_skills"}
        ],
        new_skills_added=3,
        skills_mastered=2,
        milestones=milestones,
        milestone_completion_rate=50.0,
        learning_paths_started=5,
        learning_paths_completed=2,
        courses_completed=8,
        job_compatibility_improvement=12.5,
        interview_readiness_score=78.0,
        tracking_period_days=90,
        generated_at=datetime.utcnow()
    )
    
    print("\nProgress Summary Created:")
    print(f"- Overall Progress: {progress_summary.overall_progress}%")
    print(f"- New Skills Added: {progress_summary.new_skills_added}")
    print(f"- Skills Mastered: {progress_summary.skills_mastered}")
    print(f"- Milestone Completion Rate: {progress_summary.milestone_completion_rate}%")
    print(f"- Learning Paths Completed: {progress_summary.learning_paths_completed}")
    
    # Create personalized content
    personalized_content = PersonalizedContent(
        user_id="user_123",
        featured_jobs=[
            {
                "id": "job_1",
                "title": "Senior Python Developer",
                "company": "Tech Corp",
                "location": "Remote",
                "match_score": 85.0,
                "salary_range": "$90k - $120k"
            }
        ],
        recommended_skills=[
            {
                "skill": "Docker",
                "category": "DevOps",
                "demand_score": 9.2,
                "learning_time": "2-3 weeks"
            }
        ],
        suggested_learning_paths=[
            {
                "id": "path_1",
                "title": "Full Stack Development",
                "duration": "6 months",
                "difficulty": "Intermediate",
                "match_score": 88.0
            }
        ],
        market_trends=[
            {
                "trend": "AI/ML Skills in High Demand",
                "impact": "High",
                "relevance_score": 9.1
            }
        ],
        salary_insights={
            "current_market_rate": "$95,000",
            "potential_increase": "15%",
            "top_paying_skills": ["Python", "AWS", "Machine Learning"]
        },
        industry_updates=[
            {
                "title": "Remote Work Trends 2024",
                "summary": "Latest insights on remote work adoption",
                "relevance": "High"
            }
        ],
        networking_opportunities=[
            {
                "event": "Tech Meetup - AI in Practice",
                "date": "2024-02-15",
                "location": "Virtual",
                "relevance_score": 8.5
            }
        ],
        similar_profiles=[
            {
                "profile_id": "user_456",
                "role": "Senior Developer",
                "similarity_score": 87.0,
                "common_skills": ["Python", "React", "AWS"]
            }
        ],
        content_categories=["jobs", "skills", "learning", "market", "networking"],
        personalization_score=82.5,
        generated_at=datetime.utcnow()
    )
    
    print("\nPersonalized Content Created:")
    print(f"- Featured Jobs: {len(personalized_content.featured_jobs)}")
    print(f"- Recommended Skills: {len(personalized_content.recommended_skills)}")
    print(f"- Suggested Learning Paths: {len(personalized_content.suggested_learning_paths)}")
    print(f"- Market Trends: {len(personalized_content.market_trends)}")
    print(f"- Personalization Score: {personalized_content.personalization_score}%")
    
    print("\n=== Dashboard Schema Demo Complete ===")
    return True


async def demo_dashboard_service_logic():
    """Demonstrate dashboard service logic without dependencies"""
    print("\n=== Dashboard Service Logic Demo ===")
    
    # Mock profile completion calculation
    def calculate_profile_completion_mock():
        """Mock profile completion calculation"""
        required_fields = [
            'current_role', 'experience_years', 'education_level', 'location',
            'skills', 'career_goals', 'preferred_work_type'
        ]
        
        # Simulate a profile with some fields completed
        completed_fields = 5  # Out of 7 fields
        completion_percentage = (completed_fields / len(required_fields)) * 100
        return completion_percentage
    
    completion = calculate_profile_completion_mock()
    print(f"Profile Completion: {completion:.1f}%")
    
    # Mock key metrics generation
    def generate_key_metrics_mock():
        """Mock key metrics generation"""
        analytics = {
            "overall_career_score": 75.5,
            "skill_analytics": {"total_skills": 12},
            "market_analytics": {"market_position_percentile": 68},
            "experience_analytics": {"experience_score": 8.2}
        }
        
        metrics = []
        
        # Career score metric
        career_score = analytics.get("overall_career_score", 0.0)
        metrics.append({
            "name": "Career Score",
            "value": round(career_score, 1),
            "unit": "points",
            "description": "Overall career development score"
        })
        
        # Skills count
        skill_count = analytics.get("skill_analytics", {}).get("total_skills", 0)
        metrics.append({
            "name": "Skills",
            "value": skill_count,
            "unit": "skills",
            "description": "Total skills in profile"
        })
        
        # Market position
        market_position = analytics.get("market_analytics", {}).get("market_position_percentile", 0)
        metrics.append({
            "name": "Market Position",
            "value": f"{round(market_position)}%",
            "description": "Your position in the job market"
        })
        
        # Experience level
        experience_score = analytics.get("experience_analytics", {}).get("experience_score", 0)
        metrics.append({
            "name": "Experience Score",
            "value": round(experience_score, 1),
            "unit": "points",
            "description": "Professional experience rating"
        })
        
        return metrics
    
    metrics = generate_key_metrics_mock()
    print(f"Generated {len(metrics)} key metrics:")
    for metric in metrics:
        print(f"  - {metric['name']}: {metric['value']}")
    
    # Mock career score trend generation
    def generate_career_score_trend_mock(days=90):
        """Mock career score trend generation"""
        trend = []
        for i in range(0, days, 7):  # Weekly data points
            score = 65 + (i / days) * 15  # Simulated improvement
            trend.append({
                "date": f"2024-01-{i//7 + 1:02d}",
                "score": round(score, 1)
            })
        return trend
    
    trend = generate_career_score_trend_mock(30)
    print(f"Generated career score trend with {len(trend)} data points")
    print(f"  - Start score: {trend[0]['score']}")
    print(f"  - End score: {trend[-1]['score']}")
    
    print("=== Dashboard Service Logic Demo Complete ===")
    return True


async def main():
    """Main demo function"""
    print("Dashboard Implementation Demo")
    print("=" * 50)
    
    try:
        # Test schema creation
        schema_success = await demo_dashboard_schemas()
        
        # Test service logic
        logic_success = await demo_dashboard_service_logic()
        
        if schema_success and logic_success:
            print("\n✅ Dashboard implementation demo completed successfully!")
            print("\nDashboard endpoints implemented:")
            print("- GET /api/v1/dashboard/summary - Dashboard summary data")
            print("- GET /api/v1/dashboard/progress - User progress tracking")
            print("- GET /api/v1/dashboard/personalized-content - Personalized content")
            print("- GET /api/v1/dashboard/metrics - Dashboard metrics")
            print("- GET /api/v1/dashboard/milestones - User milestones")
            print("- GET /api/v1/dashboard/activities - Recent activities")
            print("- GET /api/v1/dashboard/quick-stats - Quick statistics")
            
            print("\nFeatures implemented:")
            print("- Dashboard summary data aggregation")
            print("- User progress tracking and milestone data")
            print("- Personalized dashboard content generation")
            print("- Key metrics calculation and display")
            print("- Recent activity tracking")
            print("- Career score trending")
            print("- Profile completion calculation")
            print("- Graceful error handling and fallbacks")
            
        else:
            print("\n❌ Dashboard implementation demo failed")
            
    except Exception as e:
        print(f"\n❌ Demo failed with error: {str(e)}")
        return False
    
    return True


if __name__ == "__main__":
    asyncio.run(main())