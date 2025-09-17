"""
Demo script for market insights functionality
"""
import asyncio
import json
from datetime import datetime
from unittest.mock import AsyncMock, patch

from app.services.market_insights_service import MarketInsightsService


async def demo_market_insights():
    """Demonstrate market insights functionality"""
    
    print("🔍 Market Insights Service Demo")
    print("=" * 50)
    
    # Create service instance
    service = MarketInsightsService()
    
    # Mock database session
    mock_db = AsyncMock()
    
    # Mock some sample data
    mock_db.execute.return_value.scalars.return_value.all.return_value = []
    mock_db.execute.return_value.scalar.return_value = 0
    mock_db.execute.return_value.fetchall.return_value = []
    
    # Mock trend analyzer
    with patch.object(service.trend_analyzer, 'analyze_skill_demand_trends') as mock_trends:
        mock_trends.return_value = [
            {
                'skill_name': 'Python',
                'growth_rate_weekly': 0.15,
                'confidence': 0.85,
                'trend_direction': 'growing',
                'current_demand': 250,
                'data_points': 12
            },
            {
                'skill_name': 'React',
                'growth_rate_weekly': 0.12,
                'confidence': 0.78,
                'trend_direction': 'growing',
                'current_demand': 180,
                'data_points': 10
            }
        ]
        
        with patch.object(service.trend_analyzer, 'detect_emerging_skills') as mock_emerging:
            mock_emerging.return_value = []
            
            with patch.object(service.trend_analyzer, 'get_skill_market_data') as mock_skill_data:
                mock_skill_data.return_value = {
                    'job_count': 200,
                    'growth_trend': 0.15,
                    'avg_salary': 150000,
                    'market_competitiveness': 'medium'
                }
                
                # Test 1: Software Engineer insights
                print("\n📊 Test 1: Software Engineer Market Insights")
                print("-" * 40)
                
                insights = await service.get_comprehensive_market_insights(
                    db=mock_db,
                    role="Software Engineer",
                    skills=["Python", "React", "AWS"],
                    location="San Francisco",
                    experience_level="mid",
                    days=90
                )
                
                print(f"Demand Trend: {insights['demand_trend']}")
                print(f"Salary Growth: {insights['salary_growth']}")
                print(f"Top Skills: {', '.join(insights['top_skills'][:5])}")
                print(f"Competition Level: {insights['competition_level']}")
                print(f"Total Jobs Analyzed: {insights['market_overview']['total_jobs']}")
                print(f"Data Freshness: {insights['data_freshness']}")
                
                # Test 2: Data Scientist insights
                print("\n📊 Test 2: Data Scientist Market Insights")
                print("-" * 40)
                
                insights = await service.get_comprehensive_market_insights(
                    db=mock_db,
                    role="Data Scientist",
                    skills=["Python", "Machine Learning", "SQL"],
                    location="New York",
                    experience_level="senior",
                    days=90
                )
                
                print(f"Demand Trend: {insights['demand_trend']}")
                print(f"Salary Growth: {insights['salary_growth']}")
                print(f"Top Skills: {', '.join(insights['top_skills'][:5])}")
                print(f"Competition Level: {insights['competition_level']}")
                
                # Test 3: Skills-only analysis
                print("\n📊 Test 3: Skills-Only Market Analysis")
                print("-" * 40)
                
                insights = await service.get_comprehensive_market_insights(
                    db=mock_db,
                    role=None,
                    skills=["JavaScript", "TypeScript", "Node.js"],
                    location=None,
                    experience_level=None,
                    days=60
                )
                
                print(f"Demand Trend: {insights['demand_trend']}")
                print(f"Top Skills: {', '.join(insights['top_skills'][:5])}")
                print(f"Recommendations: {len(insights['recommendations'])} available")
                
                # Test 4: Fallback functionality
                print("\n📊 Test 4: Fallback Insights")
                print("-" * 40)
                
                fallback = service._get_fallback_insights(
                    role="DevOps Engineer",
                    skills=["Docker", "Kubernetes", "AWS"]
                )
                
                print(f"Fallback Demand Trend: {fallback['demand_trend']}")
                print(f"Fallback Competition: {fallback['competition_level']}")
                print(f"Fallback Skills: {', '.join(fallback['top_skills'][:3])}")
                
                # Test 5: Cache functionality
                print("\n📊 Test 5: Cache Key Generation")
                print("-" * 40)
                
                cache_key = service._generate_cache_key(
                    role="Full Stack Developer",
                    skills=["React", "Node.js", "MongoDB"],
                    location="Remote",
                    experience_level="mid",
                    days=90
                )
                
                print(f"Generated Cache Key: {cache_key}")
                
                # Test 6: Growth rate estimation
                print("\n📊 Test 6: Growth Rate Estimation")
                print("-" * 40)
                
                growth_rates = [
                    ("Software Engineer", ["Python", "AWS"], 150000),
                    ("Data Scientist", ["Machine Learning", "Python"], 140000),
                    ("Frontend Developer", ["React", "TypeScript"], 120000),
                    ("Marketing Manager", ["Analytics"], 80000)
                ]
                
                for role, skills, salary in growth_rates:
                    growth = service._estimate_growth_rate(role, skills, salary)
                    print(f"{role}: {growth:.1f}% estimated growth")


def demo_simple_format():
    """Demonstrate simple format for frontend compatibility"""
    
    print("\n🎯 Frontend-Compatible Format Demo")
    print("=" * 50)
    
    # Sample comprehensive insights
    comprehensive_insights = {
        'demand_trend': 'High',
        'salary_growth': '+15% YoY',
        'top_skills': ['Python', 'React', 'AWS', 'Docker', 'Kubernetes'],
        'competition_level': 'Medium'
    }
    
    # Convert to simple format (as done in the API)
    simple_format = {
        'demandTrend': comprehensive_insights['demand_trend'],
        'salaryGrowth': comprehensive_insights['salary_growth'],
        'topSkills': comprehensive_insights['top_skills'],
        'competitionLevel': comprehensive_insights['competition_level']
    }
    
    print("Simple Format (Frontend Compatible):")
    print(json.dumps(simple_format, indent=2))


def demo_api_endpoints():
    """Demonstrate available API endpoints"""
    
    print("\n🌐 Available API Endpoints")
    print("=" * 50)
    
    endpoints = [
        {
            'path': '/api/v1/market-insights/simple',
            'method': 'GET',
            'description': 'Get simple market insights (frontend compatible)',
            'params': 'role, skills, location, experience_level, days'
        },
        {
            'path': '/api/v1/market-insights/comprehensive',
            'method': 'POST',
            'description': 'Get comprehensive market insights',
            'body': 'MarketInsightsRequest'
        },
        {
            'path': '/api/v1/market-insights/role/{role}',
            'method': 'GET',
            'description': 'Get market insights for specific role',
            'params': 'location, experience_level, days'
        },
        {
            'path': '/api/v1/market-insights/skills/{skill}',
            'method': 'GET',
            'description': 'Get market insights for specific skill',
            'params': 'location, experience_level, days'
        },
        {
            'path': '/api/v1/market-insights/trending-skills',
            'method': 'GET',
            'description': 'Get trending skills in job market',
            'params': 'days, limit, location'
        },
        {
            'path': '/api/v1/market-insights/salary-analysis',
            'method': 'GET',
            'description': 'Get detailed salary analysis',
            'params': 'role, skills, location, experience_level, days'
        },
        {
            'path': '/api/v1/market-insights/competition-analysis',
            'method': 'GET',
            'description': 'Get market competition analysis',
            'params': 'role, skills, location, experience_level, days'
        }
    ]
    
    for endpoint in endpoints:
        print(f"\n{endpoint['method']} {endpoint['path']}")
        print(f"   Description: {endpoint['description']}")
        if 'params' in endpoint:
            print(f"   Parameters: {endpoint['params']}")
        if 'body' in endpoint:
            print(f"   Request Body: {endpoint['body']}")


async def main():
    """Run all demos"""
    
    print("🚀 Market Insights Backend Functionality Demo")
    print("=" * 60)
    print(f"Demo started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Run service demo
        await demo_market_insights()
        
        # Show simple format
        demo_simple_format()
        
        # Show API endpoints
        demo_api_endpoints()
        
        print("\n✅ Demo completed successfully!")
        print("\nKey Features Implemented:")
        print("• Comprehensive market insights analysis")
        print("• Salary analysis and trend calculations")
        print("• Skill demand and competition level data")
        print("• Frontend-compatible simple format")
        print("• Caching for performance")
        print("• Error handling with fallback data")
        print("• Multiple API endpoints for different use cases")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {str(e)}")
        raise


if __name__ == "__main__":
    asyncio.run(main())