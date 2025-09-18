"""
AI Analysis Service Demo

This script demonstrates the AI analysis service with Gemini integration.
It shows how to perform comprehensive profile analysis and individual analysis components.
"""
import asyncio
import os
import sys
from datetime import datetime
from pathlib import Path

# Add the backend directory to the Python path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from app.services.ai_analysis_service import AIAnalysisService, GeminiAPIClient, CompleteProfileData
from app.schemas.profile import ProfileResponse
from app.schemas.resume import ParsedResumeData, ContactInfo, WorkExperience, SkillCategory
from app.models.analysis_result import AnalysisType
from unittest.mock import AsyncMock


async def test_gemini_client():
    """Test Gemini API client functionality"""
    print("\n=== Testing Gemini API Client ===")
    
    client = GeminiAPIClient()
    
    # Check if API key is configured
    if not client.api_key:
        print("⚠️  Gemini API key not configured - using mock responses")
        return False
    
    try:
        # Test basic content generation
        print("Testing basic content generation...")
        content, request_id = await client.generate_content(
            "Explain what a software engineer does in one sentence.",
            temperature=0.1
        )
        print(f"✅ Content generated: {content[:100]}...")
        print(f"   Request ID: {request_id}")
        
        # Test structured output
        print("\nTesting structured JSON output...")
        result, request_id = await client.analyze_with_structured_output(
            "List 3 programming languages with their difficulty levels.",
            '{"languages": [{"name": "language_name", "difficulty": "beginner/intermediate/advanced"}]}'
        )
        print(f"✅ Structured output: {result}")
        print(f"   Request ID: {request_id}")
        
        return True
        
    except Exception as e:
        print(f"❌ Gemini API test failed: {str(e)}")
        return False


async def create_sample_profile_data():
    """Create sample profile data for testing"""
    print("\n=== Creating Sample Profile Data ===")
    
    # Create sample basic profile
    basic_profile = ProfileResponse(
        id="demo-profile-123",
        user_id="demo-user-123",
        current_role="Software Developer",
        experience_years=3,
        location="Bangalore, India",
        dream_job="Senior Full Stack Developer",
        industry="Technology",
        desired_role="Full Stack Developer",
        career_goals="Become a technical lead and build scalable applications",
        timeframe="12-18 months",
        salary_expectation="15-25 LPA",
        education="B.Tech Computer Science",
        certifications="AWS Solutions Architect Associate",
        languages="English, Hindi, Kannada",
        work_type="Hybrid",
        company_size="Medium (100-500 employees)",
        work_culture="Collaborative and innovation-focused",
        benefits=["Health Insurance", "Flexible Hours", "Learning Budget"],
        # Required platform fields
        github_username="johndoe",
        leetcode_id="johndoe_coder",
        linkedin_url="https://linkedin.com/in/johndoe",
        codeforces_id="johndoe_cf",
        skills={"Python": 0.8, "JavaScript": 0.7, "React": 0.75, "Node.js": 0.6, "SQL": 0.7},
        platform_data={},
        resume_data={},
        career_interests={},
        skill_gaps={},
        profile_score=0.8,
        completeness_score=0.85,
        data_last_updated=datetime.utcnow(),
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow()
    )
    
    # Create sample resume data
    resume_data = ParsedResumeData(
        contact_info=ContactInfo(
            name="John Doe",
            email="john.doe@example.com",
            phone="+91-9876543210",
            location="Bangalore, India",
            linkedin="https://linkedin.com/in/johndoe",
            github="https://github.com/johndoe"
        ),
        summary="Experienced software developer with 3+ years in full-stack development. Passionate about building scalable web applications and learning new technologies.",
        work_experience=[
            WorkExperience(
                company="TechCorp Solutions",
                position="Software Developer",
                start_date="2021-06",
                end_date="Present",
                description="Developed and maintained web applications using React and Node.js",
                technologies=["React", "Node.js", "MongoDB", "AWS"],
                achievements=[
                    "Improved application performance by 40%",
                    "Led a team of 3 junior developers",
                    "Implemented CI/CD pipeline reducing deployment time by 60%"
                ]
            ),
            WorkExperience(
                company="StartupXYZ",
                position="Junior Developer",
                start_date="2020-01",
                end_date="2021-05",
                description="Built responsive web interfaces and REST APIs",
                technologies=["JavaScript", "Python", "PostgreSQL"],
                achievements=[
                    "Developed 5+ customer-facing features",
                    "Reduced bug reports by 30% through improved testing"
                ]
            )
        ],
        skills=[
            SkillCategory(
                category="Programming Languages",
                skills=["Python", "JavaScript", "TypeScript", "Java"],
                proficiency_level="Intermediate to Advanced"
            ),
            SkillCategory(
                category="Web Technologies",
                skills=["React", "Node.js", "HTML/CSS", "REST APIs"],
                proficiency_level="Intermediate"
            ),
            SkillCategory(
                category="Databases",
                skills=["PostgreSQL", "MongoDB", "Redis"],
                proficiency_level="Intermediate"
            ),
            SkillCategory(
                category="Cloud & DevOps",
                skills=["AWS", "Docker", "Git", "CI/CD"],
                proficiency_level="Beginner to Intermediate"
            )
        ]
    )
    
    # Create sample platform data
    platform_data = {
        "github": {
            "username": "johndoe",
            "public_repos": 25,
            "followers": 45,
            "following": 30,
            "total_stars": 120,
            "languages": {"Python": 40, "JavaScript": 35, "TypeScript": 15, "Java": 10},
            "contributions_last_year": 280,
            "most_starred_repo": "awesome-web-app"
        },
        "leetcode": {
            "username": "johndoe_coder",
            "problems_solved": 150,
            "easy_solved": 80,
            "medium_solved": 60,
            "hard_solved": 10,
            "contest_rating": 1650,
            "global_ranking": 25000,
            "badges": ["50 Days Badge", "100 Problems Badge"]
        },
        "linkedin": {
            "profile_url": "https://linkedin.com/in/johndoe",
            "connections": 350,
            "endorsements": {"Python": 12, "JavaScript": 8, "React": 6},
            "recommendations": 3,
            "posts_last_month": 5,
            "profile_views": 120
        }
    }
    
    # Create career preferences
    career_preferences = {
        "desired_role": "Full Stack Developer",
        "career_goals": "Become a technical lead and build scalable applications",
        "timeframe": "12-18 months",
        "salary_expectation": "15-25 LPA",
        "work_type": "Hybrid",
        "company_size": "Medium (100-500 employees)",
        "work_culture": "Collaborative and innovation-focused"
    }
    
    complete_profile = CompleteProfileData(
        user_id="demo-user-123",
        basic_profile=basic_profile,
        resume_data=resume_data,
        platform_data=platform_data,
        career_preferences=career_preferences
    )
    
    print("✅ Sample profile data created successfully")
    print(f"   User: {resume_data.contact_info.name}")
    print(f"   Role: {basic_profile.current_role}")
    print(f"   Experience: {basic_profile.experience_years} years")
    print(f"   GitHub repos: {platform_data['github']['public_repos']}")
    print(f"   LeetCode problems: {platform_data['leetcode']['problems_solved']}")
    
    return complete_profile


async def test_skill_assessment(service, profile_data):
    """Test skill assessment generation"""
    print("\n=== Testing Skill Assessment ===")
    
    try:
        # Mock database for testing
        mock_db = AsyncMock()
        
        # Test with actual profile data
        context = profile_data.to_analysis_context()
        print(f"Profile context length: {len(context)} characters")
        
        skill_assessment = await service._generate_skill_assessment(context)
        
        print("✅ Skill Assessment Generated:")
        print(f"   Technical Skills: {list(skill_assessment.technical_skills.keys())[:5]}...")
        print(f"   Soft Skills: {list(skill_assessment.soft_skills.keys())[:3]}...")
        print(f"   Top Strengths: {skill_assessment.skill_strengths[:3]}")
        print(f"   Skill Gaps: {skill_assessment.skill_gaps[:3]}")
        print(f"   Market Relevance: {skill_assessment.market_relevance_score:.2f}")
        print(f"   Confidence: {skill_assessment.confidence_score:.2f}")
        
        return skill_assessment
        
    except Exception as e:
        print(f"❌ Skill assessment failed: {str(e)}")
        print("   Using fallback assessment...")
        return service._fallback_skill_assessment()


async def test_career_recommendations(service, profile_data, skill_assessment):
    """Test career recommendations generation"""
    print("\n=== Testing Career Recommendations ===")
    
    try:
        context = profile_data.to_analysis_context()
        recommendations = await service._generate_career_recommendations(context, skill_assessment)
        
        print("✅ Career Recommendations Generated:")
        for i, rec in enumerate(recommendations[:3], 1):
            print(f"   {i}. {rec.recommended_role} (Match: {rec.match_score:.2f})")
            print(f"      Reasoning: {rec.reasoning[:100]}...")
            print(f"      Timeline: {rec.preparation_timeline}")
            print(f"      Salary: {rec.salary_range}")
            print()
        
        return recommendations
        
    except Exception as e:
        print(f"❌ Career recommendations failed: {str(e)}")
        print("   Using fallback recommendations...")
        return service._fallback_career_recommendations()


async def test_learning_paths(service, profile_data, skill_assessment):
    """Test learning paths generation"""
    print("\n=== Testing Learning Paths ===")
    
    try:
        context = profile_data.to_analysis_context()
        learning_paths = await service._generate_learning_paths(context, skill_assessment)
        
        print("✅ Learning Paths Generated:")
        for i, path in enumerate(learning_paths[:2], 1):
            print(f"   {i}. {path.title}")
            print(f"      Description: {path.description[:100]}...")
            print(f"      Target Skills: {', '.join(path.target_skills[:3])}...")
            print(f"      Duration: {path.estimated_duration}")
            print(f"      Difficulty: {path.difficulty_level}")
            print(f"      Modules: {len(path.learning_modules)}")
            print()
        
        return learning_paths
        
    except Exception as e:
        print(f"❌ Learning paths failed: {str(e)}")
        print("   Using fallback learning paths...")
        return service._fallback_learning_paths()


async def test_project_suggestions(service, profile_data, skill_assessment):
    """Test project suggestions generation"""
    print("\n=== Testing Project Suggestions ===")
    
    try:
        context = profile_data.to_analysis_context()
        project_suggestions = await service._generate_project_suggestions(context, skill_assessment)
        
        print("✅ Project Suggestions Generated:")
        for i, proj in enumerate(project_suggestions[:3], 1):
            print(f"   {i}. {proj.title}")
            print(f"      Description: {proj.description[:100]}...")
            print(f"      Technologies: {', '.join(proj.technologies[:4])}...")
            print(f"      Difficulty: {proj.difficulty_level}")
            print(f"      Duration: {proj.estimated_duration}")
            print(f"      Portfolio Value: {proj.portfolio_value[:80]}...")
            print()
        
        return project_suggestions
        
    except Exception as e:
        print(f"❌ Project suggestions failed: {str(e)}")
        print("   Using fallback project suggestions...")
        return service._fallback_project_suggestions()


async def test_market_insights(service, profile_data):
    """Test market insights generation"""
    print("\n=== Testing Market Insights ===")
    
    try:
        context = profile_data.to_analysis_context()
        market_insights = await service._generate_market_insights(context)
        
        print("✅ Market Insights Generated:")
        print(f"   Industry Trends: {', '.join(market_insights.get('industry_trends', [])[:3])}...")
        print(f"   In-Demand Skills: {', '.join(market_insights.get('in_demand_skills', [])[:5])}...")
        print(f"   Emerging Technologies: {', '.join(market_insights.get('emerging_technologies', [])[:3])}...")
        print(f"   Market Forecast: {market_insights.get('market_forecast', 'N/A')[:100]}...")
        print(f"   Actionable Insights: {len(market_insights.get('actionable_insights', []))} insights")
        
        return market_insights
        
    except Exception as e:
        print(f"❌ Market insights failed: {str(e)}")
        print("   Using fallback market insights...")
        return service._fallback_market_insights()


async def test_complete_analysis(service, profile_data):
    """Test complete analysis workflow"""
    print("\n=== Testing Complete Analysis Workflow ===")
    
    try:
        # Mock database
        mock_db = AsyncMock()
        
        # Mock the profile data aggregation
        service._aggregate_profile_data = AsyncMock(return_value=profile_data)
        service._store_analysis_results = AsyncMock()
        
        # Run complete analysis
        analysis = await service.analyze_complete_profile("demo-user-123", mock_db)
        
        print("✅ Complete Analysis Completed:")
        print(f"   User ID: {analysis.user_id}")
        print(f"   Analysis Timestamp: {analysis.analysis_timestamp}")
        print(f"   Skill Assessment Confidence: {analysis.skill_assessment.confidence_score:.2f}")
        print(f"   Career Recommendations: {len(analysis.career_recommendations)}")
        print(f"   Learning Paths: {len(analysis.learning_paths)}")
        print(f"   Project Suggestions: {len(analysis.project_suggestions)}")
        print(f"   Market Insights Available: {'Yes' if analysis.market_insights else 'No'}")
        
        return analysis
        
    except Exception as e:
        print(f"❌ Complete analysis failed: {str(e)}")
        return None


async def test_caching_functionality(service):
    """Test analysis caching functionality"""
    print("\n=== Testing Caching Functionality ===")
    
    try:
        mock_db = AsyncMock()
        
        # Mock cached analysis result
        mock_result = AsyncMock()
        mock_result.scalar_one_or_none.return_value = AsyncMock()
        mock_result.scalar_one_or_none.return_value.result_data = {
            "technical_skills": {"Python": 0.8, "JavaScript": 0.7},
            "confidence_score": 0.9
        }
        mock_db.execute.return_value = mock_result
        
        # Test getting cached analysis
        cached_data = await service.get_cached_analysis("demo-user-123", AnalysisType.SKILL_ASSESSMENT, mock_db)
        
        if cached_data:
            print("✅ Cached analysis retrieved successfully")
            print(f"   Cached data keys: {list(cached_data.keys())}")
        else:
            print("ℹ️  No cached analysis found (expected for demo)")
        
        # Test staleness check
        is_stale = await service.is_analysis_stale("demo-user-123", AnalysisType.SKILL_ASSESSMENT, mock_db)
        print(f"   Analysis is stale: {is_stale}")
        
    except Exception as e:
        print(f"❌ Caching test failed: {str(e)}")


async def main():
    """Main demo function"""
    print("🚀 AI Analysis Service Demo")
    print("=" * 50)
    
    # Test Gemini API client
    gemini_available = await test_gemini_client()
    
    # Create sample profile data
    profile_data = await create_sample_profile_data()
    
    # Initialize AI analysis service
    service = AIAnalysisService()
    
    # Test individual analysis components
    skill_assessment = await test_skill_assessment(service, profile_data)
    career_recommendations = await test_career_recommendations(service, profile_data, skill_assessment)
    learning_paths = await test_learning_paths(service, profile_data, skill_assessment)
    project_suggestions = await test_project_suggestions(service, profile_data, skill_assessment)
    market_insights = await test_market_insights(service, profile_data)
    
    # Test complete analysis workflow
    complete_analysis = await test_complete_analysis(service, profile_data)
    
    # Test caching functionality
    await test_caching_functionality(service)
    
    print("\n" + "=" * 50)
    print("✅ AI Analysis Service Demo Completed!")
    
    if not gemini_available:
        print("\n⚠️  Note: Gemini API was not available, so fallback methods were used.")
        print("   To test with actual AI analysis, set the GEMINI_API_KEY environment variable.")
    
    print("\n📊 Summary:")
    print(f"   - Skill Assessment: {'✅' if skill_assessment else '❌'}")
    print(f"   - Career Recommendations: {'✅' if career_recommendations else '❌'}")
    print(f"   - Learning Paths: {'✅' if learning_paths else '❌'}")
    print(f"   - Project Suggestions: {'✅' if project_suggestions else '❌'}")
    print(f"   - Market Insights: {'✅' if market_insights else '❌'}")
    print(f"   - Complete Analysis: {'✅' if complete_analysis else '❌'}")


if __name__ == "__main__":
    asyncio.run(main())