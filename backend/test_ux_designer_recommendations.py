#!/usr/bin/env python3
"""
Test UX Designer Career Recommendations
Demonstrates how the system provides recommendations for any job profile
"""
import asyncio
import sys
from pathlib import Path

# Add the backend directory to Python path
backend_dir = Path(__file__).parent
if str(backend_dir) not in sys.path:
    sys.path.append(str(backend_dir))

from advanced_career_analysis_demo import GeminiCareerAnalyzer


async def test_ux_designer_recommendations():
    """Test career recommendations specifically for UX Designer profile"""
    
    print("🎨 UX Designer Career Recommendations Test")
    print("=" * 60)
    
    # UX Designer profile examples
    ux_profiles = [
        {
            "name": "Junior UX Designer",
            "current_role": "UX Designer",
            "experience_years": 1,
            "industry": "Technology",
            "location": "San Francisco",
            "career_goals": "Become a Senior UX Designer and specialize in mobile app design",
            "skills": ["Figma", "Adobe XD", "User Research", "Wireframing", "Prototyping", "Usability Testing", "HTML", "CSS", "JavaScript", "Design Systems"]
        },
        {
            "name": "Experienced UX Designer", 
            "current_role": "Senior UX Designer",
            "experience_years": 4,
            "industry": "E-commerce",
            "location": "Remote",
            "career_goals": "Transition to UX Research or Product Design leadership",
            "skills": ["User Research", "Data Analysis", "A/B Testing", "Product Strategy", "Design Thinking", "Stakeholder Management", "Figma", "Sketch", "InVision", "UserTesting", "Analytics"]
        },
        {
            "name": "Career Changer to UX",
            "current_role": "Graphic Designer", 
            "experience_years": 3,
            "industry": "Marketing",
            "location": "New York",
            "career_goals": "Successfully transition from graphic design to UX design",
            "skills": ["Adobe Creative Suite", "Visual Design", "Typography", "Branding", "Print Design", "Basic HTML/CSS", "User Psychology", "Empathy"]
        }
    ]
    
    analyzer = GeminiCareerAnalyzer()
    
    for i, profile in enumerate(ux_profiles, 1):
        print(f"\n🔍 Analysis {i}: {profile['name']}")
        print("-" * 50)
        print(f"📋 Profile Overview:")
        print(f"  Current Role: {profile['current_role']}")
        print(f"  Experience: {profile['experience_years']} years")
        print(f"  Location: {profile['location']}")
        print(f"  Goal: {profile['career_goals']}")
        print(f"  Key Skills: {', '.join(profile['skills'][:5])}")
        
        # Get career analysis
        analysis = await analyzer.analyze_career_path(profile)
        
        # Display results
        print(f"\n📊 Career Assessment:")
        career_assessment = analysis.get('career_assessment', {})
        if 'current_level' in career_assessment:
            print(f"  Current Level: {career_assessment['current_level']}")
        if 'strengths' in career_assessment:
            print(f"  Key Strengths: {', '.join(career_assessment['strengths'][:4])}")
        
        print(f"\n🚀 Recommended Career Paths:")
        paths = analysis.get('recommended_paths', [])
        for j, path in enumerate(paths[:4], 1):
            print(f"  {j}. {path.get('title', 'Career Path')}")
            print(f"     Timeline: {path.get('timeline', 'TBD')}")
            print(f"     Success Probability: {path.get('probability', 'Medium')}")
            if 'reasoning' in path:
                print(f"     Why: {path['reasoning'][:80]}...")
        
        print(f"\n📚 Skill Development Plan:")
        skill_dev = analysis.get('skill_development', {})
        if 'immediate_priorities' in skill_dev:
            priorities = skill_dev['immediate_priorities'][:3]
            print(f"  🎯 Focus Now: {', '.join(priorities)}")
        if 'medium_term_goals' in skill_dev:
            medium_goals = skill_dev.get('medium_term_goals', [])[:3]
            print(f"  📈 6-12 Months: {', '.join(medium_goals) if medium_goals else 'Advanced UX skills'}")
        if 'learning_resources' in skill_dev:
            resources = skill_dev['learning_resources'][:2]
            print(f"  📖 Recommended: {', '.join(resources)}")
        
        # Custom UX-specific recommendations
        ux_specific = generate_ux_specific_recommendations(profile)
        print(f"\n🎨 UX-Specific Insights:")
        for insight in ux_specific[:3]:
            print(f"  • {insight}")
        
        print(f"\n" + "="*50)


def generate_ux_specific_recommendations(profile):
    """Generate UX design specific career recommendations"""
    skills = [skill.lower() for skill in profile.get('skills', [])]
    experience = profile.get('experience_years', 0)
    current_role = profile.get('current_role', '').lower()
    
    recommendations = []
    
    # Experience-based recommendations
    if experience < 2:
        recommendations.extend([
            "Focus on building a strong portfolio with 3-5 case studies showing your design process",
            "Practice user research methods like interviews, surveys, and usability testing",
            "Master industry-standard tools: Figma for design, Miro for collaboration"
        ])
    elif experience < 5:
        recommendations.extend([
            "Develop expertise in a specialized area: mobile design, accessibility, or design systems",
            "Build leadership skills by mentoring junior designers or leading design projects",
            "Learn business skills: understanding metrics, ROI, and how design impacts business goals"
        ])
    else:
        recommendations.extend([
            "Consider specializing in UX strategy, service design, or design operations",
            "Develop presentation skills to communicate design value to executives",
            "Explore adjacent roles: Product Management, Design Research, or Design Leadership"
        ])
    
    # Skill-based recommendations  
    if any(skill in skills for skill in ['research', 'user research', 'testing']):
        recommendations.append("Your research skills could lead to specialized UX Research roles with higher salaries")
    
    if any(skill in skills for skill in ['html', 'css', 'javascript', 'code']):
        recommendations.append("Your coding skills make you valuable for design-to-development handoffs and prototyping")
    
    if any(skill in skills for skill in ['data', 'analytics', 'metrics']):
        recommendations.append("Data skills position you well for growth-focused UX roles and product strategy")
    
    # Role transition recommendations
    if 'graphic' in current_role:
        recommendations.extend([
            "Emphasize your visual design skills while learning interaction design and user research",
            "Take online courses in UX fundamentals: HCI, design thinking, and usability principles",
            "Start redesigning existing apps/websites to build UX portfolio pieces"
        ])
    
    return recommendations[:6]  # Return top 6 recommendations


def show_website_integration_example():
    """Show how this integrates with the website frontend"""
    print("\n🌐 Website Integration Example")
    print("=" * 60)
    print("When a user enters 'UX Designer' on your website:")
    print()
    print("1. 📝 User fills out profile form:")
    print("   - Job Title: 'UX Designer'")
    print("   - Experience: '2 years'") 
    print("   - Skills: 'Figma, User Research, Prototyping'")
    print("   - Location: 'Remote'")
    print()
    print("2. 🔄 Frontend calls backend API:")
    print("   - POST /api/v1/career-trajectory/trajectories")
    print("   - POST /api/v1/recommendations/skill-gaps/ux-designer")
    print("   - POST /api/v1/recommendations/jobs")
    print()
    print("3. 🧠 Backend processes with AI:")
    print("   - Gemini API analyzes profile")
    print("   - ML models match with job database")
    print("   - Skill gap analysis performed")
    print()
    print("4. 📊 Frontend displays results:")
    print("   - Career path recommendations")
    print("   - Skill development roadmap")
    print("   - Job matches with scores")
    print("   - Learning resource suggestions")
    print()
    print("🎯 Result: Personalized UX career guidance!")


if __name__ == "__main__":
    print("🎨 Testing UX Designer Career Recommendations")
    print("Testing if your website can provide career guidance for UX Designer profiles...")
    print()
    
    asyncio.run(test_ux_designer_recommendations())
    
    show_website_integration_example()
    
    print("\n✅ UX Designer Recommendations Test Complete!")
    print("\n💡 Your website can now provide career recommendations for:")
    print("  • UX/UI Designers at any experience level")
    print("  • Career changers transitioning to UX")
    print("  • UX professionals looking to specialize")
    print("  • Any other job profile you input!")
    print("\n🚀 Ready to test on your website at http://localhost:3000")