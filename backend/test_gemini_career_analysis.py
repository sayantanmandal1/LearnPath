#!/usr/bin/env python3
"""
Simple test script to demonstrate Gemini API integration for career analysis
"""
import asyncio
import os
import sys
import json
from pathlib import Path

# Add the backend directory to Python path
backend_dir = Path(__file__).parent
if str(backend_dir) not in sys.path:
    sys.path.append(str(backend_dir))

from app.core.config import settings
from app.services.resume_processing_service import ResumeProcessingService


async def test_gemini_career_analysis():
    """Test Gemini API integration for career analysis"""
    print("🚀 Testing Gemini API Integration for Career Analysis")
    print("=" * 60)
    
    # Check if Gemini API key is configured
    print(f"Gemini API Key configured: {'✅ Yes' if settings.GEMINI_API_KEY else '❌ No'}")
    if settings.GEMINI_API_KEY:
        print(f"API Key (first 10 chars): {settings.GEMINI_API_KEY[:10]}...")
    
    print("\n📝 Testing Resume Processing with Gemini Analysis")
    print("-" * 50)
    
    # Create a sample resume text for analysis
    sample_resume_text = """
    John Doe
    Backend Developer
    Email: john.doe@email.com
    Phone: +1-234-567-8900
    
    EXPERIENCE:
    • 3 years as Backend Developer at Tech Corp (2021-2024)
    • Developed REST APIs using Python and FastAPI
    • Worked with PostgreSQL and Redis for data storage
    • Implemented authentication systems using JWT
    • Experience with Docker and AWS deployment
    
    SKILLS:
    • Programming: Python, SQL, JavaScript
    • Frameworks: FastAPI, Django, React
    • Databases: PostgreSQL, MongoDB, Redis
    • Cloud: AWS, Docker, Kubernetes
    • Tools: Git, pytest, Linux
    
    EDUCATION:
    • Bachelor's in Computer Science (2017-2021)
    • Relevant coursework: Data Structures, Algorithms, Database Systems
    
    PROJECTS:
    • E-commerce API: Built scalable REST API serving 10k+ users
    • Authentication Service: JWT-based auth system with refresh tokens
    • Data Pipeline: ETL pipeline processing 1M+ records daily
    """
    
    try:
        # Initialize the resume processing service
        resume_service = ResumeProcessingService()
        
        print("🔍 Analyzing resume with Gemini API...")
        
        # Process the resume text with Gemini
        parsed_data = await resume_service._parse_with_gemini(sample_resume_text)
        
        print("\n✅ Gemini Analysis Results:")
        print("-" * 30)
        print(f"📧 Email: {parsed_data.email}")
        print(f"📞 Phone: {parsed_data.phone}")
        print(f"💼 Current Role: {parsed_data.current_role}")
        print(f"🏢 Company: {parsed_data.current_company}")
        print(f"📅 Experience Years: {parsed_data.experience_years}")
        
        print(f"\n🛠️ Skills ({len(parsed_data.skills)}):")
        for skill in parsed_data.skills[:10]:  # Show first 10 skills
            print(f"  • {skill.name} (Level: {skill.proficiency_level})")
        if len(parsed_data.skills) > 10:
            print(f"  ... and {len(parsed_data.skills) - 10} more skills")
        
        print(f"\n💼 Work Experience ({len(parsed_data.work_experience)}):")
        for exp in parsed_data.work_experience[:3]:  # Show first 3 experiences
            print(f"  • {exp.title} at {exp.company}")
            print(f"    Duration: {exp.start_date} - {exp.end_date}")
            print(f"    Description: {exp.description[:100]}...")
        
        print(f"\n🎓 Education ({len(parsed_data.education)}):")
        for edu in parsed_data.education:
            print(f"  • {edu.degree} - {edu.institution}")
            if edu.graduation_date:
                print(f"    Graduated: {edu.graduation_date}")
        
        print(f"\n🚀 Projects ({len(parsed_data.projects)}):")
        for project in parsed_data.projects[:3]:  # Show first 3 projects
            print(f"  • {project.name}")
            if project.description:
                print(f"    {project.description[:100]}...")
        
        print("\n🎯 Career Trajectory Analysis:")
        print("-" * 35)
        
        # Generate career recommendations based on the parsed data
        career_recommendations = await generate_career_recommendations(parsed_data)
        
        for i, rec in enumerate(career_recommendations[:5], 1):
            print(f"{i}. {rec['title']}")
            print(f"   Match Score: {rec['match_score']}/100")
            print(f"   Reasoning: {rec['reasoning'][:120]}...")
            print()
        
    except Exception as e:
        print(f"❌ Error during Gemini analysis: {str(e)}")
        print("This might be due to:")
        print("  • Missing or invalid Gemini API key")
        print("  • Network connectivity issues")
        print("  • API rate limiting")
        print("\nFalling back to basic analysis...")
        
        # Show fallback analysis
        basic_analysis = analyze_resume_basic(sample_resume_text)
        print("\n📊 Basic Analysis Results:")
        print("-" * 30)
        for key, value in basic_analysis.items():
            print(f"{key}: {value}")


async def generate_career_recommendations(parsed_data):
    """Generate career recommendations based on parsed resume data"""
    # This would normally use the Gemini API for advanced recommendations
    # For now, we'll provide some sample recommendations based on the skills
    
    skills = [skill.name.lower() for skill in parsed_data.skills]
    
    recommendations = []
    
    if any(skill in skills for skill in ['python', 'fastapi', 'api']):
        recommendations.append({
            'title': 'Senior Backend Developer',
            'match_score': 92,
            'reasoning': 'Strong match with Python and FastAPI experience. Your API development skills and database knowledge make you an excellent candidate for senior backend roles.'
        })
    
    if any(skill in skills for skill in ['aws', 'docker', 'kubernetes']):
        recommendations.append({
            'title': 'DevOps Engineer',
            'match_score': 85,
            'reasoning': 'Your cloud experience with AWS and containerization skills with Docker align well with DevOps responsibilities. Consider expanding CI/CD pipeline knowledge.'
        })
    
    if any(skill in skills for skill in ['python', 'sql', 'data']):
        recommendations.append({
            'title': 'Data Engineer',
            'match_score': 78,
            'reasoning': 'Your Python skills and database experience provide a solid foundation for data engineering. Consider learning more about big data technologies like Spark or Airflow.'
        })
    
    recommendations.append({
        'title': 'Full Stack Developer',
        'match_score': 75,
        'reasoning': 'With backend expertise and some frontend knowledge, you could transition to full-stack development. Consider strengthening React skills and learning modern frontend frameworks.'
    })
    
    recommendations.append({
        'title': 'Software Architect',
        'match_score': 70,
        'reasoning': 'Your diverse technical skills and experience with scalable systems position you well for architecture roles. Focus on system design and leadership skills.'
    })
    
    return recommendations


def analyze_resume_basic(resume_text):
    """Basic resume analysis without Gemini API"""
    text_lower = resume_text.lower()
    
    # Count skill mentions
    tech_skills = ['python', 'javascript', 'sql', 'react', 'fastapi', 'docker', 'aws', 'git']
    skill_counts = {skill: text_lower.count(skill) for skill in tech_skills if skill in text_lower}
    
    # Extract basic info
    lines = resume_text.split('\n')
    email_line = [line for line in lines if '@' in line]
    phone_line = [line for line in lines if 'phone' in line.lower()]
    
    return {
        'detected_skills': list(skill_counts.keys()),
        'primary_language': 'Python' if 'python' in skill_counts else 'Unknown',
        'experience_indicators': skill_counts,
        'email_found': len(email_line) > 0,
        'phone_found': len(phone_line) > 0,
        'total_lines': len(lines),
        'estimated_experience': 'Mid-level' if len(skill_counts) > 5 else 'Junior'
    }


if __name__ == "__main__":
    print("🎯 Career Analysis with Gemini API Integration")
    print("=" * 60)
    
    asyncio.run(test_gemini_career_analysis())
    
    print("\n" + "=" * 60)
    print("✅ Career Analysis Test Complete!")
    print("\n💡 To test with your own resume:")
    print("  1. Save your resume as a text file")
    print("  2. Modify the sample_resume_text variable")
    print("  3. Run this script again")
    print("\n🔗 API Documentation: http://127.0.0.1:8000/api/v1/docs")