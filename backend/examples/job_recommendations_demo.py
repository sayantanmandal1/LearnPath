"""
Demo script for job recommendation functionality
"""
import asyncio
from datetime import datetime
from typing import Dict, Any

from app.services.recommendation_service import RecommendationService


class MockUserProfile:
    """Mock user profile for demo purposes."""
    
    def __init__(self):
        self.user_id = "demo-user-123"
        self.current_role = "Software Developer"
        self.dream_job = "Senior Software Engineer"
        self.experience_years = 3
        self.location = "San Francisco"
        self.remote_work_preference = True
        self.skills = {
            "python": 0.8,
            "javascript": 0.7,
            "react": 0.6,
            "sql": 0.5,
            "git": 0.9
        }


class MockJobPosting:
    """Mock job posting for demo purposes."""
    
    def __init__(self, job_id: str, title: str, company: str, skills: Dict[str, float], **kwargs):
        self.id = job_id
        self.title = title
        self.company = company
        self.location = kwargs.get('location', 'San Francisco')
        self.remote_type = kwargs.get('remote_type', 'remote')
        self.employment_type = kwargs.get('employment_type', 'full-time')
        self.experience_level = kwargs.get('experience_level', 'mid')
        self.salary_min = kwargs.get('salary_min', 100000)
        self.salary_max = kwargs.get('salary_max', 130000)
        self.salary_currency = kwargs.get('salary_currency', 'USD')
        self.posted_date = datetime.now()
        self.source_url = f"https://example.com/{job_id}"
        self.description = kwargs.get('description', f"Great opportunity at {company}")
        self.processed_skills = skills


def create_sample_jobs():
    """Create sample job postings for demo."""
    jobs = [
        MockJobPosting(
            "job-1",
            "Senior Python Developer",
            "Tech Corp",
            {
                "python": 0.9,
                "javascript": 0.7,
                "react": 0.8,
                "sql": 0.6,
                "docker": 0.5
            },
            salary_min=120000,
            salary_max=150000,
            experience_level="senior"
        ),
        MockJobPosting(
            "job-2",
            "Full Stack Developer",
            "Startup Inc",
            {
                "javascript": 0.8,
                "react": 0.7,
                "node.js": 0.6,
                "mongodb": 0.5,
                "git": 0.8
            },
            salary_min=90000,
            salary_max=120000,
            experience_level="mid"
        ),
        MockJobPosting(
            "job-3",
            "Data Scientist",
            "Analytics Co",
            {
                "python": 0.8,
                "machine learning": 0.9,
                "statistics": 0.8,
                "r": 0.6,
                "tensorflow": 0.7
            },
            salary_min=130000,
            salary_max=160000,
            experience_level="senior",
            location="New York",
            remote_type="onsite"
        ),
        MockJobPosting(
            "job-4",
            "Frontend Developer",
            "Design Studio",
            {
                "javascript": 0.9,
                "react": 0.9,
                "css": 0.8,
                "html": 0.8,
                "typescript": 0.6
            },
            salary_min=85000,
            salary_max=110000,
            experience_level="mid"
        ),
        MockJobPosting(
            "job-5",
            "DevOps Engineer",
            "Cloud Company",
            {
                "docker": 0.9,
                "kubernetes": 0.8,
                "aws": 0.7,
                "python": 0.6,
                "linux": 0.8
            },
            salary_min=110000,
            salary_max=140000,
            experience_level="senior"
        )
    ]
    return jobs


async def demo_job_recommendations():
    """Demonstrate job recommendation functionality."""
    print("🚀 Job Recommendation System Demo")
    print("=" * 50)
    
    # Initialize recommendation service
    recommendation_service = RecommendationService()
    
    # Create mock user profile
    user_profile = MockUserProfile()
    print(f"\n👤 User Profile:")
    print(f"   Role: {user_profile.current_role}")
    print(f"   Dream Job: {user_profile.dream_job}")
    print(f"   Experience: {user_profile.experience_years} years")
    print(f"   Location: {user_profile.location}")
    print(f"   Remote Preference: {user_profile.remote_work_preference}")
    print(f"   Skills: {', '.join(user_profile.skills.keys())}")
    
    # Create sample jobs
    sample_jobs = create_sample_jobs()
    print(f"\n💼 Available Jobs: {len(sample_jobs)}")
    for job in sample_jobs:
        print(f"   • {job.title} at {job.company}")
    
    print(f"\n🔍 Calculating Match Scores...")
    print("-" * 30)
    
    # Calculate match scores for each job
    job_matches = []
    user_skills = user_profile.skills
    
    for job in sample_jobs:
        job_skills = job.processed_skills
        match_score = recommendation_service._calculate_simple_match_score(user_skills, job_skills)
        
        # Create match data
        match_data = {
            'job_id': job.id,
            'job_title': job.title,
            'company': job.company,
            'location': job.location,
            'remote_type': job.remote_type,
            'experience_level': job.experience_level,
            'match_score': round(match_score, 3),
            'match_percentage': round(match_score * 100, 1),
            'salary_min': job.salary_min,
            'salary_max': job.salary_max,
            'salary_currency': job.salary_currency,
            'required_skills': list(job_skills.keys()),
            'skill_overlap': list(set(user_skills.keys()).intersection(set(job_skills.keys())))
        }
        
        job_matches.append(match_data)
    
    # Sort by match score
    job_matches.sort(key=lambda x: x['match_score'], reverse=True)
    
    # Display results
    print(f"\n📊 Job Recommendations (sorted by match score):")
    print("=" * 60)
    
    for i, match in enumerate(job_matches, 1):
        print(f"\n{i}. {match['job_title']} at {match['company']}")
        print(f"   Match Score: {match['match_percentage']}%")
        print(f"   Location: {match['location']} ({match['remote_type']})")
        print(f"   Experience: {match['experience_level']}")
        print(f"   Salary: ${match['salary_min']:,} - ${match['salary_max']:,}")
        print(f"   Required Skills: {', '.join(match['required_skills'])}")
        print(f"   Your Matching Skills: {', '.join(match['skill_overlap'])}")
        
        # Calculate skill gaps
        user_skill_set = set(user_skills.keys())
        job_skill_set = set(match['required_skills'])
        missing_skills = job_skill_set - user_skill_set
        
        if missing_skills:
            print(f"   Skills to Develop: {', '.join(missing_skills)}")
    
    # Test preference-based ranking
    print(f"\n🎯 Applying User Preferences...")
    print("-" * 30)
    
    ranked_matches = recommendation_service._rank_job_recommendations(job_matches, user_profile)
    
    print(f"\n📈 Re-ranked Recommendations (with preference boost):")
    print("=" * 60)
    
    for i, match in enumerate(ranked_matches[:3], 1):  # Show top 3
        preference_boost = match.get('preference_boost', 0)
        print(f"\n{i}. {match['job_title']} at {match['company']}")
        print(f"   Final Match Score: {match['match_percentage']}%")
        print(f"   Preference Boost: +{preference_boost:.1%}")
        print(f"   Location: {match['location']} ({match['remote_type']})")
        
        # Explain why this job got a boost
        reasons = []
        if user_profile.dream_job and user_profile.dream_job.lower() in match['job_title'].lower():
            reasons.append("matches dream job")
        if user_profile.location and user_profile.location.lower() in match['location'].lower():
            reasons.append("matches preferred location")
        if user_profile.remote_work_preference and match['remote_type'] == 'remote':
            reasons.append("offers remote work")
        
        if reasons:
            print(f"   Boost Reasons: {', '.join(reasons)}")
    
    print(f"\n✅ Demo completed successfully!")
    print(f"   • Calculated match scores for {len(sample_jobs)} jobs")
    print(f"   • Applied user preference ranking")
    print(f"   • Identified skill gaps and development opportunities")


def demo_match_score_calculation():
    """Demonstrate match score calculation with different scenarios."""
    print("\n🧮 Match Score Calculation Demo")
    print("=" * 40)
    
    recommendation_service = RecommendationService()
    
    # Test scenarios
    scenarios = [
        {
            "name": "Perfect Match",
            "user_skills": {"python": 0.8, "javascript": 0.7, "react": 0.6},
            "job_skills": {"python": 0.9, "javascript": 0.8, "react": 0.7}
        },
        {
            "name": "Partial Match",
            "user_skills": {"python": 0.8, "javascript": 0.7, "react": 0.6},
            "job_skills": {"python": 0.9, "java": 0.8, "spring": 0.7}
        },
        {
            "name": "No Match",
            "user_skills": {"python": 0.8, "javascript": 0.7},
            "job_skills": {"java": 0.9, "c++": 0.8}
        },
        {
            "name": "Superset Match",
            "user_skills": {"python": 0.8, "javascript": 0.7, "react": 0.6, "sql": 0.5},
            "job_skills": {"python": 0.9, "javascript": 0.8}
        },
        {
            "name": "Empty Skills",
            "user_skills": {},
            "job_skills": {"python": 0.9, "javascript": 0.8}
        }
    ]
    
    for scenario in scenarios:
        user_skills = scenario["user_skills"]
        job_skills = scenario["job_skills"]
        
        match_score = recommendation_service._calculate_simple_match_score(user_skills, job_skills)
        
        print(f"\n📋 {scenario['name']}:")
        print(f"   User Skills: {list(user_skills.keys()) if user_skills else 'None'}")
        print(f"   Job Skills: {list(job_skills.keys()) if job_skills else 'None'}")
        print(f"   Match Score: {match_score:.3f} ({match_score * 100:.1f}%)")
        
        # Show overlap
        if user_skills and job_skills:
            overlap = set(user_skills.keys()).intersection(set(job_skills.keys()))
            print(f"   Skill Overlap: {list(overlap) if overlap else 'None'}")


if __name__ == "__main__":
    print("🎯 Job Recommendation System - Demo")
    print("This demo showcases the job matching algorithm functionality")
    print()
    
    # Run match score calculation demo
    demo_match_score_calculation()
    
    # Run main job recommendation demo
    asyncio.run(demo_job_recommendations())