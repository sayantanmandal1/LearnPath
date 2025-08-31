"""
Demo script for career trajectory recommendation service.

This script demonstrates the key features of the career trajectory service:
1. Career path matching using semantic similarity
2. Career progression modeling
3. Dream job optimization
4. Alternative career route discovery
5. Market demand integration
6. Confidence scoring and detailed reasoning
"""

import asyncio
import sys
import os
from datetime import datetime
from typing import Dict, Any

# Add the backend directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.career_trajectory_service import CareerTrajectoryService
from app.models.profile import UserProfile
from app.models.job import JobPosting
from app.core.database import get_db
from unittest.mock import Mock, AsyncMock


def create_mock_user_profile() -> UserProfile:
    """Create a mock user profile for demonstration."""
    profile = Mock(spec=UserProfile)
    profile.user_id = "demo-user-123"
    profile.dream_job = "Senior Data Scientist"
    profile.current_role = "Data Analyst"
    profile.experience_years = 2
    profile.location = "New York, NY"
    profile.skills = {
        "python": 0.8,
        "sql": 0.9,
        "pandas": 0.7,
        "numpy": 0.6,
        "matplotlib": 0.5,
        "statistics": 0.6,
        "excel": 0.9,
        "tableau": 0.7,
        "machine learning": 0.3,  # Weak skill
        "deep learning": 0.1,     # Very weak skill
        "tensorflow": 0.0,        # Missing skill
        "pytorch": 0.0,           # Missing skill
        "scikit-learn": 0.2       # Weak skill
    }
    return profile


def create_mock_job_postings():
    """Create mock job postings for demonstration."""
    jobs = []
    
    # Senior Data Scientist job
    job1 = Mock(spec=JobPosting)
    job1.id = "job-ds-senior"
    job1.title = "Senior Data Scientist"
    job1.company = "TechCorp AI"
    job1.description = "Senior data scientist role requiring Python, machine learning, and deep learning expertise"
    job1.processed_skills = {
        "python": 0.9,
        "machine learning": 0.9,
        "deep learning": 0.8,
        "tensorflow": 0.7,
        "pytorch": 0.6,
        "scikit-learn": 0.8,
        "statistics": 0.8,
        "sql": 0.7,
        "pandas": 0.7,
        "numpy": 0.6
    }
    job1.salary_min = 130000
    job1.salary_max = 180000
    job1.posted_date = datetime.utcnow()
    job1.is_active = True
    jobs.append(job1)
    
    # Machine Learning Engineer job
    job2 = Mock(spec=JobPosting)
    job2.id = "job-mle"
    job2.title = "Machine Learning Engineer"
    job2.company = "AI Innovations"
    job2.description = "ML engineer role focusing on production ML systems"
    job2.processed_skills = {
        "python": 0.9,
        "machine learning": 0.9,
        "tensorflow": 0.8,
        "pytorch": 0.7,
        "docker": 0.7,
        "kubernetes": 0.6,
        "mlops": 0.8,
        "aws": 0.6,
        "sql": 0.6
    }
    job2.salary_min = 120000
    job2.salary_max = 170000
    job2.posted_date = datetime.utcnow()
    job2.is_active = True
    jobs.append(job2)
    
    # Data Scientist (Mid-level) job
    job3 = Mock(spec=JobPosting)
    job3.id = "job-ds-mid"
    job3.title = "Data Scientist"
    job3.company = "Analytics Pro"
    job3.description = "Data scientist role with focus on statistical analysis and modeling"
    job3.processed_skills = {
        "python": 0.8,
        "machine learning": 0.7,
        "statistics": 0.9,
        "sql": 0.8,
        "pandas": 0.8,
        "numpy": 0.7,
        "scikit-learn": 0.7,
        "r": 0.6,
        "tableau": 0.5
    }
    job3.salary_min = 95000
    job3.salary_max = 130000
    job3.posted_date = datetime.utcnow()
    job3.is_active = True
    jobs.append(job3)
    
    return jobs


async def demonstrate_career_trajectory_service():
    """Demonstrate the career trajectory service functionality."""
    print("🚀 Career Trajectory Recommendation Service Demo")
    print("=" * 60)
    
    # Initialize service
    service = CareerTrajectoryService()
    
    # Create mock data
    user_profile = create_mock_user_profile()
    job_postings = create_mock_job_postings()
    
    print(f"\n👤 User Profile:")
    print(f"   Current Role: {user_profile.current_role}")
    print(f"   Dream Job: {user_profile.dream_job}")
    print(f"   Experience: {user_profile.experience_years} years")
    print(f"   Location: {user_profile.location}")
    print(f"   Skills: {len(user_profile.skills)} skills")
    
    # Display top skills
    top_skills = sorted(user_profile.skills.items(), key=lambda x: x[1], reverse=True)[:5]
    print(f"   Top Skills: {', '.join([f'{skill} ({score:.1f})' for skill, score in top_skills])}")
    
    print(f"\n📊 Available Job Market:")
    for job in job_postings:
        print(f"   • {job.title} at {job.company}")
        print(f"     Salary: ${job.salary_min:,} - ${job.salary_max:,}")
        print(f"     Key Skills: {', '.join(list(job.processed_skills.keys())[:5])}")
    
    # Mock database and repositories
    mock_db = AsyncMock()
    
    # Mock the service methods to use our test data
    def mock_get_by_user_id(db, user_id):
        return user_profile
    
    def mock_search_jobs(db, title=None, limit=None):
        if title and "senior" in title.lower():
            return [job for job in job_postings if "senior" in job.title.lower()]
        return job_postings
    
    def mock_get_recent_jobs(db, days=None, limit=None):
        return job_postings
    
    # Patch the service methods
    from unittest.mock import patch
    
    with patch('app.repositories.profile.ProfileRepository') as mock_profile_repo, \
         patch('app.repositories.job.JobRepository') as mock_job_repo, \
         patch.object(service, '_load_embedding_model') as mock_load_model, \
         patch.object(service, 'embedding_model') as mock_embedding_model:
        
        # Setup mocks
        mock_profile_repo.return_value.get_by_user_id = AsyncMock(return_value=user_profile)
        mock_job_repo.return_value.search_jobs = AsyncMock(side_effect=mock_search_jobs)
        mock_job_repo.return_value.get_recent_jobs = AsyncMock(side_effect=mock_get_recent_jobs)
        mock_job_repo.return_value.get_by_id = AsyncMock(return_value=job_postings[0])
        mock_load_model.return_value = None
        
        # Mock embedding model to return realistic similarity scores
        def mock_encode(texts):
            # Return different embeddings for different text types
            embeddings = []
            for text in texts:
                if "senior data scientist" in text.lower():
                    embeddings.append([0.8, 0.7, 0.9, 0.6, 0.5])  # High similarity to dream job
                elif "machine learning" in text.lower():
                    embeddings.append([0.6, 0.8, 0.7, 0.9, 0.4])  # Good ML similarity
                elif "data analyst" in text.lower():
                    embeddings.append([0.9, 0.5, 0.6, 0.3, 0.8])  # Current role similarity
                else:
                    embeddings.append([0.5, 0.5, 0.5, 0.5, 0.5])  # Default similarity
            return embeddings
        
        mock_embedding_model.encode = mock_encode
        
        print(f"\n🎯 Generating Career Trajectory Recommendations...")
        print("-" * 50)
        
        try:
            # Generate career trajectories
            trajectories = await service.get_career_trajectory_recommendations(
                user_id=user_profile.user_id,
                db=mock_db,
                n_recommendations=3,
                include_alternatives=True
            )
            
            print(f"\n✅ Generated {len(trajectories)} career trajectory recommendations:")
            
            for i, trajectory in enumerate(trajectories, 1):
                print(f"\n🛤️  Trajectory {i}: {trajectory.title}")
                print(f"   Target Role: {trajectory.target_role}")
                print(f"   Match Score: {trajectory.match_score:.2f}")
                print(f"   Confidence: {trajectory.confidence_score:.2f}")
                print(f"   Timeline: {trajectory.estimated_timeline_months} months")
                print(f"   Difficulty: {trajectory.difficulty_level}")
                print(f"   Market Demand: {trajectory.market_demand}")
                print(f"   Growth Potential: {trajectory.growth_potential:.2f}")
                
                print(f"\n   📈 Progression Steps:")
                for j, step in enumerate(trajectory.progression_steps, 1):
                    print(f"      {j}. {step['role']} ({step['duration_months']} months)")
                    print(f"         {step['description']}")
                
                print(f"\n   🎯 Required Skills:")
                print(f"      {', '.join(trajectory.required_skills[:5])}")
                
                if trajectory.skill_gaps:
                    print(f"\n   📚 Skill Gaps to Address:")
                    top_gaps = sorted(trajectory.skill_gaps.items(), key=lambda x: x[1], reverse=True)[:3]
                    for skill, importance in top_gaps:
                        print(f"      • {skill} (importance: {importance:.2f})")
                
                if trajectory.transferable_skills:
                    print(f"\n   ✅ Transferable Skills:")
                    print(f"      {', '.join(trajectory.transferable_skills[:5])}")
                
                print(f"\n   💡 Reasoning:")
                print(f"      {trajectory.reasoning}")
                
                if trajectory.success_factors:
                    print(f"\n   🔑 Success Factors:")
                    for factor in trajectory.success_factors[:3]:
                        print(f"      • {factor}")
                
                if trajectory.potential_challenges:
                    print(f"\n   ⚠️  Potential Challenges:")
                    for challenge in trajectory.potential_challenges[:2]:
                        print(f"      • {challenge}")
                
                if trajectory.alternative_routes:
                    print(f"\n   🔄 Alternative Routes:")
                    for alt in trajectory.alternative_routes[:2]:
                        print(f"      • {alt['approach']}: {alt['description']}")
            
            # Demonstrate skill gap analysis
            print(f"\n\n🔍 Skill Gap Analysis for Dream Job")
            print("-" * 40)
            
            gap_analysis = await service.analyze_skill_gaps(
                user_id=user_profile.user_id,
                target_role=user_profile.dream_job,
                db=mock_db
            )
            
            print(f"Target Role: {gap_analysis['target_role']}")
            print(f"Overall Readiness: {gap_analysis['readiness_percentage']:.1f}%")
            print(f"Learning Time Estimate: {gap_analysis['learning_time_estimate_weeks']} weeks")
            
            if gap_analysis['missing_skills']:
                print(f"\nMissing Skills:")
                for skill, importance in list(gap_analysis['missing_skills'].items())[:5]:
                    print(f"  • {skill} (importance: {importance:.2f})")
            
            if gap_analysis['weak_skills']:
                print(f"\nSkills Needing Improvement:")
                for skill, gap in list(gap_analysis['weak_skills'].items())[:3]:
                    print(f"  • {skill} (gap score: {gap:.2f})")
            
            if gap_analysis['strong_skills']:
                print(f"\nStrong Skills:")
                print(f"  {', '.join(gap_analysis['strong_skills'][:5])}")
            
            print(f"\nPriority Skills to Develop:")
            for skill in gap_analysis['priority_skills'][:5]:
                print(f"  • {skill}")
            
            # Demonstrate job match scoring
            print(f"\n\n🎯 Job Match Analysis")
            print("-" * 30)
            
            job_match = await service.get_job_match_score(
                user_id=user_profile.user_id,
                job_id=job_postings[0].id,
                db=mock_db
            )
            
            print(f"Job: {job_match['job_title']} at {job_match['company']}")
            print(f"Match Score: {job_match['match_percentage']:.1f}%")
            print(f"Readiness: {job_match['readiness_percentage']:.1f}%")
            
            if job_match['strong_skills']:
                print(f"Matching Skills: {', '.join(job_match['strong_skills'][:5])}")
            
            if job_match['skill_gaps']:
                print(f"Skills to Develop:")
                for skill, importance in list(job_match['skill_gaps'].items())[:3]:
                    print(f"  • {skill} (importance: {importance:.2f})")
            
            print(f"\n🎉 Demo completed successfully!")
            print(f"The career trajectory service provides comprehensive analysis including:")
            print(f"  ✅ Semantic similarity matching between profiles and jobs")
            print(f"  ✅ Career progression modeling with detailed steps")
            print(f"  ✅ Dream job optimization and path planning")
            print(f"  ✅ Alternative career route discovery")
            print(f"  ✅ Market demand integration and analysis")
            print(f"  ✅ Confidence scoring and detailed reasoning")
            
        except Exception as e:
            print(f"❌ Error during demo: {str(e)}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    print("Starting Career Trajectory Service Demo...")
    asyncio.run(demonstrate_career_trajectory_service())