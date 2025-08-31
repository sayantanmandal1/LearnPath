"""
Simple demo script for career trajectory recommendation service.

This script demonstrates the core functionality without complex mocking.
"""

import asyncio
import sys
import os
from datetime import datetime

# Add the backend directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.career_trajectory_service import (
    CareerTrajectoryService,
    CareerTrajectoryRecommendation
)


def demonstrate_career_trajectory_components():
    """Demonstrate individual components of the career trajectory service."""
    print("🚀 Career Trajectory Service Components Demo")
    print("=" * 60)
    
    # Initialize service
    service = CareerTrajectoryService()
    
    print("\n1. 🎯 Default Role Skills Analysis")
    print("-" * 40)
    
    # Test default role skills
    roles = ["software engineer", "data scientist", "product manager", "devops engineer"]
    
    for role in roles:
        skills = service._get_default_role_skills(role)
        print(f"\n{role.title()}:")
        top_skills = sorted(skills.items(), key=lambda x: x[1], reverse=True)[:5]
        for skill, importance in top_skills:
            print(f"  • {skill}: {importance:.2f}")
    
    print("\n\n2. 🔍 Skill Gap Analysis Simulation")
    print("-" * 40)
    
    # Simulate user skills (Data Analyst transitioning to Data Scientist)
    user_skills = {
        "python": 0.8,
        "sql": 0.9,
        "pandas": 0.7,
        "numpy": 0.6,
        "statistics": 0.6,
        "excel": 0.9,
        "tableau": 0.7,
        "machine learning": 0.3,  # Weak skill
        "deep learning": 0.1,     # Very weak skill
    }
    
    # Target role skills
    target_skills = service._get_default_role_skills("data scientist")
    
    print(f"User Skills ({len(user_skills)} skills):")
    for skill, level in sorted(user_skills.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  • {skill}: {level:.2f}")
    
    print(f"\nTarget Role Skills (Data Scientist):")
    for skill, importance in sorted(target_skills.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  • {skill}: {importance:.2f}")
    
    # Analyze gaps using the skill gap analyzer
    gap_analysis = service.skill_gap_analyzer.analyze_skill_gaps(
        user_skills, target_skills, "Data Scientist"
    )
    
    print(f"\n📊 Gap Analysis Results:")
    print(f"  Overall Readiness: {gap_analysis.overall_readiness:.1%}")
    print(f"  Learning Time: {gap_analysis.learning_time_estimate} weeks")
    
    if gap_analysis.missing_skills:
        print(f"  Missing Skills:")
        for skill, importance in list(gap_analysis.missing_skills.items())[:3]:
            print(f"    • {skill} (importance: {importance:.2f})")
    
    if gap_analysis.weak_skills:
        print(f"  Skills Needing Improvement:")
        for skill, gap in list(gap_analysis.weak_skills.items())[:3]:
            print(f"    • {skill} (gap: {gap:.2f})")
    
    if gap_analysis.strong_skills:
        print(f"  Strong Skills: {', '.join(gap_analysis.strong_skills[:5])}")
    
    print(f"  Priority Skills: {', '.join(gap_analysis.priority_skills[:3])}")
    
    print("\n\n3. 🛤️  Career Progression Planning")
    print("-" * 40)
    
    # Test progression step planning
    current_role = "Data Analyst"
    target_role = "Senior Data Scientist"
    
    print(f"Planning progression from {current_role} to {target_role}:")
    
    # Generate progression steps
    progression_steps = [
        {
            'role': current_role,
            'duration_months': 0,
            'description': 'Current position - strengthen foundation',
            'key_activities': ['Master SQL and Python', 'Learn statistics basics'],
            'skills_to_develop': [],
            'milestones': ['Complete current projects']
        },
        {
            'role': 'Data Scientist',
            'duration_months': 18,
            'description': 'Transition to Data Scientist role',
            'key_activities': ['Learn machine learning', 'Build ML projects', 'Get ML certification'],
            'skills_to_develop': ['machine learning', 'scikit-learn', 'statistics'],
            'milestones': ['Complete ML course', 'Deploy ML model', 'Present findings']
        },
        {
            'role': 'Senior Data Scientist',
            'duration_months': 24,
            'description': 'Advance to senior level',
            'key_activities': ['Lead ML projects', 'Mentor junior data scientists', 'Research new techniques'],
            'skills_to_develop': ['deep learning', 'tensorflow', 'leadership'],
            'milestones': ['Lead team project', 'Publish research', 'Mentor others']
        }
    ]
    
    total_timeline = service._estimate_trajectory_timeline(progression_steps, gap_analysis.learning_time_estimate)
    difficulty = service._assess_difficulty_level(gap_analysis)
    
    print(f"\nProgression Timeline: {total_timeline} months")
    print(f"Difficulty Level: {difficulty}")
    
    print(f"\nDetailed Steps:")
    for i, step in enumerate(progression_steps, 1):
        print(f"\n  Step {i}: {step['role']}")
        print(f"    Duration: {step['duration_months']} months")
        print(f"    Description: {step['description']}")
        if step['key_activities']:
            print(f"    Key Activities:")
            for activity in step['key_activities']:
                print(f"      • {activity}")
        if step['skills_to_develop']:
            print(f"    Skills to Develop: {', '.join(step['skills_to_develop'])}")
    
    print("\n\n4. 💡 Reasoning and Guidance Generation")
    print("-" * 40)
    
    # Test reasoning generation
    market_data = {
        'demand_level': 'high',
        'growth_potential': 0.8,
        'salary_trend': 'growing',
        'job_count': 45
    }
    
    reasoning = service._generate_trajectory_reasoning(
        target_role, gap_analysis, market_data, "dream_job"
    )
    
    print(f"Generated Reasoning:")
    print(f"  {reasoning}")
    
    # Test success factors and challenges
    success_factors = service._identify_success_factors(gap_analysis, market_data)
    challenges = service._identify_challenges(gap_analysis, market_data)
    
    print(f"\nSuccess Factors:")
    for factor in success_factors[:3]:
        print(f"  • {factor}")
    
    print(f"\nPotential Challenges:")
    for challenge in challenges[:3]:
        print(f"  • {challenge}")
    
    print("\n\n5. 📈 Confidence Scoring")
    print("-" * 40)
    
    # Test confidence scoring
    similarity_scores = [0.9, 0.7, 0.5, 0.3]
    readiness_scores = [0.8, 0.6, 0.4, 0.2]
    
    print("Confidence Score Analysis:")
    print("Similarity | Readiness | Market | Confidence")
    print("-" * 45)
    
    for sim in similarity_scores:
        for readiness in readiness_scores:
            confidence = service._calculate_confidence_score(sim, readiness, market_data)
            print(f"   {sim:.1f}     |    {readiness:.1f}    |  high  |   {confidence:.3f}")
    
    print("\n\n6. 🔄 Alternative Path Discovery")
    print("-" * 40)
    
    # Test alternative path discovery
    alternative_approaches = [
        {
            'approach': 'Bootcamp Route',
            'description': 'Intensive data science bootcamp',
            'advantages': ['Fast-track learning', 'Practical focus', 'Job placement support'],
            'considerations': ['Intensive time commitment', 'May lack theoretical depth'],
            'success_rate': 0.7
        },
        {
            'approach': 'Internal Transition',
            'description': 'Transition within current company',
            'advantages': ['Known environment', 'Existing relationships', 'Lower risk'],
            'considerations': ['Limited by company opportunities', 'May require patience'],
            'success_rate': 0.8
        },
        {
            'approach': 'Academic Route',
            'description': 'Master\'s degree in Data Science',
            'advantages': ['Comprehensive knowledge', 'Credibility', 'Network building'],
            'considerations': ['Time and cost intensive', 'May be too theoretical'],
            'success_rate': 0.75
        }
    ]
    
    print("Alternative Career Paths:")
    for alt in alternative_approaches:
        print(f"\n  {alt['approach']}:")
        print(f"    Description: {alt['description']}")
        print(f"    Success Rate: {alt['success_rate']:.1%}")
        print(f"    Advantages: {', '.join(alt['advantages'][:2])}")
        print(f"    Considerations: {', '.join(alt['considerations'][:2])}")
    
    print("\n\n🎉 Demo completed successfully!")
    print("\nThe career trajectory service provides:")
    print("  ✅ Comprehensive skill gap analysis")
    print("  ✅ Intelligent career progression planning")
    print("  ✅ Market-aware confidence scoring")
    print("  ✅ Detailed reasoning and guidance")
    print("  ✅ Alternative path discovery")
    print("  ✅ Timeline and difficulty assessment")
    
    print(f"\n📋 Summary for Data Analyst → Senior Data Scientist:")
    print(f"  • Current Readiness: {gap_analysis.overall_readiness:.1%}")
    print(f"  • Estimated Timeline: {total_timeline} months")
    print(f"  • Difficulty Level: {difficulty}")
    print(f"  • Priority Skills: {', '.join(gap_analysis.priority_skills[:3])}")
    print(f"  • Market Demand: {market_data['demand_level']}")


if __name__ == "__main__":
    print("Starting Career Trajectory Service Components Demo...")
    demonstrate_career_trajectory_components()