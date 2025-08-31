"""
Demo script showing the recommendation engine functionality.
"""

import asyncio
import sys
import os
import numpy as np
from datetime import datetime

# Add the parent directory to the path to access machinelearningmodel
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from machinelearningmodel.recommendation_engine import (
    HybridRecommendationEngine,
    CollaborativeFilteringEngine,
    ContentBasedFilteringEngine,
    SkillGapAnalyzer,
    RecommendationExplainer,
    CareerRecommendation,
    LearningPath,
    SkillGapAnalysis
)


def demo_collaborative_filtering():
    """Demonstrate collaborative filtering."""
    print("=== Collaborative Filtering Demo ===")
    
    # Sample user-item interaction matrix (users x jobs)
    # Ratings from 1-5, 0 means no interaction
    user_item_matrix = np.array([
        [5, 3, 0, 1, 4],  # User 1: Likes Python jobs, dislikes entry-level
        [4, 0, 0, 1, 5],  # User 2: Prefers senior roles
        [1, 1, 0, 5, 2],  # User 3: Likes data science roles
        [1, 0, 0, 4, 3],  # User 4: Mixed preferences
        [0, 1, 5, 4, 0]   # User 5: Likes frontend and data roles
    ], dtype=float)
    
    user_ids = ['user1', 'user2', 'user3', 'user4', 'user5']
    job_ids = ['python_dev', 'frontend_dev', 'mobile_dev', 'data_scientist', 'senior_dev']
    
    # Train collaborative filtering model
    cf_engine = CollaborativeFilteringEngine(n_factors=3, n_iterations=50)
    cf_engine.fit(user_item_matrix, user_ids, job_ids)
    
    # Get recommendations for user1
    recommendations = cf_engine.recommend_items('user1', n_recommendations=3)
    
    print(f"Recommendations for user1:")
    for job_id, score in recommendations:
        print(f"  {job_id}: {score:.3f}")
    
    print()


def demo_content_based_filtering():
    """Demonstrate content-based filtering."""
    print("=== Content-Based Filtering Demo ===")
    
    # Sample job data
    job_data = [
        {
            'id': 'job1',
            'title': 'Senior Python Developer',
            'description': 'Develop web applications using Python, Django, and PostgreSQL',
            'skills': ['python', 'django', 'postgresql', 'rest-api']
        },
        {
            'id': 'job2',
            'title': 'Data Scientist',
            'description': 'Build machine learning models using Python, scikit-learn, and TensorFlow',
            'skills': ['python', 'machine-learning', 'tensorflow', 'pandas']
        },
        {
            'id': 'job3',
            'title': 'Frontend Developer',
            'description': 'Create responsive web interfaces using React and TypeScript',
            'skills': ['javascript', 'react', 'typescript', 'css']
        }
    ]
    
    # Sample user data
    user_data = [
        {
            'id': 'user1',
            'current_role': 'Junior Python Developer',
            'dream_job': 'Senior Python Developer',
            'skills': ['python', 'flask', 'mysql']
        },
        {
            'id': 'user2',
            'current_role': 'Data Analyst',
            'dream_job': 'Data Scientist',
            'skills': ['python', 'pandas', 'sql']
        }
    ]
    
    # Note: This would normally use actual sentence transformers
    # For demo purposes, we'll create a mock version
    print("Content-based filtering would analyze job descriptions and user profiles")
    print("to find semantic similarities using transformer models.")
    print("Sample job matches for user1 (Python developer):")
    print("  job1 (Senior Python Developer): 0.85 similarity")
    print("  job2 (Data Scientist): 0.62 similarity")
    print("  job3 (Frontend Developer): 0.23 similarity")
    print()


def demo_skill_gap_analysis():
    """Demonstrate skill gap analysis."""
    print("=== Skill Gap Analysis Demo ===")
    
    # User's current skills
    user_skills = {
        'python': 0.8,
        'sql': 0.6,
        'git': 0.7,
        'javascript': 0.4
    }
    
    # Target job requirements
    target_job_skills = {
        'python': 0.9,
        'django': 0.8,
        'postgresql': 0.7,
        'redis': 0.5,
        'docker': 0.6,
        'git': 0.8
    }
    
    # Analyze skill gaps
    analyzer = SkillGapAnalyzer()
    gap_analysis = analyzer.analyze_skill_gaps(
        user_skills, target_job_skills, 'Senior Python Developer'
    )
    
    print(f"Skill Gap Analysis for: {gap_analysis.target_role}")
    print(f"Overall Readiness: {gap_analysis.overall_readiness:.1%}")
    print(f"Learning Time Estimate: {gap_analysis.learning_time_estimate} weeks")
    
    print("\nMissing Skills:")
    for skill, importance in gap_analysis.missing_skills.items():
        print(f"  {skill}: {importance:.2f} importance")
    
    print("\nWeak Skills (need improvement):")
    for skill, gap in gap_analysis.weak_skills.items():
        print(f"  {skill}: {gap:.2f} gap score")
    
    print("\nStrong Skills:")
    for skill in gap_analysis.strong_skills:
        print(f"  {skill}")
    
    print("\nPriority Learning Order:")
    for i, skill in enumerate(gap_analysis.priority_skills, 1):
        print(f"  {i}. {skill}")
    
    print()


def demo_recommendation_explanations():
    """Demonstrate recommendation explanations."""
    print("=== Recommendation Explanations Demo ===")
    
    # Sample recommendation
    recommendation = CareerRecommendation(
        job_title="Senior Python Developer",
        match_score=0.85,
        required_skills=["python", "django", "postgresql"],
        skill_gaps={"django": 0.3, "postgresql": 0.4},
        salary_range=(90000, 130000),
        growth_potential=0.8,
        market_demand="high",
        reasoning="",
        confidence_score=0.85,
        alternative_paths=["DevOps Engineer", "Full Stack Developer"]
    )
    
    user_skills = {"python": 0.9, "sql": 0.6, "git": 0.8}
    market_data = {"demand_level": "high", "salary_trend": "increasing"}
    
    explainer = RecommendationExplainer()
    explanation = explainer.explain_career_recommendation(
        recommendation, user_skills, market_data
    )
    
    print("Career Recommendation Explanation:")
    print(f"Job: {recommendation.job_title}")
    print(f"Match Score: {recommendation.match_score:.1%}")
    print(f"Explanation: {explanation}")
    print()


def demo_hybrid_system():
    """Demonstrate hybrid recommendation system."""
    print("=== Hybrid Recommendation System Demo ===")
    
    # Sample data for hybrid system
    user_item_matrix = np.array([
        [5, 3, 0, 1],
        [4, 0, 0, 1],
        [1, 1, 0, 5]
    ], dtype=float)
    
    user_ids = ['user1', 'user2', 'user3']
    
    job_data = [
        {'id': 'job1', 'title': 'Python Developer', 'skills': ['python', 'django']},
        {'id': 'job2', 'title': 'Frontend Developer', 'skills': ['javascript', 'react']},
        {'id': 'job3', 'title': 'Mobile Developer', 'skills': ['swift', 'kotlin']},
        {'id': 'job4', 'title': 'Data Scientist', 'skills': ['python', 'tensorflow']}
    ]
    
    user_data = [
        {'id': 'user1', 'skills': ['python', 'flask']},
        {'id': 'user2', 'skills': ['javascript', 'vue']},
        {'id': 'user3', 'skills': ['python', 'pandas']}
    ]
    
    print("Hybrid system combines multiple approaches:")
    print("1. Collaborative Filtering (30% weight)")
    print("2. Content-Based Filtering (40% weight)")
    print("3. Neural Collaborative Filtering (20% weight)")
    print("4. Skill-Based Matching (10% weight)")
    print()
    
    print("Sample hybrid recommendations for user1:")
    print("  Senior Python Developer: 0.87 (high skill match + user similarity)")
    print("  Data Scientist: 0.73 (good skill overlap)")
    print("  Full Stack Developer: 0.65 (partial skill match)")
    print()


def demo_learning_paths():
    """Demonstrate learning path generation."""
    print("=== Learning Path Generation Demo ===")
    
    # Sample skill gaps
    skill_gaps = {
        'django': 0.8,
        'postgresql': 0.7,
        'docker': 0.6,
        'redis': 0.4
    }
    
    # Generate sample learning path
    learning_path = LearningPath(
        path_id="path_django_mastery",
        title="Master Django for Senior Python Developer",
        target_skills=["django", "postgresql"],
        estimated_duration_weeks=12,
        difficulty_level="intermediate",
        resources=[
            {
                'title': 'Django Complete Course',
                'type': 'course',
                'provider': 'Coursera',
                'rating': 4.5,
                'duration_hours': 40,
                'cost': 49.99
            },
            {
                'title': 'PostgreSQL Fundamentals',
                'type': 'course',
                'provider': 'Udemy',
                'rating': 4.3,
                'duration_hours': 25,
                'cost': 29.99
            }
        ],
        milestones=[
            {
                'title': 'Complete Django Basics',
                'description': 'Learn Django fundamentals and create first app',
                'estimated_weeks': 3
            },
            {
                'title': 'Database Integration',
                'description': 'Master PostgreSQL integration with Django',
                'estimated_weeks': 4
            },
            {
                'title': 'Build Portfolio Project',
                'description': 'Create a full-stack web application',
                'estimated_weeks': 5
            }
        ],
        priority_score=0.9,
        reasoning="Django is the most critical missing skill for your target role"
    )
    
    print(f"Learning Path: {learning_path.title}")
    print(f"Duration: {learning_path.estimated_duration_weeks} weeks")
    print(f"Difficulty: {learning_path.difficulty_level}")
    print(f"Priority Score: {learning_path.priority_score:.1f}")
    print(f"Reasoning: {learning_path.reasoning}")
    
    print("\nResources:")
    for resource in learning_path.resources:
        print(f"  - {resource['title']} ({resource['provider']}) - {resource['duration_hours']}h - ${resource['cost']}")
    
    print("\nMilestones:")
    for milestone in learning_path.milestones:
        print(f"  {milestone['estimated_weeks']} weeks: {milestone['title']}")
        print(f"    {milestone['description']}")
    
    print()


def main():
    """Run all demos."""
    print("AI Career Recommendation Engine Demo")
    print("=" * 50)
    print()
    
    demo_collaborative_filtering()
    demo_content_based_filtering()
    demo_skill_gap_analysis()
    demo_recommendation_explanations()
    demo_hybrid_system()
    demo_learning_paths()
    
    print("Demo completed! The recommendation engine provides:")
    print("✓ Collaborative filtering for user-based recommendations")
    print("✓ Content-based filtering using semantic similarity")
    print("✓ Neural collaborative filtering with deep learning")
    print("✓ Comprehensive skill gap analysis")
    print("✓ Detailed recommendation explanations")
    print("✓ Personalized learning path generation")
    print("✓ Hybrid approach combining multiple algorithms")


if __name__ == "__main__":
    main()