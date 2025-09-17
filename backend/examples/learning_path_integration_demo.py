"""
Demo script for learning path generation backend functionality.

This script demonstrates the enhanced learning path generation capabilities
for frontend integration, including:
- User-specific learning path generation
- Skill-based resource recommendations
- Custom learning path creation
- Profile-based path generation
"""

import asyncio
import json
import sys
import os

# Add the backend directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app.services.learning_path_service import LearningPathService
from app.schemas.learning_path import DifficultyLevel


async def demo_user_learning_paths():
    """Demo getting learning paths for a user."""
    print("=" * 60)
    print("DEMO: User Learning Paths")
    print("=" * 60)
    
    service = LearningPathService()
    
    # Get learning paths for a user
    user_id = "demo-user-123"
    paths = await service.get_user_learning_paths(user_id, limit=3)
    
    print(f"Generated {len(paths)} learning paths for user {user_id}:")
    print()
    
    for i, path in enumerate(paths, 1):
        print(f"{i}. {path.title}")
        print(f"   Target Role: {path.target_role}")
        print(f"   Difficulty: {path.difficulty_level.value}")
        print(f"   Duration: {path.estimated_duration_weeks} weeks ({path.estimated_duration_hours} hours)")
        print(f"   Skills: {', '.join(path.target_skills[:5])}{'...' if len(path.target_skills) > 5 else ''}")
        print(f"   Milestones: {len(path.milestones)}")
        print(f"   Resources: {len(path.resources)}")
        print(f"   Confidence: {path.confidence_score:.1%}")
        print()


async def demo_skill_based_recommendations():
    """Demo getting skill-based learning resources."""
    print("=" * 60)
    print("DEMO: Skill-Based Resource Recommendations")
    print("=" * 60)
    
    service = LearningPathService()
    
    # Get resources for specific skills
    skills = ["python", "machine_learning", "react"]
    resources = await service.get_skill_based_recommendations(skills, DifficultyLevel.INTERMEDIATE)
    
    print(f"Found {len(resources)} resources for skills: {', '.join(skills)}")
    print()
    
    for i, resource in enumerate(resources[:5], 1):  # Show top 5
        print(f"{i}. {resource.title}")
        print(f"   Provider: {resource.provider.value.title()}")
        print(f"   Type: {resource.type.value.title()}")
        print(f"   Difficulty: {resource.difficulty_level.value.title()}")
        print(f"   Duration: {resource.duration_hours} hours")
        print(f"   Cost: ${resource.cost:.2f}" if resource.cost else "   Cost: Free")
        print(f"   Rating: {resource.rating}/5" if resource.rating else "   Rating: N/A")
        print(f"   Skills: {', '.join(resource.skills_taught)}")
        print(f"   Quality Score: {resource.quality_score:.1%}" if resource.quality_score else "   Quality Score: N/A")
        print()


async def demo_custom_learning_path():
    """Demo creating a custom learning path."""
    print("=" * 60)
    print("DEMO: Custom Learning Path Creation")
    print("=" * 60)
    
    service = LearningPathService()
    
    # Create a custom learning path
    title = "Full Stack Web Development Bootcamp"
    skills = ["html", "css", "javascript", "react", "node.js", "mongodb"]
    preferences = {
        "difficulty": "intermediate",
        "hours_per_week": 20
    }
    
    path = await service.create_custom_learning_path(title, skills, preferences)
    
    print(f"Created custom learning path: {path.title}")
    print(f"Target Skills: {', '.join(path.target_skills)}")
    print(f"Difficulty: {path.difficulty_level.value}")
    print(f"Estimated Duration: {path.estimated_duration_weeks} weeks ({path.estimated_duration_hours} hours)")
    print(f"Confidence Score: {path.confidence_score:.1%}")
    print()
    
    print("Milestones:")
    for milestone in path.milestones:
        print(f"  {milestone.order + 1}. {milestone.title}")
        print(f"     Skills: {', '.join(milestone.skills_to_acquire)}")
        print(f"     Duration: {milestone.estimated_duration_hours} hours")
        print(f"     Criteria: {len(milestone.completion_criteria)} completion criteria")
        print()
    
    print(f"Learning Resources: {len(path.resources)} resources available")
    for resource in path.resources[:3]:  # Show first 3
        print(f"  - {resource.title} ({resource.provider.value.title()})")


async def demo_profile_based_generation():
    """Demo generating learning paths from user profile."""
    print("=" * 60)
    print("DEMO: Profile-Based Learning Path Generation")
    print("=" * 60)
    
    service = LearningPathService()
    
    # Sample user profile
    profile_data = {
        "user_id": "profile-demo-user",
        "skills": ["python", "sql", "excel"],
        "target_role": "Data Scientist",
        "experience_level": "intermediate",
        "time_commitment_hours_per_week": 15,
        "learning_style": "hands_on",
        "target_skills": ["machine_learning", "data_visualization", "statistics", "deep_learning"],
        "budget_limit": 200
    }
    
    paths = await service.generate_learning_paths_for_profile(profile_data)
    
    print("User Profile:")
    print(f"  Current Skills: {', '.join(profile_data['skills'])}")
    print(f"  Target Role: {profile_data['target_role']}")
    print(f"  Experience Level: {profile_data['experience_level']}")
    print(f"  Time Commitment: {profile_data['time_commitment_hours_per_week']} hours/week")
    print(f"  Budget: ${profile_data['budget_limit']}")
    print()
    
    print(f"Generated {len(paths)} personalized learning paths:")
    print()
    
    for i, path in enumerate(paths, 1):
        print(f"{i}. {path.title}")
        print(f"   Difficulty: {path.difficulty_level.value}")
        print(f"   Duration: {path.estimated_duration_weeks} weeks")
        print(f"   Target Skills: {', '.join(path.target_skills[:4])}{'...' if len(path.target_skills) > 4 else ''}")
        print(f"   Confidence: {path.confidence_score:.1%}")
        print()


async def demo_simplified_format():
    """Demo the simplified format for frontend integration."""
    print("=" * 60)
    print("DEMO: Simplified Format for Frontend")
    print("=" * 60)
    
    service = LearningPathService()
    
    # Get learning paths
    paths = await service.get_user_learning_paths("frontend-demo-user", limit=3)
    
    # Convert to simplified format (similar to what the API endpoint does)
    simplified_paths = []
    for path in paths:
        # Get primary provider from resources
        primary_provider = "Mixed"
        if path.resources:
            provider_counts = {}
            for resource in path.resources:
                provider = resource.provider.value.title()
                provider_counts[provider] = provider_counts.get(provider, 0) + 1
            primary_provider = max(provider_counts, key=provider_counts.get)
        
        # Format duration
        duration_str = f"{path.estimated_duration_weeks} weeks"
        if path.estimated_duration_weeks == 1:
            duration_str = "1 week"
        elif path.estimated_duration_weeks > 52:
            years = path.estimated_duration_weeks // 52
            duration_str = f"{years} year{'s' if years > 1 else ''}"
        
        simplified_path = {
            "id": path.id,
            "title": path.title,
            "provider": primary_provider,
            "duration": duration_str,
            "difficulty": path.difficulty_level.value.title(),
            "description": path.description,
            "target_role": path.target_role,
            "skills": path.target_skills,
            "confidence_score": path.confidence_score,
            "estimated_hours": path.estimated_duration_hours
        }
        simplified_paths.append(simplified_path)
    
    print("Learning Paths in Frontend Format:")
    print(json.dumps(simplified_paths, indent=2, default=str))


async def main():
    """Run all demos."""
    print("Learning Path Generation Backend Demo")
    print("=====================================")
    print()
    
    try:
        await demo_user_learning_paths()
        await demo_skill_based_recommendations()
        await demo_custom_learning_path()
        await demo_profile_based_generation()
        await demo_simplified_format()
        
        print("=" * 60)
        print("DEMO COMPLETED SUCCESSFULLY")
        print("=" * 60)
        print()
        print("Key Features Demonstrated:")
        print("✓ User-specific learning path generation")
        print("✓ Skill-based resource recommendations")
        print("✓ Custom learning path creation")
        print("✓ Profile-based path generation")
        print("✓ Frontend-compatible simplified format")
        print()
        print("The learning path generation backend is ready for frontend integration!")
        
    except Exception as e:
        print(f"Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())