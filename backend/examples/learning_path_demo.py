"""
Learning Path Generation Demo

This script demonstrates the learning path generation functionality including:
- Skill gap identification and prioritization
- Learning resource integration from multiple platforms
- Project recommendations from GitHub
- Timeline estimation and milestone tracking
- Resource quality scoring and filtering
"""

import asyncio
import json
from datetime import datetime
from typing import Dict, Any

# Add the parent directory to the path to import our modules
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.learning_path_service import LearningPathService
from app.schemas.learning_path import (
    LearningPathRequest, DifficultyLevel, ResourceProvider
)


async def demo_basic_learning_path_generation():
    """Demonstrate basic learning path generation."""
    print("=" * 60)
    print("LEARNING PATH GENERATION DEMO")
    print("=" * 60)
    
    # Initialize the service
    service = LearningPathService()
    
    # Create a sample request for a software engineer career path
    request = LearningPathRequest(
        user_id="demo_user_001",
        target_role="software_engineer",
        target_skills=["python", "javascript", "react", "sql", "docker"],
        current_skills={
            "html": 0.8,
            "css": 0.7,
            "git": 0.6,
            "python": 0.3,
            "javascript": 0.4
        },
        time_commitment_hours_per_week=12,
        budget_limit=300.0,
        include_free_only=False,
        preferred_providers=[ResourceProvider.COURSERA, ResourceProvider.UDEMY, ResourceProvider.FREECODECAMP],
        difficulty_preference=DifficultyLevel.INTERMEDIATE,
        include_certifications=True,
        include_projects=True
    )
    
    print(f"User Profile:")
    print(f"  Target Role: {request.target_role}")
    print(f"  Target Skills: {', '.join(request.target_skills)}")
    print(f"  Current Skills: {request.current_skills}")
    print(f"  Time Commitment: {request.time_commitment_hours_per_week} hours/week")
    print(f"  Budget: ${request.budget_limit}")
    print()
    
    try:
        # Generate learning paths
        print("Generating personalized learning paths...")
        response = await service.generate_learning_paths(request)
        
        print(f"✅ Generated {response.total_paths} learning paths")
        print(f"✅ Identified {len(response.skill_gaps_identified)} skill gaps")
        print()
        
        # Display skill gaps
        print("SKILL GAP ANALYSIS:")
        print("-" * 40)
        for i, gap in enumerate(response.skill_gaps_identified[:5], 1):
            print(f"{i}. {gap.skill_name.title()}")
            print(f"   Current Level: {gap.current_level:.1f}/1.0")
            print(f"   Target Level: {gap.target_level:.1f}/1.0")
            print(f"   Gap Size: {gap.gap_size:.1f}")
            print(f"   Priority: {gap.priority:.2f}")
            print(f"   Estimated Hours: {gap.estimated_learning_hours}")
            print(f"   Difficulty: {gap.difficulty}")
            print()
        
        # Display learning paths
        print("GENERATED LEARNING PATHS:")
        print("-" * 40)
        for i, path in enumerate(response.learning_paths, 1):
            print(f"{i}. {path.title}")
            print(f"   Description: {path.description}")
            print(f"   Duration: {path.estimated_duration_weeks} weeks ({path.estimated_duration_hours} hours)")
            print(f"   Difficulty: {path.difficulty_level}")
            print(f"   Confidence: {path.confidence_score:.2f}")
            print(f"   Target Skills: {', '.join(path.target_skills[:5])}{'...' if len(path.target_skills) > 5 else ''}")
            print(f"   Milestones: {len(path.milestones)}")
            print(f"   Resources: {len(path.resources)}")
            print()
        
        # Display detailed view of the first path
        if response.learning_paths:
            path = response.learning_paths[0]
            print("DETAILED VIEW - PRIMARY LEARNING PATH:")
            print("-" * 50)
            print(f"Title: {path.title}")
            print(f"Description: {path.description}")
            print()
            
            # Show milestones
            print("Milestones:")
            for milestone in path.milestones:
                print(f"  • {milestone.title}")
                print(f"    Skills: {', '.join(milestone.skills_to_acquire)}")
                print(f"    Duration: {milestone.estimated_duration_hours} hours")
                print(f"    Order: {milestone.order}")
                print()
            
            # Show top resources
            print("Top Learning Resources:")
            for i, resource in enumerate(path.resources[:5], 1):
                print(f"  {i}. {resource.title}")
                print(f"     Provider: {resource.provider}")
                print(f"     Type: {resource.type}")
                print(f"     Duration: {resource.duration_hours} hours")
                print(f"     Cost: ${resource.cost:.2f}" if resource.cost else "     Cost: Free")
                print(f"     Rating: {resource.rating}/5.0" if resource.rating else "     Rating: N/A")
                print(f"     Quality Score: {resource.quality_score:.2f}" if resource.quality_score else "     Quality Score: N/A")
                print(f"     Skills: {', '.join(resource.skills_taught)}")
                print()
        
        return response
        
    except Exception as e:
        print(f"❌ Error generating learning paths: {e}")
        return None


async def demo_project_recommendations():
    """Demonstrate project recommendations."""
    print("=" * 60)
    print("PROJECT RECOMMENDATIONS DEMO")
    print("=" * 60)
    
    service = LearningPathService()
    
    skills = ["python", "javascript", "react", "machine_learning"]
    difficulty = DifficultyLevel.INTERMEDIATE
    
    print(f"Getting project recommendations for skills: {', '.join(skills)}")
    print(f"Difficulty level: {difficulty}")
    print()
    
    try:
        projects = await service.get_project_recommendations(skills, difficulty)
        
        print(f"✅ Found {len(projects)} project recommendations")
        print()
        
        print("PROJECT RECOMMENDATIONS:")
        print("-" * 40)
        for i, project in enumerate(projects, 1):
            print(f"{i}. {project.title}")
            print(f"   Description: {project.description}")
            print(f"   Difficulty: {project.difficulty_level}")
            print(f"   Duration: {project.estimated_duration_hours} hours")
            print(f"   Skills Practiced: {', '.join(project.skills_practiced)}")
            print(f"   Technologies: {', '.join(project.technologies)}")
            print(f"   Learning Value: {project.learning_value:.2f}" if project.learning_value else "   Learning Value: N/A")
            print(f"   Market Relevance: {project.market_relevance:.2f}" if project.market_relevance else "   Market Relevance: N/A")
            if project.repository_url:
                print(f"   Repository: {project.repository_url}")
            print()
        
        return projects
        
    except Exception as e:
        print(f"❌ Error getting project recommendations: {e}")
        return None


async def demo_different_user_scenarios():
    """Demonstrate learning paths for different user scenarios."""
    print("=" * 60)
    print("DIFFERENT USER SCENARIOS DEMO")
    print("=" * 60)
    
    service = LearningPathService()
    
    scenarios = [
        {
            "name": "Complete Beginner",
            "request": LearningPathRequest(
                user_id="beginner_001",
                target_role="frontend_developer",
                target_skills=["html", "css", "javascript", "react"],
                current_skills={},  # No current skills
                time_commitment_hours_per_week=8,
                budget_limit=100.0,
                include_free_only=False,
                difficulty_preference=DifficultyLevel.BEGINNER,
                include_certifications=True,
                include_projects=True
            )
        },
        {
            "name": "Career Changer (Budget Conscious)",
            "request": LearningPathRequest(
                user_id="career_changer_001",
                target_role="data_scientist",
                target_skills=["python", "machine_learning", "statistics", "sql"],
                current_skills={"excel": 0.9, "business_analysis": 0.8},
                time_commitment_hours_per_week=20,
                budget_limit=50.0,  # Low budget
                include_free_only=True,  # Free resources only
                difficulty_preference=DifficultyLevel.INTERMEDIATE,
                include_certifications=True,
                include_projects=True
            )
        },
        {
            "name": "Experienced Developer (Upskilling)",
            "request": LearningPathRequest(
                user_id="experienced_001",
                target_role="machine_learning_engineer",
                target_skills=["machine_learning", "tensorflow", "mlops", "kubernetes"],
                current_skills={
                    "python": 0.9,
                    "javascript": 0.8,
                    "sql": 0.8,
                    "docker": 0.7,
                    "git": 0.9
                },
                time_commitment_hours_per_week=10,
                budget_limit=500.0,
                include_free_only=False,
                preferred_providers=[ResourceProvider.COURSERA, ResourceProvider.UDACITY],
                difficulty_preference=DifficultyLevel.ADVANCED,
                include_certifications=True,
                include_projects=True
            )
        }
    ]
    
    for scenario in scenarios:
        print(f"SCENARIO: {scenario['name']}")
        print("-" * 30)
        
        request = scenario['request']
        print(f"Target Role: {request.target_role}")
        print(f"Current Skills: {request.current_skills}")
        print(f"Time Commitment: {request.time_commitment_hours_per_week} hours/week")
        print(f"Budget: ${request.budget_limit}")
        print(f"Free Only: {request.include_free_only}")
        print()
        
        try:
            response = await service.generate_learning_paths(request)
            
            print(f"✅ Generated {len(response.learning_paths)} paths")
            print(f"✅ Identified {len(response.skill_gaps_identified)} skill gaps")
            
            if response.learning_paths:
                primary_path = response.learning_paths[0]
                print(f"Primary Path: {primary_path.title}")
                print(f"Duration: {primary_path.estimated_duration_weeks} weeks")
                print(f"Total Hours: {primary_path.estimated_duration_hours}")
                print(f"Resources: {len(primary_path.resources)}")
                
                # Show cost breakdown
                total_cost = sum(r.cost or 0 for r in primary_path.resources)
                free_resources = sum(1 for r in primary_path.resources if (r.cost or 0) == 0)
                print(f"Total Cost: ${total_cost:.2f}")
                print(f"Free Resources: {free_resources}/{len(primary_path.resources)}")
            
        except Exception as e:
            print(f"❌ Error: {e}")
        
        print()


async def demo_resource_quality_scoring():
    """Demonstrate resource quality scoring and filtering."""
    print("=" * 60)
    print("RESOURCE QUALITY SCORING DEMO")
    print("=" * 60)
    
    service = LearningPathService()
    
    # Create requests with different preferences
    high_quality_request = LearningPathRequest(
        user_id="quality_focused_001",
        target_skills=["python"],
        preferred_providers=[ResourceProvider.COURSERA, ResourceProvider.EDX],
        budget_limit=200.0,
        include_certifications=True,
        include_projects=True
    )
    
    budget_request = LearningPathRequest(
        user_id="budget_focused_001",
        target_skills=["python"],
        budget_limit=30.0,
        include_free_only=False,
        include_certifications=False,
        include_projects=False
    )
    
    free_request = LearningPathRequest(
        user_id="free_focused_001",
        target_skills=["python"],
        include_free_only=True,
        include_certifications=True,
        include_projects=True
    )
    
    requests = [
        ("High Quality Focus", high_quality_request),
        ("Budget Focus", budget_request),
        ("Free Resources Only", free_request)
    ]
    
    for name, request in requests:
        print(f"SCENARIO: {name}")
        print("-" * 30)
        
        try:
            # Get resources for Python
            resources = await service._gather_learning_resources(["python"], request)
            
            print(f"Found {len(resources)} resources")
            
            if resources:
                print("Top 3 Resources:")
                for i, resource in enumerate(resources[:3], 1):
                    print(f"  {i}. {resource.title}")
                    print(f"     Provider: {resource.provider}")
                    print(f"     Cost: ${resource.cost:.2f}" if resource.cost else "     Cost: Free")
                    print(f"     Rating: {resource.rating}/5.0" if resource.rating else "     Rating: N/A")
                    print(f"     Quality Score: {resource.quality_score:.2f}" if resource.quality_score else "     Quality Score: N/A")
                    print(f"     Duration: {resource.duration_hours} hours" if resource.duration_hours else "     Duration: N/A")
                    print(f"     Certificate: {'Yes' if resource.certificate_available else 'No'}")
                    print(f"     Projects: {'Yes' if resource.hands_on_projects else 'No'}")
                    print()
                
                # Show statistics
                avg_cost = sum(r.cost or 0 for r in resources) / len(resources)
                avg_quality = sum(r.quality_score or 0 for r in resources) / len(resources)
                free_count = sum(1 for r in resources if (r.cost or 0) == 0)
                
                print(f"Statistics:")
                print(f"  Average Cost: ${avg_cost:.2f}")
                print(f"  Average Quality Score: {avg_quality:.2f}")
                print(f"  Free Resources: {free_count}/{len(resources)}")
            
        except Exception as e:
            print(f"❌ Error: {e}")
        
        print()


async def main():
    """Run all demos."""
    print("🚀 Starting Learning Path Generation Demos")
    print()
    
    # Run basic demo
    await demo_basic_learning_path_generation()
    
    print("\n" + "="*80 + "\n")
    
    # Run project recommendations demo
    await demo_project_recommendations()
    
    print("\n" + "="*80 + "\n")
    
    # Run different scenarios demo
    await demo_different_user_scenarios()
    
    print("\n" + "="*80 + "\n")
    
    # Run resource quality demo
    await demo_resource_quality_scoring()
    
    print("\n🎉 All demos completed!")


if __name__ == "__main__":
    # Run the demo
    asyncio.run(main())