"""Demo script for enhanced external API integration features."""

import asyncio
import json
import sys
import os
from datetime import datetime

# Add the backend directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.external_apis import (
    ExternalAPIIntegrationService,
    ProfileExtractionRequest,
    ProfileMerger,
    circuit_breaker_manager
)


async def demo_profile_extraction():
    """Demonstrate enhanced profile extraction with merging and error handling."""
    print("🚀 Enhanced External API Integration Demo")
    print("=" * 50)
    
    # Initialize service
    service = ExternalAPIIntegrationService(enable_caching=True)
    
    # Demo 1: Profile extraction with merging
    print("\n📊 Demo 1: Profile Extraction with Automatic Merging")
    print("-" * 50)
    
    request = ProfileExtractionRequest(
        github_username="octocat",  # Public GitHub user
        timeout_seconds=30,
        enable_validation=True,
        enable_graceful_degradation=True
    )
    
    try:
        result = await service.extract_comprehensive_profile(request)
        
        print(f"✅ Extraction successful: {result.success}")
        print(f"📈 Sources attempted: {result.sources_attempted}")
        print(f"✨ Sources successful: {result.sources_successful}")
        print(f"⏱️  Extraction time: {result.extraction_time:.2f}s")
        
        if result.merged_profile:
            merged = result.merged_profile
            print(f"\n🔗 Merged Profile Summary:")
            print(f"   Name: {merged.get('name', 'N/A')}")
            print(f"   Username: {merged.get('username', 'N/A')}")
            print(f"   Location: {merged.get('location', 'N/A')}")
            print(f"   Company: {merged.get('current_company', 'N/A')}")
            print(f"   Programming Languages: {len(merged.get('programming_languages', {}))}")
            print(f"   Technical Skills: {len(merged.get('technical_skills', {}))}")
            print(f"   Data Quality: {merged.get('confidence_level', 'N/A')}")
            print(f"   Quality Score: {merged.get('data_quality_score', 0):.2f}")
        
        if result.warnings:
            print(f"\n⚠️  Warnings: {len(result.warnings)}")
            for warning in result.warnings[:3]:  # Show first 3
                print(f"   - {warning[:100]}...")
        
        if result.errors:
            print(f"\n❌ Errors: {len(result.errors)}")
            for source, error in result.errors.items():
                print(f"   {source}: {error[:100]}...")
    
    except Exception as e:
        print(f"❌ Demo failed: {str(e)}")
    
    # Demo 2: Circuit breaker statistics
    print("\n⚡ Demo 2: Circuit Breaker Statistics")
    print("-" * 50)
    
    cb_stats = service.get_circuit_breaker_stats()
    if cb_stats:
        for service_name, stats in cb_stats.items():
            print(f"🔌 {service_name.title()} Service:")
            print(f"   State: {stats['state']}")
            print(f"   Total Requests: {stats['total_requests']}")
            print(f"   Success Rate: {stats['total_successes']}/{stats['total_requests']} "
                  f"({stats['total_successes']/max(1, stats['total_requests'])*100:.1f}%)")
    else:
        print("   No circuit breaker activity yet")
    
    # Demo 3: Cache statistics
    print("\n💾 Demo 3: Cache Statistics")
    print("-" * 50)
    
    cache_stats = service.get_cache_stats()
    print(f"📦 Cache Entries: {cache_stats['total_entries']}")
    print(f"✅ Valid Entries: {cache_stats['valid_entries']}")
    print(f"⏰ Expired Entries: {cache_stats['expired_entries']}")
    print(f"🕐 TTL: {cache_stats['cache_ttl_seconds']}s")
    
    # Demo 4: Profile merging standalone
    print("\n🔀 Demo 4: Standalone Profile Merging")
    print("-" * 50)
    
    # Sample data for demonstration
    sample_github = {
        "username": "developer",
        "name": "Jane Developer",
        "bio": "Full-stack developer",
        "company": "Tech Corp",
        "location": "San Francisco",
        "public_repos": 42,
        "followers": 150,
        "languages": {"Python": 5000, "JavaScript": 3000, "Go": 1000},
        "total_stars": 200
    }
    
    sample_leetcode = {
        "username": "developer",
        "real_name": "Jane Developer",
        "country": "United States",
        "stats": {
            "total_solved": 300,
            "easy_solved": 150,
            "medium_solved": 120,
            "hard_solved": 30,
            "acceptance_rate": 88.5
        },
        "languages_used": {"Python": 180, "Java": 80, "C++": 40},
        "skill_tags": {"Array": 60, "Dynamic Programming": 45, "Tree": 35}
    }
    
    merger = ProfileMerger()
    merged_profile = merger.merge_profiles(
        github_profile=sample_github,
        leetcode_profile=sample_leetcode
    )
    
    print(f"🎯 Merged Profile Results:")
    print(f"   Name: {merged_profile.name}")
    print(f"   Username: {merged_profile.username}")
    print(f"   Company: {merged_profile.current_company}")
    print(f"   Programming Languages: {list(merged_profile.programming_languages.keys())[:5]}")
    print(f"   Technical Skills: {list(merged_profile.technical_skills.keys())[:5]}")
    print(f"   GitHub Repos: {merged_profile.total_repositories}")
    print(f"   LeetCode Problems: {merged_profile.total_problems_solved}")
    print(f"   Data Sources: {merged_profile.data_sources}")
    print(f"   Confidence Level: {merged_profile.confidence_level}")
    print(f"   Quality Score: {merged_profile.data_quality_score:.2f}")
    
    print("\n🎉 Demo completed successfully!")


async def demo_error_handling():
    """Demonstrate enhanced error handling and resilience."""
    print("\n🛡️  Enhanced Error Handling Demo")
    print("=" * 50)
    
    service = ExternalAPIIntegrationService(enable_caching=False)
    
    # Test with non-existent user
    print("\n🔍 Testing with non-existent user...")
    request = ProfileExtractionRequest(
        github_username="nonexistentuser12345xyz",
        enable_graceful_degradation=True
    )
    
    result = await service.extract_comprehensive_profile(request)
    print(f"Result: {result.success}")
    print(f"Errors: {result.errors}")
    print(f"Graceful degradation: {'✅' if result.success else '❌'}")
    
    # Test circuit breaker reset
    print("\n🔄 Testing circuit breaker management...")
    await service.reset_circuit_breakers()
    print("✅ Circuit breakers reset successfully")
    
    # Show final stats
    print("\n📊 Final Statistics:")
    cb_stats = service.get_circuit_breaker_stats()
    for service_name, stats in cb_stats.items():
        if stats['total_requests'] > 0:
            print(f"   {service_name}: {stats['total_requests']} requests, "
                  f"{stats['total_failures']} failures")


if __name__ == "__main__":
    print("Starting Enhanced External API Integration Demo...")
    
    try:
        asyncio.run(demo_profile_extraction())
        asyncio.run(demo_error_handling())
    except KeyboardInterrupt:
        print("\n👋 Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed with error: {str(e)}")
        import traceback
        traceback.print_exc()