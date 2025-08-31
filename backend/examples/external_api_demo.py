"""Demo script showing how to use external API integration services."""

import asyncio
import logging
from typing import Optional

from app.services.external_apis import (
    ExternalAPIIntegrationService,
    ProfileExtractionRequest,
    GitHubClient,
    LeetCodeScraper,
    LinkedInScraper
)


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def demo_github_client():
    """Demonstrate GitHub API client usage."""
    print("\n=== GitHub API Client Demo ===")
    
    # Initialize client (optionally with token for higher rate limits)
    github_token = None  # Set your GitHub token here if available
    client = GitHubClient(api_token=github_token)
    
    try:
        async with client:
            # Get a public user profile
            username = "octocat"  # GitHub's mascot account
            print(f"Fetching GitHub profile for: {username}")
            
            profile = await client.get_user_profile(username)
            
            print(f"Username: {profile.username}")
            print(f"Name: {profile.name}")
            print(f"Bio: {profile.bio}")
            print(f"Company: {profile.company}")
            print(f"Public Repos: {profile.public_repos}")
            print(f"Followers: {profile.followers}")
            print(f"Total Stars: {profile.total_stars}")
            print(f"Languages: {list(profile.languages.keys())[:5]}")  # Top 5 languages
            print(f"Repository Count: {len(profile.repositories)}")
            
            if profile.repositories:
                print(f"Latest Repository: {profile.repositories[0].name}")
    
    except Exception as e:
        print(f"Error fetching GitHub profile: {str(e)}")


async def demo_leetcode_scraper():
    """Demonstrate LeetCode scraper usage."""
    print("\n=== LeetCode Scraper Demo ===")
    
    scraper = LeetCodeScraper()
    
    try:
        async with scraper:
            # Note: LeetCode scraping may be blocked or rate-limited
            # This is a demonstration of the API structure
            username = "testuser"  # Replace with a real username if testing
            print(f"Attempting to fetch LeetCode profile for: {username}")
            
            # First validate if username exists
            is_valid = await scraper.validate_username(username)
            print(f"Username valid: {is_valid}")
            
            if is_valid:
                profile = await scraper.get_user_profile(username)
                
                print(f"Username: {profile.username}")
                print(f"Real Name: {profile.real_name}")
                print(f"Country: {profile.country}")
                print(f"Company: {profile.company}")
                print(f"Total Solved: {profile.stats.total_solved}")
                print(f"Easy: {profile.stats.easy_solved}")
                print(f"Medium: {profile.stats.medium_solved}")
                print(f"Hard: {profile.stats.hard_solved}")
                print(f"Acceptance Rate: {profile.stats.acceptance_rate}%")
                print(f"Languages Used: {list(profile.languages_used.keys())}")
                print(f"Top Skill Tags: {list(profile.skill_tags.keys())[:5]}")
    
    except Exception as e:
        print(f"Error fetching LeetCode profile: {str(e)}")
        print("Note: LeetCode scraping may be blocked by anti-bot measures")


async def demo_linkedin_scraper():
    """Demonstrate LinkedIn scraper usage."""
    print("\n=== LinkedIn Scraper Demo ===")
    
    scraper = LinkedInScraper()
    
    try:
        async with scraper:
            # Note: LinkedIn scraping is heavily restricted
            # This demonstrates the API structure with mock data
            profile_url = "https://www.linkedin.com/in/example-user"
            print(f"Attempting to extract LinkedIn profile from: {profile_url}")
            
            # Validate URL format
            profile_id = scraper._extract_profile_id_from_url(profile_url)
            print(f"Extracted profile ID: {profile_id}")
            
            if profile_id:
                # This will return mock data due to LinkedIn's restrictions
                profile = await scraper.get_profile_safely(profile_url)
                
                if profile:
                    print(f"Name: {profile.name}")
                    print(f"Headline: {profile.headline}")
                    print(f"Location: {profile.location}")
                    print(f"Current Company: {profile.current_company}")
                    print(f"Experience Entries: {len(profile.experience)}")
                    print(f"Skills: {len(profile.skills)}")
                    print("Note: This is mock data due to LinkedIn's ToS restrictions")
                else:
                    print("Profile extraction returned no data")
    
    except Exception as e:
        print(f"Error with LinkedIn scraper: {str(e)}")


async def demo_integration_service():
    """Demonstrate the comprehensive integration service."""
    print("\n=== Integration Service Demo ===")
    
    # Initialize the integration service
    github_token = None  # Set your GitHub token here if available
    service = ExternalAPIIntegrationService(
        github_token=github_token,
        enable_caching=True
    )
    
    # Create extraction request
    request = ProfileExtractionRequest(
        github_username="octocat",  # Public GitHub user
        leetcode_username="testuser",  # This may fail due to restrictions
        linkedin_url="https://www.linkedin.com/in/example-user",  # Mock data
        timeout_seconds=60,
        enable_validation=True,
        enable_graceful_degradation=True
    )
    
    try:
        print("Starting comprehensive profile extraction...")
        result = await service.extract_comprehensive_profile(request)
        
        print(f"Extraction Success: {result.success}")
        print(f"Extraction Time: {result.extraction_time:.2f} seconds")
        print(f"Sources Attempted: {result.sources_attempted}")
        print(f"Sources Successful: {result.sources_successful}")
        
        if result.errors:
            print(f"Errors: {result.errors}")
        
        if result.warnings:
            print(f"Warnings: {result.warnings}")
        
        # Display extracted data
        if result.github_profile:
            print(f"\nGitHub Profile Extracted:")
            print(f"  Username: {result.github_profile.get('username')}")
            print(f"  Public Repos: {result.github_profile.get('public_repos')}")
            print(f"  Languages: {list(result.github_profile.get('languages', {}).keys())[:3]}")
        
        if result.leetcode_profile:
            print(f"\nLeetCode Profile Extracted:")
            print(f"  Username: {result.leetcode_profile.get('username')}")
            print(f"  Total Solved: {result.leetcode_profile.get('stats', {}).get('total_solved')}")
        
        if result.linkedin_profile:
            print(f"\nLinkedIn Profile Extracted:")
            print(f"  Name: {result.linkedin_profile.get('name')}")
            print(f"  Company: {result.linkedin_profile.get('current_company')}")
        
        # Display validation results
        if result.validation_results:
            print(f"\nValidation Results:")
            for source, validation in result.validation_results.items():
                print(f"  {source.title()}: Quality={validation.quality.value}, Score={validation.confidence_score:.2f}")
        
        # Show cache stats
        cache_stats = service.get_cache_stats()
        print(f"\nCache Stats: {cache_stats}")
    
    except Exception as e:
        print(f"Error in integration service: {str(e)}")


async def demo_profile_validation():
    """Demonstrate profile source validation."""
    print("\n=== Profile Source Validation Demo ===")
    
    service = ExternalAPIIntegrationService()
    
    try:
        print("Validating profile sources...")
        
        validation_results = await service.validate_profile_sources(
            github_username="octocat",  # Should be valid
            leetcode_username="nonexistentuser12345",  # Should be invalid
            linkedin_url="https://www.linkedin.com/in/example-user"
        )
        
        print("Validation Results:")
        for source, is_valid in validation_results.items():
            status = "✓ Valid" if is_valid else "✗ Invalid"
            print(f"  {source.title()}: {status}")
    
    except Exception as e:
        print(f"Error in profile validation: {str(e)}")


async def main():
    """Run all demos."""
    print("External API Integration Service Demo")
    print("=" * 50)
    
    # Run individual client demos
    await demo_github_client()
    await demo_leetcode_scraper()
    await demo_linkedin_scraper()
    
    # Run integration service demos
    await demo_integration_service()
    await demo_profile_validation()
    
    print("\n" + "=" * 50)
    print("Demo completed!")
    print("\nNotes:")
    print("- GitHub API works best with an authentication token")
    print("- LeetCode scraping may be blocked by anti-bot measures")
    print("- LinkedIn scraping is heavily restricted and returns mock data")
    print("- All services include comprehensive error handling and graceful degradation")


if __name__ == "__main__":
    asyncio.run(main())