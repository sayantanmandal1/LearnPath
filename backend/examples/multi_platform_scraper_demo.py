"""Demo script for multi-platform scraper infrastructure."""

import asyncio
import json
from datetime import datetime
from typing import List

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.external_apis.multi_platform_scraper import (
    MultiPlatformScraper,
    PlatformType,
    PlatformAccount
)


async def demo_single_platform_scraping():
    """Demonstrate scraping a single platform."""
    print("=== Single Platform Scraping Demo ===")
    
    scraper = MultiPlatformScraper(
        github_token=None,  # Add your GitHub token here for better rate limits
        timeout_per_platform=30.0
    )
    
    try:
        # Test GitHub scraping
        print("Testing GitHub scraping...")
        github_result = await scraper.scrape_single_platform_safe(
            PlatformType.GITHUB,
            username="octocat"  # GitHub's mascot account
        )
        
        print(f"GitHub scraping result: {github_result.success}")
        if github_result.success:
            print(f"Data keys: {list(github_result.data.keys()) if github_result.data else 'None'}")
        else:
            print(f"Error: {github_result.error_message}")
        
        # Test LeetCode scraping
        print("\nTesting LeetCode scraping...")
        leetcode_result = await scraper.scrape_single_platform_safe(
            PlatformType.LEETCODE,
            username="testuser"  # This will likely fail as it's a test username
        )
        
        print(f"LeetCode scraping result: {leetcode_result.success}")
        if not leetcode_result.success:
            print(f"Expected error: {leetcode_result.error_message}")
        
        # Test Codeforces scraping
        print("\nTesting Codeforces scraping...")
        codeforces_result = await scraper.scrape_single_platform_safe(
            PlatformType.CODEFORCES,
            username="tourist"  # Famous competitive programmer
        )
        
        print(f"Codeforces scraping result: {codeforces_result.success}")
        if codeforces_result.success:
            print(f"Data keys: {list(codeforces_result.data.keys()) if codeforces_result.data else 'None'}")
        else:
            print(f"Error: {codeforces_result.error_message}")
        
    finally:
        await scraper.cleanup()


async def demo_multi_platform_scraping():
    """Demonstrate scraping multiple platforms concurrently."""
    print("\n=== Multi-Platform Scraping Demo ===")
    
    scraper = MultiPlatformScraper(
        github_token=None,  # Add your GitHub token here
        max_concurrent_scrapers=3,
        timeout_per_platform=30.0
    )
    
    # Define platform accounts to scrape
    platform_accounts = [
        PlatformAccount(
            platform=PlatformType.GITHUB,
            username="octocat",
            is_active=True
        ),
        PlatformAccount(
            platform=PlatformType.LEETCODE,
            username="testuser",  # This will likely fail
            is_active=True
        ),
        PlatformAccount(
            platform=PlatformType.CODEFORCES,
            username="tourist",
            is_active=True
        ),
        PlatformAccount(
            platform=PlatformType.LINKEDIN,
            profile_url="https://linkedin.com/in/testuser",  # This will likely fail
            is_active=True
        ),
        PlatformAccount(
            platform=PlatformType.KAGGLE,
            username="testuser",  # This will likely fail
            is_active=True
        )
    ]
    
    try:
        print(f"Scraping {len(platform_accounts)} platforms concurrently...")
        start_time = datetime.utcnow()
        
        result = await scraper.scrape_all_platforms(
            platform_accounts=platform_accounts,
            user_id="demo_user_123"
        )
        
        end_time = datetime.utcnow()
        processing_time = (end_time - start_time).total_seconds()
        
        print(f"\nScraping completed in {processing_time:.2f} seconds")
        print(f"Total platforms: {result.total_platforms}")
        print(f"Successful: {result.successful_platforms}")
        print(f"Failed: {result.failed_platforms}")
        print(f"Total processing time: {result.total_processing_time:.2f} seconds")
        
        # Show results for each platform
        print("\nPlatform Results:")
        for platform, platform_result in result.platforms.items():
            status = "✓" if platform_result.success else "✗"
            print(f"  {status} {platform.value}: {platform_result.processing_time:.2f}s")
            if not platform_result.success:
                print(f"    Error: {platform_result.error_message}")
            elif platform_result.data:
                # Show a few key data points
                data_keys = list(platform_result.data.keys())[:3]
                print(f"    Data keys: {data_keys}...")
        
    finally:
        await scraper.cleanup()


async def demo_platform_connectivity():
    """Demonstrate platform connectivity testing."""
    print("\n=== Platform Connectivity Demo ===")
    
    scraper = MultiPlatformScraper()
    
    try:
        print("Testing connectivity to all platforms...")
        connectivity = await scraper.test_platform_connectivity()
        
        print("\nConnectivity Results:")
        for platform, is_connected in connectivity.items():
            status = "✓" if is_connected else "✗"
            print(f"  {status} {platform.value}")
        
        # Test rate limit status
        print("\nRate Limit Status:")
        rate_limits = await scraper.get_platform_rate_limits()
        
        for platform, limit_info in rate_limits.items():
            is_limited = limit_info.get("is_limited", False)
            reset_time = limit_info.get("reset_in_seconds", 0)
            
            if is_limited:
                print(f"  ⚠ {platform.value}: Rate limited, reset in {reset_time:.0f}s")
            else:
                print(f"  ✓ {platform.value}: No rate limit")
        
    finally:
        await scraper.cleanup()


async def demo_error_handling():
    """Demonstrate error handling and graceful degradation."""
    print("\n=== Error Handling Demo ===")
    
    scraper = MultiPlatformScraper(timeout_per_platform=5.0)  # Short timeout for demo
    
    # Create accounts with invalid data to trigger errors
    problematic_accounts = [
        PlatformAccount(
            platform=PlatformType.GITHUB,
            username="",  # Empty username
            is_active=True
        ),
        PlatformAccount(
            platform=PlatformType.LINKEDIN,
            profile_url="invalid-url",  # Invalid URL
            is_active=True
        ),
        PlatformAccount(
            platform=PlatformType.LEETCODE,
            username="nonexistent_user_12345",  # Non-existent user
            is_active=True
        )
    ]
    
    try:
        print("Testing error handling with problematic accounts...")
        result = await scraper.scrape_all_platforms(
            platform_accounts=problematic_accounts,
            user_id="error_demo_user"
        )
        
        print(f"\nResults with errors:")
        print(f"Total platforms: {result.total_platforms}")
        print(f"Successful: {result.successful_platforms}")
        print(f"Failed: {result.failed_platforms}")
        
        print("\nError Details:")
        for platform, platform_result in result.platforms.items():
            if not platform_result.success:
                print(f"  ✗ {platform.value}: {platform_result.error_message}")
        
    finally:
        await scraper.cleanup()


async def demo_account_validation():
    """Demonstrate account validation."""
    print("\n=== Account Validation Demo ===")
    
    scraper = MultiPlatformScraper()
    
    # Mix of valid and invalid accounts
    test_accounts = [
        PlatformAccount(platform=PlatformType.GITHUB, username="octocat", is_active=True),
        PlatformAccount(platform=PlatformType.GITHUB, is_active=True),  # Missing username
        PlatformAccount(platform=PlatformType.LINKEDIN, profile_url="https://linkedin.com/in/test", is_active=True),
        PlatformAccount(platform=PlatformType.LINKEDIN, is_active=True),  # Missing URL
        PlatformAccount(platform=PlatformType.CODEFORCES, username="tourist", is_active=False),  # Inactive
    ]
    
    try:
        print(f"Validating {len(test_accounts)} accounts...")
        validated = await scraper._validate_accounts(test_accounts)
        
        print(f"Valid accounts: {len(validated)}")
        print("Validated accounts:")
        for account in validated:
            print(f"  ✓ {account.platform.value}: {account.username or account.profile_url}")
        
    finally:
        await scraper.cleanup()


def save_demo_results(results, filename="demo_results.json"):
    """Save demo results to a JSON file."""
    try:
        # Convert results to JSON-serializable format
        json_data = {
            "user_id": results.user_id,
            "total_platforms": results.total_platforms,
            "successful_platforms": results.successful_platforms,
            "failed_platforms": results.failed_platforms,
            "total_processing_time": results.total_processing_time,
            "aggregated_at": results.aggregated_at.isoformat(),
            "platforms": {}
        }
        
        for platform, result in results.platforms.items():
            json_data["platforms"][platform.value] = {
                "success": result.success,
                "error_message": result.error_message,
                "processing_time": result.processing_time,
                "scraped_at": result.scraped_at.isoformat(),
                "data_available": result.data is not None
            }
        
        with open(filename, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        print(f"\nResults saved to {filename}")
        
    except Exception as e:
        print(f"Failed to save results: {str(e)}")


async def main():
    """Run all demo functions."""
    print("Multi-Platform Scraper Infrastructure Demo")
    print("=" * 50)
    
    try:
        # Run individual demos
        await demo_single_platform_scraping()
        await demo_platform_connectivity()
        await demo_account_validation()
        await demo_error_handling()
        await demo_multi_platform_scraping()
        
        print("\n" + "=" * 50)
        print("Demo completed successfully!")
        print("\nKey Features Demonstrated:")
        print("✓ Single platform scraping")
        print("✓ Multi-platform concurrent scraping")
        print("✓ Platform connectivity testing")
        print("✓ Account validation")
        print("✓ Error handling and graceful degradation")
        print("✓ Rate limiting and retry mechanisms")
        print("✓ Comprehensive result aggregation")
        
    except Exception as e:
        print(f"\nDemo failed with error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Run the demo
    asyncio.run(main())