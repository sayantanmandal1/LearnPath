#!/usr/bin/env python3
"""
Demo script for testing additional platform scrapers (Codeforces, AtCoder, HackerRank, Kaggle).
This script demonstrates the functionality of the four additional coding platform scrapers.
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, Any

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.external_apis.codeforces_scraper import CodeforcesScraper
from app.services.external_apis.atcoder_scraper import AtCoderScraper
from app.services.external_apis.hackerrank_scraper import HackerRankScraper
from app.services.external_apis.kaggle_scraper import KaggleScraper
from app.services.external_apis.multi_platform_scraper import (
    MultiPlatformScraper, 
    PlatformAccount, 
    PlatformType
)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_codeforces_scraper():
    """Test Codeforces scraper functionality."""
    logger.info("=== Testing Codeforces Scraper ===")
    
    scraper = CodeforcesScraper()
    
    try:
        # Test with a known public handle (tourist is a famous competitive programmer)
        test_handle = "tourist"
        
        async with scraper:
            # Test handle validation
            is_valid = await scraper.validate_handle(test_handle)
            logger.info(f"Handle '{test_handle}' is valid: {is_valid}")
            
            if is_valid:
                # Get user profile
                profile = await scraper.get_user_profile(test_handle)
                
                logger.info(f"Profile for {test_handle}:")
                logger.info(f"  - Handle: {profile.handle}")
                logger.info(f"  - Current Rating: {profile.stats.current_rating}")
                logger.info(f"  - Max Rating: {profile.stats.max_rating}")
                logger.info(f"  - Rank: {profile.stats.rank}")
                logger.info(f"  - Contests Participated: {profile.stats.contests_participated}")
                logger.info(f"  - Problems Solved: {profile.stats.problems_solved}")
                logger.info(f"  - Top Problem Tags: {list(profile.problem_tags.keys())[:5]}")
                logger.info(f"  - Languages Used: {list(profile.languages_used.keys())[:3]}")
                
                return True
            else:
                logger.warning(f"Handle '{test_handle}' not found")
                return False
                
    except Exception as e:
        logger.error(f"Codeforces scraper test failed: {str(e)}")
        return False


async def test_atcoder_scraper():
    """Test AtCoder scraper functionality."""
    logger.info("=== Testing AtCoder Scraper ===")
    
    scraper = AtCoderScraper()
    
    try:
        # Test with a known public username
        test_username = "tourist"
        
        async with scraper:
            # Test username validation
            is_valid = await scraper.validate_username(test_username)
            logger.info(f"Username '{test_username}' is valid: {is_valid}")
            
            if is_valid:
                # Get user profile
                profile = await scraper.get_user_profile(test_username)
                
                logger.info(f"Profile for {test_username}:")
                logger.info(f"  - Username: {profile.username}")
                logger.info(f"  - Current Rating: {profile.stats.current_rating}")
                logger.info(f"  - Max Rating: {profile.stats.max_rating}")
                logger.info(f"  - Rank: {profile.stats.rank}")
                logger.info(f"  - Contests Participated: {profile.stats.contests_participated}")
                logger.info(f"  - Problems Solved: {profile.stats.problems_solved}")
                logger.info(f"  - Country: {profile.country}")
                logger.info(f"  - Difficulty Distribution: {profile.difficulty_distribution}")
                
                return True
            else:
                logger.warning(f"Username '{test_username}' not found")
                return False
                
    except Exception as e:
        logger.error(f"AtCoder scraper test failed: {str(e)}")
        return False


async def test_hackerrank_scraper():
    """Test HackerRank scraper functionality."""
    logger.info("=== Testing HackerRank Scraper ===")
    
    scraper = HackerRankScraper()
    
    try:
        # Test with a known public username
        test_username = "shashank21j"  # A known HackerRank user
        
        async with scraper:
            # Test username validation
            is_valid = await scraper.validate_username(test_username)
            logger.info(f"Username '{test_username}' is valid: {is_valid}")
            
            if is_valid:
                # Get user profile
                profile = await scraper.get_user_profile(test_username)
                
                logger.info(f"Profile for {test_username}:")
                logger.info(f"  - Username: {profile.username}")
                logger.info(f"  - Name: {profile.name}")
                logger.info(f"  - Country: {profile.country}")
                logger.info(f"  - Total Score: {profile.stats.total_score}")
                logger.info(f"  - Challenges Solved: {profile.stats.challenges_solved}")
                logger.info(f"  - Certifications Earned: {profile.stats.certifications_earned}")
                logger.info(f"  - Badges Earned: {profile.stats.badges_earned}")
                logger.info(f"  - Domain Scores: {profile.domain_scores}")
                logger.info(f"  - Skill Levels: {profile.skill_levels}")
                
                return True
            else:
                logger.warning(f"Username '{test_username}' not found")
                return False
                
    except Exception as e:
        logger.error(f"HackerRank scraper test failed: {str(e)}")
        return False


async def test_kaggle_scraper():
    """Test Kaggle scraper functionality."""
    logger.info("=== Testing Kaggle Scraper ===")
    
    scraper = KaggleScraper()
    
    try:
        # Test with a known public username
        test_username = "sudalairajkumar"  # A known Kaggle user
        
        async with scraper:
            # Test username validation
            is_valid = await scraper.validate_username(test_username)
            logger.info(f"Username '{test_username}' is valid: {is_valid}")
            
            if is_valid:
                # Get user profile
                profile = await scraper.get_user_profile(test_username)
                
                logger.info(f"Profile for {test_username}:")
                logger.info(f"  - Username: {profile.username}")
                logger.info(f"  - Display Name: {profile.display_name}")
                logger.info(f"  - Location: {profile.location}")
                logger.info(f"  - Current Tier: {profile.stats.current_tier}")
                logger.info(f"  - Competitions Entered: {profile.stats.competitions_entered}")
                logger.info(f"  - Datasets Created: {profile.stats.datasets_created}")
                logger.info(f"  - Notebooks Created: {profile.stats.notebooks_created}")
                logger.info(f"  - Total Medals: {profile.stats.total_medals}")
                logger.info(f"  - Followers: {profile.stats.followers}")
                logger.info(f"  - Skills: {profile.skills[:5]}")
                
                return True
            else:
                logger.warning(f"Username '{test_username}' not found")
                return False
                
    except Exception as e:
        logger.error(f"Kaggle scraper test failed: {str(e)}")
        return False


async def test_multi_platform_integration():
    """Test integration of additional platforms with MultiPlatformScraper."""
    logger.info("=== Testing Multi-Platform Integration ===")
    
    scraper = MultiPlatformScraper(max_concurrent_scrapers=2)
    
    try:
        # Create test accounts for all additional platforms
        test_accounts = [
            PlatformAccount(
                platform=PlatformType.CODEFORCES,
                username="tourist",
                is_active=True
            ),
            PlatformAccount(
                platform=PlatformType.ATCODER,
                username="tourist",
                is_active=True
            ),
            PlatformAccount(
                platform=PlatformType.HACKERRANK,
                username="shashank21j",
                is_active=True
            ),
            PlatformAccount(
                platform=PlatformType.KAGGLE,
                username="sudalairajkumar",
                is_active=True
            )
        ]
        
        # Test concurrent scraping
        result = await scraper.scrape_all_platforms(test_accounts, "test_user_123")
        
        logger.info(f"Multi-platform scraping results:")
        logger.info(f"  - Total Platforms: {result.total_platforms}")
        logger.info(f"  - Successful: {result.successful_platforms}")
        logger.info(f"  - Failed: {result.failed_platforms}")
        logger.info(f"  - Total Processing Time: {result.total_processing_time:.2f}s")
        
        # Show results for each platform
        for platform, scraping_result in result.platforms.items():
            logger.info(f"  - {platform.value}: {'✓' if scraping_result.success else '✗'} "
                       f"({scraping_result.processing_time:.2f}s)")
            if not scraping_result.success:
                logger.warning(f"    Error: {scraping_result.error_message}")
        
        # Test platform connectivity
        connectivity = await scraper.test_platform_connectivity()
        logger.info(f"Platform connectivity:")
        for platform, is_connected in connectivity.items():
            logger.info(f"  - {platform.value}: {'✓' if is_connected else '✗'}")
        
        # Test rate limit status
        rate_limits = await scraper.get_platform_rate_limits()
        logger.info(f"Rate limit status:")
        for platform, limit_info in rate_limits.items():
            is_limited = limit_info.get('is_limited', False)
            reset_time = limit_info.get('reset_in_seconds', 0)
            logger.info(f"  - {platform.value}: {'Limited' if is_limited else 'OK'} "
                       f"(reset in {reset_time}s)")
        
        await scraper.cleanup()
        return result.successful_platforms > 0
        
    except Exception as e:
        logger.error(f"Multi-platform integration test failed: {str(e)}")
        return False


async def test_error_handling():
    """Test error handling for invalid usernames and network issues."""
    logger.info("=== Testing Error Handling ===")
    
    test_results = []
    
    # Test Codeforces with invalid handle
    try:
        scraper = CodeforcesScraper()
        async with scraper:
            profile = await scraper.get_user_profile("nonexistent_user_12345")
        test_results.append(False)  # Should not reach here
    except Exception as e:
        logger.info(f"Codeforces error handling: ✓ (caught: {type(e).__name__})")
        test_results.append(True)
    
    # Test AtCoder with invalid username
    try:
        scraper = AtCoderScraper()
        async with scraper:
            profile = await scraper.get_user_profile("nonexistent_user_12345")
        test_results.append(False)  # Should not reach here
    except Exception as e:
        logger.info(f"AtCoder error handling: ✓ (caught: {type(e).__name__})")
        test_results.append(True)
    
    # Test HackerRank with invalid username
    try:
        scraper = HackerRankScraper()
        async with scraper:
            profile = await scraper.get_user_profile("nonexistent_user_12345")
        test_results.append(False)  # Should not reach here
    except Exception as e:
        logger.info(f"HackerRank error handling: ✓ (caught: {type(e).__name__})")
        test_results.append(True)
    
    # Test Kaggle with invalid username
    try:
        scraper = KaggleScraper()
        async with scraper:
            profile = await scraper.get_user_profile("nonexistent_user_12345")
        test_results.append(False)  # Should not reach here
    except Exception as e:
        logger.info(f"Kaggle error handling: ✓ (caught: {type(e).__name__})")
        test_results.append(True)
    
    return all(test_results)


async def main():
    """Run all tests for additional platform scrapers."""
    logger.info("Starting Additional Platform Scrapers Demo")
    logger.info("=" * 60)
    
    test_results = []
    
    # Test individual scrapers
    test_results.append(await test_codeforces_scraper())
    test_results.append(await test_atcoder_scraper())
    test_results.append(await test_hackerrank_scraper())
    test_results.append(await test_kaggle_scraper())
    
    # Test multi-platform integration
    test_results.append(await test_multi_platform_integration())
    
    # Test error handling
    test_results.append(await test_error_handling())
    
    # Summary
    logger.info("=" * 60)
    logger.info("Test Summary:")
    logger.info(f"  - Codeforces Scraper: {'✓' if test_results[0] else '✗'}")
    logger.info(f"  - AtCoder Scraper: {'✓' if test_results[1] else '✗'}")
    logger.info(f"  - HackerRank Scraper: {'✓' if test_results[2] else '✗'}")
    logger.info(f"  - Kaggle Scraper: {'✓' if test_results[3] else '✗'}")
    logger.info(f"  - Multi-Platform Integration: {'✓' if test_results[4] else '✗'}")
    logger.info(f"  - Error Handling: {'✓' if test_results[5] else '✗'}")
    
    successful_tests = sum(test_results)
    total_tests = len(test_results)
    
    logger.info(f"Overall: {successful_tests}/{total_tests} tests passed")
    
    if successful_tests == total_tests:
        logger.info("🎉 All additional platform scrapers are working correctly!")
    else:
        logger.warning("⚠️  Some tests failed. Check the logs above for details.")
    
    return successful_tests == total_tests


if __name__ == "__main__":
    asyncio.run(main())