"""
Data collection background tasks
"""
import asyncio
from typing import Dict, List, Any
import structlog
from datetime import datetime

from app.core.celery_app import celery_app, TASK_PRIORITIES
from app.services.cache_service import get_cache_service, CacheKeyBuilder

logger = structlog.get_logger()


@celery_app.task(
    name="app.tasks.data_collection_tasks.refresh_job_market_data",
    priority=TASK_PRIORITIES["MEDIUM"]
)
def refresh_job_market_data() -> Dict[str, Any]:
    """Refresh job market data from external sources"""
    try:
        logger.info("Starting job market data refresh")
        
        # Import services (lazy import to avoid circular dependencies)
        from app.services.data_collection_pipeline import DataCollectionPipeline
        from app.services.market_trend_analyzer import MarketTrendAnalyzer
        
        # Initialize services
        pipeline = DataCollectionPipeline()
        analyzer = MarketTrendAnalyzer()
        
        # Run async operations in sync context
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # Collect fresh job data
            job_data = loop.run_until_complete(pipeline.collect_job_postings())
            
            # Analyze trends
            trend_analysis = loop.run_until_complete(analyzer.analyze_market_trends())
            
            # Invalidate related cache
            cache_service = loop.run_until_complete(get_cache_service())
            loop.run_until_complete(
                cache_service.invalidation.invalidate_market_data_cache()
            )
            
            logger.info(
                "Job market data refresh completed",
                jobs_collected=len(job_data),
                trends_analyzed=len(trend_analysis)
            )
            
            return {
                "status": "success",
                "jobs_collected": len(job_data),
                "trends_analyzed": len(trend_analysis),
                "refresh_timestamp": datetime.utcnow().isoformat()
            }
            
        finally:
            loop.close()
            
    except Exception as e:
        logger.error("Job market data refresh failed", error=str(e))
        raise


@celery_app.task(
    name="app.tasks.data_collection_tasks.update_skill_taxonomy",
    priority=TASK_PRIORITIES["LOW"]
)
def update_skill_taxonomy() -> Dict[str, Any]:
    """Update skill taxonomy based on market trends"""
    try:
        logger.info("Starting skill taxonomy update")
        
        from app.services.market_trend_analyzer import MarketTrendAnalyzer
        from app.core.database import AsyncSessionLocal
        from app.repositories.skill import SkillRepository
        
        analyzer = MarketTrendAnalyzer()
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # Analyze emerging skills
            emerging_skills = loop.run_until_complete(
                analyzer.identify_emerging_skills()
            )
            
            # Update skill database
            async def update_skills():
                async with AsyncSessionLocal() as session:
                    skill_repo = SkillRepository(session)
                    updated_count = 0
                    
                    for skill_data in emerging_skills:
                        skill = await skill_repo.create_or_update_skill(skill_data)
                        if skill:
                            updated_count += 1
                    
                    await session.commit()
                    return updated_count
            
            updated_count = loop.run_until_complete(update_skills())
            
            logger.info(
                "Skill taxonomy update completed",
                skills_updated=updated_count
            )
            
            return {
                "status": "success",
                "skills_updated": updated_count,
                "update_timestamp": datetime.utcnow().isoformat()
            }
            
        finally:
            loop.close()
            
    except Exception as e:
        logger.error("Skill taxonomy update failed", error=str(e))
        raise


@celery_app.task(
    bind=True,
    name="app.tasks.data_collection_tasks.refresh_external_profiles",
    priority=TASK_PRIORITIES["MEDIUM"],
    max_retries=2
)
def refresh_external_profiles(self, user_ids: List[str]) -> Dict[str, Any]:
    """Refresh external profile data for specified users with concurrent processing"""
    try:
        logger.info("Starting concurrent external profile refresh", user_count=len(user_ids))
        
        from app.services.concurrent_processing_service import get_concurrent_processing_service
        from app.services.intelligent_retry_service import get_intelligent_retry_service
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            concurrent_service = loop.run_until_complete(get_concurrent_processing_service())
            retry_service = get_intelligent_retry_service()
            
            results = {}
            total_users = len(user_ids)
            
            for i, user_id in enumerate(user_ids):
                # Update task progress
                self.update_state(
                    state="PROGRESS",
                    meta={
                        "current": i + 1,
                        "total": total_users,
                        "status": f"Processing user {user_id}"
                    }
                )
                
                try:
                    # Get user's platform configurations
                    platform_configs = loop.run_until_complete(
                        _get_user_platform_configs(user_id)
                    )
                    
                    if platform_configs:
                        # Process platforms concurrently with intelligent retry
                        platform_results = loop.run_until_complete(
                            concurrent_service.process_multiple_platforms(
                                user_id, platform_configs, use_cache=False
                            )
                        )
                        
                        successful_platforms = [
                            p for p, r in platform_results.items() if r.success
                        ]
                        
                        results[user_id] = {
                            "status": "success",
                            "platforms_updated": len(successful_platforms),
                            "total_platforms": len(platform_configs),
                            "updated_platforms": successful_platforms
                        }
                        
                        # Invalidate user cache
                        cache_service = loop.run_until_complete(get_cache_service())
                        loop.run_until_complete(
                            cache_service.invalidation.invalidate_user_cache(user_id)
                        )
                    else:
                        results[user_id] = {
                            "status": "skipped",
                            "reason": "No platform configurations found"
                        }
                    
                except Exception as e:
                    logger.error("Failed to refresh profile", user_id=user_id, error=str(e))
                    results[user_id] = {
                        "status": "error",
                        "error": str(e)
                    }
            
            successful_updates = len([r for r in results.values() if r["status"] == "success"])
            
            logger.info(
                "Concurrent external profile refresh completed",
                successful_updates=successful_updates,
                total_users=len(user_ids)
            )
            
            return {
                "status": "success",
                "successful_updates": successful_updates,
                "total_users": len(user_ids),
                "results": results,
                "refresh_timestamp": datetime.utcnow().isoformat()
            }
            
        finally:
            loop.close()
            
    except Exception as e:
        logger.error("External profile refresh failed", error=str(e))
        raise self.retry(exc=e, countdown=300, max_retries=2)


@celery_app.task(
    bind=True,
    name="app.tasks.data_collection_tasks.concurrent_analysis_pipeline",
    priority=TASK_PRIORITIES["HIGH"],
    max_retries=2
)
def concurrent_analysis_pipeline(
    self, 
    user_id: str, 
    analysis_types: List[str] = None,
    use_cache: bool = True
) -> Dict[str, Any]:
    """Run multiple analysis types concurrently for a user"""
    try:
        if analysis_types is None:
            analysis_types = [
                "skill_assessment",
                "career_recommendations", 
                "learning_path",
                "job_matching"
            ]
        
        logger.info(
            "Starting concurrent analysis pipeline",
            user_id=user_id,
            analysis_types=analysis_types
        )
        
        from app.services.concurrent_processing_service import get_concurrent_processing_service
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            concurrent_service = loop.run_until_complete(get_concurrent_processing_service())
            
            # Update task progress
            self.update_state(
                state="PROGRESS",
                meta={"current": 20, "total": 100, "status": "Gathering platform data"}
            )
            
            # Get user's platform data
            platform_data = loop.run_until_complete(_get_user_platform_data(user_id))
            
            self.update_state(
                state="PROGRESS",
                meta={"current": 50, "total": 100, "status": "Running concurrent analysis"}
            )
            
            # Run analysis pipeline concurrently
            analysis_results = loop.run_until_complete(
                concurrent_service.process_analysis_pipeline(
                    user_id, platform_data, analysis_types
                )
            )
            
            self.update_state(
                state="PROGRESS",
                meta={"current": 90, "total": 100, "status": "Caching results"}
            )
            
            # Cache analysis results with enhanced metadata
            if use_cache:
                cache_service = loop.run_until_complete(get_cache_service())
                for analysis_type, result in analysis_results.items():
                    if result["success"]:
                        # Enhanced cache data with metadata
                        cache_data = {
                            "data": result["data"],
                            "analysis_type": analysis_type,
                            "user_id": user_id,
                            "generated_at": datetime.utcnow().isoformat(),
                            "processing_time": result.get("processing_time", 0),
                            "data_sources": list(platform_data.keys()) if platform_data else [],
                            "cache_version": "v2"
                        }
                        
                        cache_key = f"analysis:{user_id}:{analysis_type}"
                        ttl = _get_analysis_cache_ttl(analysis_type)
                        
                        loop.run_until_complete(
                            cache_service.set(cache_key, cache_data, ttl=ttl)
                        )
            
            successful_analyses = len([r for r in analysis_results.values() if r["success"]])
            
            return {
                "status": "success",
                "user_id": user_id,
                "successful_analyses": successful_analyses,
                "total_analyses": len(analysis_types),
                "results": analysis_results,
                "processing_timestamp": datetime.utcnow().isoformat()
            }
            
        finally:
            loop.close()
            
    except Exception as e:
        logger.error("Concurrent analysis pipeline failed", user_id=user_id, error=str(e))
        raise self.retry(exc=e, countdown=180, max_retries=2)


@celery_app.task(
    bind=True,
    name="app.tasks.data_collection_tasks.intelligent_cache_warming",
    priority=TASK_PRIORITIES["MEDIUM"],
    max_retries=1
)
def intelligent_cache_warming(self, user_ids: List[str], analysis_types: List[str] = None) -> Dict[str, Any]:
    """Intelligently warm cache for active users with frequently accessed analysis"""
    try:
        if analysis_types is None:
            analysis_types = ["skill_assessment", "career_recommendations", "dashboard_data"]
        
        logger.info(
            "Starting intelligent cache warming",
            user_count=len(user_ids),
            analysis_types=analysis_types
        )
        
        from app.services.concurrent_processing_service import get_concurrent_processing_service
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            concurrent_service = loop.run_until_complete(get_concurrent_processing_service())
            cache_service = loop.run_until_complete(get_cache_service())
            
            warmed_count = 0
            failed_count = 0
            total_users = len(user_ids)
            
            for i, user_id in enumerate(user_ids):
                # Update task progress
                self.update_state(
                    state="PROGRESS",
                    meta={
                        "current": i + 1,
                        "total": total_users,
                        "status": f"Warming cache for user {user_id}"
                    }
                )
                
                try:
                    # Check if user has recent activity (would need to implement this check)
                    # For now, warm cache for all provided users
                    
                    # Get user's platform data
                    platform_data = loop.run_until_complete(_get_user_platform_data(user_id))
                    
                    if platform_data:
                        # Run analysis pipeline to warm cache
                        analysis_results = loop.run_until_complete(
                            concurrent_service.process_analysis_pipeline(
                                user_id, platform_data, analysis_types
                            )
                        )
                        
                        successful_analyses = len([r for r in analysis_results.values() if r["success"]])
                        if successful_analyses > 0:
                            warmed_count += 1
                            
                            logger.debug(
                                "Cache warmed for user",
                                user_id=user_id,
                                successful_analyses=successful_analyses
                            )
                    
                except Exception as e:
                    logger.error("Failed to warm cache for user", user_id=user_id, error=str(e))
                    failed_count += 1
                
                # Add small delay to avoid overwhelming the system
                await asyncio.sleep(0.1)
            
            logger.info(
                "Intelligent cache warming completed",
                warmed_users=warmed_count,
                failed_users=failed_count,
                total_users=total_users
            )
            
            return {
                "status": "success",
                "warmed_users": warmed_count,
                "failed_users": failed_count,
                "total_users": total_users,
                "analysis_types": analysis_types,
                "warming_timestamp": datetime.utcnow().isoformat()
            }
            
        finally:
            loop.close()
            
    except Exception as e:
        logger.error("Intelligent cache warming failed", error=str(e))
        raise self.retry(exc=e, countdown=300, max_retries=1)


@celery_app.task(
    name="app.tasks.data_collection_tasks.batch_platform_scraping",
    priority=TASK_PRIORITIES["LOW"]
)
def batch_platform_scraping(platform: str, user_configs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Batch scrape data from a specific platform for multiple users"""
    try:
        logger.info(
            "Starting batch platform scraping",
            platform=platform,
            user_count=len(user_configs)
        )
        
        from app.services.concurrent_processing_service import get_concurrent_processing_service
        from app.services.intelligent_retry_service import get_intelligent_retry_service
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            concurrent_service = loop.run_until_complete(get_concurrent_processing_service())
            retry_service = get_intelligent_retry_service()
            
            results = {}
            
            # Process users in batches to avoid overwhelming the platform
            batch_size = 5  # Process 5 users at a time
            for i in range(0, len(user_configs), batch_size):
                batch = user_configs[i:i + batch_size]
                
                batch_results = {}
                for user_config in batch:
                    user_id = user_config["user_id"]
                    platform_config = {platform: user_config["config"]}
                    
                    try:
                        platform_results = loop.run_until_complete(
                            concurrent_service.process_multiple_platforms(
                                user_id, platform_config, use_cache=True
                            )
                        )
                        
                        batch_results[user_id] = platform_results.get(platform)
                        
                    except Exception as e:
                        logger.error(
                            "Failed to scrape platform for user",
                            platform=platform,
                            user_id=user_id,
                            error=str(e)
                        )
                        batch_results[user_id] = {
                            "success": False,
                            "error": str(e)
                        }
                
                results.update(batch_results)
                
                # Add delay between batches to be respectful to the platform
                if i + batch_size < len(user_configs):
                    await asyncio.sleep(2)
            
            successful_scrapes = len([r for r in results.values() if r and r.success])
            
            logger.info(
                "Batch platform scraping completed",
                platform=platform,
                successful_scrapes=successful_scrapes,
                total_users=len(user_configs)
            )
            
            return {
                "status": "success",
                "platform": platform,
                "successful_scrapes": successful_scrapes,
                "total_users": len(user_configs),
                "results": results,
                "scraping_timestamp": datetime.utcnow().isoformat()
            }
            
        finally:
            loop.close()
            
    except Exception as e:
        logger.error("Batch platform scraping failed", platform=platform, error=str(e))
        raise


async def _get_user_platform_configs(user_id: str) -> Dict[str, Dict[str, Any]]:
    """Get platform configurations for a user"""
    try:
        from app.core.database import AsyncSessionLocal
        from app.repositories.profile import ProfileRepository
        
        async with AsyncSessionLocal() as session:
            profile_repo = ProfileRepository(session)
            profile = await profile_repo.get_by_user_id(user_id)
            
            if not profile or not profile.platform_accounts:
                return {}
            
            platform_configs = {}
            for account in profile.platform_accounts:
                platform_configs[account.platform] = {
                    "username": account.username,
                    "profile_url": account.profile_url,
                    "priority": 5,
                    "timeout": 30.0,
                    "max_retries": 3
                }
            
            return platform_configs
            
    except Exception as e:
        logger.error("Failed to get user platform configs", user_id=user_id, error=str(e))
        return {}


async def _get_user_platform_data(user_id: str) -> Dict[str, Any]:
    """Get cached platform data for a user"""
    try:
        cache_service = await get_cache_service()
        
        # Try to get cached platform data
        platforms = ["github", "leetcode", "linkedin", "codeforces", "atcoder", "hackerrank", "kaggle"]
        platform_data = {}
        
        for platform in platforms:
            cache_key = CacheKeyBuilder.external_api(platform, user_id)
            data = await cache_service.get(cache_key)
            if data:
                platform_data[platform] = data
        
        return platform_data
        
    except Exception as e:
        logger.error("Failed to get user platform data", user_id=user_id, error=str(e))
        return {}


def _get_analysis_cache_ttl(analysis_type: str) -> int:
    """Get appropriate cache TTL for different analysis types"""
    ttl_mapping = {
        "skill_assessment": 7200,      # 2 hours - skills change slowly
        "career_recommendations": 3600, # 1 hour - recommendations may change
        "learning_path": 14400,        # 4 hours - learning paths are stable
        "job_matching": 1800,          # 30 minutes - job market changes frequently
        "resume_analysis": 86400,      # 24 hours - resume analysis is expensive
        "market_insights": 3600,       # 1 hour - market data changes regularly
        "dashboard_data": 1800         # 30 minutes - dashboard should be fresh
    }
    
    return ttl_mapping.get(analysis_type, 3600)  # Default 1 hour