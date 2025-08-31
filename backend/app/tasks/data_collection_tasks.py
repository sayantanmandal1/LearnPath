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
    name="app.tasks.data_collection_tasks.refresh_external_profiles",
    priority=TASK_PRIORITIES["MEDIUM"]
)
def refresh_external_profiles(user_ids: List[str]) -> Dict[str, Any]:
    """Refresh external profile data for specified users"""
    try:
        logger.info("Starting external profile refresh", user_count=len(user_ids))
        
        from app.services.external_apis.integration_service import IntegrationService
        
        integration_service = IntegrationService()
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            results = {}
            
            for user_id in user_ids:
                try:
                    # Refresh user's external profiles
                    profile_data = loop.run_until_complete(
                        integration_service.refresh_user_profiles(user_id)
                    )
                    
                    results[user_id] = {
                        "status": "success",
                        "profiles_updated": len(profile_data)
                    }
                    
                    # Invalidate user cache
                    cache_service = loop.run_until_complete(get_cache_service())
                    loop.run_until_complete(
                        cache_service.invalidation.invalidate_user_cache(user_id)
                    )
                    
                except Exception as e:
                    logger.error("Failed to refresh profile", user_id=user_id, error=str(e))
                    results[user_id] = {
                        "status": "error",
                        "error": str(e)
                    }
            
            successful_updates = len([r for r in results.values() if r["status"] == "success"])
            
            logger.info(
                "External profile refresh completed",
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
        raise