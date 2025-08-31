"""
Analytics and reporting background tasks
"""
import asyncio
from typing import Dict, List, Any
import structlog
from datetime import datetime, timedelta

from app.core.celery_app import celery_app, TASK_PRIORITIES
from app.services.cache_service import get_cache_service, CacheKeyBuilder

logger = structlog.get_logger()


@celery_app.task(
    name="app.tasks.analytics_tasks.generate_daily_analytics",
    priority=TASK_PRIORITIES["LOW"]
)
def generate_daily_analytics() -> Dict[str, Any]:
    """Generate daily analytics reports"""
    try:
        logger.info("Starting daily analytics generation")
        
        from app.services.analytics_service import AnalyticsService
        from app.core.database import AsyncSessionLocal
        
        analytics_service = AnalyticsService()
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # Generate system-wide analytics
            system_analytics = loop.run_until_complete(
                analytics_service.generate_system_analytics()
            )
            
            # Generate user engagement analytics
            engagement_analytics = loop.run_until_complete(
                analytics_service.generate_engagement_analytics()
            )
            
            # Generate recommendation performance analytics
            recommendation_analytics = loop.run_until_complete(
                analytics_service.generate_recommendation_analytics()
            )
            
            # Cache the analytics data
            cache_service = loop.run_until_complete(get_cache_service())
            
            analytics_data = {
                "system": system_analytics,
                "engagement": engagement_analytics,
                "recommendations": recommendation_analytics,
                "generated_at": datetime.utcnow().isoformat()
            }
            
            cache_key = f"daily_analytics:{datetime.utcnow().strftime('%Y-%m-%d')}"
            loop.run_until_complete(
                cache_service.set(cache_key, analytics_data, ttl=86400)  # 24 hours
            )
            
            logger.info("Daily analytics generation completed")
            
            return {
                "status": "success",
                "analytics_generated": True,
                "cache_key": cache_key,
                "generation_timestamp": datetime.utcnow().isoformat()
            }
            
        finally:
            loop.close()
            
    except Exception as e:
        logger.error("Daily analytics generation failed", error=str(e))
        raise


@celery_app.task(
    name="app.tasks.analytics_tasks.generate_user_report",
    priority=TASK_PRIORITIES["MEDIUM"]
)
def generate_user_report(user_id: str, report_type: str = "comprehensive") -> Dict[str, Any]:
    """Generate detailed user analytics report"""
    try:
        logger.info("Starting user report generation", user_id=user_id, report_type=report_type)
        
        from app.services.analytics_service import AnalyticsService
        from app.services.pdf_report_service import PDFReportService
        
        analytics_service = AnalyticsService()
        pdf_service = PDFReportService()
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # Generate user analytics
            user_analytics = loop.run_until_complete(
                analytics_service.generate_user_analytics(user_id)
            )
            
            # Generate PDF report if requested
            pdf_path = None
            if report_type == "comprehensive":
                pdf_path = loop.run_until_complete(
                    pdf_service.generate_career_analysis_report(user_id, user_analytics)
                )
            
            # Cache the report
            cache_service = loop.run_until_complete(get_cache_service())
            cache_key = CacheKeyBuilder.analytics_data(user_id, report_type)
            
            report_data = {
                "user_id": user_id,
                "report_type": report_type,
                "analytics": user_analytics,
                "pdf_path": pdf_path,
                "generated_at": datetime.utcnow().isoformat()
            }
            
            loop.run_until_complete(
                cache_service.set(cache_key, report_data, ttl=3600)  # 1 hour
            )
            
            logger.info(
                "User report generation completed",
                user_id=user_id,
                has_pdf=pdf_path is not None
            )
            
            return {
                "status": "success",
                "user_id": user_id,
                "report_type": report_type,
                "pdf_generated": pdf_path is not None,
                "cache_key": cache_key,
                "generation_timestamp": datetime.utcnow().isoformat()
            }
            
        finally:
            loop.close()
            
    except Exception as e:
        logger.error("User report generation failed", user_id=user_id, error=str(e))
        raise


@celery_app.task(
    name="app.tasks.analytics_tasks.update_recommendation_metrics",
    priority=TASK_PRIORITIES["MEDIUM"]
)
def update_recommendation_metrics() -> Dict[str, Any]:
    """Update recommendation system performance metrics"""
    try:
        logger.info("Starting recommendation metrics update")
        
        from app.services.analytics_service import AnalyticsService
        from app.core.database import AsyncSessionLocal
        from app.repositories.user import UserRepository
        
        analytics_service = AnalyticsService()
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # Get recommendation performance data
            metrics = loop.run_until_complete(
                analytics_service.calculate_recommendation_metrics()
            )
            
            # Update cached metrics
            cache_service = loop.run_until_complete(get_cache_service())
            cache_key = "recommendation_metrics:latest"
            
            loop.run_until_complete(
                cache_service.set(cache_key, metrics, ttl=3600)  # 1 hour
            )
            
            logger.info(
                "Recommendation metrics update completed",
                accuracy_score=metrics.get("accuracy_score", 0),
                user_satisfaction=metrics.get("user_satisfaction", 0)
            )
            
            return {
                "status": "success",
                "metrics": metrics,
                "update_timestamp": datetime.utcnow().isoformat()
            }
            
        finally:
            loop.close()
            
    except Exception as e:
        logger.error("Recommendation metrics update failed", error=str(e))
        raise


@celery_app.task(
    name="app.tasks.analytics_tasks.cleanup_old_analytics",
    priority=TASK_PRIORITIES["LOW"]
)
def cleanup_old_analytics() -> Dict[str, Any]:
    """Clean up old analytics data to save storage"""
    try:
        logger.info("Starting analytics cleanup")
        
        from app.core.database import AsyncSessionLocal
        from sqlalchemy import text
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # Clean up analytics data older than 90 days
            cutoff_date = datetime.utcnow() - timedelta(days=90)
            
            async def cleanup_database():
                async with AsyncSessionLocal() as session:
                    # Clean up old analytics records (if you have an analytics table)
                    cleanup_query = text("""
                        DELETE FROM analytics_records 
                        WHERE created_at < :cutoff_date
                    """)
                    
                    try:
                        result = await session.execute(cleanup_query, {"cutoff_date": cutoff_date})
                        await session.commit()
                        return result.rowcount if hasattr(result, 'rowcount') else 0
                    except Exception:
                        # Table might not exist yet
                        return 0
            
            deleted_records = loop.run_until_complete(cleanup_database())
            
            # Clean up old cache entries
            cache_service = loop.run_until_complete(get_cache_service())
            redis_manager = loop.run_until_complete(cache_service.redis)
            
            # Clean up old daily analytics cache
            old_keys = []
            for i in range(90, 365):  # Clean analytics older than 90 days
                old_date = (datetime.utcnow() - timedelta(days=i)).strftime('%Y-%m-%d')
                old_key = f"daily_analytics:{old_date}"
                old_keys.append(old_key)
            
            cleaned_cache_keys = 0
            for key in old_keys:
                if loop.run_until_complete(cache_service.delete(key)):
                    cleaned_cache_keys += 1
            
            logger.info(
                "Analytics cleanup completed",
                deleted_records=deleted_records,
                cleaned_cache_keys=cleaned_cache_keys
            )
            
            return {
                "status": "success",
                "deleted_records": deleted_records,
                "cleaned_cache_keys": cleaned_cache_keys,
                "cleanup_timestamp": datetime.utcnow().isoformat()
            }
            
        finally:
            loop.close()
            
    except Exception as e:
        logger.error("Analytics cleanup failed", error=str(e))
        raise