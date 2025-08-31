"""
Automated Job Posting Collection Pipeline
Collects job postings from multiple sources with daily updates.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import logging

from app.core.logging import get_logger
from app.services.job_scrapers.scraper_manager import ScraperManager
from app.services.data_pipeline.pipeline_monitor import get_pipeline_monitor
from app.services.data_pipeline.data_quality_validator import DataQualityValidator
from app.repositories.job import JobRepository
from app.core.database import get_db_session

logger = get_logger(__name__)


class JobCollectionPipeline:
    """
    Automated pipeline for collecting job postings from multiple sources
    """
    
    def __init__(self):
        self.scraper_manager = ScraperManager()
        self.data_validator = DataQualityValidator()
        self.monitor = None
        
    async def execute(self, metadata: Dict[str, Any] = None):
        """Execute the job collection pipeline"""
        execution_id = metadata.get('execution_id', f"job_collection_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}")
        
        self.monitor = await get_pipeline_monitor()
        
        try:
            logger.info(f"Starting job collection pipeline: {execution_id}")
            
            # Initialize metrics
            await self.monitor.update_job_metrics(
                execution_id,
                status="running",
                records_processed=0,
                records_failed=0
            )
            
            # Get collection parameters
            sources = metadata.get('sources', ['linkedin', 'indeed', 'glassdoor'])
            max_jobs_per_source = metadata.get('max_jobs_per_source', 1000)
            search_terms = metadata.get('search_terms', self._get_default_search_terms())
            
            total_collected = 0
            total_failed = 0
            collection_results = {}
            
            # Collect from each source
            for source in sources:
                try:
                    logger.info(f"Collecting jobs from {source}")
                    
                    source_results = await self._collect_from_source(
                        source, search_terms, max_jobs_per_source
                    )
                    
                    collection_results[source] = source_results
                    total_collected += source_results['collected']
                    total_failed += source_results['failed']
                    
                    # Update metrics
                    await self.monitor.update_job_metrics(
                        execution_id,
                        records_processed=total_collected,
                        records_failed=total_failed
                    )
                    
                except Exception as e:
                    logger.error(f"Failed to collect from {source}: {e}")
                    total_failed += 1
                    collection_results[source] = {'collected': 0, 'failed': 1, 'error': str(e)}
            
            # Validate collected data
            validation_results = await self._validate_collected_data(total_collected)
            
            # Update final metrics
            await self.monitor.update_job_metrics(
                execution_id,
                status="completed",
                records_processed=total_collected,
                records_failed=total_failed,
                data_quality_score=validation_results['quality_score']
            )
            
            # Store collection summary
            await self._store_collection_summary(execution_id, collection_results, validation_results)
            
            logger.info(f"Job collection pipeline completed: {total_collected} jobs collected")
            
        except Exception as e:
            logger.error(f"Job collection pipeline failed: {e}")
            
            await self.monitor.update_job_metrics(
                execution_id,
                status="failed",
                error_count=1
            )
            raise
    
    async def _collect_from_source(self, source: str, search_terms: List[str], max_jobs: int) -> Dict[str, Any]:
        """Collect jobs from a specific source"""
        collected = 0
        failed = 0
        
        try:
            scraper = await self.scraper_manager.get_scraper(source)
            if not scraper:
                raise ValueError(f"No scraper available for source: {source}")
            
            async with get_db_session() as db:
                job_repo = JobRepository(db)
                
                for search_term in search_terms:
                    try:
                        # Get jobs for this search term
                        jobs = await scraper.search_jobs(
                            query=search_term,
                            limit=max_jobs // len(search_terms)
                        )
                        
                        for job_data in jobs:
                            try:
                                # Validate job data
                                if await self.data_validator.validate_job_posting(job_data):
                                    # Check if job already exists
                                    existing_job = await job_repo.get_by_external_id(
                                        job_data.get('external_id'),
                                        source
                                    )
                                    
                                    if not existing_job:
                                        # Create new job posting
                                        await job_repo.create_job_posting(job_data)
                                        collected += 1
                                    else:
                                        # Update existing job if needed
                                        await job_repo.update_job_posting(existing_job.id, job_data)
                                        collected += 1
                                else:
                                    failed += 1
                                    
                            except Exception as e:
                                logger.error(f"Failed to process job from {source}: {e}")
                                failed += 1
                                
                    except Exception as e:
                        logger.error(f"Failed to search jobs for '{search_term}' on {source}: {e}")
                        failed += 1
                        
        except Exception as e:
            logger.error(f"Failed to collect from {source}: {e}")
            failed += 1
        
        return {
            'collected': collected,
            'failed': failed,
            'source': source
        }
    
    async def _validate_collected_data(self, total_collected: int) -> Dict[str, Any]:
        """Validate the quality of collected job data"""
        try:
            # Check if we met minimum collection threshold
            min_jobs_threshold = 100  # Minimum jobs per day
            meets_threshold = total_collected >= min_jobs_threshold
            
            # Calculate quality score based on collection success
            if total_collected == 0:
                quality_score = 0.0
            elif total_collected < min_jobs_threshold:
                quality_score = total_collected / min_jobs_threshold * 0.8  # Max 0.8 if below threshold
            else:
                quality_score = 1.0
            
            # Additional data quality checks
            async with get_db_session() as db:
                job_repo = JobRepository(db)
                
                # Check for duplicate jobs
                recent_jobs = await job_repo.get_recent_jobs(hours=24)
                duplicate_count = await self._count_duplicates(recent_jobs)
                
                # Adjust quality score based on duplicates
                if len(recent_jobs) > 0:
                    duplicate_rate = duplicate_count / len(recent_jobs)
                    quality_score *= (1.0 - min(duplicate_rate, 0.5))  # Max 50% penalty for duplicates
            
            return {
                'quality_score': quality_score,
                'meets_threshold': meets_threshold,
                'total_collected': total_collected,
                'duplicate_count': duplicate_count,
                'validation_timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to validate collected data: {e}")
            return {
                'quality_score': 0.5,  # Default score on validation failure
                'meets_threshold': False,
                'total_collected': total_collected,
                'validation_error': str(e)
            }
    
    async def _count_duplicates(self, jobs: List[Any]) -> int:
        """Count duplicate job postings"""
        seen_jobs = set()
        duplicates = 0
        
        for job in jobs:
            # Create a signature for the job
            signature = f"{job.title}_{job.company}_{job.location}"
            
            if signature in seen_jobs:
                duplicates += 1
            else:
                seen_jobs.add(signature)
        
        return duplicates
    
    async def _store_collection_summary(self, execution_id: str, results: Dict[str, Any], validation: Dict[str, Any]):
        """Store collection summary for reporting"""
        try:
            from app.core.redis import get_redis_client
            
            redis_client = await get_redis_client()
            
            summary = {
                'execution_id': execution_id,
                'timestamp': datetime.utcnow().isoformat(),
                'collection_results': results,
                'validation_results': validation,
                'total_collected': sum(r.get('collected', 0) for r in results.values()),
                'total_failed': sum(r.get('failed', 0) for r in results.values())
            }
            
            # Store summary
            summary_key = f"job_collection_summary:{execution_id}"
            await redis_client.set(
                summary_key,
                json.dumps(summary),
                ex=86400 * 30  # Keep for 30 days
            )
            
            # Update daily stats
            today = datetime.utcnow().strftime('%Y-%m-%d')
            daily_stats_key = f"job_collection_daily:{today}"
            
            await redis_client.hincrby(daily_stats_key, 'total_collected', summary['total_collected'])
            await redis_client.hincrby(daily_stats_key, 'total_failed', summary['total_failed'])
            await redis_client.expire(daily_stats_key, 86400 * 90)  # Keep for 90 days
            
        except Exception as e:
            logger.error(f"Failed to store collection summary: {e}")
    
    def _get_default_search_terms(self) -> List[str]:
        """Get default search terms for job collection"""
        return [
            "software engineer",
            "data scientist",
            "product manager",
            "frontend developer",
            "backend developer",
            "full stack developer",
            "devops engineer",
            "machine learning engineer",
            "data analyst",
            "ui ux designer",
            "cybersecurity analyst",
            "cloud architect",
            "mobile developer",
            "qa engineer",
            "business analyst"
        ]
    
    async def get_collection_stats(self, days: int = 7) -> Dict[str, Any]:
        """Get job collection statistics for the last N days"""
        try:
            from app.core.redis import get_redis_client
            
            redis_client = await get_redis_client()
            stats = {
                'daily_stats': {},
                'total_collected': 0,
                'total_failed': 0,
                'average_per_day': 0,
                'success_rate': 0.0
            }
            
            # Get daily stats
            for i in range(days):
                date = (datetime.utcnow() - timedelta(days=i)).strftime('%Y-%m-%d')
                daily_key = f"job_collection_daily:{date}"
                
                daily_data = await redis_client.hgetall(daily_key)
                if daily_data:
                    collected = int(daily_data.get('total_collected', 0))
                    failed = int(daily_data.get('total_failed', 0))
                    
                    stats['daily_stats'][date] = {
                        'collected': collected,
                        'failed': failed
                    }
                    
                    stats['total_collected'] += collected
                    stats['total_failed'] += failed
            
            # Calculate averages and rates
            if days > 0:
                stats['average_per_day'] = stats['total_collected'] / days
            
            total_attempts = stats['total_collected'] + stats['total_failed']
            if total_attempts > 0:
                stats['success_rate'] = stats['total_collected'] / total_attempts
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            return {}


import json