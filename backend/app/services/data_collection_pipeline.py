"""
Automated data collection pipeline for job market data
"""
import asyncio
import logging
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import json

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from app.core.database import get_db
from app.services.job_scrapers.scraper_manager import JobScraperManager
from app.services.job_scrapers.base_job_scraper import JobSearchParams
from app.services.job_analysis_service import JobAnalysisService
from app.services.market_trend_analyzer import MarketTrendAnalyzer
from app.repositories.job import JobRepository
from app.core.exceptions import ServiceException as ProcessingError

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for data collection pipeline"""
    name: str
    search_params: JobSearchParams
    platforms: List[str]
    schedule_hours: int  # How often to run (in hours)
    max_jobs_per_run: int
    enable_analysis: bool
    enable_trend_analysis: bool


@dataclass
class PipelineRun:
    """Result of a pipeline run"""
    config_name: str
    start_time: datetime
    end_time: Optional[datetime]
    status: str  # 'running', 'completed', 'failed'
    jobs_scraped: int
    jobs_stored: int
    jobs_processed: int
    errors: List[str]
    stats: Dict[str, Any]


class DataCollectionPipeline:
    """Automated data collection and processing pipeline"""
    
    def __init__(self):
        self.scraper_manager = JobScraperManager()
        self.analysis_service = JobAnalysisService()
        self.trend_analyzer = MarketTrendAnalyzer()
        self.job_repository = JobRepository()
        
        # Pipeline state
        self.running_pipelines = {}
        self.pipeline_history = []
        
        # Default configurations
        self.default_configs = [
            PipelineConfig(
                name="tech_jobs_general",
                search_params=JobSearchParams(
                    keywords="software engineer developer programmer",
                    location=None,
                    remote=True,
                    posted_days=1,
                    limit=200
                ),
                platforms=["linkedin", "indeed", "glassdoor"],
                schedule_hours=6,
                max_jobs_per_run=500,
                enable_analysis=True,
                enable_trend_analysis=False
            ),
            PipelineConfig(
                name="data_science_jobs",
                search_params=JobSearchParams(
                    keywords="data scientist machine learning AI",
                    location=None,
                    remote=True,
                    posted_days=1,
                    limit=150
                ),
                platforms=["linkedin", "indeed"],
                schedule_hours=8,
                max_jobs_per_run=300,
                enable_analysis=True,
                enable_trend_analysis=False
            ),
            PipelineConfig(
                name="frontend_jobs",
                search_params=JobSearchParams(
                    keywords="frontend developer react angular vue",
                    location=None,
                    remote=True,
                    posted_days=1,
                    limit=150
                ),
                platforms=["linkedin", "indeed"],
                schedule_hours=12,
                max_jobs_per_run=300,
                enable_analysis=True,
                enable_trend_analysis=False
            ),
            PipelineConfig(
                name="backend_jobs",
                search_params=JobSearchParams(
                    keywords="backend developer API microservices",
                    location=None,
                    remote=True,
                    posted_days=1,
                    limit=150
                ),
                platforms=["linkedin", "indeed"],
                schedule_hours=12,
                max_jobs_per_run=300,
                enable_analysis=True,
                enable_trend_analysis=False
            )
        ]
    
    async def start_pipeline_scheduler(
        self,
        configs: Optional[List[PipelineConfig]] = None
    ):
        """
        Start the automated pipeline scheduler
        
        Args:
            configs: Pipeline configurations (uses defaults if None)
        """
        if configs is None:
            configs = self.default_configs
        
        logger.info(f"Starting pipeline scheduler with {len(configs)} configurations")
        
        # Start each pipeline as a separate task
        tasks = []
        for config in configs:
            task = asyncio.create_task(
                self._run_scheduled_pipeline(config),
                name=f"pipeline_{config.name}"
            )
            tasks.append(task)
        
        # Wait for all pipelines (they run indefinitely)
        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            logger.error(f"Pipeline scheduler failed: {str(e)}")
            raise
    
    async def _run_scheduled_pipeline(self, config: PipelineConfig):
        """Run a single pipeline on schedule"""
        logger.info(f"Starting scheduled pipeline: {config.name}")
        
        while True:
            try:
                # Run the pipeline
                await self.run_pipeline(config)
                
                # Wait for next scheduled run
                logger.info(f"Pipeline {config.name} completed. Next run in {config.schedule_hours} hours")
                await asyncio.sleep(config.schedule_hours * 3600)
                
            except Exception as e:
                logger.error(f"Pipeline {config.name} failed: {str(e)}")
                # Wait 1 hour before retrying
                await asyncio.sleep(3600)
    
    async def run_pipeline(self, config: PipelineConfig) -> PipelineRun:
        """
        Run a single data collection pipeline
        
        Args:
            config: Pipeline configuration
            
        Returns:
            Pipeline run result
        """
        run = PipelineRun(
            config_name=config.name,
            start_time=datetime.utcnow(),
            end_time=None,
            status='running',
            jobs_scraped=0,
            jobs_stored=0,
            jobs_processed=0,
            errors=[],
            stats={}
        )
        
        self.running_pipelines[config.name] = run
        
        try:
            logger.info(f"Starting pipeline run: {config.name}")
            
            # Step 1: Scrape jobs
            async with get_db() as db:
                scraping_stats = await self.scraper_manager.scrape_and_store_jobs(
                    db=db,
                    search_params=config.search_params,
                    platforms=config.platforms,
                    deduplicate=True
                )
                
                # Update run statistics
                run.jobs_scraped = sum(
                    stats['scraped'] for stats in scraping_stats.values()
                )
                run.jobs_stored = sum(
                    stats['stored'] for stats in scraping_stats.values()
                )
                run.stats['scraping'] = scraping_stats
                
                logger.info(f"Pipeline {config.name}: Scraped {run.jobs_scraped}, Stored {run.jobs_stored}")
                
                # Step 2: Process jobs for skill extraction (if enabled)
                if config.enable_analysis:
                    try:
                        processing_stats = await self.analysis_service.process_unprocessed_jobs(
                            db=db,
                            batch_size=50,
                            max_jobs=config.max_jobs_per_run
                        )
                        
                        run.jobs_processed = processing_stats['processed']
                        run.stats['processing'] = processing_stats
                        
                        logger.info(f"Pipeline {config.name}: Processed {run.jobs_processed} jobs")
                        
                    except Exception as e:
                        error_msg = f"Job processing failed: {str(e)}"
                        run.errors.append(error_msg)
                        logger.error(error_msg)
                
                # Step 3: Run trend analysis (if enabled)
                if config.enable_trend_analysis:
                    try:
                        trend_stats = await self._run_trend_analysis(db)
                        run.stats['trend_analysis'] = trend_stats
                        
                        logger.info(f"Pipeline {config.name}: Trend analysis completed")
                        
                    except Exception as e:
                        error_msg = f"Trend analysis failed: {str(e)}"
                        run.errors.append(error_msg)
                        logger.error(error_msg)
                
                # Step 4: Cleanup old data
                await self._cleanup_old_data(db)
            
            run.status = 'completed'
            run.end_time = datetime.utcnow()
            
            logger.info(f"Pipeline {config.name} completed successfully")
            
        except Exception as e:
            run.status = 'failed'
            run.end_time = datetime.utcnow()
            error_msg = f"Pipeline failed: {str(e)}"
            run.errors.append(error_msg)
            logger.error(error_msg)
        
        finally:
            # Remove from running pipelines and add to history
            if config.name in self.running_pipelines:
                del self.running_pipelines[config.name]
            
            self.pipeline_history.append(run)
            
            # Keep only last 100 runs in history
            if len(self.pipeline_history) > 100:
                self.pipeline_history = self.pipeline_history[-100:]
        
        return run
    
    async def _run_trend_analysis(self, db: AsyncSession) -> Dict[str, Any]:
        """Run trend analysis as part of pipeline"""
        
        # Analyze skill demand trends
        skill_trends = await self.trend_analyzer.analyze_skill_demand_trends(
            db, days=30
        )
        
        # Detect emerging skills
        emerging_skills = await self.trend_analyzer.detect_emerging_skills(
            db, days=30
        )
        
        # Get salary predictions for top skills
        top_skills = [trend['skill_name'] for trend in skill_trends[:10]]
        salary_predictions = await self.trend_analyzer.predict_salaries(
            db, top_skills
        )
        
        return {
            'skill_trends_analyzed': len(skill_trends),
            'emerging_skills_detected': len(emerging_skills),
            'salary_predictions_generated': len(salary_predictions),
            'top_emerging_skills': [
                skill.skill_name for skill in emerging_skills[:5]
            ]
        }
    
    async def _cleanup_old_data(self, db: AsyncSession, days_to_keep: int = 365):
        """Clean up old job postings to manage database size"""
        
        cutoff_date = datetime.utcnow() - timedelta(days=days_to_keep)
        
        try:
            # Count jobs to be deleted
            count_query = select(func.count(JobPosting.id)).where(
                JobPosting.posted_date < cutoff_date
            )
            result = await db.execute(count_query)
            old_jobs_count = result.scalar() or 0
            
            if old_jobs_count > 0:
                # For now, just mark as inactive instead of deleting
                # to preserve historical data for trend analysis
                from sqlalchemy import update
                
                update_query = update(JobPosting).where(
                    JobPosting.posted_date < cutoff_date
                ).values(is_active=False)
                
                await db.execute(update_query)
                await db.commit()
                
                logger.info(f"Marked {old_jobs_count} old jobs as inactive")
            
        except Exception as e:
            logger.error(f"Cleanup failed: {str(e)}")
    
    async def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current status of all pipelines"""
        
        return {
            'running_pipelines': {
                name: {
                    'config_name': run.config_name,
                    'start_time': run.start_time,
                    'status': run.status,
                    'jobs_scraped': run.jobs_scraped,
                    'jobs_stored': run.jobs_stored,
                    'jobs_processed': run.jobs_processed,
                    'errors': run.errors
                }
                for name, run in self.running_pipelines.items()
            },
            'recent_runs': [
                {
                    'config_name': run.config_name,
                    'start_time': run.start_time,
                    'end_time': run.end_time,
                    'status': run.status,
                    'jobs_scraped': run.jobs_scraped,
                    'jobs_stored': run.jobs_stored,
                    'jobs_processed': run.jobs_processed,
                    'duration_minutes': (
                        (run.end_time - run.start_time).total_seconds() / 60
                        if run.end_time else None
                    ),
                    'error_count': len(run.errors)
                }
                for run in self.pipeline_history[-10:]  # Last 10 runs
            ]
        }
    
    async def run_manual_collection(
        self,
        keywords: str,
        location: Optional[str] = None,
        platforms: Optional[List[str]] = None,
        limit: int = 100
    ) -> Dict[str, Any]:
        """
        Run manual job collection for specific parameters
        
        Args:
            keywords: Search keywords
            location: Target location
            platforms: Platforms to scrape
            limit: Maximum jobs to collect
            
        Returns:
            Collection results
        """
        if platforms is None:
            platforms = ["linkedin", "indeed"]
        
        search_params = JobSearchParams(
            keywords=keywords,
            location=location,
            posted_days=7,
            limit=limit
        )
        
        config = PipelineConfig(
            name=f"manual_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            search_params=search_params,
            platforms=platforms,
            schedule_hours=0,  # Not scheduled
            max_jobs_per_run=limit,
            enable_analysis=True,
            enable_trend_analysis=False
        )
        
        run = await self.run_pipeline(config)
        
        return {
            'status': run.status,
            'jobs_scraped': run.jobs_scraped,
            'jobs_stored': run.jobs_stored,
            'jobs_processed': run.jobs_processed,
            'errors': run.errors,
            'stats': run.stats,
            'duration_minutes': (
                (run.end_time - run.start_time).total_seconds() / 60
                if run.end_time else None
            )
        }
    
    async def get_collection_metrics(self, days: int = 30) -> Dict[str, Any]:
        """
        Get metrics about data collection performance
        
        Args:
            days: Number of days to analyze
            
        Returns:
            Collection metrics
        """
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        # Filter recent runs
        recent_runs = [
            run for run in self.pipeline_history
            if run.start_time >= cutoff_date and run.status == 'completed'
        ]
        
        if not recent_runs:
            return {
                'period_days': days,
                'total_runs': 0,
                'metrics': {}
            }
        
        # Calculate metrics
        total_jobs_scraped = sum(run.jobs_scraped for run in recent_runs)
        total_jobs_stored = sum(run.jobs_stored for run in recent_runs)
        total_jobs_processed = sum(run.jobs_processed for run in recent_runs)
        
        avg_jobs_per_run = total_jobs_scraped / len(recent_runs)
        storage_rate = total_jobs_stored / total_jobs_scraped if total_jobs_scraped > 0 else 0
        processing_rate = total_jobs_processed / total_jobs_stored if total_jobs_stored > 0 else 0
        
        # Platform performance
        platform_stats = {}
        for run in recent_runs:
            scraping_stats = run.stats.get('scraping', {})
            for platform, stats in scraping_stats.items():
                if platform not in platform_stats:
                    platform_stats[platform] = {'scraped': 0, 'stored': 0, 'runs': 0}
                
                platform_stats[platform]['scraped'] += stats.get('scraped', 0)
                platform_stats[platform]['stored'] += stats.get('stored', 0)
                platform_stats[platform]['runs'] += 1
        
        return {
            'period_days': days,
            'total_runs': len(recent_runs),
            'metrics': {
                'total_jobs_scraped': total_jobs_scraped,
                'total_jobs_stored': total_jobs_stored,
                'total_jobs_processed': total_jobs_processed,
                'avg_jobs_per_run': avg_jobs_per_run,
                'storage_rate': storage_rate,
                'processing_rate': processing_rate,
                'platform_performance': platform_stats
            }
        }
    
    def stop_pipeline(self, config_name: str) -> bool:
        """
        Stop a running pipeline
        
        Args:
            config_name: Name of the pipeline to stop
            
        Returns:
            True if pipeline was stopped, False if not running
        """
        if config_name in self.running_pipelines:
            # In a real implementation, you'd need to cancel the asyncio task
            # For now, just mark as failed
            run = self.running_pipelines[config_name]
            run.status = 'stopped'
            run.end_time = datetime.utcnow()
            run.errors.append("Pipeline stopped manually")
            
            del self.running_pipelines[config_name]
            self.pipeline_history.append(run)
            
            logger.info(f"Pipeline {config_name} stopped manually")
            return True
        
        return False