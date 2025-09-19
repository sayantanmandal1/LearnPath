"""
Concurrent processing service for multi-platform data collection and analysis
"""
import asyncio
import time
import hashlib
from typing import Dict, List, Any, Optional, Callable, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
import structlog

from app.services.cache_service import get_cache_service, CacheKeyBuilder
from app.services.external_apis.circuit_breaker import circuit_breaker_manager
from app.core.config import settings

logger = structlog.get_logger()


@dataclass
class ProcessingTask:
    """Represents a processing task with metadata"""
    task_id: str
    platform: str
    user_id: str
    priority: int = 5  # 1-10, higher is more priority
    retry_count: int = 0
    max_retries: int = 3
    timeout: float = 30.0
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()


@dataclass
class ProcessingResult:
    """Result of a processing task"""
    task_id: str
    platform: str
    user_id: str
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    processing_time: float = 0.0
    retry_count: int = 0


class ConcurrentProcessingService:
    """Service for concurrent processing of multiple platform data collection tasks"""
    
    def __init__(self, max_workers: int = None):
        self.max_workers = max_workers or settings.MAX_CONCURRENT_WORKERS
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        self.active_tasks: Dict[str, ProcessingTask] = {}
        self.completed_tasks: Dict[str, ProcessingResult] = {}
        self._task_semaphore = asyncio.Semaphore(self.max_workers)
        
    async def process_multiple_platforms(
        self,
        user_id: str,
        platform_configs: Dict[str, Dict[str, Any]],
        use_cache: bool = True
    ) -> Dict[str, ProcessingResult]:
        """
        Process multiple platform data collection concurrently
        
        Args:
            user_id: User identifier
            platform_configs: Dict mapping platform names to their config
            use_cache: Whether to use cached results
            
        Returns:
            Dict mapping platform names to processing results
        """
        logger.info(
            "Starting concurrent platform processing",
            user_id=user_id,
            platforms=list(platform_configs.keys())
        )
        
        # Create processing tasks
        tasks = []
        for platform, config in platform_configs.items():
            task = ProcessingTask(
                task_id=f"{user_id}_{platform}_{int(time.time())}",
                platform=platform,
                user_id=user_id,
                priority=config.get('priority', 5),
                timeout=config.get('timeout', 30.0),
                max_retries=config.get('max_retries', 3)
            )
            tasks.append(task)
        
        # Sort tasks by priority (higher priority first)
        tasks.sort(key=lambda t: t.priority, reverse=True)
        
        # Process tasks concurrently
        results = await self._execute_concurrent_tasks(tasks, platform_configs, use_cache)
        
        logger.info(
            "Concurrent platform processing completed",
            user_id=user_id,
            successful_platforms=len([r for r in results.values() if r.success]),
            total_platforms=len(results)
        )
        
        return results
    
    async def _execute_concurrent_tasks(
        self,
        tasks: List[ProcessingTask],
        platform_configs: Dict[str, Dict[str, Any]],
        use_cache: bool
    ) -> Dict[str, ProcessingResult]:
        """Execute tasks concurrently with proper resource management"""
        
        # Create semaphore for rate limiting
        semaphore = asyncio.Semaphore(self.max_workers)
        
        async def process_single_task(task: ProcessingTask) -> ProcessingResult:
            async with semaphore:
                return await self._process_platform_task(
                    task, 
                    platform_configs[task.platform], 
                    use_cache
                )
        
        # Execute all tasks concurrently
        task_coroutines = [process_single_task(task) for task in tasks]
        results = await asyncio.gather(*task_coroutines, return_exceptions=True)
        
        # Process results
        result_dict = {}
        for i, result in enumerate(results):
            task = tasks[i]
            if isinstance(result, Exception):
                result_dict[task.platform] = ProcessingResult(
                    task_id=task.task_id,
                    platform=task.platform,
                    user_id=task.user_id,
                    success=False,
                    error=str(result)
                )
            else:
                result_dict[task.platform] = result
        
        return result_dict
    
    async def _process_platform_task(
        self,
        task: ProcessingTask,
        config: Dict[str, Any],
        use_cache: bool
    ) -> ProcessingResult:
        """Process a single platform data collection task"""
        start_time = time.time()
        
        try:
            # Check cache first if enabled
            if use_cache:
                cached_data = await self._get_cached_platform_data(task.user_id, task.platform)
                if cached_data:
                    return ProcessingResult(
                        task_id=task.task_id,
                        platform=task.platform,
                        user_id=task.user_id,
                        success=True,
                        data=cached_data,
                        processing_time=time.time() - start_time
                    )
            
            # Get circuit breaker for this platform
            circuit_breaker = circuit_breaker_manager.get_breaker(task.platform)
            
            # Execute platform-specific data collection
            platform_data = await circuit_breaker.call(
                self._collect_platform_data,
                task.platform,
                task.user_id,
                config
            )
            
            # Cache the result
            if use_cache and platform_data:
                await self._cache_platform_data(task.user_id, task.platform, platform_data)
            
            return ProcessingResult(
                task_id=task.task_id,
                platform=task.platform,
                user_id=task.user_id,
                success=True,
                data=platform_data,
                processing_time=time.time() - start_time,
                retry_count=task.retry_count
            )
            
        except Exception as e:
            logger.error(
                "Platform processing task failed",
                task_id=task.task_id,
                platform=task.platform,
                user_id=task.user_id,
                error=str(e),
                retry_count=task.retry_count
            )
            
            # Retry logic
            if task.retry_count < task.max_retries:
                task.retry_count += 1
                await asyncio.sleep(2 ** task.retry_count)  # Exponential backoff
                return await self._process_platform_task(task, config, use_cache)
            
            return ProcessingResult(
                task_id=task.task_id,
                platform=task.platform,
                user_id=task.user_id,
                success=False,
                error=str(e),
                processing_time=time.time() - start_time,
                retry_count=task.retry_count
            )
    
    async def _collect_platform_data(
        self,
        platform: str,
        user_id: str,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Collect data from a specific platform"""
        
        # Import platform-specific scrapers
        if platform == "github":
            from app.services.external_apis.github_client import GitHubClient
            client = GitHubClient()
            return await client.get_user_profile(config.get('username'))
            
        elif platform == "leetcode":
            from app.services.external_apis.leetcode_scraper import LeetCodeScraper
            scraper = LeetCodeScraper()
            return await scraper.scrape_user_profile(config.get('username'))
            
        elif platform == "linkedin":
            from app.services.external_apis.linkedin_scraper import LinkedInScraper
            scraper = LinkedInScraper()
            return await scraper.scrape_profile(config.get('profile_url'))
            
        elif platform == "codeforces":
            from app.services.external_apis.codeforces_scraper import CodeforcesScraper
            scraper = CodeforcesScraper()
            return await scraper.scrape_user_profile(config.get('handle'))
            
        elif platform == "atcoder":
            from app.services.external_apis.atcoder_scraper import AtCoderScraper
            scraper = AtCoderScraper()
            return await scraper.scrape_user_profile(config.get('username'))
            
        elif platform == "hackerrank":
            from app.services.external_apis.hackerrank_scraper import HackerRankScraper
            scraper = HackerRankScraper()
            return await scraper.scrape_user_profile(config.get('username'))
            
        elif platform == "kaggle":
            from app.services.external_apis.kaggle_scraper import KaggleScraper
            scraper = KaggleScraper()
            return await scraper.scrape_user_profile(config.get('username'))
            
        else:
            raise ValueError(f"Unsupported platform: {platform}")
    
    async def _get_cached_platform_data(
        self,
        user_id: str,
        platform: str
    ) -> Optional[Dict[str, Any]]:
        """Get cached platform data if available"""
        try:
            cache_service = await get_cache_service()
            cache_key = CacheKeyBuilder.external_api(platform, user_id)
            return await cache_service.get(cache_key)
        except Exception as e:
            logger.error("Failed to get cached platform data", error=str(e))
            return None
    
    async def _cache_platform_data(
        self,
        user_id: str,
        platform: str,
        data: Dict[str, Any]
    ) -> None:
        """Cache platform data with enhanced metadata"""
        try:
            cache_service = await get_cache_service()
            
            # Enhanced data with metadata
            cached_data = {
                "data": data,
                "cached_at": datetime.utcnow().isoformat(),
                "platform": platform,
                "user_id": user_id,
                "data_quality_score": self._calculate_data_quality_score(data),
                "freshness_score": 100.0  # Fresh data gets max score
            }
            
            cache_key = CacheKeyBuilder.external_api(platform, user_id)
            # Cache for 6 hours (platform data doesn't change frequently)
            await cache_service.set(cache_key, cached_data, ttl=21600)
            
            # Also cache with platform-specific key for easier access
            platform_key = CacheKeyBuilder.platform_data(user_id, platform)
            await cache_service.set(platform_key, cached_data, ttl=21600)
            
        except Exception as e:
            logger.error("Failed to cache platform data", error=str(e))
    
    def _calculate_data_quality_score(self, data: Dict[str, Any]) -> float:
        """Calculate data quality score based on completeness and validity"""
        if not data:
            return 0.0
        
        score = 0.0
        max_score = 100.0
        
        # Check for essential fields
        essential_fields = ["username", "profile_data", "last_updated"]
        present_fields = sum(1 for field in essential_fields if field in data and data[field])
        score += (present_fields / len(essential_fields)) * 40
        
        # Check data richness
        if isinstance(data.get("profile_data"), dict):
            profile_data = data["profile_data"]
            richness_indicators = ["skills", "projects", "contributions", "achievements"]
            rich_fields = sum(1 for field in richness_indicators if field in profile_data and profile_data[field])
            score += (rich_fields / len(richness_indicators)) * 30
        
        # Check data freshness (if timestamp available)
        if "last_updated" in data:
            try:
                last_updated = datetime.fromisoformat(data["last_updated"].replace("Z", "+00:00"))
                age_hours = (datetime.utcnow() - last_updated.replace(tzinfo=None)).total_seconds() / 3600
                freshness_score = max(0, 30 - (age_hours / 24) * 5)  # Decrease score over time
                score += freshness_score
            except:
                pass
        
        return min(score, max_score)
    
    async def process_analysis_pipeline(
        self,
        user_id: str,
        platform_data: Dict[str, Any],
        analysis_types: List[str]
    ) -> Dict[str, Any]:
        """Process multiple analysis types concurrently"""
        
        logger.info(
            "Starting concurrent analysis pipeline",
            user_id=user_id,
            analysis_types=analysis_types
        )
        
        # Create analysis tasks
        analysis_tasks = []
        for analysis_type in analysis_types:
            if analysis_type == "skill_assessment":
                analysis_tasks.append(
                    self._run_skill_assessment(user_id, platform_data)
                )
            elif analysis_type == "career_recommendations":
                analysis_tasks.append(
                    self._run_career_recommendations(user_id, platform_data)
                )
            elif analysis_type == "learning_path":
                analysis_tasks.append(
                    self._run_learning_path_generation(user_id, platform_data)
                )
            elif analysis_type == "job_matching":
                analysis_tasks.append(
                    self._run_job_matching(user_id, platform_data)
                )
        
        # Execute analysis tasks concurrently
        results = await asyncio.gather(*analysis_tasks, return_exceptions=True)
        
        # Process results
        analysis_results = {}
        for i, result in enumerate(results):
            analysis_type = analysis_types[i]
            if isinstance(result, Exception):
                analysis_results[analysis_type] = {
                    "success": False,
                    "error": str(result)
                }
            else:
                analysis_results[analysis_type] = {
                    "success": True,
                    "data": result
                }
        
        logger.info(
            "Concurrent analysis pipeline completed",
            user_id=user_id,
            successful_analyses=len([r for r in analysis_results.values() if r["success"]])
        )
        
        return analysis_results
    
    async def _run_skill_assessment(
        self,
        user_id: str,
        platform_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run skill assessment analysis with caching"""
        try:
            # Check cache first
            cache_service = await get_cache_service()
            params_hash = hashlib.md5(str(platform_data).encode()).hexdigest()
            cache_key = CacheKeyBuilder.analysis_result(user_id, "skill_assessment", params_hash)
            
            cached_result = await cache_service.get(cache_key)
            if cached_result:
                logger.info("Using cached skill assessment", user_id=user_id)
                return cached_result
            
            # Run analysis
            from app.services.ai_analysis_service import AIAnalysisService
            ai_service = AIAnalysisService()
            result = await ai_service.analyze_skills(user_id, platform_data)
            
            # Cache result for 2 hours
            await cache_service.set(cache_key, result, ttl=7200)
            
            return result
            
        except Exception as e:
            logger.error("Skill assessment failed", user_id=user_id, error=str(e))
            raise
    
    async def _run_career_recommendations(
        self,
        user_id: str,
        platform_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run career recommendations analysis with caching"""
        try:
            # Check cache first
            cache_service = await get_cache_service()
            params_hash = hashlib.md5(str(platform_data).encode()).hexdigest()
            cache_key = CacheKeyBuilder.analysis_result(user_id, "career_recommendations", params_hash)
            
            cached_result = await cache_service.get(cache_key)
            if cached_result:
                logger.info("Using cached career recommendations", user_id=user_id)
                return cached_result
            
            # Run analysis
            from app.services.recommendation_service import RecommendationService
            rec_service = RecommendationService()
            result = await rec_service.generate_recommendations(user_id, platform_data)
            
            # Cache result for 1 hour (recommendations may change more frequently)
            await cache_service.set(cache_key, result, ttl=3600)
            
            return result
            
        except Exception as e:
            logger.error("Career recommendations failed", user_id=user_id, error=str(e))
            raise
    
    async def _run_learning_path_generation(
        self,
        user_id: str,
        platform_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run learning path generation with caching"""
        try:
            # Check cache first
            cache_service = await get_cache_service()
            params_hash = hashlib.md5(str(platform_data).encode()).hexdigest()
            cache_key = CacheKeyBuilder.analysis_result(user_id, "learning_path", params_hash)
            
            cached_result = await cache_service.get(cache_key)
            if cached_result:
                logger.info("Using cached learning path", user_id=user_id)
                return cached_result
            
            # Run analysis
            from app.services.learning_path_service import LearningPathService
            learning_service = LearningPathService()
            result = await learning_service.generate_learning_path(user_id, platform_data)
            
            # Cache result for 4 hours (learning paths are relatively stable)
            await cache_service.set(cache_key, result, ttl=14400)
            
            return result
            
        except Exception as e:
            logger.error("Learning path generation failed", user_id=user_id, error=str(e))
            raise
    
    async def _run_job_matching(
        self,
        user_id: str,
        platform_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run job matching analysis with caching"""
        try:
            # Check cache first
            cache_service = await get_cache_service()
            params_hash = hashlib.md5(str(platform_data).encode()).hexdigest()
            cache_key = CacheKeyBuilder.analysis_result(user_id, "job_matching", params_hash)
            
            cached_result = await cache_service.get(cache_key)
            if cached_result:
                logger.info("Using cached job matching", user_id=user_id)
                return cached_result
            
            # Run analysis
            from app.services.job_matching_service import JobMatchingService
            job_service = JobMatchingService()
            result = await job_service.match_jobs(user_id, platform_data)
            
            # Cache result for 30 minutes (job market changes frequently)
            await cache_service.set(cache_key, result, ttl=1800)
            
            return result
            
        except Exception as e:
            logger.error("Job matching failed", user_id=user_id, error=str(e))
            raise
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        return {
            "max_workers": self.max_workers,
            "active_tasks": len(self.active_tasks),
            "completed_tasks": len(self.completed_tasks),
            "success_rate": self._calculate_success_rate(),
            "average_processing_time": self._calculate_average_processing_time()
        }
    
    def _calculate_success_rate(self) -> float:
        """Calculate success rate of completed tasks"""
        if not self.completed_tasks:
            return 0.0
        
        successful = len([t for t in self.completed_tasks.values() if t.success])
        return (successful / len(self.completed_tasks)) * 100
    
    def _calculate_average_processing_time(self) -> float:
        """Calculate average processing time"""
        if not self.completed_tasks:
            return 0.0
        
        total_time = sum(t.processing_time for t in self.completed_tasks.values())
        return total_time / len(self.completed_tasks)
    
    async def cleanup(self):
        """Cleanup resources"""
        self.executor.shutdown(wait=True)
        self.active_tasks.clear()
        self.completed_tasks.clear()


# Global concurrent processing service instance
_concurrent_service: Optional[ConcurrentProcessingService] = None


async def get_concurrent_processing_service() -> ConcurrentProcessingService:
    """Get global concurrent processing service instance"""
    global _concurrent_service
    if _concurrent_service is None:
        _concurrent_service = ConcurrentProcessingService()
    return _concurrent_service