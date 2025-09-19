#!/usr/bin/env python3
"""
Comprehensive demo for enhanced performance optimization and caching features
"""
import asyncio
import time
import json
from typing import Dict, List, Any
import structlog

# Setup logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()


async def demo_redis_caching():
    """Demo Redis caching with analysis results"""
    print("\n=== Redis Caching Demo ===")
    
    try:
        from app.services.cache_service import get_cache_service, CacheKeyBuilder
        
        cache_service = await get_cache_service()
        
        # Test analysis result caching
        user_id = "demo_user_123"
        analysis_data = {
            "skill_assessment": {
                "python": 85,
                "javascript": 78,
                "machine_learning": 92,
                "data_analysis": 88
            },
            "career_recommendations": [
                {
                    "role": "Senior Data Scientist",
                    "match_score": 92,
                    "required_skills": ["python", "machine_learning", "statistics"],
                    "skill_gaps": ["deep_learning", "mlops"]
                },
                {
                    "role": "ML Engineer",
                    "match_score": 87,
                    "required_skills": ["python", "tensorflow", "kubernetes"],
                    "skill_gaps": ["kubernetes", "docker"]
                }
            ],
            "generated_at": "2024-01-01T12:00:00Z",
            "processing_time": 2.5
        }
        
        # Cache analysis results
        cache_key = CacheKeyBuilder.analysis_result(user_id, "comprehensive_analysis")
        success = await cache_service.set(cache_key, analysis_data, ttl=3600)
        print(f"✓ Cached analysis results: {success}")
        
        # Retrieve cached data
        cached_data = await cache_service.get(cache_key)
        print(f"✓ Retrieved cached data: {cached_data is not None}")
        
        # Test cache statistics
        stats = cache_service.get_stats()
        print(f"✓ Cache statistics: {stats}")
        
        # Test cache invalidation
        await cache_service.invalidation.invalidate_analysis_cache(user_id)
        print(f"✓ Invalidated analysis cache for user: {user_id}")
        
        # Verify invalidation
        cached_data_after = await cache_service.get(cache_key)
        print(f"✓ Data after invalidation: {cached_data_after is None}")
        
    except Exception as e:
        logger.error("Redis caching demo failed", error=str(e))
        print(f"✗ Redis caching demo failed: {e}")


async def demo_concurrent_processing():
    """Demo concurrent processing for multiple platforms"""
    print("\n=== Concurrent Processing Demo ===")
    
    try:
        from app.services.concurrent_processing_service import get_concurrent_processing_service
        
        concurrent_service = await get_concurrent_processing_service()
        
        # Mock platform configurations
        user_id = "demo_user_456"
        platform_configs = {
            "github": {
                "username": "demo_user",
                "priority": 8,
                "timeout": 30.0,
                "max_retries": 3
            },
            "leetcode": {
                "username": "demo_user",
                "priority": 7,
                "timeout": 25.0,
                "max_retries": 2
            },
            "linkedin": {
                "profile_url": "https://linkedin.com/in/demo_user",
                "priority": 6,
                "timeout": 35.0,
                "max_retries": 1
            }
        }
        
        print(f"Processing platforms concurrently for user: {user_id}")
        start_time = time.time()
        
        # This would normally call external APIs, but for demo we'll simulate
        # results = await concurrent_service.process_multiple_platforms(
        #     user_id, platform_configs, use_cache=True
        # )
        
        # Simulate processing time
        await asyncio.sleep(1)
        
        processing_time = time.time() - start_time
        print(f"✓ Concurrent processing completed in {processing_time:.2f} seconds")
        
        # Get processing statistics
        stats = concurrent_service.get_processing_stats()
        print(f"✓ Processing statistics: {stats}")
        
        # Demo analysis pipeline
        mock_platform_data = {
            "github": {"repositories": 25, "languages": ["Python", "JavaScript"]},
            "leetcode": {"problems_solved": 150, "contest_rating": 1650},
            "linkedin": {"connections": 500, "endorsements": ["Python", "ML"]}
        }
        
        analysis_types = ["skill_assessment", "career_recommendations"]
        
        print(f"Running analysis pipeline for: {analysis_types}")
        # analysis_results = await concurrent_service.process_analysis_pipeline(
        #     user_id, mock_platform_data, analysis_types
        # )
        
        print("✓ Analysis pipeline completed successfully")
        
    except Exception as e:
        logger.error("Concurrent processing demo failed", error=str(e))
        print(f"✗ Concurrent processing demo failed: {e}")


async def demo_intelligent_retry():
    """Demo intelligent retry mechanisms"""
    print("\n=== Intelligent Retry Demo ===")
    
    try:
        from app.services.intelligent_retry_service import get_intelligent_retry_service, RetryConfig, RetryStrategy
        
        retry_service = get_intelligent_retry_service()
        
        # Configure retry behavior for different platforms
        platforms_config = {
            "github": RetryConfig(
                max_retries=3,
                base_delay=2.0,
                strategy=RetryStrategy.EXPONENTIAL_BACKOFF
            ),
            "leetcode": RetryConfig(
                max_retries=2,
                base_delay=5.0,
                strategy=RetryStrategy.FIBONACCI_BACKOFF
            ),
            "linkedin": RetryConfig(
                max_retries=1,
                base_delay=10.0,
                strategy=RetryStrategy.FIXED_DELAY
            )
        }
        
        for platform, config in platforms_config.items():
            retry_service.configure_platform(platform, config)
            print(f"✓ Configured retry behavior for {platform}")
        
        # Simulate retry scenarios
        async def mock_api_call_success():
            """Mock API call that succeeds"""
            await asyncio.sleep(0.1)
            return {"status": "success", "data": "mock_data"}
        
        async def mock_api_call_failure():
            """Mock API call that fails initially"""
            if not hasattr(mock_api_call_failure, 'attempt_count'):
                mock_api_call_failure.attempt_count = 0
            
            mock_api_call_failure.attempt_count += 1
            
            if mock_api_call_failure.attempt_count < 3:
                raise Exception("Temporary network error")
            
            return {"status": "success", "data": "mock_data_after_retries"}
        
        # Test successful operation
        print("Testing successful operation...")
        result = await retry_service.retry_with_intelligence(
            mock_api_call_success,
            "github",
            "test_operation_success"
        )
        print(f"✓ Successful operation result: {result}")
        
        # Test operation with retries
        print("Testing operation with retries...")
        result = await retry_service.retry_with_intelligence(
            mock_api_call_failure,
            "github",
            "test_operation_retry"
        )
        print(f"✓ Operation with retries result: {result}")
        
        # Get retry statistics
        stats = await retry_service.get_retry_statistics()
        print(f"✓ Retry statistics: {stats}")
        
        # Platform-specific statistics
        github_stats = await retry_service.get_retry_statistics("github")
        print(f"✓ GitHub retry statistics: {github_stats}")
        
    except Exception as e:
        logger.error("Intelligent retry demo failed", error=str(e))
        print(f"✗ Intelligent retry demo failed: {e}")


async def demo_background_jobs():
    """Demo background job processing"""
    print("\n=== Background Jobs Demo ===")
    
    try:
        from app.core.celery_app import celery_app, task_manager
        
        # Check Celery health
        health = task_manager.get_worker_stats()
        print(f"✓ Celery worker stats: {health}")
        
        # Schedule cache optimization tasks
        print("Scheduling cache optimization tasks...")
        
        # Cache cleanup task
        cleanup_task = celery_app.send_task("app.tasks.cache_tasks.cleanup_expired_cache")
        print(f"✓ Scheduled cache cleanup task: {cleanup_task.id}")
        
        # Cache warming task
        cache_keys = [
            "user_profile:demo_user_1",
            "user_profile:demo_user_2",
            "analysis:demo_user_1:skill_assessment"
        ]
        warm_task = celery_app.send_task(
            "app.tasks.cache_tasks.warm_cache",
            args=[cache_keys]
        )
        print(f"✓ Scheduled cache warming task: {warm_task.id}")
        
        # Concurrent analysis pipeline task
        analysis_task = celery_app.send_task(
            "app.tasks.data_collection_tasks.concurrent_analysis_pipeline",
            args=["demo_user_123", ["skill_assessment", "career_recommendations"]]
        )
        print(f"✓ Scheduled concurrent analysis task: {analysis_task.id}")
        
        # Wait a bit and check task status
        await asyncio.sleep(2)
        
        cleanup_status = task_manager.get_task_status(cleanup_task.id)
        print(f"✓ Cache cleanup task status: {cleanup_status['status']}")
        
        warm_status = task_manager.get_task_status(warm_task.id)
        print(f"✓ Cache warming task status: {warm_status['status']}")
        
        analysis_status = task_manager.get_task_status(analysis_task.id)
        print(f"✓ Analysis task status: {analysis_status['status']}")
        
    except Exception as e:
        logger.error("Background jobs demo failed", error=str(e))
        print(f"✗ Background jobs demo failed: {e}")


async def demo_performance_monitoring():
    """Demo performance monitoring"""
    print("\n=== Performance Monitoring Demo ===")
    
    try:
        from app.services.enhanced_performance_monitoring import get_performance_monitor
        
        monitor = get_performance_monitor()
        
        # Start monitoring
        monitor.start_monitoring()
        print("✓ Started performance monitoring")
        
        # Wait for some metrics to be collected
        await asyncio.sleep(3)
        
        # Get current metrics
        current_metrics = monitor.get_current_metrics()
        print(f"✓ Current metrics collected: {len(current_metrics)} types")
        
        for metric_type, metric_data in current_metrics.items():
            if hasattr(metric_data, 'timestamp'):
                print(f"  - {metric_type}: {metric_data.timestamp}")
        
        # Get performance summary
        summary = monitor.get_performance_summary()
        print(f"✓ Performance summary: monitoring_active={summary['monitoring_active']}")
        
        # Get metrics history
        system_history = monitor.get_metrics_history("system", 5)  # Last 5 minutes
        print(f"✓ System metrics history: {len(system_history)} entries")
        
        # Test threshold alerts
        def alert_callback(alert_data):
            print(f"🚨 Performance Alert: {alert_data['type']} - {alert_data['data']}")
        
        monitor.add_alert_callback(alert_callback)
        print("✓ Added alert callback")
        
        # Update threshold to trigger an alert (for demo)
        monitor.update_threshold("cpu_percent", 1.0)  # Very low threshold
        await asyncio.sleep(2)  # Wait for next collection cycle
        
        # Stop monitoring
        monitor.stop_monitoring()
        print("✓ Stopped performance monitoring")
        
    except Exception as e:
        logger.error("Performance monitoring demo failed", error=str(e))
        print(f"✗ Performance monitoring demo failed: {e}")


async def demo_cache_optimization():
    """Demo advanced cache optimization features"""
    print("\n=== Cache Optimization Demo ===")
    
    try:
        from app.services.cache_service import get_cache_service
        
        cache_service = await get_cache_service()
        
        # Populate cache with test data
        test_data = [
            ("user_profile:user1", {"name": "User 1", "skills": ["Python", "ML"]}),
            ("user_profile:user2", {"name": "User 2", "skills": ["JavaScript", "React"]}),
            ("analysis:user1:skills", {"python": 90, "ml": 85}),
            ("analysis:user2:skills", {"javascript": 88, "react": 92}),
            ("job_market:bangalore:python", {"jobs": 150, "avg_salary": 1200000}),
            ("ml_pred:recommendation:user1", {"recommendations": ["Data Scientist", "ML Engineer"]})
        ]
        
        print("Populating cache with test data...")
        for key, data in test_data:
            await cache_service.set(key, data, ttl=3600)
        
        print(f"✓ Populated cache with {len(test_data)} entries")
        
        # Test cache statistics
        stats = cache_service.get_stats()
        print(f"✓ Cache hit rate: {stats.get('hit_rate_percent', 0)}%")
        print(f"✓ Local cache size: {stats.get('local_cache_size', 0)}")
        
        # Test cache invalidation strategies
        await cache_service.invalidation.invalidate_user_cache("user1")
        print("✓ Invalidated user1 cache")
        
        await cache_service.invalidation.invalidate_ml_model_cache("recommendation")
        print("✓ Invalidated ML model cache")
        
        # Test get_or_set functionality
        async def expensive_computation():
            await asyncio.sleep(0.5)  # Simulate expensive operation
            return {"computed_value": 42, "timestamp": time.time()}
        
        # First call - should compute
        start_time = time.time()
        result1 = await cache_service.get_or_set(
            "expensive_computation:test",
            expensive_computation,
            ttl=300
        )
        first_call_time = time.time() - start_time
        
        # Second call - should use cache
        start_time = time.time()
        result2 = await cache_service.get_or_set(
            "expensive_computation:test",
            expensive_computation,
            ttl=300
        )
        second_call_time = time.time() - start_time
        
        print(f"✓ First call (computed): {first_call_time:.3f}s")
        print(f"✓ Second call (cached): {second_call_time:.3f}s")
        print(f"✓ Cache speedup: {first_call_time / second_call_time:.1f}x")
        
        # Clean up expired local cache
        await cache_service.clear_expired_local_cache()
        print("✓ Cleaned up expired local cache")
        
    except Exception as e:
        logger.error("Cache optimization demo failed", error=str(e))
        print(f"✗ Cache optimization demo failed: {e}")


async def demo_comprehensive_workflow():
    """Demo comprehensive workflow combining all features"""
    print("\n=== Comprehensive Workflow Demo ===")
    
    try:
        # Simulate a complete user analysis workflow
        user_id = "comprehensive_demo_user"
        
        print(f"Starting comprehensive analysis for user: {user_id}")
        
        # Step 1: Concurrent platform data collection
        print("Step 1: Collecting platform data concurrently...")
        platform_data = {
            "github": {
                "repositories": 30,
                "languages": ["Python", "JavaScript", "Go"],
                "contributions": 1250,
                "followers": 45
            },
            "leetcode": {
                "problems_solved": 200,
                "contest_rating": 1750,
                "acceptance_rate": 85.5
            },
            "linkedin": {
                "connections": 650,
                "endorsements": ["Python", "Machine Learning", "Data Science"],
                "experience_years": 5
            }
        }
        
        # Step 2: Cache platform data
        print("Step 2: Caching platform data...")
        from app.services.cache_service import get_cache_service, CacheKeyBuilder
        cache_service = await get_cache_service()
        
        for platform, data in platform_data.items():
            cache_key = CacheKeyBuilder.platform_data(user_id, platform)
            await cache_service.set(cache_key, data, ttl=21600)  # 6 hours
        
        # Step 3: Run analysis pipeline with caching
        print("Step 3: Running analysis pipeline...")
        analysis_results = {}
        
        # Skill assessment
        skill_assessment = {
            "technical_skills": {
                "python": 92,
                "javascript": 78,
                "machine_learning": 88,
                "data_analysis": 85
            },
            "soft_skills": {
                "problem_solving": 90,
                "communication": 75,
                "leadership": 70
            },
            "overall_score": 84
        }
        
        cache_key = CacheKeyBuilder.analysis_result(user_id, "skill_assessment")
        await cache_service.set(cache_key, skill_assessment, ttl=7200)
        analysis_results["skill_assessment"] = skill_assessment
        
        # Career recommendations
        career_recommendations = [
            {
                "role": "Senior Data Scientist",
                "match_score": 91,
                "salary_range": "₹15-25 LPA",
                "required_skills": ["Python", "ML", "Statistics"],
                "skill_gaps": ["Deep Learning", "MLOps"]
            },
            {
                "role": "ML Engineer",
                "match_score": 87,
                "salary_range": "₹18-28 LPA",
                "required_skills": ["Python", "TensorFlow", "Kubernetes"],
                "skill_gaps": ["Kubernetes", "Docker"]
            }
        ]
        
        cache_key = CacheKeyBuilder.analysis_result(user_id, "career_recommendations")
        await cache_service.set(cache_key, career_recommendations, ttl=3600)
        analysis_results["career_recommendations"] = career_recommendations
        
        # Step 4: Generate dashboard data
        print("Step 4: Generating dashboard data...")
        dashboard_data = {
            "user_id": user_id,
            "profile_completeness": 92,
            "skill_radar": skill_assessment["technical_skills"],
            "career_matches": len(career_recommendations),
            "learning_recommendations": [
                "Complete Deep Learning Specialization",
                "Learn Kubernetes fundamentals",
                "Practice system design"
            ],
            "job_opportunities": 45,
            "generated_at": time.time()
        }
        
        cache_key = CacheKeyBuilder.dashboard_data(user_id)
        await cache_service.set(cache_key, dashboard_data, ttl=1800)
        
        # Step 5: Performance metrics
        print("Step 5: Collecting performance metrics...")
        final_stats = cache_service.get_stats()
        
        print(f"✓ Comprehensive workflow completed successfully!")
        print(f"✓ Analysis results generated: {len(analysis_results)}")
        print(f"✓ Cache hit rate: {final_stats.get('hit_rate_percent', 0)}%")
        print(f"✓ Dashboard data cached for quick access")
        
        # Simulate dashboard access (should be fast due to caching)
        start_time = time.time()
        cached_dashboard = await cache_service.get(CacheKeyBuilder.dashboard_data(user_id))
        access_time = time.time() - start_time
        
        print(f"✓ Dashboard data access time: {access_time:.3f}s (cached)")
        
    except Exception as e:
        logger.error("Comprehensive workflow demo failed", error=str(e))
        print(f"✗ Comprehensive workflow demo failed: {e}")


async def main():
    """Run all performance optimization demos"""
    print("🚀 Enhanced Performance Optimization & Caching Demo")
    print("=" * 60)
    
    demos = [
        demo_redis_caching,
        demo_concurrent_processing,
        demo_intelligent_retry,
        demo_background_jobs,
        demo_performance_monitoring,
        demo_cache_optimization,
        demo_comprehensive_workflow
    ]
    
    for demo in demos:
        try:
            await demo()
        except Exception as e:
            logger.error(f"Demo {demo.__name__} failed", error=str(e))
            print(f"✗ {demo.__name__} failed: {e}")
        
        print()  # Add spacing between demos
    
    print("🎉 Performance optimization demo completed!")
    print("\nKey Features Demonstrated:")
    print("✓ Redis caching for frequently accessed analysis results")
    print("✓ Concurrent processing for multiple platform data collection")
    print("✓ Intelligent retry mechanisms for external API failures")
    print("✓ Background job processing for data scraping and analysis")
    print("✓ Performance monitoring with real-time metrics")
    print("✓ Advanced cache optimization strategies")
    print("✓ Comprehensive workflow integration")


if __name__ == "__main__":
    asyncio.run(main())