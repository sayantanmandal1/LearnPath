#!/usr/bin/env python3
"""
Performance optimization features demonstration
"""
import asyncio
import time
import sys
from pathlib import Path

# Add the backend directory to Python path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from app.services.cache_service import AdvancedCacheService, CacheKeyBuilder, cached
from app.services.performance_monitoring import PerformanceMonitor, performance_context
from app.core.database_optimization import DatabasePerformanceMonitor
from app.core.redis import RedisManager


async def demo_cache_service():
    """Demonstrate advanced caching features"""
    print("🚀 Advanced Cache Service Demo")
    print("=" * 50)
    
    # Create mock Redis manager for demo
    class MockRedisManager:
        def __init__(self):
            self._storage = {}
        
        async def get(self, key):
            return self._storage.get(key)
        
        async def set(self, key, value, ttl=None):
            self._storage[key] = value
            return True
        
        async def delete(self, key):
            if key in self._storage:
                del self._storage[key]
                return True
            return False
        
        @property
        def redis(self):
            mock_redis = type('MockRedis', (), {})()
            mock_redis.scan = lambda cursor, match, count: (0, [])
            return mock_redis
    
    # Initialize cache service
    redis_manager = MockRedisManager()
    cache_service = AdvancedCacheService(redis_manager)
    
    # Test basic cache operations
    print("\n1. Basic Cache Operations")
    print("-" * 30)
    
    test_key = CacheKeyBuilder.user_profile("demo_user_123")
    test_data = {
        "user_id": "demo_user_123",
        "skills": ["Python", "FastAPI", "Redis"],
        "experience_years": 5,
        "dream_job": "Senior Software Engineer"
    }
    
    # Set cache
    await cache_service.set(test_key, test_data, ttl=300)
    print(f"✅ Cached user profile: {test_key}")
    
    # Get cache
    cached_data = await cache_service.get(test_key)
    print(f"✅ Retrieved from cache: {cached_data}")
    
    # Test cache statistics
    print("\n2. Cache Statistics")
    print("-" * 30)
    
    # Simulate some cache operations
    for i in range(10):
        key = f"test_key_{i}"
        await cache_service.set(key, f"value_{i}")
        await cache_service.get(key)
    
    stats = cache_service.get_stats()
    print(f"Cache Statistics: {stats}")
    
    # Test get_or_set functionality
    print("\n3. Get-or-Set Pattern")
    print("-" * 30)
    
    def expensive_computation():
        time.sleep(0.1)  # Simulate expensive operation
        return {"computed_at": time.time(), "result": "expensive_result"}
    
    start_time = time.time()
    result = await cache_service.get_or_set("expensive_key", expensive_computation, ttl=600)
    first_call_time = time.time() - start_time
    
    start_time = time.time()
    result2 = await cache_service.get_or_set("expensive_key", expensive_computation, ttl=600)
    second_call_time = time.time() - start_time
    
    print(f"First call (cache miss): {first_call_time:.3f}s")
    print(f"Second call (cache hit): {second_call_time:.3f}s")
    if second_call_time > 0:
        print(f"Speed improvement: {first_call_time / second_call_time:.1f}x faster")
    else:
        print("Speed improvement: Cache hit was instantaneous!")


async def demo_cached_decorator():
    """Demonstrate the @cached decorator"""
    print("\n🎯 Cached Decorator Demo")
    print("=" * 50)
    
    call_count = 0
    
    @cached(ttl=300, key_prefix="demo")
    async def fibonacci(n: int) -> int:
        nonlocal call_count
        call_count += 1
        print(f"  Computing fibonacci({n}) - Call #{call_count}")
        
        if n <= 1:
            return n
        return await fibonacci(n-1) + await fibonacci(n-2)
    
    # Mock the cache service for the decorator
    class MockCacheService:
        def __init__(self):
            self._cache = {}
        
        async def get(self, key, use_local_cache=True):
            return self._cache.get(key)
        
        async def set(self, key, value, ttl, use_local_cache=True):
            self._cache[key] = value
            return True
    
    # Patch the get_cache_service function
    import app.services.cache_service
    original_get_cache = app.services.cache_service.get_cache_service
    
    async def mock_get_cache():
        mock_redis = type('MockRedis', (), {})()
        return MockCacheService()
    
    app.services.cache_service.get_cache_service = mock_get_cache
    
    try:
        print("\nCalculating fibonacci(10) with caching:")
        start_time = time.time()
        result = await fibonacci(10)
        end_time = time.time()
        
        print(f"Result: {result}")
        print(f"Total function calls: {call_count}")
        print(f"Execution time: {end_time - start_time:.3f}s")
        
    finally:
        # Restore original function
        app.services.cache_service.get_cache_service = original_get_cache


async def demo_performance_monitoring():
    """Demonstrate performance monitoring"""
    print("\n📊 Performance Monitoring Demo")
    print("=" * 50)
    
    monitor = PerformanceMonitor()
    
    # Simulate some operations with performance context
    print("\n1. Performance Context Manager")
    print("-" * 30)
    
    async with performance_context("database_query"):
        await asyncio.sleep(0.05)  # Simulate database query
    
    async with performance_context("api_call"):
        await asyncio.sleep(0.1)  # Simulate API call
    
    async with performance_context("ml_inference"):
        await asyncio.sleep(0.2)  # Simulate ML model inference
    
    print("✅ Performance measurements logged")
    
    # Test bottleneck analysis (with mock data)
    print("\n2. Bottleneck Analysis")
    print("-" * 30)
    
    # Add some mock metrics to demonstrate analysis
    from app.services.performance_monitoring import PerformanceMetrics
    
    current_time = time.time()
    for i in range(5):
        metrics = PerformanceMetrics(
            timestamp=current_time - (i * 60),
            cpu_percent=85.0 + i,  # High CPU usage
            memory_percent=90.0 + i,  # High memory usage
            memory_used_mb=7000.0 + i * 100,
            disk_io_read_mb=100.0,
            disk_io_write_mb=50.0,
            network_sent_mb=10.0,
            network_recv_mb=20.0,
            active_connections=10,
            response_time_ms=2500.0 + i * 100,  # Slow response times
            cache_hit_rate=50.0 - i,  # Declining cache performance
            queue_size=200 + i * 50  # Growing queue
        )
        monitor.metrics_history.append(metrics)
    
    analysis = monitor.get_bottleneck_analysis()
    
    print("Detected Bottlenecks:")
    for bottleneck in analysis["bottlenecks"]:
        print(f"  - {bottleneck['type'].upper()}: {bottleneck['description']}")
    
    print("\nRecommendations:")
    for rec in analysis["recommendations"]:
        print(f"  - {rec['type']}: {rec['suggestion']}")


def demo_database_monitoring():
    """Demonstrate database performance monitoring"""
    print("\n🗄️ Database Performance Monitoring Demo")
    print("=" * 50)
    
    db_monitor = DatabasePerformanceMonitor()
    
    # Simulate some database queries
    queries = [
        ("SELECT * FROM users WHERE id = %s", 0.05, 1),
        ("SELECT * FROM posts ORDER BY created_at DESC LIMIT 20", 0.15, 20),
        ("SELECT COUNT(*) FROM user_profiles", 0.02, 1),
        ("SELECT * FROM jobs WHERE location = %s AND salary > %s", 0.8, 150),
        ("UPDATE user_profiles SET last_login = NOW() WHERE user_id = %s", 0.03, 1),
        ("SELECT * FROM recommendations WHERE user_id = %s", 1.2, 50),  # Slow query
    ]
    
    print("\n1. Recording Query Performance")
    print("-" * 30)
    
    for query, exec_time, result_count in queries:
        db_monitor.record_query(query, exec_time, result_count)
        status = "⚠️ SLOW" if exec_time > 1.0 else "✅ FAST"
        print(f"{status} {exec_time:.3f}s - {query[:50]}...")
    
    # Generate performance report
    print("\n2. Performance Report")
    print("-" * 30)
    
    report = db_monitor.get_performance_report()
    
    print(f"Total Queries: {report['total_queries']}")
    print(f"Unique Queries: {report['unique_queries']}")
    print(f"Total Execution Time: {report['total_execution_time']:.3f}s")
    print(f"Slow Queries Detected: {len(report['recent_slow_queries'])}")
    
    if report['recent_slow_queries']:
        print("\nSlow Queries:")
        for slow_query in report['recent_slow_queries']:
            print(f"  - {slow_query['execution_time']:.3f}s: {slow_query['query'][:60]}...")


async def main():
    """Run all demos"""
    print("🚀 AI Career Recommender - Performance Optimization Demo")
    print("=" * 60)
    
    try:
        await demo_cache_service()
        await demo_cached_decorator()
        await demo_performance_monitoring()
        demo_database_monitoring()
        
        print("\n✅ All demos completed successfully!")
        print("\n📝 Summary:")
        print("- Advanced caching with multi-layer support")
        print("- Intelligent cache invalidation strategies")
        print("- Performance monitoring and bottleneck detection")
        print("- Database query optimization and analysis")
        print("- Background task processing with Celery")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())