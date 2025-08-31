"""
Tests for performance optimization features
"""
import pytest
import asyncio
import time
from unittest.mock import Mock, patch, AsyncMock

from app.services.cache_service import (
    AdvancedCacheService, CacheKeyBuilder, CacheInvalidationStrategy,
    cached, cache_key_hash, get_cache_service
)
from app.services.performance_monitoring import (
    PerformanceMonitor, PerformanceMetrics, performance_context
)
from app.core.database_optimization import (
    DatabasePerformanceMonitor, DatabaseOptimizer, get_database_health
)
from app.core.celery_app import celery_app, CeleryTaskManager
from app.core.redis import RedisManager


class TestAdvancedCacheService:
    """Test advanced caching functionality"""
    
    @pytest.fixture
    async def redis_manager(self):
        """Mock Redis manager for testing"""
        mock_redis = Mock()
        mock_redis.get = AsyncMock(return_value=None)
        mock_redis.set = AsyncMock(return_value=True)
        mock_redis.delete = AsyncMock(return_value=True)
        mock_redis.redis = Mock()
        mock_redis.redis.scan = AsyncMock(return_value=(0, []))
        
        return mock_redis
    
    @pytest.fixture
    def cache_service(self, redis_manager):
        """Create cache service with mocked Redis"""
        return AdvancedCacheService(redis_manager)
    
    @pytest.mark.asyncio
    async def test_cache_key_builder(self):
        """Test cache key generation"""
        user_id = "user123"
        params_hash = "abc123"
        
        # Test different key types
        assert CacheKeyBuilder.user_profile(user_id) == f"user_profile:{user_id}"
        assert CacheKeyBuilder.career_recommendations(user_id, params_hash) == f"career_rec:{user_id}:{params_hash}"
        assert CacheKeyBuilder.learning_path(user_id, params_hash) == f"learning_path:{user_id}:{params_hash}"
        assert CacheKeyBuilder.ml_prediction("model1", params_hash) == f"ml_pred:model1:{params_hash}"
    
    @pytest.mark.asyncio
    async def test_cache_get_set(self, cache_service):
        """Test basic cache get/set operations"""
        key = "test_key"
        value = {"data": "test_value", "number": 42}
        
        # Test set
        result = await cache_service.set(key, value, ttl=300)
        assert result is True
        
        # Test get (should return None since Redis is mocked)
        cached_value = await cache_service.get(key)
        assert cached_value is None  # Mocked to return None
    
    @pytest.mark.asyncio
    async def test_cache_get_or_set(self, cache_service):
        """Test get_or_set functionality"""
        key = "test_factory_key"
        expected_value = {"generated": True, "timestamp": time.time()}
        
        def factory_function():
            return expected_value
        
        # Should call factory since cache miss
        result = await cache_service.get_or_set(key, factory_function, ttl=300)
        assert result == expected_value
    
    @pytest.mark.asyncio
    async def test_cache_stats(self, cache_service):
        """Test cache statistics"""
        # Simulate some cache operations
        cache_service._cache_stats["hits"] = 10
        cache_service._cache_stats["misses"] = 5
        cache_service._cache_stats["sets"] = 8
        
        stats = cache_service.get_stats()
        
        assert stats["hits"] == 10
        assert stats["misses"] == 5
        assert stats["sets"] == 8
        assert stats["hit_rate_percent"] == 66.67  # 10/(10+5) * 100
    
    @pytest.mark.asyncio
    async def test_cache_invalidation(self, redis_manager):
        """Test cache invalidation strategies"""
        invalidation = CacheInvalidationStrategy(redis_manager)
        
        # Test user cache invalidation
        await invalidation.invalidate_user_cache("user123")
        
        # Test market data cache invalidation
        await invalidation.invalidate_market_data_cache()
        
        # Test ML model cache invalidation
        await invalidation.invalidate_ml_model_cache("recommendation_engine")
    
    def test_cache_key_hash(self):
        """Test cache key hashing"""
        # Same inputs should produce same hash
        hash1 = cache_key_hash("arg1", "arg2", param1="value1", param2="value2")
        hash2 = cache_key_hash("arg1", "arg2", param1="value1", param2="value2")
        assert hash1 == hash2
        
        # Different inputs should produce different hashes
        hash3 = cache_key_hash("arg1", "arg2", param1="different", param2="value2")
        assert hash1 != hash3
    
    @pytest.mark.asyncio
    async def test_cached_decorator(self):
        """Test the cached decorator"""
        call_count = 0
        
        @cached(ttl=300, key_prefix="test")
        async def expensive_function(param1: str, param2: int):
            nonlocal call_count
            call_count += 1
            return f"result_{param1}_{param2}"
        
        # Mock the cache service
        with patch('app.services.cache_service.get_cache_service') as mock_get_cache:
            mock_cache = Mock()
            mock_cache.get = AsyncMock(return_value=None)  # Cache miss
            mock_cache.set = AsyncMock(return_value=True)
            mock_get_cache.return_value = mock_cache
            
            # First call should execute function
            result1 = await expensive_function("test", 123)
            assert result1 == "result_test_123"
            assert call_count == 1
            
            # Second call should also execute since cache returns None
            result2 = await expensive_function("test", 123)
            assert result2 == "result_test_123"
            assert call_count == 2


class TestPerformanceMonitoring:
    """Test performance monitoring functionality"""
    
    @pytest.fixture
    def performance_monitor(self):
        """Create performance monitor for testing"""
        return PerformanceMonitor()
    
    @pytest.mark.asyncio
    async def test_collect_metrics(self, performance_monitor):
        """Test metrics collection"""
        with patch('psutil.cpu_percent', return_value=50.0), \
             patch('psutil.virtual_memory') as mock_memory, \
             patch('psutil.disk_io_counters') as mock_disk, \
             patch('psutil.net_io_counters') as mock_network, \
             patch('app.services.performance_monitoring.get_database_health') as mock_db_health, \
             patch('app.services.performance_monitoring.get_cache_service') as mock_cache:
            
            # Mock memory info
            mock_memory.return_value = Mock(percent=60.0, used=1024*1024*1024)  # 1GB
            
            # Mock disk I/O
            mock_disk.return_value = Mock(read_bytes=1024*1024*100, write_bytes=1024*1024*50)
            
            # Mock network I/O
            mock_network.return_value = Mock(bytes_sent=1024*1024*10, bytes_recv=1024*1024*20)
            
            # Mock database health
            mock_db_health.return_value = {
                "performance_report": {
                    "connection_stats": {"active_connections": 5}
                }
            }
            
            # Mock cache service
            mock_cache_service = Mock()
            mock_cache_service.get_stats.return_value = {"hit_rate_percent": 85.0}
            mock_cache.return_value = mock_cache_service
            
            # Mock response time measurement
            with patch.object(performance_monitor, '_measure_response_time', return_value=150.0):
                metrics = await performance_monitor.collect_metrics()
                
                assert isinstance(metrics, PerformanceMetrics)
                assert metrics.cpu_percent == 50.0
                assert metrics.memory_percent == 60.0
                assert metrics.response_time_ms == 150.0
                assert metrics.cache_hit_rate == 85.0
                assert metrics.active_connections == 5
    
    @pytest.mark.asyncio
    async def test_performance_summary(self, performance_monitor):
        """Test performance summary generation"""
        # Add some mock metrics
        current_time = time.time()
        for i in range(5):
            metrics = PerformanceMetrics(
                timestamp=current_time - (i * 60),  # 1 minute intervals
                cpu_percent=50.0 + i,
                memory_percent=60.0 + i,
                memory_used_mb=1000.0 + i * 10,
                disk_io_read_mb=100.0,
                disk_io_write_mb=50.0,
                network_sent_mb=10.0,
                network_recv_mb=20.0,
                active_connections=5,
                response_time_ms=150.0 + i * 10,
                cache_hit_rate=85.0 - i,
                queue_size=10 + i
            )
            performance_monitor.metrics_history.append(metrics)
        
        summary = performance_monitor.get_performance_summary(hours=1)
        
        assert "averages" in summary
        assert "peaks" in summary
        assert "alerts" in summary
        assert "current_status" in summary
        assert summary["metrics_count"] == 5
    
    @pytest.mark.asyncio
    async def test_bottleneck_analysis(self, performance_monitor):
        """Test bottleneck analysis"""
        # Add metrics with high resource usage
        current_time = time.time()
        for i in range(10):
            metrics = PerformanceMetrics(
                timestamp=current_time - (i * 60),
                cpu_percent=90.0,  # High CPU
                memory_percent=95.0,  # High memory
                memory_used_mb=8000.0,
                disk_io_read_mb=100.0,
                disk_io_write_mb=50.0,
                network_sent_mb=10.0,
                network_recv_mb=20.0,
                active_connections=5,
                response_time_ms=3000.0,  # Slow response
                cache_hit_rate=40.0,  # Low cache hit rate
                queue_size=500  # High queue size
            )
            performance_monitor.metrics_history.append(metrics)
        
        analysis = performance_monitor.get_bottleneck_analysis()
        
        assert "bottlenecks" in analysis
        assert "recommendations" in analysis
        assert len(analysis["bottlenecks"]) > 0
        assert len(analysis["recommendations"]) > 0
        
        # Check for expected bottlenecks
        bottleneck_types = [b["type"] for b in analysis["bottlenecks"]]
        assert "cpu" in bottleneck_types
        assert "memory" in bottleneck_types
        assert "response_time" in bottleneck_types
    
    @pytest.mark.asyncio
    async def test_performance_context(self):
        """Test performance context manager"""
        with patch('structlog.get_logger') as mock_logger:
            mock_log = Mock()
            mock_logger.return_value = mock_log
            
            async with performance_context("test_operation"):
                await asyncio.sleep(0.01)  # Small delay
            
            # Should have logged performance info
            mock_log.info.assert_called_once()
            call_args = mock_log.info.call_args
            assert "Operation performance" in call_args[0]
            assert call_args[1]["operation"] == "test_operation"


class TestDatabaseOptimization:
    """Test database optimization functionality"""
    
    @pytest.fixture
    def db_monitor(self):
        """Create database performance monitor"""
        return DatabasePerformanceMonitor()
    
    @pytest.fixture
    def db_optimizer(self):
        """Create database optimizer"""
        return DatabaseOptimizer()
    
    def test_query_recording(self, db_monitor):
        """Test query performance recording"""
        query = "SELECT * FROM users WHERE id = 1"
        execution_time = 0.5
        result_count = 1
        
        db_monitor.record_query(query, execution_time, result_count)
        
        assert len(db_monitor.query_stats) == 1
        query_hash = hash(query)
        assert query_hash in db_monitor.query_stats
        
        stats = db_monitor.query_stats[query_hash]
        assert stats["execution_count"] == 1
        assert stats["total_time"] == execution_time
        assert stats["avg_time"] == execution_time
        assert stats["max_time"] == execution_time
        assert stats["min_time"] == execution_time
    
    def test_slow_query_tracking(self, db_monitor):
        """Test slow query tracking"""
        slow_query = "SELECT * FROM large_table ORDER BY created_at"
        execution_time = 2.5  # > 1 second threshold
        
        db_monitor.record_query(slow_query, execution_time)
        
        assert len(db_monitor.slow_queries) == 1
        assert db_monitor.slow_queries[0]["query"] == slow_query
        assert db_monitor.slow_queries[0]["execution_time"] == execution_time
    
    def test_performance_report(self, db_monitor):
        """Test performance report generation"""
        # Add some query data
        queries = [
            ("SELECT * FROM users", 0.1, 10),
            ("SELECT * FROM posts", 0.5, 50),
            ("SELECT * FROM comments", 1.2, 100),
        ]
        
        for query, time, count in queries:
            db_monitor.record_query(query, time, count)
        
        report = db_monitor.get_performance_report()
        
        assert "total_queries" in report
        assert "unique_queries" in report
        assert "total_execution_time" in report
        assert "slowest_queries" in report
        assert report["total_queries"] == 3
        assert report["unique_queries"] == 3
    
    @pytest.mark.asyncio
    async def test_database_health(self):
        """Test database health check"""
        with patch('app.core.database_optimization.AsyncSessionLocal') as mock_session:
            mock_session.return_value.__aenter__ = AsyncMock()
            mock_session.return_value.__aexit__ = AsyncMock()
            mock_session.return_value.execute = AsyncMock()
            
            health = await get_database_health()
            
            assert "connection_status" in health
            assert "performance_report" in health
            assert "timestamp" in health


class TestCeleryIntegration:
    """Test Celery task management"""
    
    def test_celery_task_manager(self):
        """Test Celery task manager functionality"""
        manager = CeleryTaskManager()
        
        # Test task status (will return UNKNOWN for non-existent task)
        status = manager.get_task_status("fake_task_id")
        assert status["status"] == "UNKNOWN"
        assert "error" in status
    
    def test_celery_health_check(self):
        """Test Celery health check"""
        from app.core.celery_app import celery_health_check
        
        # This will likely return unhealthy in test environment
        health = celery_health_check()
        assert "status" in health
        assert health["status"] in ["healthy", "unhealthy"]


@pytest.mark.asyncio
async def test_integration_cache_and_performance():
    """Integration test for cache and performance monitoring"""
    with patch('app.services.cache_service.get_redis') as mock_get_redis:
        # Mock Redis manager
        mock_redis = Mock()
        mock_redis.get = AsyncMock(return_value=None)
        mock_redis.set = AsyncMock(return_value=True)
        mock_redis.delete = AsyncMock(return_value=True)
        mock_redis.redis = Mock()
        mock_redis.redis.scan = AsyncMock(return_value=(0, []))
        mock_get_redis.return_value = mock_redis
        
        # Test cache service creation
        cache_service = await get_cache_service()
        assert cache_service is not None
        
        # Test cache operations
        test_key = "integration_test_key"
        test_value = {"test": "data", "number": 123}
        
        result = await cache_service.set(test_key, test_value, ttl=300)
        assert result is True
        
        # Test performance monitoring with cache
        with patch('app.services.performance_monitoring.get_cache_service', return_value=cache_service):
            monitor = PerformanceMonitor()
            
            # Mock other dependencies
            with patch('psutil.cpu_percent', return_value=25.0), \
                 patch('psutil.virtual_memory') as mock_memory, \
                 patch('app.services.performance_monitoring.get_database_health') as mock_db_health:
                
                mock_memory.return_value = Mock(percent=30.0, used=512*1024*1024)
                mock_db_health.return_value = {
                    "performance_report": {"connection_stats": {"active_connections": 2}}
                }
                
                with patch.object(monitor, '_measure_response_time', return_value=100.0):
                    metrics = await monitor.collect_metrics()
                    
                    assert metrics.cpu_percent == 25.0
                    assert metrics.memory_percent == 30.0
                    assert metrics.response_time_ms == 100.0