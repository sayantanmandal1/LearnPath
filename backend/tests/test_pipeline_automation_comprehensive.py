"""
Comprehensive tests for the data pipeline automation system.
Tests all components: scheduling, monitoring, data quality, backup/recovery.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
import json

from app.services.data_pipeline.pipeline_scheduler import PipelineScheduler, PipelineJob
from app.services.data_pipeline.pipeline_monitor import PipelineMonitor, PipelineMetrics, Alert, AlertLevel
from app.services.data_pipeline.data_quality_validator import DataQualityValidator
from app.services.data_pipeline.backup_recovery import BackupRecoveryManager
from app.services.data_pipeline.pipeline_initializer import PipelineInitializer
from app.services.data_pipeline.pipeline_config import PipelineConfig, DefaultPipelineSchedules


class TestPipelineScheduler:
    """Test pipeline scheduling functionality"""
    
    @pytest.fixture
    async def scheduler(self):
        """Create a test scheduler instance"""
        scheduler = PipelineScheduler()
        scheduler.redis_client = AsyncMock()
        scheduler.scheduler = Mock()
        scheduler.scheduler.running = True
        return scheduler
    
    @pytest.mark.asyncio
    async def test_schedule_job_cron(self, scheduler):
        """Test scheduling a job with cron trigger"""
        job = PipelineJob(
            job_id="test_cron_job",
            pipeline_name="job_collection",
            schedule_type="cron",
            schedule_config={"hour": 2, "minute": 0},
            enabled=True,
            max_retries=3,
            retry_delay=300,
            timeout=3600
        )
        
        scheduler.scheduler.add_job = Mock()
        
        result = await scheduler.schedule_job(job)
        
        assert result is True
        scheduler.scheduler.add_job.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_schedule_job_interval(self, scheduler):
        """Test scheduling a job with interval trigger"""
        job = PipelineJob(
            job_id="test_interval_job",
            pipeline_name="profile_refresh",
            schedule_type="interval",
            schedule_config={"hours": 6},
            enabled=True,
            max_retries=2,
            retry_delay=600,
            timeout=1800
        )
        
        scheduler.scheduler.add_job = Mock()
        
        result = await scheduler.schedule_job(job)
        
        assert result is True
        scheduler.scheduler.add_job.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_unschedule_job(self, scheduler):
        """Test unscheduling a job"""
        job_id = "test_job"
        scheduler.scheduler.remove_job = Mock()
        
        result = await scheduler.unschedule_job(job_id)
        
        assert result is True
        scheduler.scheduler.remove_job.assert_called_once_with(job_id)
    
    @pytest.mark.asyncio
    async def test_get_scheduled_jobs(self, scheduler):
        """Test getting list of scheduled jobs"""
        mock_job = Mock()
        mock_job.id = "test_job"
        mock_job.name = "Test Job"
        mock_job.next_run_time = datetime.utcnow()
        mock_job.trigger = "cron"
        
        scheduler.scheduler.get_jobs = Mock(return_value=[mock_job])
        
        jobs = await scheduler.get_scheduled_jobs()
        
        assert len(jobs) == 1
        assert jobs[0]['id'] == "test_job"
        assert jobs[0]['name'] == "Test Job"
    
    @pytest.mark.asyncio
    async def test_job_execution_success(self, scheduler):
        """Test successful job execution"""
        job = PipelineJob(
            job_id="test_execution",
            pipeline_name="job_collection",
            schedule_type="cron",
            schedule_config={"hour": 2},
            enabled=True
        )
        
        # Mock the pipeline execution
        scheduler._run_pipeline = AsyncMock()
        scheduler.monitor = AsyncMock()
        
        await scheduler._execute_pipeline_job(job)
        
        scheduler._run_pipeline.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_job_execution_failure_with_retry(self, scheduler):
        """Test job execution failure with retry mechanism"""
        job = PipelineJob(
            job_id="test_failure",
            pipeline_name="job_collection",
            schedule_type="cron",
            schedule_config={"hour": 2},
            enabled=True,
            max_retries=2
        )
        
        # Mock pipeline execution to fail
        scheduler._run_pipeline = AsyncMock(side_effect=Exception("Pipeline failed"))
        scheduler.monitor = AsyncMock()
        scheduler._schedule_retry = AsyncMock()
        
        await scheduler._execute_pipeline_job(job)
        
        scheduler._schedule_retry.assert_called_once()


class TestPipelineMonitor:
    """Test pipeline monitoring functionality"""
    
    @pytest.fixture
    async def monitor(self):
        """Create a test monitor instance"""
        monitor = PipelineMonitor()
        monitor.redis_client = AsyncMock()
        return monitor
    
    @pytest.mark.asyncio
    async def test_start_job_monitoring(self, monitor):
        """Test starting job monitoring"""
        execution_id = "test_execution"
        pipeline_name = "job_collection"
        
        await monitor.start_job_monitoring(execution_id, pipeline_name)
        
        assert execution_id in monitor.active_monitors
        assert monitor.active_monitors[execution_id]['metrics'].pipeline_name == pipeline_name
    
    @pytest.mark.asyncio
    async def test_update_job_metrics(self, monitor):
        """Test updating job metrics"""
        execution_id = "test_execution"
        pipeline_name = "job_collection"
        
        await monitor.start_job_monitoring(execution_id, pipeline_name)
        
        await monitor.update_job_metrics(
            execution_id,
            records_processed=100,
            records_failed=5,
            status="running"
        )
        
        metrics = monitor.active_monitors[execution_id]['metrics']
        assert metrics.records_processed == 100
        assert metrics.records_failed == 5
        assert metrics.status == "running"
    
    @pytest.mark.asyncio
    async def test_stop_job_monitoring(self, monitor):
        """Test stopping job monitoring"""
        execution_id = "test_execution"
        pipeline_name = "job_collection"
        
        await monitor.start_job_monitoring(execution_id, pipeline_name)
        await monitor.stop_job_monitoring(execution_id)
        
        assert execution_id not in monitor.active_monitors
    
    @pytest.mark.asyncio
    async def test_alert_generation_low_quality(self, monitor):
        """Test alert generation for low data quality"""
        execution_id = "test_execution"
        pipeline_name = "job_collection"
        
        await monitor.start_job_monitoring(execution_id, pipeline_name)
        
        # Mock alert sending
        monitor._send_alert = AsyncMock()
        
        # Update metrics with low quality score
        await monitor.update_job_metrics(
            execution_id,
            data_quality_score=0.5,  # Low quality score
            records_processed=100
        )
        
        # Check if alert was sent
        monitor._send_alert.assert_called()
    
    @pytest.mark.asyncio
    async def test_alert_generation_high_error_rate(self, monitor):
        """Test alert generation for high error rate"""
        execution_id = "test_execution"
        pipeline_name = "job_collection"
        
        await monitor.start_job_monitoring(execution_id, pipeline_name)
        
        # Mock alert sending
        monitor._send_alert = AsyncMock()
        
        # Update metrics with high error rate
        await monitor.update_job_metrics(
            execution_id,
            records_processed=100,
            records_failed=25,  # 25% error rate
            status="running"
        )
        
        # Check if alert was sent
        monitor._send_alert.assert_called()
    
    @pytest.mark.asyncio
    async def test_get_system_health(self, monitor):
        """Test getting system health metrics"""
        # Mock Redis data
        monitor.redis_client.keys = AsyncMock(return_value=[
            "pipeline_metrics:exec1",
            "pipeline_metrics:exec2"
        ])
        
        mock_metrics = {
            'pipeline_name': 'job_collection',
            'start_time': datetime.utcnow().isoformat(),
            'status': 'completed',
            'records_processed': 100,
            'error_count': 2,
            'data_quality_score': 0.95,
            'duration_seconds': 120
        }
        
        monitor.redis_client.get = AsyncMock(return_value=json.dumps(mock_metrics))
        
        health = await monitor.get_system_health()
        
        assert 'active_jobs' in health
        assert 'success_rate_24h' in health
        assert 'data_quality_score_24h' in health


class TestDataQualityValidator:
    """Test data quality validation functionality"""
    
    @pytest.fixture
    def validator(self):
        """Create a test validator instance"""
        return DataQualityValidator()
    
    @pytest.mark.asyncio
    async def test_validate_job_posting_valid(self, validator):
        """Test validation of a valid job posting"""
        job_data = {
            'title': 'Senior Software Engineer',
            'company': 'Tech Corp',
            'description': 'We are looking for a skilled software engineer with experience in Python, JavaScript, and cloud technologies. The role involves developing scalable applications and working with cross-functional teams.',
            'location': 'San Francisco, CA',
            'required_skills': ['Python', 'JavaScript', 'AWS'],
            'experience_level': 'Senior'
        }
        
        result = await validator.validate_job_posting(job_data)
        
        assert result is True
    
    @pytest.mark.asyncio
    async def test_validate_job_posting_invalid(self, validator):
        """Test validation of an invalid job posting"""
        job_data = {
            'title': '$$$URGENT$$$',  # Suspicious title
            'company': '',  # Missing company
            'description': 'Make money fast!',  # Spam description
            'location': 'Remote'
        }
        
        result = await validator.validate_job_posting(job_data)
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_validate_profile_data_valid(self, validator):
        """Test validation of valid profile data"""
        profile_data = {
            'user_id': 'user123',
            'skills': {
                'python': 0.9,
                'javascript': 0.8,
                'react': 0.7
            },
            'github_username': 'testuser',
            'linkedin_url': 'https://linkedin.com/in/testuser',
            'updated_at': datetime.utcnow().isoformat()
        }
        
        result = await validator.validate_profile_data(profile_data)
        
        assert result['is_valid'] is True
        assert result['quality_score'] > 0.8
    
    @pytest.mark.asyncio
    async def test_validate_profile_data_invalid(self, validator):
        """Test validation of invalid profile data"""
        profile_data = {
            'skills': {
                'python': 1.5,  # Invalid confidence score
                'invalid@skill': 0.8  # Invalid skill name
            },
            'github_username': 'invalid-username-',  # Invalid format
            'linkedin_url': 'not-a-url'  # Invalid URL
        }
        
        result = await validator.validate_profile_data(profile_data)
        
        assert result['is_valid'] is False
        assert len(result['issues']) > 0
    
    @pytest.mark.asyncio
    async def test_validate_skill_taxonomy(self, validator):
        """Test skill taxonomy validation"""
        # Mock database session and repository
        with patch('app.core.database.get_db_session') as mock_db:
            mock_session = AsyncMock()
            mock_db.return_value.__aenter__.return_value = mock_session
            
            # Mock skill repository
            with patch('app.repositories.skill.SkillRepository') as mock_repo_class:
                mock_repo = AsyncMock()
                mock_repo_class.return_value = mock_repo
                
                # Mock skills data
                mock_skills = [
                    Mock(name='Python', category='Programming', demand_score=0.9, last_updated=datetime.utcnow()),
                    Mock(name='JavaScript', category='Programming', demand_score=0.8, last_updated=datetime.utcnow()),
                    Mock(name='python', category='Programming', demand_score=0.9, last_updated=datetime.utcnow())  # Duplicate
                ]
                
                mock_repo.get_all_skills.return_value = mock_skills
                
                result = await validator.validate_skill_taxonomy()
                
                assert 'quality_score' in result
                assert 'total_skills' in result
                assert 'statistics' in result
                assert len(result['issues']) > 0  # Should detect duplicate


class TestBackupRecoveryManager:
    """Test backup and recovery functionality"""
    
    @pytest.fixture
    def backup_manager(self):
        """Create a test backup manager instance"""
        manager = BackupRecoveryManager()
        manager.monitor = AsyncMock()
        return manager
    
    @pytest.mark.asyncio
    async def test_backup_database_success(self, backup_manager):
        """Test successful database backup"""
        backup_dir = Mock()
        backup_type = "full"
        
        # Mock subprocess for pg_dump
        with patch('subprocess.run') as mock_run:
            mock_run.return_value.returncode = 0
            
            # Mock file operations
            with patch('pathlib.Path.stat') as mock_stat:
                mock_stat.return_value.st_size = 1024 * 1024 * 100  # 100MB
                
                result = await backup_manager._backup_database(backup_dir, backup_type)
                
                assert result['success'] is True
                assert result['size_mb'] > 0
    
    @pytest.mark.asyncio
    async def test_backup_database_failure(self, backup_manager):
        """Test database backup failure"""
        backup_dir = Mock()
        backup_type = "full"
        
        # Mock subprocess failure
        with patch('subprocess.run') as mock_run:
            mock_run.return_value.returncode = 1
            mock_run.return_value.stderr = "Connection failed"
            
            result = await backup_manager._backup_database(backup_dir, backup_type)
            
            assert result['success'] is False
            assert 'errors' in result
    
    @pytest.mark.asyncio
    async def test_backup_redis_success(self, backup_manager):
        """Test successful Redis backup"""
        backup_dir = Mock()
        
        # Mock Redis client
        with patch('app.core.redis.get_redis_client') as mock_redis:
            mock_client = AsyncMock()
            mock_client.keys.return_value = ['key1', 'key2']
            mock_client.type.return_value = 'string'
            mock_client.get.return_value = 'value'
            mock_client.ttl.return_value = -1
            mock_redis.return_value = mock_client
            
            # Mock file operations
            with patch('builtins.open', create=True) as mock_open:
                mock_open.return_value.__enter__.return_value = Mock()
                
                with patch('pathlib.Path.stat') as mock_stat:
                    mock_stat.return_value.st_size = 1024 * 1024  # 1MB
                    
                    result = await backup_manager._backup_redis(backup_dir)
                    
                    assert result['success'] is True
                    assert result['keys_backed_up'] == 2
    
    @pytest.mark.asyncio
    async def test_execute_backup_full(self, backup_manager):
        """Test full backup execution"""
        metadata = {
            'execution_id': 'test_backup',
            'backup_type': 'full',
            'include_ml_models': True,
            'compress_backup': False
        }
        
        # Mock all backup methods
        backup_manager._backup_database = AsyncMock(return_value={
            'success': True, 'size_mb': 100, 'errors': []
        })
        backup_manager._backup_redis = AsyncMock(return_value={
            'success': True, 'size_mb': 10, 'errors': []
        })
        backup_manager._backup_ml_models = AsyncMock(return_value={
            'success': True, 'size_mb': 50, 'errors': []
        })
        backup_manager._backup_configurations = AsyncMock(return_value={
            'success': True, 'size_mb': 5, 'errors': []
        })
        backup_manager._store_backup_metadata = AsyncMock()
        backup_manager._cleanup_old_backups = AsyncMock()
        
        # Mock Path operations
        with patch('pathlib.Path.mkdir'):
            result = await backup_manager.execute_backup(metadata)
            
            assert result['success'] is True
            assert result['total_size_mb'] == 165
            assert len(result['components']) == 4


class TestPipelineInitializer:
    """Test pipeline system initialization"""
    
    @pytest.fixture
    def initializer(self):
        """Create a test initializer instance"""
        return PipelineInitializer()
    
    @pytest.mark.asyncio
    async def test_initialize_pipeline_system(self, initializer):
        """Test pipeline system initialization"""
        # Mock all dependencies
        with patch('app.services.data_pipeline.pipeline_scheduler.get_pipeline_scheduler') as mock_scheduler:
            mock_scheduler_instance = AsyncMock()
            mock_scheduler.return_value = mock_scheduler_instance
            
            with patch('app.services.data_pipeline.pipeline_monitor.get_pipeline_monitor') as mock_monitor:
                mock_monitor_instance = AsyncMock()
                mock_monitor.return_value = mock_monitor_instance
                
                initializer._setup_default_schedules = AsyncMock()
                
                await initializer.initialize_pipeline_system()
                
                mock_scheduler_instance.initialize.assert_called_once()
                mock_monitor_instance.initialize.assert_called_once()
                mock_scheduler_instance.start.assert_called_once()
                initializer._setup_default_schedules.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_health_check(self, initializer):
        """Test system health check"""
        # Mock scheduler
        with patch('app.services.data_pipeline.pipeline_scheduler.get_pipeline_scheduler') as mock_scheduler:
            mock_scheduler_instance = AsyncMock()
            mock_scheduler_instance.scheduler = Mock()
            mock_scheduler_instance.scheduler.running = True
            mock_scheduler_instance.get_scheduled_jobs.return_value = [{'id': 'job1'}, {'id': 'job2'}]
            mock_scheduler.return_value = mock_scheduler_instance
            
            # Mock monitor
            with patch('app.services.data_pipeline.pipeline_monitor.get_pipeline_monitor') as mock_monitor:
                mock_monitor_instance = AsyncMock()
                mock_monitor_instance.active_monitors = {'exec1': {}, 'exec2': {}}
                mock_monitor.return_value = mock_monitor_instance
                
                health = await initializer.health_check()
                
                assert health['scheduler_status'] == 'running'
                assert health['monitor_status'] == 'running'
                assert health['scheduled_jobs_count'] == 2
                assert health['active_executions'] == 2
                assert health['overall_health'] == 'healthy'


class TestPipelineConfig:
    """Test pipeline configuration management"""
    
    def test_pipeline_config_validation(self):
        """Test pipeline configuration validation"""
        # Test valid configuration
        config = PipelineConfig(
            max_concurrent_jobs=5,
            default_timeout=3600,
            min_job_postings_per_day=100
        )
        
        assert config.max_concurrent_jobs == 5
        assert config.default_timeout == 3600
        
        # Test invalid configuration
        with pytest.raises(ValueError):
            PipelineConfig(max_concurrent_jobs=0)  # Should raise error
    
    def test_default_schedules(self):
        """Test default pipeline schedules"""
        schedules = DefaultPipelineSchedules()
        
        job_schedule = schedules.get_job_collection_schedule()
        assert job_schedule['schedule_type'] == 'cron'
        assert 'schedule_config' in job_schedule
        
        skill_schedule = schedules.get_skill_taxonomy_schedule()
        assert skill_schedule['schedule_type'] == 'cron'
        
        profile_schedule = schedules.get_profile_refresh_schedule()
        assert profile_schedule['schedule_type'] == 'interval'


class TestIntegration:
    """Integration tests for the complete pipeline system"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_pipeline_execution(self):
        """Test complete pipeline execution flow"""
        # This would test the entire flow from scheduling to execution to monitoring
        # Mock all external dependencies
        
        with patch('app.core.redis.get_redis_client') as mock_redis:
            mock_redis.return_value = AsyncMock()
            
            with patch('app.core.database.get_db_session') as mock_db:
                mock_db.return_value.__aenter__.return_value = AsyncMock()
                
                # Create scheduler and monitor
                scheduler = PipelineScheduler()
                monitor = PipelineMonitor()
                
                # Initialize
                await scheduler.initialize()
                await monitor.initialize()
                
                # Create and schedule a job
                job = PipelineJob(
                    job_id="integration_test",
                    pipeline_name="job_collection",
                    schedule_type="interval",
                    schedule_config={"minutes": 1},
                    enabled=True
                )
                
                # Mock the actual pipeline execution
                scheduler._run_pipeline = AsyncMock()
                
                # Schedule job
                scheduler.scheduler = Mock()
                scheduler.scheduler.add_job = Mock()
                
                result = await scheduler.schedule_job(job)
                assert result is True
                
                # Test monitoring
                execution_id = "test_execution"
                await monitor.start_job_monitoring(execution_id, "job_collection")
                
                await monitor.update_job_metrics(
                    execution_id,
                    records_processed=100,
                    status="completed"
                )
                
                await monitor.stop_job_monitoring(execution_id)
                
                # Verify execution was tracked
                assert execution_id not in monitor.active_monitors
    
    @pytest.mark.asyncio
    async def test_error_handling_and_recovery(self):
        """Test error handling and recovery mechanisms"""
        # Test that the system handles various error conditions gracefully
        
        scheduler = PipelineScheduler()
        scheduler.redis_client = AsyncMock()
        
        # Test scheduling with invalid job data
        invalid_job = PipelineJob(
            job_id="invalid_job",
            pipeline_name="nonexistent_pipeline",
            schedule_type="invalid_type",
            schedule_config={},
            enabled=True
        )
        
        scheduler.scheduler = Mock()
        scheduler.scheduler.add_job = Mock(side_effect=Exception("Invalid schedule"))
        
        result = await scheduler.schedule_job(invalid_job)
        assert result is False
    
    @pytest.mark.asyncio
    async def test_performance_under_load(self):
        """Test system performance under load"""
        # Test multiple concurrent operations
        
        monitor = PipelineMonitor()
        monitor.redis_client = AsyncMock()
        
        # Start monitoring multiple jobs concurrently
        tasks = []
        for i in range(10):
            execution_id = f"load_test_{i}"
            task = monitor.start_job_monitoring(execution_id, "job_collection")
            tasks.append(task)
        
        await asyncio.gather(*tasks)
        
        # Verify all jobs are being monitored
        assert len(monitor.active_monitors) == 10
        
        # Update metrics for all jobs concurrently
        update_tasks = []
        for i in range(10):
            execution_id = f"load_test_{i}"
            task = monitor.update_job_metrics(
                execution_id,
                records_processed=i * 10,
                status="running"
            )
            update_tasks.append(task)
        
        await asyncio.gather(*update_tasks)
        
        # Verify all updates were processed
        for i in range(10):
            execution_id = f"load_test_{i}"
            metrics = monitor.active_monitors[execution_id]['metrics']
            assert metrics.records_processed == i * 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])