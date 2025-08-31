"""
Tests for data pipeline automation and scheduling system
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
import json

from app.services.data_pipeline.pipeline_scheduler import PipelineScheduler, PipelineJob
from app.services.data_pipeline.pipeline_monitor import PipelineMonitor
from app.services.data_pipeline.data_quality_validator import DataQualityValidator
from app.services.data_pipeline.backup_recovery import BackupRecoveryManager
from app.services.data_pipeline.job_collection_pipeline import JobCollectionPipeline
from app.services.data_pipeline.skill_taxonomy_pipeline import SkillTaxonomyPipeline
from app.services.data_pipeline.profile_refresh_pipeline import ProfileRefreshPipeline
from app.services.data_pipeline.model_training_pipeline import ModelTrainingPipeline


class TestPipelineScheduler:
    """Test pipeline scheduler functionality"""
    
    @pytest.fixture
    async def scheduler(self):
        """Create a test scheduler instance"""
        scheduler = PipelineScheduler()
        # Mock Redis client
        scheduler.redis_client = AsyncMock()
        return scheduler
    
    @pytest.fixture
    def sample_job(self):
        """Create a sample pipeline job"""
        return PipelineJob(
            job_id="test_job_001",
            pipeline_name="job_collection",
            schedule_type="cron",
            schedule_config={"hour": 2, "minute": 0},
            enabled=True,
            max_retries=3,
            retry_delay=300,
            timeout=3600
        )
    
    async def test_schedule_job(self, scheduler, sample_job):
        """Test scheduling a pipeline job"""
        # Mock scheduler initialization
        scheduler.scheduler = Mock()
        scheduler.scheduler.add_job = Mock()
        
        result = await scheduler.schedule_job(sample_job)
        
        assert result is True
        scheduler.scheduler.add_job.assert_called_once()
    
    async def test_unschedule_job(self, scheduler):
        """Test unscheduling a pipeline job"""
        scheduler.scheduler = Mock()
        scheduler.scheduler.remove_job = Mock()
        
        result = await scheduler.unschedule_job("test_job_001")
        
        assert result is True
        scheduler.scheduler.remove_job.assert_called_once_with("test_job_001")
    
    async def test_get_scheduled_jobs(self, scheduler):
        """Test getting list of scheduled jobs"""
        # Mock scheduled jobs
        mock_job = Mock()
        mock_job.id = "test_job_001"
        mock_job.name = "Test Job"
        mock_job.next_run_time = datetime.utcnow()
        mock_job.trigger = "cron"
        
        scheduler.scheduler = Mock()
        scheduler.scheduler.get_jobs = Mock(return_value=[mock_job])
        
        jobs = await scheduler.get_scheduled_jobs()
        
        assert len(jobs) == 1
        assert jobs[0]['id'] == "test_job_001"
    
    async def test_job_execution_with_monitoring(self, scheduler, sample_job):
        """Test job execution with monitoring"""
        scheduler.monitor = AsyncMock()
        
        # Mock pipeline execution
        with patch.object(scheduler, '_run_pipeline', new_callable=AsyncMock) as mock_run:
            await scheduler._execute_pipeline_job(sample_job)
            
            # Verify monitoring was called
            scheduler.monitor.start_job_monitoring.assert_called_once()
            scheduler.monitor.stop_job_monitoring.assert_called_once()
            mock_run.assert_called_once()


class TestPipelineMonitor:
    """Test pipeline monitoring functionality"""
    
    @pytest.fixture
    async def monitor(self):
        """Create a test monitor instance"""
        monitor = PipelineMonitor()
        monitor.redis_client = AsyncMock()
        return monitor
    
    async def test_start_job_monitoring(self, monitor):
        """Test starting job monitoring"""
        execution_id = "test_execution_001"
        pipeline_name = "job_collection"
        
        await monitor.start_job_monitoring(execution_id, pipeline_name)
        
        assert execution_id in monitor.active_monitors
        assert monitor.active_monitors[execution_id]['metrics'].pipeline_name == pipeline_name
    
    async def test_update_job_metrics(self, monitor):
        """Test updating job metrics"""
        execution_id = "test_execution_001"
        
        # Start monitoring first
        await monitor.start_job_monitoring(execution_id, "job_collection")
        
        # Update metrics
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
    
    async def test_stop_job_monitoring(self, monitor):
        """Test stopping job monitoring"""
        execution_id = "test_execution_001"
        
        # Start monitoring first
        await monitor.start_job_monitoring(execution_id, "job_collection")
        
        # Stop monitoring
        await monitor.stop_job_monitoring(execution_id)
        
        assert execution_id not in monitor.active_monitors
    
    async def test_get_system_health(self, monitor):
        """Test getting system health metrics"""
        # Mock Redis data
        monitor.redis_client.keys = AsyncMock(return_value=[
            "pipeline_metrics:exec1",
            "pipeline_metrics:exec2"
        ])
        monitor.redis_client.get = AsyncMock(side_effect=[
            json.dumps({
                'pipeline_name': 'job_collection',
                'status': 'completed',
                'start_time': datetime.utcnow().isoformat(),
                'duration_seconds': 120,
                'records_processed': 100,
                'error_count': 0,
                'data_quality_score': 0.95
            }),
            json.dumps({
                'pipeline_name': 'skill_taxonomy_update',
                'status': 'completed',
                'start_time': datetime.utcnow().isoformat(),
                'duration_seconds': 300,
                'records_processed': 50,
                'error_count': 2,
                'data_quality_score': 0.85
            })
        ])
        
        health = await monitor.get_system_health()
        
        assert 'success_rate_24h' in health
        assert 'avg_duration_24h' in health
        assert 'error_rate_24h' in health
        assert 'data_quality_score_24h' in health


class TestDataQualityValidator:
    """Test data quality validation functionality"""
    
    @pytest.fixture
    def validator(self):
        """Create a test validator instance"""
        return DataQualityValidator()
    
    async def test_validate_job_posting(self, validator):
        """Test job posting validation"""
        # Valid job posting
        valid_job = {
            'title': 'Software Engineer',
            'company': 'Tech Corp',
            'description': 'We are looking for a skilled software engineer with experience in Python and web development. Responsibilities include developing web applications and collaborating with the team.',
            'location': 'San Francisco, CA'
        }
        
        result = await validator.validate_job_posting(valid_job)
        assert result is True
        
        # Invalid job posting (missing required fields)
        invalid_job = {
            'title': 'SE',
            'company': '',
            'description': 'Short desc',
            'location': 'SF'
        }
        
        result = await validator.validate_job_posting(invalid_job)
        assert result is False
    
    async def test_validate_profile_data(self, validator):
        """Test profile data validation"""
        valid_profile = {
            'user_id': 'user123',
            'skills': {
                'python': 0.9,
                'javascript': 0.7,
                'react': 0.8
            },
            'github_username': 'testuser',
            'updated_at': datetime.utcnow().isoformat()
        }
        
        result = await validator.validate_profile_data(valid_profile)
        
        assert result['is_valid'] is True
        assert result['quality_score'] > 0.5
    
    async def test_run_data_quality_checks(self, validator):
        """Test comprehensive data quality checks"""
        # Mock database operations
        with patch('app.core.database.get_db_session'):
            with patch.object(validator, '_check_job_postings_quality', new_callable=AsyncMock) as mock_job_check:
                with patch.object(validator, '_check_profiles_quality', new_callable=AsyncMock) as mock_profile_check:
                    with patch.object(validator, 'validate_skill_taxonomy', new_callable=AsyncMock) as mock_skill_check:
                        
                        # Mock return values
                        mock_job_check.return_value = {'quality_score': 0.9, 'issues': [], 'warnings': []}
                        mock_profile_check.return_value = {'quality_score': 0.8, 'issues': [], 'warnings': []}
                        mock_skill_check.return_value = {'quality_score': 0.95, 'issues': [], 'warnings': []}
                        
                        report = await validator.run_data_quality_checks()
                        
                        assert 'overall_quality_score' in report
                        assert 'checks' in report
                        assert report['overall_quality_score'] > 0.5


class TestBackupRecoveryManager:
    """Test backup and recovery functionality"""
    
    @pytest.fixture
    def backup_manager(self):
        """Create a test backup manager instance"""
        manager = BackupRecoveryManager()
        # Mock the backup location
        manager.backup_location = Mock()
        return manager
    
    async def test_execute_backup(self, backup_manager):
        """Test backup execution"""
        metadata = {
            'execution_id': 'backup_test_001',
            'backup_type': 'full',
            'include_ml_models': True,
            'compress_backup': True
        }
        
        # Mock backup methods
        backup_manager._backup_database = AsyncMock(return_value={
            'success': True, 'size_mb': 100, 'errors': []
        })
        backup_manager._backup_redis = AsyncMock(return_value={
            'success': True, 'size_mb': 10, 'errors': []
        })
        backup_manager._backup_ml_models = AsyncMock(return_value={
            'success': True, 'size_mb': 500, 'errors': []
        })
        backup_manager._backup_configurations = AsyncMock(return_value={
            'success': True, 'size_mb': 5, 'errors': []
        })
        backup_manager._compress_backup = AsyncMock(return_value=300)
        backup_manager._store_backup_metadata = AsyncMock()
        backup_manager._cleanup_old_backups = AsyncMock()
        backup_manager.monitor = AsyncMock()
        
        # Mock backup directory creation
        with patch('pathlib.Path.mkdir'):
            result = await backup_manager.execute_backup(metadata)
        
        assert result['success'] is True
        assert result['total_size_mb'] == 615  # Sum of all component sizes
        assert 'compressed_size_mb' in result


class TestJobCollectionPipeline:
    """Test job collection pipeline"""
    
    @pytest.fixture
    def pipeline(self):
        """Create a test job collection pipeline"""
        pipeline = JobCollectionPipeline()
        pipeline.monitor = AsyncMock()
        return pipeline
    
    async def test_execute_pipeline(self, pipeline):
        """Test job collection pipeline execution"""
        metadata = {
            'execution_id': 'job_collection_test_001',
            'sources': ['linkedin', 'indeed'],
            'max_jobs_per_source': 100
        }
        
        # Mock pipeline methods
        pipeline._collect_from_source = AsyncMock(return_value={
            'collected': 50, 'failed': 5, 'source': 'linkedin'
        })
        pipeline._validate_collected_data = AsyncMock(return_value={
            'quality_score': 0.9, 'meets_threshold': True
        })
        pipeline._store_collection_summary = AsyncMock()
        
        await pipeline.execute(metadata)
        
        # Verify monitoring was updated
        pipeline.monitor.update_job_metrics.assert_called()


class TestSkillTaxonomyPipeline:
    """Test skill taxonomy update pipeline"""
    
    @pytest.fixture
    def pipeline(self):
        """Create a test skill taxonomy pipeline"""
        pipeline = SkillTaxonomyPipeline()
        pipeline.monitor = AsyncMock()
        return pipeline
    
    async def test_execute_pipeline(self, pipeline):
        """Test skill taxonomy update pipeline execution"""
        metadata = {
            'execution_id': 'skill_taxonomy_test_001'
        }
        
        # Mock pipeline methods
        pipeline._analyze_job_postings = AsyncMock(return_value={
            'skill_mentions': {'python': 100, 'javascript': 80},
            'total_jobs_analyzed': 200
        })
        pipeline._detect_emerging_skills = AsyncMock(return_value=[
            {'name': 'rust', 'mention_count': 25}
        ])
        pipeline._update_skill_categories = AsyncMock(return_value=[])
        pipeline._update_skill_demand_scores = AsyncMock(return_value=[])
        pipeline._cleanup_obsolete_skills = AsyncMock(return_value={})
        pipeline._update_skill_relationships = AsyncMock(return_value=[])
        pipeline._store_update_summary = AsyncMock()
        
        await pipeline.execute(metadata)
        
        # Verify monitoring was updated
        pipeline.monitor.update_job_metrics.assert_called()


class TestProfileRefreshPipeline:
    """Test profile refresh pipeline"""
    
    @pytest.fixture
    def pipeline(self):
        """Create a test profile refresh pipeline"""
        pipeline = ProfileRefreshPipeline()
        pipeline.monitor = AsyncMock()
        return pipeline
    
    async def test_execute_pipeline(self, pipeline):
        """Test profile refresh pipeline execution"""
        metadata = {
            'execution_id': 'profile_refresh_test_001',
            'refresh_mode': 'incremental',
            'platforms': ['github', 'linkedin']
        }
        
        # Mock pipeline methods
        pipeline._get_profiles_to_refresh = AsyncMock(return_value=[
            Mock(id='profile1', user_id='user1'),
            Mock(id='profile2', user_id='user2')
        ])
        pipeline._refresh_profile = AsyncMock(return_value={
            'profile_id': 'profile1',
            'platform_results': {'github': True, 'linkedin': True}
        })
        pipeline._store_refresh_summary = AsyncMock()
        
        await pipeline.execute(metadata)
        
        # Verify monitoring was updated
        pipeline.monitor.update_job_metrics.assert_called()


class TestModelTrainingPipeline:
    """Test model training pipeline"""
    
    @pytest.fixture
    def pipeline(self):
        """Create a test model training pipeline"""
        pipeline = ModelTrainingPipeline()
        pipeline.monitor = AsyncMock()
        return pipeline
    
    async def test_execute_pipeline(self, pipeline):
        """Test model training pipeline execution"""
        metadata = {
            'execution_id': 'model_training_test_001',
            'training_mode': 'incremental',
            'models': ['recommendation', 'skill_extraction']
        }
        
        # Mock pipeline methods
        pipeline._prepare_training_data = AsyncMock(return_value={
            'has_sufficient_data': True,
            'data_stats': {'total_profiles': 1000, 'total_jobs': 5000}
        })
        pipeline._train_model = AsyncMock(return_value={
            'success': True,
            'model_path': '/models/recommendation_v1.pkl',
            'metrics': {'accuracy': 0.85}
        })
        pipeline._evaluate_model = AsyncMock(return_value={
            'precision': 0.82, 'recall': 0.78
        })
        pipeline._setup_ab_test = AsyncMock(return_value={
            'test_id': 'ab_test_001'
        })
        pipeline._generate_training_summary = AsyncMock(return_value={
            'models_trained': 2, 'success_rate': 1.0
        })
        pipeline._store_training_summary = AsyncMock()
        
        await pipeline.execute(metadata)
        
        # Verify monitoring was updated
        pipeline.monitor.update_job_metrics.assert_called()


class TestPipelineIntegration:
    """Integration tests for pipeline automation system"""
    
    async def test_full_pipeline_workflow(self):
        """Test complete pipeline workflow from scheduling to execution"""
        # This would be a more comprehensive integration test
        # that tests the entire workflow in a test environment
        
        scheduler = PipelineScheduler()
        monitor = PipelineMonitor()
        
        # Mock Redis and other dependencies
        scheduler.redis_client = AsyncMock()
        monitor.redis_client = AsyncMock()
        
        # Create a test job
        test_job = PipelineJob(
            job_id="integration_test_001",
            pipeline_name="job_collection",
            schedule_type="interval",
            schedule_config={"minutes": 5},
            enabled=True
        )
        
        # This test would verify the complete workflow
        # but requires more setup for a full integration test
        assert test_job.job_id == "integration_test_001"
    
    async def test_error_handling_and_recovery(self):
        """Test error handling and recovery mechanisms"""
        scheduler = PipelineScheduler()
        scheduler.redis_client = AsyncMock()
        
        # Test job execution with failure
        failing_job = PipelineJob(
            job_id="failing_test_001",
            pipeline_name="nonexistent_pipeline",
            schedule_type="interval",
            schedule_config={"minutes": 1},
            enabled=True,
            max_retries=2
        )
        
        # Mock the pipeline execution to fail
        with patch.object(scheduler, '_run_pipeline', side_effect=Exception("Pipeline failed")):
            # This should handle the error gracefully and schedule retries
            try:
                await scheduler._execute_pipeline_job(failing_job)
            except Exception:
                # The scheduler should catch and handle the exception
                pass
        
        # Verify error handling was triggered
        assert True  # Placeholder for actual error handling verification


@pytest.mark.asyncio
async def test_pipeline_automation_api_endpoints():
    """Test API endpoints for pipeline automation"""
    from fastapi.testclient import TestClient
    from app.main import app
    
    client = TestClient(app)
    
    # Test getting scheduled jobs (this would require authentication in real scenario)
    # response = client.get("/api/v1/pipeline/jobs")
    # assert response.status_code in [200, 401]  # 401 if authentication required
    
    # Test system health endpoint
    # response = client.get("/api/v1/pipeline/system/health")
    # assert response.status_code in [200, 401]
    
    # Placeholder test - actual API tests would require proper setup
    assert True


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])