#!/usr/bin/env python3
"""
Data Pipeline Automation Demo
Demonstrates the automated data pipeline system functionality.
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime

# Add the backend directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.data_pipeline.pipeline_initializer import get_pipeline_initializer
from app.services.data_pipeline.pipeline_scheduler import get_pipeline_scheduler, PipelineJob
from app.services.data_pipeline.pipeline_monitor import get_pipeline_monitor
from app.services.data_pipeline.data_quality_validator import DataQualityValidator
from app.services.data_pipeline.backup_recovery import BackupRecoveryManager
from app.core.logging import get_logger

logger = get_logger(__name__)


async def demo_pipeline_automation():
    """Demonstrate the pipeline automation system"""
    
    print("🚀 Data Pipeline Automation Demo")
    print("=" * 50)
    
    try:
        # 1. Initialize the pipeline system
        print("\n1. Initializing Pipeline System...")
        initializer = await get_pipeline_initializer()
        
        # Mock the Redis connection for demo
        from unittest.mock import AsyncMock
        
        # Create mock Redis client
        mock_redis = AsyncMock()
        mock_redis.keys.return_value = []
        mock_redis.get.return_value = None
        mock_redis.set.return_value = True
        mock_redis.delete.return_value = True
        
        # Get scheduler and set mock Redis
        scheduler = await get_pipeline_scheduler()
        scheduler.redis_client = mock_redis
        
        # Mock the APScheduler for demo
        from unittest.mock import Mock
        scheduler.scheduler = Mock()
        scheduler.scheduler.running = True
        scheduler.scheduler.add_job = Mock()
        scheduler.scheduler.remove_job = Mock()
        scheduler.scheduler.get_jobs = Mock(return_value=[])
        
        print("✅ Pipeline system initialized")
        
        # 2. Create a sample pipeline job
        print("\n2. Creating Sample Pipeline Job...")
        
        sample_job = PipelineJob(
            job_id="demo_job_collection",
            pipeline_name="job_collection",
            schedule_type="cron",
            schedule_config={"hour": 2, "minute": 0},
            enabled=True,
            max_retries=3,
            retry_delay=300,
            timeout=3600,
            metadata={
                "description": "Demo job collection pipeline",
                "sources": ["linkedin", "indeed"],
                "max_jobs_per_source": 100
            }
        )
        
        success = await scheduler.schedule_job(sample_job)
        print(f"✅ Sample job scheduled: {success}")
        
        # 3. Demonstrate monitoring
        print("\n3. Demonstrating Pipeline Monitoring...")
        
        monitor = await get_pipeline_monitor()
        monitor.redis_client = mock_redis
        
        # Start monitoring a demo execution
        execution_id = f"demo_execution_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        await monitor.start_job_monitoring(execution_id, "job_collection")
        
        # Update metrics
        await monitor.update_job_metrics(
            execution_id,
            records_processed=150,
            records_failed=5,
            memory_usage_mb=256.5,
            cpu_usage_percent=45.2,
            data_quality_score=0.95
        )
        
        print(f"✅ Monitoring started for execution: {execution_id}")
        
        # 4. Demonstrate data quality validation
        print("\n4. Demonstrating Data Quality Validation...")
        
        validator = DataQualityValidator()
        
        # Validate sample job posting
        sample_job_posting = {
            "title": "Senior Software Engineer",
            "company": "Tech Corp",
            "description": "We are looking for an experienced software engineer...",
            "location": "San Francisco, CA",
            "salary_range": "$120,000 - $180,000",
            "required_skills": ["Python", "JavaScript", "React", "PostgreSQL"]
        }
        
        is_valid = await validator.validate_job_posting(sample_job_posting)
        print(f"✅ Job posting validation result: {is_valid}")
        
        # 5. Demonstrate backup system
        print("\n5. Demonstrating Backup System...")
        
        backup_manager = BackupRecoveryManager()
        
        # Mock backup execution
        backup_metadata = {
            "execution_id": f"demo_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "backup_type": "incremental",
            "include_ml_models": True,
            "compress_backup": True
        }
        
        print("✅ Backup system ready (demo mode)")
        
        # 6. Show system health
        print("\n6. System Health Check...")
        
        health_status = await initializer.health_check()
        
        print("📊 System Health Status:")
        print(f"  - Overall Health: {health_status.get('overall_health', 'unknown')}")
        print(f"  - Scheduler Status: {health_status.get('scheduler_status', 'unknown')}")
        print(f"  - Monitor Status: {health_status.get('monitor_status', 'unknown')}")
        print(f"  - Scheduled Jobs: {health_status.get('scheduled_jobs_count', 0)}")
        print(f"  - Active Executions: {health_status.get('active_executions', 0)}")
        
        # 7. Show available pipelines
        print("\n7. Available Pipeline Types:")
        pipeline_types = [
            "job_collection - Daily automated job posting collection",
            "skill_taxonomy_update - Weekly skill taxonomy updates",
            "profile_refresh - Regular profile data refresh",
            "model_training - Automated model retraining",
            "data_quality_check - Data quality monitoring",
            "backup - System backup and recovery"
        ]
        
        for pipeline in pipeline_types:
            print(f"  📋 {pipeline}")
        
        # 8. Show configuration
        print("\n8. Pipeline Configuration Features:")
        features = [
            "✅ Automated job scheduling with cron and interval triggers",
            "✅ Real-time pipeline monitoring and alerting",
            "✅ Data quality validation and reporting",
            "✅ Automated backup and disaster recovery",
            "✅ Performance monitoring and optimization",
            "✅ Retry mechanisms with exponential backoff",
            "✅ Comprehensive logging and audit trails",
            "✅ REST API for pipeline management"
        ]
        
        for feature in features:
            print(f"  {feature}")
        
        print("\n🎉 Pipeline Automation Demo Completed Successfully!")
        print("=" * 50)
        
        return True
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        logger.error(f"Pipeline automation demo failed: {e}")
        return False


async def demo_api_endpoints():
    """Demonstrate API endpoint functionality"""
    
    print("\n🌐 API Endpoints Available:")
    print("=" * 30)
    
    endpoints = [
        "GET /api/v1/pipeline/jobs - List scheduled jobs",
        "POST /api/v1/pipeline/jobs - Schedule new job",
        "DELETE /api/v1/pipeline/jobs/{job_id} - Remove job",
        "GET /api/v1/pipeline/jobs/{job_id}/status - Get job status",
        "POST /api/v1/pipeline/jobs/{job_id}/execute - Execute job",
        "GET /api/v1/pipeline/executions/{execution_id} - Get execution status",
        "GET /api/v1/pipeline/pipelines/{name}/history - Get pipeline history",
        "GET /api/v1/pipeline/system/health - System health check",
        "GET /api/v1/pipeline/data-quality/report - Data quality report",
        "POST /api/v1/pipeline/backup - Create backup",
        "GET /api/v1/pipeline/backup/history - Backup history",
        "POST /api/v1/pipeline/backup/{id}/restore - Restore from backup"
    ]
    
    for endpoint in endpoints:
        print(f"  📡 {endpoint}")


async def main():
    """Main demo function"""
    
    # Run the main demo
    success = await demo_pipeline_automation()
    
    if success:
        # Show API endpoints
        await demo_api_endpoints()
        
        print("\n📚 Next Steps:")
        print("  1. Start the FastAPI server: uvicorn app.main:app --reload")
        print("  2. Access API docs: http://localhost:8000/docs")
        print("  3. Run pipeline automation: python scripts/run_pipeline_automation.py")
        print("  4. Monitor system health via API endpoints")
        
    return success


if __name__ == "__main__":
    # Run the demo
    try:
        result = asyncio.run(main())
        if result:
            print("\n✨ Demo completed successfully!")
        else:
            print("\n❌ Demo failed!")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n👋 Demo interrupted by user")
    except Exception as e:
        print(f"\n💥 Demo crashed: {e}")
        sys.exit(1)