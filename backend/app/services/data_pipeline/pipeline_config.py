"""
Pipeline Configuration Management
Handles configuration for data pipeline automation and scheduling.
"""

import os
from typing import Dict, Any, List
from dataclasses import dataclass
from pathlib import Path
import yaml
import json

from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for data pipeline automation"""
    
    # Redis configuration
    redis_host: str = os.getenv("REDIS_HOST", "localhost")
    redis_port: int = int(os.getenv("REDIS_PORT", "6379"))
    redis_db: int = int(os.getenv("REDIS_DB", "0"))
    redis_password: str = os.getenv("REDIS_PASSWORD", "")
    
    # Pipeline execution settings
    max_concurrent_jobs: int = int(os.getenv("MAX_CONCURRENT_PIPELINE_JOBS", "5"))
    default_timeout: int = int(os.getenv("DEFAULT_PIPELINE_TIMEOUT", "3600"))
    default_retry_count: int = int(os.getenv("DEFAULT_RETRY_COUNT", "3"))
    default_retry_delay: int = int(os.getenv("DEFAULT_RETRY_DELAY", "300"))
    
    # Data quality thresholds
    min_job_postings_per_day: int = int(os.getenv("MIN_JOB_POSTINGS_PER_DAY", "100"))
    max_skill_extraction_error_rate: float = float(os.getenv("MAX_SKILL_EXTRACTION_ERROR_RATE", "0.1"))
    min_profile_refresh_success_rate: float = float(os.getenv("MIN_PROFILE_REFRESH_SUCCESS_RATE", "0.8"))
    
    # Backup and recovery settings
    backup_retention_days: int = int(os.getenv("BACKUP_RETENTION_DAYS", "30"))
    backup_location: str = os.getenv("BACKUP_LOCATION", "/app/backups")
    enable_auto_backup: bool = os.getenv("ENABLE_AUTO_BACKUP", "true").lower() == "true"
    
    # Monitoring and alerting
    enable_monitoring: bool = os.getenv("ENABLE_PIPELINE_MONITORING", "true").lower() == "true"
    alert_webhook_url: str = os.getenv("ALERT_WEBHOOK_URL", "")
    alert_email: str = os.getenv("ALERT_EMAIL", "")
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        self._validate_config()
    
    def _validate_config(self):
        """Validate configuration values"""
        if self.max_concurrent_jobs <= 0:
            raise ValueError("max_concurrent_jobs must be positive")
        
        if self.default_timeout <= 0:
            raise ValueError("default_timeout must be positive")
        
        if self.min_job_postings_per_day <= 0:
            raise ValueError("min_job_postings_per_day must be positive")
        
        if not (0 <= self.max_skill_extraction_error_rate <= 1):
            raise ValueError("max_skill_extraction_error_rate must be between 0 and 1")
        
        if not (0 <= self.min_profile_refresh_success_rate <= 1):
            raise ValueError("min_profile_refresh_success_rate must be between 0 and 1")
    
    @classmethod
    def from_file(cls, config_path: str) -> 'PipelineConfig':
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)
            
            return cls(**config_data)
        except Exception as e:
            logger.error(f"Failed to load config from {config_path}: {e}")
            return cls()  # Return default config
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            'redis_host': self.redis_host,
            'redis_port': self.redis_port,
            'redis_db': self.redis_db,
            'max_concurrent_jobs': self.max_concurrent_jobs,
            'default_timeout': self.default_timeout,
            'default_retry_count': self.default_retry_count,
            'default_retry_delay': self.default_retry_delay,
            'min_job_postings_per_day': self.min_job_postings_per_day,
            'max_skill_extraction_error_rate': self.max_skill_extraction_error_rate,
            'min_profile_refresh_success_rate': self.min_profile_refresh_success_rate,
            'backup_retention_days': self.backup_retention_days,
            'backup_location': self.backup_location,
            'enable_auto_backup': self.enable_auto_backup,
            'enable_monitoring': self.enable_monitoring,
            'alert_webhook_url': self.alert_webhook_url,
            'alert_email': self.alert_email
        }


class DefaultPipelineSchedules:
    """Default scheduling configurations for different pipelines"""
    
    @staticmethod
    def get_job_collection_schedule() -> Dict[str, Any]:
        """Daily job posting collection at 2 AM UTC"""
        return {
            'schedule_type': 'cron',
            'schedule_config': {
                'hour': 2,
                'minute': 0,
                'timezone': 'UTC'
            },
            'max_retries': 3,
            'retry_delay': 1800,  # 30 minutes
            'timeout': 7200  # 2 hours
        }
    
    @staticmethod
    def get_skill_taxonomy_schedule() -> Dict[str, Any]:
        """Weekly skill taxonomy update on Sundays at 3 AM UTC"""
        return {
            'schedule_type': 'cron',
            'schedule_config': {
                'day_of_week': 'sun',
                'hour': 3,
                'minute': 0,
                'timezone': 'UTC'
            },
            'max_retries': 2,
            'retry_delay': 3600,  # 1 hour
            'timeout': 10800  # 3 hours
        }
    
    @staticmethod
    def get_profile_refresh_schedule() -> Dict[str, Any]:
        """Profile refresh every 6 hours"""
        return {
            'schedule_type': 'interval',
            'schedule_config': {
                'hours': 6
            },
            'max_retries': 2,
            'retry_delay': 1800,  # 30 minutes
            'timeout': 3600  # 1 hour
        }
    
    @staticmethod
    def get_model_training_schedule() -> Dict[str, Any]:
        """Model retraining weekly on Saturdays at 1 AM UTC"""
        return {
            'schedule_type': 'cron',
            'schedule_config': {
                'day_of_week': 'sat',
                'hour': 1,
                'minute': 0,
                'timezone': 'UTC'
            },
            'max_retries': 1,
            'retry_delay': 7200,  # 2 hours
            'timeout': 14400  # 4 hours
        }
    
    @staticmethod
    def get_data_quality_check_schedule() -> Dict[str, Any]:
        """Data quality checks every 4 hours"""
        return {
            'schedule_type': 'interval',
            'schedule_config': {
                'hours': 4
            },
            'max_retries': 1,
            'retry_delay': 900,  # 15 minutes
            'timeout': 1800  # 30 minutes
        }
    
    @staticmethod
    def get_backup_schedule() -> Dict[str, Any]:
        """Daily backup at 4 AM UTC"""
        return {
            'schedule_type': 'cron',
            'schedule_config': {
                'hour': 4,
                'minute': 0,
                'timezone': 'UTC'
            },
            'max_retries': 2,
            'retry_delay': 1800,  # 30 minutes
            'timeout': 3600  # 1 hour
        }


class PipelineConfigManager:
    """Manages pipeline configurations and schedules"""
    
    def __init__(self, config_dir: str = "/app/config/pipelines"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.schedules = DefaultPipelineSchedules()
    
    def get_default_schedules(self) -> Dict[str, Dict[str, Any]]:
        """Get all default pipeline schedules"""
        return {
            'job_collection': self.schedules.get_job_collection_schedule(),
            'skill_taxonomy_update': self.schedules.get_skill_taxonomy_schedule(),
            'profile_refresh': self.schedules.get_profile_refresh_schedule(),
            'model_training': self.schedules.get_model_training_schedule(),
            'data_quality_check': self.schedules.get_data_quality_check_schedule(),
            'backup': self.schedules.get_backup_schedule()
        }
    
    def save_schedule_config(self, pipeline_name: str, config: Dict[str, Any]):
        """Save pipeline schedule configuration to file"""
        try:
            config_file = self.config_dir / f"{pipeline_name}_schedule.yaml"
            with open(config_file, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
            logger.info(f"Saved schedule config for {pipeline_name}")
        except Exception as e:
            logger.error(f"Failed to save schedule config for {pipeline_name}: {e}")
    
    def load_schedule_config(self, pipeline_name: str) -> Dict[str, Any]:
        """Load pipeline schedule configuration from file"""
        try:
            config_file = self.config_dir / f"{pipeline_name}_schedule.yaml"
            if config_file.exists():
                with open(config_file, 'r') as f:
                    return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load schedule config for {pipeline_name}: {e}")
        
        # Return default schedule if file doesn't exist or fails to load
        default_schedules = self.get_default_schedules()
        return default_schedules.get(pipeline_name, {})
    
    def get_all_pipeline_configs(self) -> Dict[str, Dict[str, Any]]:
        """Get configurations for all pipelines"""
        configs = {}
        default_schedules = self.get_default_schedules()
        
        for pipeline_name in default_schedules.keys():
            configs[pipeline_name] = self.load_schedule_config(pipeline_name)
        
        return configs
    
    def update_schedule_config(self, pipeline_name: str, updates: Dict[str, Any]):
        """Update specific fields in a pipeline schedule configuration"""
        try:
            current_config = self.load_schedule_config(pipeline_name)
            current_config.update(updates)
            self.save_schedule_config(pipeline_name, current_config)
            logger.info(f"Updated schedule config for {pipeline_name}")
        except Exception as e:
            logger.error(f"Failed to update schedule config for {pipeline_name}: {e}")


# Global configuration instances
pipeline_config = PipelineConfig()
config_manager = PipelineConfigManager()


def get_pipeline_config() -> PipelineConfig:
    """Get the global pipeline configuration"""
    return pipeline_config


def get_config_manager() -> PipelineConfigManager:
    """Get the global configuration manager"""
    return config_manager