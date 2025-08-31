"""
Pydantic schemas for pipeline automation API
"""

from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
from datetime import datetime


class PipelineJobCreate(BaseModel):
    """Schema for creating a new pipeline job"""
    job_id: str = Field(..., description="Unique identifier for the job")
    pipeline_name: str = Field(..., description="Name of the pipeline to execute")
    schedule_type: str = Field(..., description="Type of schedule (cron or interval)")
    schedule_config: Dict[str, Any] = Field(..., description="Schedule configuration parameters")
    enabled: bool = Field(True, description="Whether the job is enabled")
    max_retries: int = Field(3, description="Maximum number of retries on failure")
    retry_delay: int = Field(300, description="Delay between retries in seconds")
    timeout: int = Field(3600, description="Job timeout in seconds")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata for the job")


class PipelineJobResponse(BaseModel):
    """Schema for pipeline job response"""
    job_id: str
    name: str
    next_run_time: Optional[str]
    trigger: str
    status: str


class PipelineExecutionResponse(BaseModel):
    """Schema for pipeline execution response"""
    execution_id: str
    pipeline_name: str
    status: str
    start_time: str
    end_time: Optional[str]
    duration_seconds: Optional[float]
    records_processed: int
    records_failed: int
    data_quality_score: float
    error_count: int


class DataQualityCheck(BaseModel):
    """Schema for individual data quality check"""
    quality_score: float
    issues: List[str]
    warnings: List[str]
    statistics: Optional[Dict[str, Any]] = None


class DataQualityReport(BaseModel):
    """Schema for comprehensive data quality report"""
    timestamp: str
    overall_quality_score: float
    checks: Dict[str, DataQualityCheck]
    issues: List[str]
    warnings: List[str]
    recommendations: List[str]


class BackupRequest(BaseModel):
    """Schema for backup request"""
    backup_type: str = Field("full", description="Type of backup (full or incremental)")
    include_ml_models: bool = Field(True, description="Whether to include ML models in backup")
    compress_backup: bool = Field(True, description="Whether to compress the backup")


class BackupResponse(BaseModel):
    """Schema for backup response"""
    backup_id: str
    status: str
    message: str
    timestamp: str


class SystemHealthResponse(BaseModel):
    """Schema for system health response"""
    active_jobs: int
    total_pipelines: int
    success_rate_24h: float
    avg_duration_24h: float
    error_rate_24h: float
    data_quality_score_24h: float
    alerts_24h: int


class PipelineStats(BaseModel):
    """Schema for pipeline statistics"""
    total_executions: int
    successful_executions: int
    failed_executions: int
    success_rate: float
    average_duration: float
    last_execution: Optional[str]


class JobCollectionStats(BaseModel):
    """Schema for job collection pipeline statistics"""
    daily_stats: Dict[str, Dict[str, int]]
    total_collected: int
    total_failed: int
    average_per_day: float
    success_rate: float


class SkillTaxonomyStats(BaseModel):
    """Schema for skill taxonomy update statistics"""
    monthly_stats: Dict[str, Dict[str, int]]
    total_updates: int
    total_emerging_skills: int


class ProfileRefreshStats(BaseModel):
    """Schema for profile refresh statistics"""
    daily_stats: Dict[str, Dict[str, Any]]
    platform_stats: Dict[str, Dict[str, Any]]
    total_refreshed: int
    total_failed: int
    success_rate: float


class ModelTrainingStats(BaseModel):
    """Schema for model training statistics"""
    monthly_stats: Dict[str, Dict[str, Any]]
    total_trainings: int
    total_models_trained: int
    total_models_failed: int
    success_rate: float


class ProfileFreshness(BaseModel):
    """Schema for profile freshness information"""
    profile_id: str
    last_updated: Optional[str]
    platform_freshness: Dict[str, Dict[str, Any]]


class BackupVerification(BaseModel):
    """Schema for backup verification result"""
    backup_id: str
    timestamp: str
    integrity_check: bool
    components_verified: List[str]
    issues: List[str]


class RestoreRequest(BaseModel):
    """Schema for restore request"""
    backup_id: str
    components: Optional[List[str]] = Field(None, description="Components to restore")


class RestoreResponse(BaseModel):
    """Schema for restore response"""
    backup_id: str
    timestamp: str
    components_restored: List[str]
    components_failed: List[str]
    success: bool
    errors: List[str]