"""
Application configuration settings
"""
from typing import Dict, List, Optional

from pydantic import Field, validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings"""
    
    # Application
    PROJECT_NAME: str = "AI Career Recommender"
    VERSION: str = "0.1.0"
    DEBUG: bool = False
    SECRET_KEY: str = Field(..., min_length=32)
    API_V1_STR: str = "/api/v1"
    ALLOWED_HOSTS: List[str] = ["*"]
    
    # CORS Configuration
    CORS_ORIGINS: List[str] = [
        "http://localhost:3000",  # React dev server
        "http://localhost:5173",  # Vite dev server
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
    ]
    CORS_ALLOW_CREDENTIALS: bool = True
    CORS_ALLOW_METHODS: List[str] = ["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"]
    CORS_ALLOW_HEADERS: List[str] = [
        "Accept",
        "Accept-Language",
        "Content-Language",
        "Content-Type",
        "Authorization",
        "X-Requested-With",
    ]
    
    # Database
    DATABASE_URL: str = Field(..., description="PostgreSQL database URL")
    DATABASE_POOL_SIZE: int = 20
    DATABASE_MAX_OVERFLOW: int = 30
    
    # Redis
    REDIS_URL: Optional[str] = Field(None, description="Redis connection URL")
    REDIS_CACHE_TTL: int = 300  # 5 minutes
    
    # JWT Authentication
    JWT_SECRET_KEY: str = Field(..., min_length=32)
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    
    # External APIs
    GITHUB_TOKEN: Optional[str] = None
    LINKEDIN_CLIENT_ID: Optional[str] = None
    LINKEDIN_CLIENT_SECRET: Optional[str] = None
    GEMINI_API_KEY: Optional[str] = None
    
    # Supabase Configuration
    SUPABASE_URL: Optional[str] = Field(None, description="Supabase project URL")
    SUPABASE_ANON_KEY: Optional[str] = Field(None, description="Supabase anonymous key")
    SUPABASE_SERVICE_ROLE_KEY: Optional[str] = Field(None, description="Supabase service role key")
    SUPABASE_JWT_SECRET: Optional[str] = Field(None, description="Supabase JWT secret")
    
    # Data Sync Configuration
    ENABLE_DATA_SYNC: bool = True
    DATA_SYNC_INTERVAL_MINUTES: int = 60  # How often to run sync checks
    CONFLICT_RESOLUTION_STRATEGY: str = "hybrid"  # hybrid, supabase_priority, postgresql_priority
    MAX_SYNC_RETRIES: int = 3
    SYNC_BATCH_SIZE: int = 100
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "json"
    
    # Monitoring
    ENABLE_METRICS: bool = True
    ENABLE_ALERTING: bool = True
    
    # Alerting Configuration
    SMTP_HOST: Optional[str] = None
    SMTP_PORT: int = 587
    SMTP_USERNAME: Optional[str] = None
    SMTP_PASSWORD: Optional[str] = None
    SMTP_USE_TLS: bool = True
    ALERT_FROM_EMAIL: str = "alerts@aicareer.com"
    ALERT_TO_EMAILS: List[str] = ["admin@aicareer.com"]
    
    SLACK_WEBHOOK_URL: Optional[str] = None
    SLACK_CHANNEL: str = "#alerts"
    SLACK_USERNAME: str = "AI Career Bot"
    
    ALERT_WEBHOOK_URL: Optional[str] = None
    ALERT_WEBHOOK_HEADERS: Dict[str, str] = {}
    
    # Performance Optimization
    CACHE_LOCAL_TTL: int = 60  # Local cache TTL in seconds
    PERFORMANCE_MONITORING_INTERVAL: int = 60  # Performance metrics collection interval
    MAX_METRICS_HISTORY: int = 1000  # Maximum number of metrics to keep in memory
    
    # Celery Configuration
    CELERY_BROKER_URL: Optional[str] = None  # Defaults to REDIS_URL
    CELERY_RESULT_BACKEND: Optional[str] = None  # Defaults to REDIS_URL
    CELERY_WORKER_CONCURRENCY: int = 4
    CELERY_TASK_SOFT_TIME_LIMIT: int = 300  # 5 minutes
    CELERY_TASK_TIME_LIMIT: int = 600  # 10 minutes
    
    # Database Optimization
    DB_QUERY_TIMEOUT: int = 30  # Database query timeout in seconds
    DB_SLOW_QUERY_THRESHOLD: float = 1.0  # Slow query threshold in seconds
    DB_CONNECTION_TIMEOUT: int = 5  # Connection timeout in seconds
    
    @property
    def celery_broker_url(self) -> str:
        """Get Celery broker URL, defaulting to Redis URL"""
        return self.CELERY_BROKER_URL or self.REDIS_URL
    
    @property
    def celery_result_backend(self) -> str:
        """Get Celery result backend URL, defaulting to Redis URL"""
        return self.CELERY_RESULT_BACKEND or self.REDIS_URL
    
    @validator("ALLOWED_HOSTS", pre=True)
    def assemble_allowed_hosts(cls, v):
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",")]
        elif isinstance(v, (list, str)):
            return v
        raise ValueError(v)
    
    @validator("CORS_ORIGINS", pre=True)
    def assemble_cors_origins(cls, v):
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",")]
        elif isinstance(v, (list, str)):
            return v
        raise ValueError(v)
    
    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()