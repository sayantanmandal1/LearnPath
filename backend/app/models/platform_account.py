"""
Platform account models for storing external platform connections and scraped data
"""
from datetime import datetime
from typing import Optional
from uuid import uuid4

from sqlalchemy import DateTime, ForeignKey, JSON, String, Text, Boolean, Enum
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func
import enum

from app.core.database import Base


class PlatformType(str, enum.Enum):
    """Supported platform types"""
    GITHUB = "github"
    LEETCODE = "leetcode"
    LINKEDIN = "linkedin"
    CODEFORCES = "codeforces"
    ATCODER = "atcoder"
    HACKERRANK = "hackerrank"
    KAGGLE = "kaggle"


class ScrapingStatus(str, enum.Enum):
    """Platform data scraping status"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    RATE_LIMITED = "rate_limited"
    UNAUTHORIZED = "unauthorized"
    NOT_FOUND = "not_found"


class PlatformAccount(Base):
    """Model for storing platform account connections and scraped data"""
    
    __tablename__ = "platform_accounts"
    
    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        primary_key=True,
        default=lambda: str(uuid4()),
        index=True
    )
    user_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    
    # Platform identification
    platform: Mapped[PlatformType] = mapped_column(
        Enum(PlatformType),
        nullable=False,
        index=True
    )
    username: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        comment="Username or handle on the platform"
    )
    profile_url: Mapped[Optional[str]] = mapped_column(
        String(500),
        nullable=True,
        comment="Full URL to user's profile"
    )
    
    # Connection status
    is_active: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        nullable=False,
        comment="Whether this account connection is active"
    )
    is_verified: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
        comment="Whether the account ownership has been verified"
    )
    
    # Scraping information
    scraping_status: Mapped[ScrapingStatus] = mapped_column(
        Enum(ScrapingStatus),
        nullable=False,
        default=ScrapingStatus.PENDING
    )
    last_scraped_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True
    )
    next_scrape_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="Scheduled time for next data scraping"
    )
    scrape_frequency_hours: Mapped[int] = mapped_column(
        default=24,
        nullable=False,
        comment="How often to scrape data in hours"
    )
    
    # Scraped data storage
    raw_data: Mapped[Optional[dict]] = mapped_column(
        JSON,
        nullable=True,
        comment="Raw data scraped from platform"
    )
    processed_data: Mapped[Optional[dict]] = mapped_column(
        JSON,
        nullable=True,
        comment="Processed and normalized platform data"
    )
    skills_data: Mapped[Optional[dict]] = mapped_column(
        JSON,
        nullable=True,
        comment="Skills extracted from platform data"
    )
    achievements_data: Mapped[Optional[dict]] = mapped_column(
        JSON,
        nullable=True,
        comment="Achievements and certifications from platform"
    )
    statistics: Mapped[Optional[dict]] = mapped_column(
        JSON,
        nullable=True,
        comment="Platform-specific statistics and metrics"
    )
    
    # Error handling
    last_error: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        comment="Last error encountered during scraping"
    )
    error_count: Mapped[int] = mapped_column(
        default=0,
        nullable=False,
        comment="Number of consecutive scraping errors"
    )
    
    # Rate limiting information
    rate_limit_reset_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="When rate limit will reset"
    )
    requests_remaining: Mapped[Optional[int]] = mapped_column(
        nullable=True,
        comment="API requests remaining before rate limit"
    )
    
    # Data quality metrics
    data_completeness_score: Mapped[Optional[float]] = mapped_column(
        nullable=True,
        comment="Score indicating how complete the scraped data is (0-1)"
    )
    data_freshness_score: Mapped[Optional[float]] = mapped_column(
        nullable=True,
        comment="Score indicating how fresh the data is (0-1)"
    )
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False
    )
    
    # Relationships
    user = relationship("User", backref="platform_accounts")
    
    def __repr__(self) -> str:
        return f"<PlatformAccount(id={self.id}, platform={self.platform}, username={self.username})>"


class PlatformScrapingLog(Base):
    """Model for logging platform scraping activities"""
    
    __tablename__ = "platform_scraping_logs"
    
    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        primary_key=True,
        default=lambda: str(uuid4()),
        index=True
    )
    platform_account_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("platform_accounts.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    
    # Scraping session information
    scraping_started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False
    )
    scraping_completed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True
    )
    status: Mapped[ScrapingStatus] = mapped_column(
        Enum(ScrapingStatus),
        nullable=False
    )
    
    # Data collected
    data_points_collected: Mapped[int] = mapped_column(
        default=0,
        nullable=False,
        comment="Number of data points successfully collected"
    )
    api_requests_made: Mapped[int] = mapped_column(
        default=0,
        nullable=False,
        comment="Number of API requests made during scraping"
    )
    
    # Error information
    error_message: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True
    )
    error_details: Mapped[Optional[dict]] = mapped_column(
        JSON,
        nullable=True,
        comment="Detailed error information"
    )
    
    # Performance metrics
    processing_time_seconds: Mapped[Optional[float]] = mapped_column(
        nullable=True,
        comment="Time taken to complete scraping"
    )
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False
    )
    
    # Relationships
    platform_account = relationship("PlatformAccount", backref="scraping_logs")
    
    def __repr__(self) -> str:
        return f"<PlatformScrapingLog(id={self.id}, status={self.status})>"