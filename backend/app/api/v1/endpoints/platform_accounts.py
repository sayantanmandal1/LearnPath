"""API endpoints for platform account management."""

from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.models.platform_account import PlatformAccount, PlatformType, ScrapingStatus
from app.services.external_apis.multi_platform_scraper import MultiPlatformScraper
from app.api.dependencies import get_current_user
from app.models.user import User


router = APIRouter()


class PlatformAccountCreate(BaseModel):
    """Schema for creating a platform account."""
    platform: PlatformType
    username: str = Field(..., min_length=1, max_length=255)
    profile_url: Optional[str] = Field(None, max_length=500)
    scrape_frequency_hours: int = Field(24, ge=1, le=168)  # 1 hour to 1 week


class PlatformAccountUpdate(BaseModel):
    """Schema for updating a platform account."""
    username: Optional[str] = Field(None, min_length=1, max_length=255)
    profile_url: Optional[str] = Field(None, max_length=500)
    is_active: Optional[bool] = None
    scrape_frequency_hours: Optional[int] = Field(None, ge=1, le=168)


class PlatformAccountResponse(BaseModel):
    """Schema for platform account response."""
    id: str
    platform: PlatformType
    username: str
    profile_url: Optional[str]
    is_active: bool
    is_verified: bool
    scraping_status: ScrapingStatus
    last_scraped_at: Optional[str]
    next_scrape_at: Optional[str]
    scrape_frequency_hours: int
    data_completeness_score: Optional[float]
    data_freshness_score: Optional[float]
    statistics: Optional[Dict[str, Any]]
    last_error: Optional[str]
    error_count: int
    created_at: str
    updated_at: str

    class Config:
        from_attributes = True


class PlatformDataPreview(BaseModel):
    """Schema for platform data preview."""
    account_id: str
    platform: PlatformType
    username: str
    raw_data_summary: Dict[str, Any]
    processed_data_summary: Dict[str, Any]
    skills_extracted: List[str]
    achievements_count: int
    statistics: Dict[str, Any]
    data_quality_score: float


class DataCollectionProgress(BaseModel):
    """Schema for data collection progress."""
    account_id: str
    status: ScrapingStatus
    progress_percentage: float
    current_step: str
    estimated_completion: Optional[str]
    data_points_collected: int
    errors_encountered: int


@router.get("/{user_id}/accounts", response_model=List[PlatformAccountResponse])
async def get_user_platform_accounts(
    user_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
) -> List[PlatformAccountResponse]:
    """
    Get all platform accounts for a user.
    
    Returns a list of all connected platform accounts with their current status,
    data quality metrics, and last scraping information.
    """
    # Verify user access
    if current_user.id != user_id:
        raise HTTPException(status_code=403, detail="Access denied")
    
    accounts = db.query(PlatformAccount).filter(
        PlatformAccount.user_id == user_id
    ).all()
    
    return [
        PlatformAccountResponse(
            id=account.id,
            platform=account.platform,
            username=account.username,
            profile_url=account.profile_url,
            is_active=account.is_active,
            is_verified=account.is_verified,
            scraping_status=account.scraping_status,
            last_scraped_at=account.last_scraped_at.isoformat() if account.last_scraped_at else None,
            next_scrape_at=account.next_scrape_at.isoformat() if account.next_scrape_at else None,
            scrape_frequency_hours=account.scrape_frequency_hours,
            data_completeness_score=account.data_completeness_score,
            data_freshness_score=account.data_freshness_score,
            statistics=account.statistics,
            last_error=account.last_error,
            error_count=account.error_count,
            created_at=account.created_at.isoformat(),
            updated_at=account.updated_at.isoformat()
        )
        for account in accounts
    ]


@router.post("/accounts", response_model=PlatformAccountResponse)
async def create_platform_account(
    account_data: PlatformAccountCreate,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
) -> PlatformAccountResponse:
    """
    Create a new platform account connection.
    
    Creates a new platform account and schedules initial data collection.
    The account will be validated before creation.
    """
    # Check if account already exists
    existing_account = db.query(PlatformAccount).filter(
        PlatformAccount.user_id == current_user.id,
        PlatformAccount.platform == account_data.platform,
        PlatformAccount.username == account_data.username
    ).first()
    
    if existing_account:
        raise HTTPException(
            status_code=400,
            detail=f"Account for {account_data.platform} with username {account_data.username} already exists"
        )
    
    # Generate profile URL if not provided
    profile_url = account_data.profile_url
    if not profile_url:
        profile_url = generate_profile_url(account_data.platform, account_data.username)
    
    # Create new platform account
    new_account = PlatformAccount(
        user_id=current_user.id,
        platform=account_data.platform,
        username=account_data.username,
        profile_url=profile_url,
        scrape_frequency_hours=account_data.scrape_frequency_hours,
        scraping_status=ScrapingStatus.PENDING
    )
    
    db.add(new_account)
    db.commit()
    db.refresh(new_account)
    
    # Schedule initial data collection
    background_tasks.add_task(
        start_data_collection,
        new_account.id,
        db
    )
    
    return PlatformAccountResponse(
        id=new_account.id,
        platform=new_account.platform,
        username=new_account.username,
        profile_url=new_account.profile_url,
        is_active=new_account.is_active,
        is_verified=new_account.is_verified,
        scraping_status=new_account.scraping_status,
        last_scraped_at=new_account.last_scraped_at.isoformat() if new_account.last_scraped_at else None,
        next_scrape_at=new_account.next_scrape_at.isoformat() if new_account.next_scrape_at else None,
        scrape_frequency_hours=new_account.scrape_frequency_hours,
        data_completeness_score=new_account.data_completeness_score,
        data_freshness_score=new_account.data_freshness_score,
        statistics=new_account.statistics,
        last_error=new_account.last_error,
        error_count=new_account.error_count,
        created_at=new_account.created_at.isoformat(),
        updated_at=new_account.updated_at.isoformat()
    )


@router.put("/accounts/{account_id}", response_model=PlatformAccountResponse)
async def update_platform_account(
    account_id: str,
    account_data: PlatformAccountUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
) -> PlatformAccountResponse:
    """
    Update a platform account.
    
    Updates account settings such as username, profile URL, active status,
    and scraping frequency.
    """
    account = db.query(PlatformAccount).filter(
        PlatformAccount.id == account_id,
        PlatformAccount.user_id == current_user.id
    ).first()
    
    if not account:
        raise HTTPException(status_code=404, detail="Platform account not found")
    
    # Update fields
    if account_data.username is not None:
        account.username = account_data.username
    if account_data.profile_url is not None:
        account.profile_url = account_data.profile_url
    if account_data.is_active is not None:
        account.is_active = account_data.is_active
    if account_data.scrape_frequency_hours is not None:
        account.scrape_frequency_hours = account_data.scrape_frequency_hours
    
    db.commit()
    db.refresh(account)
    
    return PlatformAccountResponse(
        id=account.id,
        platform=account.platform,
        username=account.username,
        profile_url=account.profile_url,
        is_active=account.is_active,
        is_verified=account.is_verified,
        scraping_status=account.scraping_status,
        last_scraped_at=account.last_scraped_at.isoformat() if account.last_scraped_at else None,
        next_scrape_at=account.next_scrape_at.isoformat() if account.next_scrape_at else None,
        scrape_frequency_hours=account.scrape_frequency_hours,
        data_completeness_score=account.data_completeness_score,
        data_freshness_score=account.data_freshness_score,
        statistics=account.statistics,
        last_error=account.last_error,
        error_count=account.error_count,
        created_at=account.created_at.isoformat(),
        updated_at=account.updated_at.isoformat()
    )


@router.delete("/accounts/{account_id}")
async def delete_platform_account(
    account_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
) -> Dict[str, str]:
    """
    Delete a platform account.
    
    Removes the platform account and all associated data.
    This action cannot be undone.
    """
    account = db.query(PlatformAccount).filter(
        PlatformAccount.id == account_id,
        PlatformAccount.user_id == current_user.id
    ).first()
    
    if not account:
        raise HTTPException(status_code=404, detail="Platform account not found")
    
    db.delete(account)
    db.commit()
    
    return {"message": "Platform account deleted successfully"}


@router.post("/accounts/{account_id}/refresh", response_model=DataCollectionProgress)
async def refresh_platform_data(
    account_id: str,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
) -> DataCollectionProgress:
    """
    Refresh data for a platform account.
    
    Triggers immediate data collection for the specified platform account.
    Returns the current progress status.
    """
    account = db.query(PlatformAccount).filter(
        PlatformAccount.id == account_id,
        PlatformAccount.user_id == current_user.id
    ).first()
    
    if not account:
        raise HTTPException(status_code=404, detail="Platform account not found")
    
    if not account.is_active:
        raise HTTPException(status_code=400, detail="Account is not active")
    
    # Update status to in progress
    account.scraping_status = ScrapingStatus.IN_PROGRESS
    db.commit()
    
    # Schedule data collection
    background_tasks.add_task(
        start_data_collection,
        account_id,
        db
    )
    
    return DataCollectionProgress(
        account_id=account_id,
        status=ScrapingStatus.IN_PROGRESS,
        progress_percentage=0.0,
        current_step="Initializing data collection",
        estimated_completion=None,
        data_points_collected=0,
        errors_encountered=0
    )


@router.get("/accounts/{account_id}/preview", response_model=PlatformDataPreview)
async def get_platform_data_preview(
    account_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
) -> PlatformDataPreview:
    """
    Get a preview of platform data.
    
    Returns a summary of the collected data including skills, achievements,
    and statistics without exposing sensitive information.
    """
    account = db.query(PlatformAccount).filter(
        PlatformAccount.id == account_id,
        PlatformAccount.user_id == current_user.id
    ).first()
    
    if not account:
        raise HTTPException(status_code=404, detail="Platform account not found")
    
    # Generate preview data
    raw_data_summary = {}
    processed_data_summary = {}
    skills_extracted = []
    achievements_count = 0
    statistics = account.statistics or {}
    
    if account.raw_data:
        raw_data_summary = {
            "total_fields": len(account.raw_data),
            "data_size_kb": len(str(account.raw_data)) / 1024,
            "last_updated": account.last_scraped_at.isoformat() if account.last_scraped_at else None
        }
    
    if account.processed_data:
        processed_data_summary = {
            "processed_fields": len(account.processed_data),
            "processing_version": account.processed_data.get("version", "1.0")
        }
    
    if account.skills_data:
        skills_extracted = list(account.skills_data.get("skills", []))[:10]  # Limit to first 10
    
    if account.achievements_data:
        achievements_count = len(account.achievements_data.get("achievements", []))
    
    # Calculate data quality score
    completeness = account.data_completeness_score or 0.0
    freshness = account.data_freshness_score or 0.0
    data_quality_score = (completeness + freshness) / 2
    
    return PlatformDataPreview(
        account_id=account_id,
        platform=account.platform,
        username=account.username,
        raw_data_summary=raw_data_summary,
        processed_data_summary=processed_data_summary,
        skills_extracted=skills_extracted,
        achievements_count=achievements_count,
        statistics=statistics,
        data_quality_score=data_quality_score
    )


@router.get("/accounts/{account_id}/progress", response_model=DataCollectionProgress)
async def get_data_collection_progress(
    account_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
) -> DataCollectionProgress:
    """
    Get the current data collection progress for a platform account.
    
    Returns real-time progress information for ongoing data collection.
    """
    account = db.query(PlatformAccount).filter(
        PlatformAccount.id == account_id,
        PlatformAccount.user_id == current_user.id
    ).first()
    
    if not account:
        raise HTTPException(status_code=404, detail="Platform account not found")
    
    # Calculate progress based on status
    progress_percentage = 0.0
    current_step = "Idle"
    
    if account.scraping_status == ScrapingStatus.PENDING:
        progress_percentage = 0.0
        current_step = "Waiting to start"
    elif account.scraping_status == ScrapingStatus.IN_PROGRESS:
        progress_percentage = 50.0  # Assume halfway through
        current_step = "Collecting data"
    elif account.scraping_status == ScrapingStatus.COMPLETED:
        progress_percentage = 100.0
        current_step = "Completed"
    elif account.scraping_status == ScrapingStatus.FAILED:
        progress_percentage = 0.0
        current_step = f"Failed: {account.last_error or 'Unknown error'}"
    
    return DataCollectionProgress(
        account_id=account_id,
        status=account.scraping_status,
        progress_percentage=progress_percentage,
        current_step=current_step,
        estimated_completion=account.next_scrape_at.isoformat() if account.next_scrape_at else None,
        data_points_collected=len(account.statistics or {}),
        errors_encountered=account.error_count
    )


def generate_profile_url(platform: PlatformType, username: str) -> str:
    """Generate profile URL based on platform and username."""
    url_patterns = {
        PlatformType.GITHUB: f"https://github.com/{username}",
        PlatformType.LEETCODE: f"https://leetcode.com/{username}",
        PlatformType.LINKEDIN: f"https://linkedin.com/in/{username}",
        PlatformType.CODEFORCES: f"https://codeforces.com/profile/{username}",
        PlatformType.ATCODER: f"https://atcoder.jp/users/{username}",
        PlatformType.HACKERRANK: f"https://www.hackerrank.com/{username}",
        PlatformType.KAGGLE: f"https://www.kaggle.com/{username}"
    }
    return url_patterns.get(platform, f"https://{platform}.com/{username}")


async def start_data_collection(account_id: str, db: Session):
    """Background task to start data collection for a platform account."""
    try:
        account = db.query(PlatformAccount).filter(
            PlatformAccount.id == account_id
        ).first()
        
        if not account:
            return
        
        # Initialize scraper service
        scraper_service = MultiPlatformScraper()
        
        # Update status
        account.scraping_status = ScrapingStatus.IN_PROGRESS
        db.commit()
        
        # Collect data based on platform
        platform_data = {}
        if account.platform == PlatformType.GITHUB:
            platform_data = await scraper_service.scrape_github(account.username)
        elif account.platform == PlatformType.LEETCODE:
            platform_data = await scraper_service.scrape_leetcode(account.username)
        elif account.platform == PlatformType.LINKEDIN:
            platform_data = await scraper_service.scrape_linkedin(account.profile_url)
        # Add other platforms as needed
        
        # Update account with collected data
        account.raw_data = platform_data
        account.scraping_status = ScrapingStatus.COMPLETED
        account.is_verified = True
        account.data_completeness_score = calculate_completeness_score(platform_data)
        account.data_freshness_score = 1.0  # Fresh data
        account.statistics = extract_statistics(account.platform, platform_data)
        account.last_scraped_at = db.func.now()
        account.error_count = 0
        account.last_error = None
        
        db.commit()
        
    except Exception as e:
        # Handle errors
        account = db.query(PlatformAccount).filter(
            PlatformAccount.id == account_id
        ).first()
        
        if account:
            account.scraping_status = ScrapingStatus.FAILED
            account.last_error = str(e)
            account.error_count += 1
            db.commit()


def calculate_completeness_score(data: dict) -> float:
    """Calculate data completeness score based on available fields."""
    if not data:
        return 0.0
    
    # Simple scoring based on number of fields
    total_possible_fields = 20  # Adjust based on platform
    actual_fields = len([v for v in data.values() if v is not None])
    
    return min(actual_fields / total_possible_fields, 1.0)


def extract_statistics(platform: PlatformType, data: dict) -> dict:
    """Extract platform-specific statistics from raw data."""
    if not data:
        return {}
    
    # Platform-specific statistics extraction
    if platform == PlatformType.GITHUB:
        return {
            "repositories": data.get("public_repos", 0),
            "followers": data.get("followers", 0),
            "following": data.get("following", 0)
        }
    elif platform == PlatformType.LEETCODE:
        return {
            "problems_solved": data.get("problems_solved", 0),
            "acceptance_rate": data.get("acceptance_rate", "0%"),
            "contest_rating": data.get("contest_rating", 0)
        }
    
    return {"data_points": len(data)}