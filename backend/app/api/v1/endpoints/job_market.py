"""
Job market data collection and analysis API endpoints
"""
from typing import List, Optional, Dict, Any
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel, Field

from app.core.database import get_db
from app.services.job_scrapers.scraper_manager import JobScraperManager
from app.services.job_scrapers.base_job_scraper import JobSearchParams
from app.services.job_analysis_service import JobAnalysisService
from app.services.market_trend_analyzer import MarketTrendAnalyzer
from app.services.data_collection_pipeline import DataCollectionPipeline
from app.repositories.job import JobRepository
from app.schemas.job import JobPostingResponse, JobSearch
from app.api.dependencies import get_current_user

router = APIRouter()

# Service instances
scraper_manager = JobScraperManager()
analysis_service = JobAnalysisService()
trend_analyzer = MarketTrendAnalyzer()
pipeline = DataCollectionPipeline()
job_repository = JobRepository()


# Request/Response Models
class JobScrapingRequest(BaseModel):
    """Request model for job scraping"""
    keywords: str = Field(..., description="Search keywords")
    location: Optional[str] = Field(None, description="Target location")
    platforms: List[str] = Field(["linkedin", "indeed"], description="Platforms to scrape")
    remote: bool = Field(False, description="Include remote jobs")
    experience_level: Optional[str] = Field(None, description="Experience level filter")
    job_type: Optional[str] = Field(None, description="Job type filter")
    posted_days: int = Field(7, ge=1, le=30, description="Days since posted")
    limit: int = Field(100, ge=1, le=500, description="Maximum jobs to scrape")


class TrendAnalysisRequest(BaseModel):
    """Request model for trend analysis"""
    skills: Optional[List[str]] = Field(None, description="Specific skills to analyze")
    days: int = Field(90, ge=7, le=365, description="Analysis period in days")
    location: Optional[str] = Field(None, description="Location filter")
    experience_level: Optional[str] = Field(None, description="Experience level filter")


class SalaryPredictionRequest(BaseModel):
    """Request model for salary prediction"""
    skills: List[str] = Field(..., description="Skills to predict salaries for")
    location: Optional[str] = Field(None, description="Target location")
    experience_level: Optional[str] = Field(None, description="Experience level")
    days: int = Field(365, ge=30, le=730, description="Historical data period")


class MarketReportResponse(BaseModel):
    """Response model for market reports"""
    report_generated: datetime
    analysis_period_days: int
    market_overview: Dict[str, Any]
    skill_trends: List[Dict[str, Any]]
    emerging_skills: List[Dict[str, Any]]
    salary_predictions: List[Dict[str, Any]]
    top_growing_skills: List[Dict[str, Any]]
    top_declining_skills: List[Dict[str, Any]]


@router.post("/scrape-jobs")
async def scrape_jobs(
    request: JobScrapingRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """
    Scrape jobs from specified platforms
    """
    try:
        # Convert request to search parameters
        search_params = JobSearchParams(
            keywords=request.keywords,
            location=request.location,
            remote=request.remote,
            experience_level=request.experience_level,
            job_type=request.job_type,
            posted_days=request.posted_days,
            limit=request.limit
        )
        
        # Run scraping and storage
        stats = await scraper_manager.scrape_and_store_jobs(
            db=db,
            search_params=search_params,
            platforms=request.platforms,
            deduplicate=True
        )
        
        # Schedule background processing if jobs were stored
        total_stored = sum(platform_stats['stored'] for platform_stats in stats.values())
        if total_stored > 0:
            background_tasks.add_task(
                _process_jobs_background,
                total_stored
            )
        
        return {
            "status": "success",
            "message": f"Job scraping completed",
            "stats": stats,
            "total_scraped": sum(platform_stats['scraped'] for platform_stats in stats.values()),
            "total_stored": total_stored
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Job scraping failed: {str(e)}")


@router.post("/analyze-trends")
async def analyze_trends(
    request: TrendAnalysisRequest,
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """
    Analyze job market trends for specified skills
    """
    try:
        # Analyze skill demand trends
        trends = await trend_analyzer.analyze_skill_demand_trends(
            db=db,
            skill_names=request.skills,
            days=request.days
        )
        
        # Get emerging skills
        emerging_skills = await trend_analyzer.detect_emerging_skills(
            db=db,
            days=request.days
        )
        
        return {
            "status": "success",
            "analysis_period_days": request.days,
            "skill_trends": trends,
            "emerging_skills": [
                {
                    "skill_name": skill.skill_name,
                    "growth_rate": skill.growth_rate,
                    "current_demand": skill.current_demand,
                    "trend_score": skill.trend_score,
                    "confidence": skill.confidence,
                    "related_skills": skill.related_skills
                }
                for skill in emerging_skills
            ]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Trend analysis failed: {str(e)}")


@router.post("/predict-salaries")
async def predict_salaries(
    request: SalaryPredictionRequest,
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """
    Predict salaries for specified skills
    """
    try:
        predictions = await trend_analyzer.predict_salaries(
            db=db,
            skill_names=request.skills,
            location=request.location,
            experience_level=request.experience_level,
            days=request.days
        )
        
        return {
            "status": "success",
            "predictions": [
                {
                    "skill_name": pred.skill_name,
                    "location": pred.location,
                    "experience_level": pred.experience_level,
                    "predicted_salary": pred.predicted_salary,
                    "confidence_interval": pred.confidence_interval,
                    "model_accuracy": pred.model_accuracy,
                    "sample_size": pred.sample_size
                }
                for pred in predictions
            ]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Salary prediction failed: {str(e)}")


@router.get("/market-report")
async def get_market_report(
    days: int = Query(90, ge=7, le=365, description="Analysis period in days"),
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_user)
) -> MarketReportResponse:
    """
    Generate comprehensive market analysis report
    """
    try:
        report = await trend_analyzer.generate_market_report(db=db, days=days)
        return MarketReportResponse(**report)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Market report generation failed: {str(e)}")


@router.get("/trending-skills")
async def get_trending_skills(
    days: int = Query(30, ge=7, le=90, description="Analysis period in days"),
    limit: int = Query(20, ge=1, le=100, description="Number of skills to return"),
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """
    Get trending skills from recent job postings
    """
    try:
        trending_skills = await scraper_manager.get_trending_skills(
            db=db,
            days=days,
            limit=limit
        )
        
        return {
            "status": "success",
            "analysis_period_days": days,
            "trending_skills": trending_skills
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get trending skills: {str(e)}")


@router.get("/salary-trends")
async def get_salary_trends(
    skill: Optional[str] = Query(None, description="Specific skill to analyze"),
    location: Optional[str] = Query(None, description="Location filter"),
    days: int = Query(90, ge=30, le=365, description="Analysis period in days"),
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """
    Get salary trends for skills or locations
    """
    try:
        salary_trends = await scraper_manager.get_salary_trends(
            db=db,
            skill=skill,
            location=location,
            days=days
        )
        
        return {
            "status": "success",
            "salary_trends": salary_trends
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get salary trends: {str(e)}")


@router.post("/process-jobs")
async def process_unprocessed_jobs(
    batch_size: int = Query(50, ge=1, le=200, description="Batch size for processing"),
    max_jobs: Optional[int] = Query(None, ge=1, description="Maximum jobs to process"),
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """
    Process unprocessed job postings to extract skills
    """
    try:
        stats = await analysis_service.process_unprocessed_jobs(
            db=db,
            batch_size=batch_size,
            max_jobs=max_jobs
        )
        
        return {
            "status": "success",
            "processing_stats": stats
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Job processing failed: {str(e)}")


@router.get("/skill-recommendations")
async def get_skill_recommendations(
    user_skills: List[str] = Query(..., description="User's current skills"),
    target_role: Optional[str] = Query(None, description="Target job role"),
    location: Optional[str] = Query(None, description="Target location"),
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """
    Get skill recommendations based on market demand
    """
    try:
        recommendations = await analysis_service.generate_skill_recommendations(
            db=db,
            user_skills=user_skills,
            target_role=target_role,
            location=location
        )
        
        return {
            "status": "success",
            "recommendations": recommendations
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Skill recommendation failed: {str(e)}")


@router.get("/jobs/search")
async def search_jobs(
    search: JobSearch = Depends(),
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_user)
) -> List[JobPostingResponse]:
    """
    Search stored job postings
    """
    try:
        jobs = await job_repository.search_jobs(
            db=db,
            title=search.title,
            company=search.company,
            location=search.location,
            remote_type=search.remote_type,
            experience_level=search.experience_level,
            min_salary=search.min_salary,
            max_salary=search.max_salary,
            skip=search.skip,
            limit=search.limit
        )
        
        return [JobPostingResponse.from_orm(job) for job in jobs]
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Job search failed: {str(e)}")


@router.get("/jobs/recent")
async def get_recent_jobs(
    days: int = Query(7, ge=1, le=30, description="Days to look back"),
    limit: int = Query(50, ge=1, le=200, description="Maximum jobs to return"),
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_user)
) -> List[JobPostingResponse]:
    """
    Get recently posted jobs
    """
    try:
        jobs = await job_repository.get_recent_jobs(db=db, days=days, limit=limit)
        return [JobPostingResponse.from_orm(job) for job in jobs]
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get recent jobs: {str(e)}")


# Pipeline Management Endpoints
@router.get("/pipeline/status")
async def get_pipeline_status(current_user = Depends(get_current_user)):
    """
    Get current status of data collection pipelines
    """
    try:
        status = await pipeline.get_pipeline_status()
        return {
            "status": "success",
            "pipeline_status": status
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get pipeline status: {str(e)}")


@router.post("/pipeline/run-manual")
async def run_manual_collection(
    background_tasks: BackgroundTasks,
    current_user = Depends(get_current_user),
    keywords: str = Query(..., description="Search keywords"),
    location: Optional[str] = Query(None, description="Target location"),
    platforms: List[str] = Query(["linkedin", "indeed"], description="Platforms to scrape"),
    limit: int = Query(100, ge=1, le=500, description="Maximum jobs to collect")
):
    """
    Run manual data collection
    """
    try:
        # Run collection in background
        background_tasks.add_task(
            _run_manual_collection_background,
            keywords, location, platforms, limit
        )
        
        return {
            "status": "success",
            "message": "Manual collection started in background",
            "parameters": {
                "keywords": keywords,
                "location": location,
                "platforms": platforms,
                "limit": limit
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start manual collection: {str(e)}")


@router.get("/pipeline/metrics")
async def get_collection_metrics(
    days: int = Query(30, ge=1, le=90, description="Analysis period in days"),
    current_user = Depends(get_current_user)
):
    """
    Get data collection performance metrics
    """
    try:
        metrics = await pipeline.get_collection_metrics(days=days)
        return {
            "status": "success",
            "metrics": metrics
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get collection metrics: {str(e)}")


# Background task functions
async def _process_jobs_background(job_count: int):
    """Background task to process newly scraped jobs"""
    try:
        async with get_db() as db:
            await analysis_service.process_unprocessed_jobs(
                db=db,
                batch_size=50,
                max_jobs=job_count
            )
    except Exception as e:
        logger.error(f"Background job processing failed: {str(e)}")


async def _run_manual_collection_background(
    keywords: str,
    location: Optional[str],
    platforms: List[str],
    limit: int
):
    """Background task for manual collection"""
    try:
        result = await pipeline.run_manual_collection(
            keywords=keywords,
            location=location,
            platforms=platforms,
            limit=limit
        )
        logger.info(f"Manual collection completed: {result}")
    except Exception as e:
        logger.error(f"Manual collection failed: {str(e)}")


# Health check endpoint
@router.get("/health")
async def health_check():
    """
    Health check for job market services
    """
    return {
        "status": "healthy",
        "services": {
            "scraper_manager": "available",
            "analysis_service": "available",
            "trend_analyzer": "available",
            "pipeline": "available"
        },
        "timestamp": datetime.utcnow()
    }