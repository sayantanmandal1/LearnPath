# Job Market Data Collection and Analysis System

This system provides comprehensive job market data collection, analysis, and trend prediction capabilities for the AI Career Recommender platform.

## Overview

The job market data collection system consists of several interconnected components:

1. **Job Scrapers** - Collect job postings from multiple platforms
2. **Job Analysis Service** - Extract skills and analyze job requirements
3. **Market Trend Analyzer** - Perform time series analysis and predictions
4. **Data Collection Pipeline** - Orchestrate automated data collection
5. **API Endpoints** - Provide REST API access to all functionality

## Components

### Job Scrapers

#### Base Job Scraper (`base_job_scraper.py`)
- Abstract base class for all job scrapers
- Provides common functionality like rate limiting, error handling, and data parsing
- Defines the `ScrapedJob` data structure for consistent job data representation

#### Platform-Specific Scrapers
- **LinkedIn Jobs Scraper** (`linkedin_jobs_scraper.py`) - Scrapes LinkedIn job postings
- **Indeed Scraper** (`indeed_scraper.py`) - Scrapes Indeed job postings  
- **Glassdoor Scraper** (`glassdoor_scraper.py`) - Scrapes Glassdoor job postings

#### Scraper Manager (`scraper_manager.py`)
- Coordinates multiple scrapers concurrently
- Handles deduplication and data storage
- Provides trending skills analysis and salary trend analysis

### Job Analysis Service (`job_analysis_service.py`)

Processes job postings to extract structured information:

- **Skill Extraction** - Uses NLP to identify required skills from job descriptions
- **Importance Classification** - Determines if skills are required, preferred, or nice-to-have
- **Experience Analysis** - Extracts years of experience requirements
- **Market Analysis** - Analyzes job market trends and skill demand

### Market Trend Analyzer (`market_trend_analyzer.py`)

Provides advanced analytics and predictions:

- **Time Series Analysis** - Analyzes skill demand trends over time
- **Salary Prediction** - Uses machine learning to predict salaries for skills
- **Emerging Skills Detection** - Identifies trending and emerging skills using anomaly detection
- **Market Reports** - Generates comprehensive market analysis reports

### Data Collection Pipeline (`data_collection_pipeline.py`)

Orchestrates automated data collection:

- **Scheduled Collection** - Runs data collection on configurable schedules
- **Pipeline Management** - Manages multiple concurrent collection pipelines
- **Error Handling** - Provides robust error handling and recovery
- **Metrics Tracking** - Tracks collection performance and success rates

## Usage

### Manual Job Collection

```python
from app.services.job_scrapers.scraper_manager import JobScraperManager
from app.services.job_scrapers.base_job_scraper import JobSearchParams

# Initialize scraper manager
scraper_manager = JobScraperManager()

# Define search parameters
search_params = JobSearchParams(
    keywords="python developer",
    location="San Francisco",
    remote=True,
    posted_days=7,
    limit=100
)

# Scrape jobs from multiple platforms
async with get_db() as db:
    stats = await scraper_manager.scrape_and_store_jobs(
        db=db,
        search_params=search_params,
        platforms=["linkedin", "indeed", "glassdoor"]
    )
    print(f"Scraped and stored: {stats}")
```

### Job Analysis and Skill Extraction

```python
from app.services.job_analysis_service import JobAnalysisService

# Initialize analysis service
analysis_service = JobAnalysisService()

# Process unprocessed jobs to extract skills
async with get_db() as db:
    stats = await analysis_service.process_unprocessed_jobs(
        db=db,
        batch_size=50,
        max_jobs=500
    )
    print(f"Processing stats: {stats}")
```

### Market Trend Analysis

```python
from app.services.market_trend_analyzer import MarketTrendAnalyzer

# Initialize trend analyzer
trend_analyzer = MarketTrendAnalyzer()

# Analyze skill demand trends
async with get_db() as db:
    trends = await trend_analyzer.analyze_skill_demand_trends(
        db=db,
        skill_names=["Python", "JavaScript", "React"],
        days=90
    )
    
    # Generate market report
    report = await trend_analyzer.generate_market_report(db=db, days=90)
    print(f"Market report: {report}")
```

### Automated Pipeline

```python
from app.services.data_collection_pipeline import DataCollectionPipeline

# Initialize pipeline
pipeline = DataCollectionPipeline()

# Start scheduled data collection
await pipeline.start_pipeline_scheduler()
```

### CLI Usage

The system includes a CLI script for easy management:

```bash
# Run single collection
python scripts/run_job_pipeline.py collect "python developer" --location "San Francisco" --limit 100

# Run scheduled pipeline
python scripts/run_job_pipeline.py schedule --configs tech_jobs data_science

# Show pipeline status
python scripts/run_job_pipeline.py status

# Show collection metrics
python scripts/run_job_pipeline.py metrics --days 30
```

### API Endpoints

The system provides REST API endpoints:

```bash
# Scrape jobs
POST /api/v1/job-market/scrape-jobs
{
    "keywords": "python developer",
    "location": "San Francisco",
    "platforms": ["linkedin", "indeed"],
    "limit": 100
}

# Analyze trends
POST /api/v1/job-market/analyze-trends
{
    "skills": ["Python", "Django", "FastAPI"],
    "days": 90
}

# Predict salaries
POST /api/v1/job-market/predict-salaries
{
    "skills": ["Python", "Machine Learning"],
    "location": "San Francisco",
    "experience_level": "senior"
}

# Get market report
GET /api/v1/job-market/market-report?days=90

# Get trending skills
GET /api/v1/job-market/trending-skills?days=30&limit=20
```

## Configuration

### Pipeline Configurations

Default pipeline configurations are defined in `DataCollectionPipeline`:

```python
PipelineConfig(
    name="tech_jobs_general",
    search_params=JobSearchParams(
        keywords="software engineer developer programmer",
        location=None,
        remote=True,
        posted_days=1,
        limit=200
    ),
    platforms=["linkedin", "indeed", "glassdoor"],
    schedule_hours=6,  # Run every 6 hours
    max_jobs_per_run=500,
    enable_analysis=True,
    enable_trend_analysis=False
)
```

### Rate Limiting

Each scraper has configurable rate limiting:

- LinkedIn: 2.0 seconds between requests
- Indeed: 1.5 seconds between requests  
- Glassdoor: 2.5 seconds between requests

### Database Models

The system uses several database models:

- `JobPosting` - Stores job posting data
- `JobSkill` - Links jobs to required skills
- `Skill` - Master skill taxonomy
- `Company` - Company information

## Features

### Advanced Analytics

- **Time Series Analysis** - Tracks skill demand over time
- **Seasonality Detection** - Identifies seasonal patterns in job postings
- **Growth Rate Calculation** - Measures skill demand growth rates
- **Volatility Analysis** - Measures market stability

### Machine Learning

- **Salary Prediction** - Uses regression models to predict salaries
- **Skill Classification** - Categorizes skills using NLP
- **Emerging Skills Detection** - Uses anomaly detection to identify trending skills
- **Feature Engineering** - Creates features for ML models from job data

### Data Quality

- **Deduplication** - Removes duplicate job postings across platforms
- **Data Validation** - Validates scraped data quality
- **Error Handling** - Robust error handling and recovery
- **Quality Scoring** - Assigns quality scores to job postings

### Performance Optimization

- **Concurrent Scraping** - Scrapes multiple platforms simultaneously
- **Batch Processing** - Processes jobs in configurable batches
- **Caching** - Caches API responses and ML predictions
- **Background Processing** - Runs analysis tasks in background

## Monitoring and Metrics

The system provides comprehensive monitoring:

- **Pipeline Status** - Real-time status of running pipelines
- **Collection Metrics** - Performance metrics and success rates
- **Error Tracking** - Detailed error logging and reporting
- **Platform Performance** - Per-platform scraping statistics

## Error Handling

- **Graceful Degradation** - Continues operation when individual scrapers fail
- **Retry Mechanisms** - Intelligent retry with exponential backoff
- **Rate Limit Handling** - Respects platform rate limits
- **Data Recovery** - Recovers from partial failures

## Security and Compliance

- **Rate Limiting** - Respects platform terms of service
- **User Agent Rotation** - Uses appropriate user agents
- **Data Privacy** - Handles job posting data responsibly
- **Error Logging** - Secure logging without sensitive data

## Testing

The system includes comprehensive tests:

- Unit tests for individual components
- Integration tests for complete workflows
- Mock data for testing scrapers
- Performance tests for large datasets

Run tests with:
```bash
pytest backend/tests/test_job_market_pipeline.py -v
```

## Dependencies

Key dependencies:
- `httpx` - Async HTTP client for web scraping
- `beautifulsoup4` - HTML parsing
- `scikit-learn` - Machine learning models
- `pandas` - Data manipulation
- `numpy` - Numerical computing
- `sqlalchemy` - Database ORM
- `fastapi` - API framework

## Future Enhancements

Planned improvements:
- Additional job platforms (Stack Overflow Jobs, AngelList)
- Enhanced NLP models for skill extraction
- Real-time job alerts and notifications
- Geographic salary analysis
- Company culture analysis
- Job matching algorithms
- Advanced visualization dashboards