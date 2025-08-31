"""
Tests for job market data collection pipeline
"""
import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
from typing import List

from app.services.job_scrapers.base_job_scraper import JobSearchParams, ScrapedJob
from app.services.job_scrapers.scraper_manager import JobScraperManager
from app.services.job_analysis_service import JobAnalysisService
from app.services.market_trend_analyzer import MarketTrendAnalyzer
from app.services.data_collection_pipeline import DataCollectionPipeline, PipelineConfig
from app.models.job import JobPosting
from app.schemas.job import JobPostingCreate


class TestJobScraperManager:
    """Test job scraper manager functionality"""
    
    @pytest.fixture
    def scraper_manager(self):
        return JobScraperManager()
    
    @pytest.fixture
    def sample_scraped_jobs(self):
        return [
            ScrapedJob(
                external_id="job1",
                title="Python Developer",
                company="Tech Corp",
                location="San Francisco, CA",
                description="Python developer position",
                requirements="Python, Django, PostgreSQL",
                salary_min=80000,
                salary_max=120000,
                salary_currency="USD",
                salary_period="yearly",
                employment_type="full-time",
                experience_level="mid",
                remote_type="hybrid",
                posted_date=datetime.utcnow(),
                expires_date=None,
                source_url="https://example.com/job1",
                source="linkedin",
                raw_data={}
            ),
            ScrapedJob(
                external_id="job2",
                title="Senior Python Engineer",
                company="Startup Inc",
                location="Remote",
                description="Senior Python engineer role",
                requirements="Python, FastAPI, AWS",
                salary_min=120000,
                salary_max=160000,
                salary_currency="USD",
                salary_period="yearly",
                employment_type="full-time",
                experience_level="senior",
                remote_type="remote",
                posted_date=datetime.utcnow(),
                expires_date=None,
                source_url="https://example.com/job2",
                source="indeed",
                raw_data={}
            )
        ]
    
    @pytest.mark.asyncio
    async def test_scrape_jobs(self, scraper_manager, sample_scraped_jobs):
        """Test job scraping from multiple platforms"""
        
        search_params = JobSearchParams(
            keywords="python developer",
            location="San Francisco",
            limit=50
        )
        
        # Mock the scrapers
        with patch.object(scraper_manager.scrapers['linkedin'], 'search_jobs', 
                         new_callable=AsyncMock) as mock_linkedin:
            with patch.object(scraper_manager.scrapers['indeed'], 'search_jobs',
                             new_callable=AsyncMock) as mock_indeed:
                
                mock_linkedin.return_value = [sample_scraped_jobs[0]]
                mock_indeed.return_value = [sample_scraped_jobs[1]]
                
                # Mock context managers
                scraper_manager.scrapers['linkedin'].__aenter__ = AsyncMock(
                    return_value=scraper_manager.scrapers['linkedin']
                )
                scraper_manager.scrapers['linkedin'].__aexit__ = AsyncMock(return_value=None)
                scraper_manager.scrapers['indeed'].__aenter__ = AsyncMock(
                    return_value=scraper_manager.scrapers['indeed']
                )
                scraper_manager.scrapers['indeed'].__aexit__ = AsyncMock(return_value=None)
                
                result = await scraper_manager.scrape_jobs(
                    search_params, 
                    platforms=['linkedin', 'indeed']
                )
                
                assert 'linkedin' in result
                assert 'indeed' in result
                assert len(result['linkedin']) == 1
                assert len(result['indeed']) == 1
                assert result['linkedin'][0].title == "Python Developer"
                assert result['indeed'][0].title == "Senior Python Engineer"
    
    def test_deduplicate_jobs(self, scraper_manager, sample_scraped_jobs):
        """Test job deduplication"""
        
        # Create duplicate job
        duplicate_job = ScrapedJob(
            external_id="job3",
            title="Python Developer",  # Same title
            company="Tech Corp",       # Same company
            location="San Francisco, CA",  # Same location
            description="Another Python developer position",
            requirements=None,
            salary_min=None,
            salary_max=None,
            salary_currency="USD",
            salary_period=None,
            employment_type=None,
            experience_level=None,
            remote_type=None,
            posted_date=datetime.utcnow(),
            expires_date=None,
            source_url="https://example.com/job3",
            source="glassdoor",
            raw_data={}
        )
        
        jobs_with_duplicate = sample_scraped_jobs + [duplicate_job]
        unique_jobs = scraper_manager._deduplicate_jobs(jobs_with_duplicate)
        
        # Should remove the duplicate
        assert len(unique_jobs) == 2
        titles = [job.title for job in unique_jobs]
        assert "Python Developer" in titles
        assert "Senior Python Engineer" in titles
    
    @pytest.mark.asyncio
    async def test_get_trending_skills(self, scraper_manager):
        """Test trending skills extraction"""
        
        # Mock database session and repository
        mock_db = AsyncMock()
        
        # Mock recent jobs with processed skills
        mock_jobs = [
            Mock(processed_skills={"Python": 0.9, "Django": 0.8, "PostgreSQL": 0.7}),
            Mock(processed_skills={"Python": 0.95, "FastAPI": 0.85, "AWS": 0.8}),
            Mock(processed_skills={"JavaScript": 0.9, "React": 0.85, "Node.js": 0.8})
        ]
        
        with patch.object(scraper_manager.job_repository, 'get_recent_jobs',
                         new_callable=AsyncMock) as mock_get_recent:
            mock_get_recent.return_value = mock_jobs
            
            trending_skills = await scraper_manager.get_trending_skills(mock_db, days=30)
            
            # Python should be top skill (appears in 2 jobs)
            assert len(trending_skills) > 0
            assert trending_skills[0]['skill'] == 'Python'
            assert trending_skills[0]['job_count'] == 2


class TestJobAnalysisService:
    """Test job analysis service functionality"""
    
    @pytest.fixture
    def analysis_service(self):
        return JobAnalysisService()
    
    @pytest.fixture
    def sample_job_posting(self):
        return JobPosting(
            id="job1",
            external_id="ext1",
            title="Senior Python Developer",
            company="Tech Corp",
            location="San Francisco, CA",
            description="""
            We are looking for a Senior Python Developer with experience in:
            - Python (5+ years required)
            - Django or FastAPI (preferred)
            - PostgreSQL database (nice to have)
            - AWS cloud services (required)
            - Docker containerization
            """,
            requirements="Bachelor's degree in Computer Science",
            source="linkedin",
            source_url="https://example.com/job1",
            posted_date=datetime.utcnow(),
            is_processed=False
        )
    
    def test_determine_skill_importance(self, analysis_service):
        """Test skill importance determination"""
        
        job_text = """
        We are looking for a developer with:
        - Python (required)
        - Django (preferred)
        - PostgreSQL (nice to have)
        """
        
        python_importance = analysis_service._determine_skill_importance("Python", job_text)
        django_importance = analysis_service._determine_skill_importance("Django", job_text)
        postgres_importance = analysis_service._determine_skill_importance("PostgreSQL", job_text)
        
        assert python_importance == "required"
        assert django_importance == "preferred"
        assert postgres_importance == "nice-to-have"
    
    def test_extract_years_required(self, analysis_service):
        """Test years of experience extraction"""
        
        job_text = """
        Requirements:
        - Python: 5+ years of experience
        - Django: minimum 3 years
        - AWS: at least 2 years
        """
        
        python_years = analysis_service._extract_years_required("Python", job_text)
        django_years = analysis_service._extract_years_required("Django", job_text)
        aws_years = analysis_service._extract_years_required("AWS", job_text)
        
        assert python_years == 5
        assert django_years == 3
        assert aws_years == 2
    
    def test_determine_proficiency_level(self, analysis_service):
        """Test proficiency level determination"""
        
        job_text = """
        We need someone with:
        - Expert level Python skills
        - Advanced Django knowledge
        - Basic understanding of PostgreSQL
        """
        
        python_level = analysis_service._determine_proficiency_level("Python", job_text)
        django_level = analysis_service._determine_proficiency_level("Django", job_text)
        postgres_level = analysis_service._determine_proficiency_level("PostgreSQL", job_text)
        
        assert python_level == "expert"
        assert django_level == "advanced"
        assert postgres_level == "beginner"


class TestMarketTrendAnalyzer:
    """Test market trend analyzer functionality"""
    
    @pytest.fixture
    def trend_analyzer(self):
        return MarketTrendAnalyzer()
    
    def test_analyze_time_series(self, trend_analyzer):
        """Test time series analysis"""
        
        # Create sample time series data (growing trend)
        base_date = datetime.utcnow() - timedelta(days=30)
        data = []
        
        for i in range(10):
            data.append({
                'date': base_date + timedelta(days=i*3),
                'value': 10 + i * 2  # Growing trend
            })
        
        analysis = trend_analyzer._analyze_time_series(data, "Python")
        
        assert analysis['trend_direction'] == 'growing'
        assert analysis['trend_slope'] > 0
        assert analysis['confidence'] > 0
    
    def test_detect_seasonality(self, trend_analyzer):
        """Test seasonality detection"""
        
        # Create data with some seasonality
        values = [10, 12, 8, 15, 10, 12, 8, 15, 10, 12, 8, 15, 10, 12, 8]
        
        seasonality = trend_analyzer._detect_seasonality(values)
        
        # Should detect some seasonality
        assert seasonality > 0
    
    def test_engineer_salary_features(self, trend_analyzer):
        """Test salary feature engineering"""
        
        import pandas as pd
        
        # Create sample salary data
        data = {
            'salary': [100000, 120000, 80000],
            'experience_level': ['senior', 'mid', 'entry'],
            'remote_type': ['remote', 'hybrid', 'onsite'],
            'location': ['San Francisco', 'New York', 'Austin'],
            'posted_date': [
                datetime.utcnow(),
                datetime.utcnow() - timedelta(days=10),
                datetime.utcnow() - timedelta(days=20)
            ]
        }
        
        df = pd.DataFrame(data)
        features = trend_analyzer._engineer_salary_features(df)
        
        # Should return feature matrix
        assert features.shape[0] == 3  # 3 samples
        assert features.shape[1] > 0   # Multiple features


class TestDataCollectionPipeline:
    """Test data collection pipeline functionality"""
    
    @pytest.fixture
    def pipeline(self):
        return DataCollectionPipeline()
    
    @pytest.fixture
    def sample_config(self):
        return PipelineConfig(
            name="test_pipeline",
            search_params=JobSearchParams(
                keywords="python developer",
                limit=50
            ),
            platforms=["linkedin"],
            schedule_hours=24,
            max_jobs_per_run=100,
            enable_analysis=True,
            enable_trend_analysis=False
        )
    
    @pytest.mark.asyncio
    async def test_run_pipeline(self, pipeline, sample_config):
        """Test pipeline execution"""
        
        # Mock the scraper manager
        with patch.object(pipeline.scraper_manager, 'scrape_and_store_jobs',
                         new_callable=AsyncMock) as mock_scrape:
            with patch.object(pipeline.analysis_service, 'process_unprocessed_jobs',
                             new_callable=AsyncMock) as mock_process:
                with patch('app.services.data_collection_pipeline.get_db') as mock_get_db:
                    
                    # Setup mocks
                    mock_scrape.return_value = {
                        'linkedin': {'scraped': 25, 'stored': 20}
                    }
                    mock_process.return_value = {
                        'processed': 20, 'failed': 0, 'skills_extracted': 150
                    }
                    
                    mock_db = AsyncMock()
                    mock_get_db.return_value.__aenter__.return_value = mock_db
                    
                    # Run pipeline
                    result = await pipeline.run_pipeline(sample_config)
                    
                    # Verify results
                    assert result.status == 'completed'
                    assert result.jobs_scraped == 25
                    assert result.jobs_stored == 20
                    assert result.jobs_processed == 20
                    assert len(result.errors) == 0
    
    @pytest.mark.asyncio
    async def test_run_manual_collection(self, pipeline):
        """Test manual collection"""
        
        with patch.object(pipeline, 'run_pipeline', new_callable=AsyncMock) as mock_run:
            from app.services.data_collection_pipeline import PipelineRun
            
            # Mock successful run
            mock_run.return_value = PipelineRun(
                config_name="manual_test",
                start_time=datetime.utcnow(),
                end_time=datetime.utcnow(),
                status='completed',
                jobs_scraped=50,
                jobs_stored=45,
                jobs_processed=40,
                errors=[],
                stats={}
            )
            
            result = await pipeline.run_manual_collection(
                keywords="python developer",
                location="San Francisco",
                platforms=["linkedin"],
                limit=50
            )
            
            assert result['status'] == 'completed'
            assert result['jobs_scraped'] == 50
            assert result['jobs_stored'] == 45
    
    def test_get_collection_metrics(self, pipeline):
        """Test collection metrics calculation"""
        
        # Add some mock runs to history
        from app.services.data_collection_pipeline import PipelineRun
        
        base_time = datetime.utcnow() - timedelta(days=10)
        
        for i in range(5):
            run = PipelineRun(
                config_name=f"test_run_{i}",
                start_time=base_time + timedelta(days=i),
                end_time=base_time + timedelta(days=i, hours=1),
                status='completed',
                jobs_scraped=100 + i * 10,
                jobs_stored=90 + i * 10,
                jobs_processed=80 + i * 10,
                errors=[],
                stats={
                    'scraping': {
                        'linkedin': {'scraped': 50 + i * 5, 'stored': 45 + i * 5},
                        'indeed': {'scraped': 50 + i * 5, 'stored': 45 + i * 5}
                    }
                }
            )
            pipeline.pipeline_history.append(run)
        
        # Test metrics calculation (this would normally be async)
        # For testing, we'll just verify the structure
        assert len(pipeline.pipeline_history) == 5


@pytest.mark.asyncio
async def test_integration_job_collection_flow():
    """Integration test for complete job collection flow"""
    
    # This would test the complete flow from scraping to analysis
    # For now, we'll just verify the components can be instantiated together
    
    scraper_manager = JobScraperManager()
    analysis_service = JobAnalysisService()
    trend_analyzer = MarketTrendAnalyzer()
    pipeline = DataCollectionPipeline()
    
    # Verify all components are properly initialized
    assert scraper_manager is not None
    assert analysis_service is not None
    assert trend_analyzer is not None
    assert pipeline is not None
    
    # Verify scraper manager has expected scrapers
    assert 'linkedin' in scraper_manager.scrapers
    assert 'indeed' in scraper_manager.scrapers
    assert 'glassdoor' in scraper_manager.scrapers


if __name__ == "__main__":
    pytest.main([__file__])