"""
Tests for market insights service and endpoints
"""
import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, patch

from app.services.market_insights_service import MarketInsightsService
from app.models.job import JobPosting
from app.models.skill import Skill


class TestMarketInsightsService:
    """Test cases for MarketInsightsService"""
    
    @pytest.fixture
    def market_service(self):
        """Create market insights service instance"""
        return MarketInsightsService()
    
    @pytest.fixture
    def mock_db_session(self):
        """Mock database session"""
        return AsyncMock()
    
    @pytest.fixture
    def sample_job_data(self):
        """Sample job posting data"""
        return [
            JobPosting(
                id=1,
                title="Software Engineer",
                company="Tech Corp",
                location="San Francisco, CA",
                salary_min=120000,
                salary_max=180000,
                salary_period="yearly",
                experience_level="mid",
                remote_type="hybrid",
                posted_date=datetime.utcnow() - timedelta(days=5),
                is_active=True,
                processed_skills={"python": 0.9, "react": 0.8, "aws": 0.7}
            ),
            JobPosting(
                id=2,
                title="Senior Software Engineer",
                company="Startup Inc",
                location="New York, NY",
                salary_min=140000,
                salary_max=200000,
                salary_period="yearly",
                experience_level="senior",
                remote_type="remote",
                posted_date=datetime.utcnow() - timedelta(days=3),
                is_active=True,
                processed_skills={"python": 0.95, "kubernetes": 0.8, "golang": 0.7}
            )
        ]
    
    @pytest.mark.asyncio
    async def test_get_comprehensive_market_insights_success(
        self, market_service, mock_db_session, sample_job_data
    ):
        """Test successful comprehensive market insights generation"""
        
        # Mock database queries
        mock_db_session.execute.return_value.scalars.return_value.all.return_value = sample_job_data
        mock_db_session.execute.return_value.scalar.return_value = 1000  # Total jobs
        mock_db_session.execute.return_value.fetchall.return_value = [
            type('Row', (), {
                'salary_min': 120000,
                'salary_max': 180000,
                'salary_period': 'yearly',
                'posted_date': datetime.utcnow()
            })(),
            type('Row', (), {
                'salary_min': 140000,
                'salary_max': 200000,
                'salary_period': 'yearly',
                'posted_date': datetime.utcnow()
            })()
        ]
        
        # Mock trend analyzer
        with patch.object(market_service.trend_analyzer, 'analyze_skill_demand_trends') as mock_trends:
            mock_trends.return_value = [
                {
                    'skill_name': 'python',
                    'growth_rate_weekly': 0.12,
                    'confidence': 0.85,
                    'trend_direction': 'growing',
                    'current_demand': 150,
                    'data_points': 10
                }
            ]
            
            with patch.object(market_service.trend_analyzer, 'detect_emerging_skills') as mock_emerging:
                mock_emerging.return_value = []
                
                with patch.object(market_service.trend_analyzer, 'get_skill_market_data') as mock_skill_data:
                    mock_skill_data.return_value = {
                        'job_count': 150,
                        'growth_trend': 0.12,
                        'avg_salary': 150000,
                        'market_competitiveness': 'medium'
                    }
                    
                    # Test the service
                    result = await market_service.get_comprehensive_market_insights(
                        db=mock_db_session,
                        role="Software Engineer",
                        skills=["python", "react"],
                        location="San Francisco",
                        experience_level="mid",
                        days=90
                    )
                    
                    # Assertions
                    assert result is not None
                    assert 'demand_trend' in result
                    assert 'salary_growth' in result
                    assert 'top_skills' in result
                    assert 'competition_level' in result
                    assert 'market_overview' in result
                    assert 'skill_analysis' in result
                    assert 'geographic_data' in result
                    assert 'industry_trends' in result
                    assert 'recommendations' in result
                    
                    # Check market overview
                    market_overview = result['market_overview']
                    assert 'total_jobs' in market_overview
                    assert 'demand_score' in market_overview
                    assert 'growth_rate' in market_overview
    
    @pytest.mark.asyncio
    async def test_get_comprehensive_market_insights_no_data(
        self, market_service, mock_db_session
    ):
        """Test market insights generation with no data"""
        
        # Mock empty database responses
        mock_db_session.execute.return_value.scalars.return_value.all.return_value = []
        mock_db_session.execute.return_value.scalar.return_value = 0
        mock_db_session.execute.return_value.fetchall.return_value = []
        
        # Mock trend analyzer
        with patch.object(market_service.trend_analyzer, 'analyze_skill_demand_trends') as mock_trends:
            mock_trends.return_value = []
            
            with patch.object(market_service.trend_analyzer, 'detect_emerging_skills') as mock_emerging:
                mock_emerging.return_value = []
                
                # Test the service
                result = await market_service.get_comprehensive_market_insights(
                    db=mock_db_session,
                    role="Rare Job Title",
                    skills=["obscure_skill"],
                    days=90
                )
                
                # Should still return valid structure
                assert result is not None
                assert result['market_overview']['total_jobs'] == 0
                assert result['competition_level'] == 'Very High'  # No jobs = high competition
    
    @pytest.mark.asyncio
    async def test_salary_analysis(self, market_service, mock_db_session):
        """Test salary analysis functionality"""
        
        # Mock salary data
        mock_db_session.execute.return_value.fetchall.return_value = [
            type('Row', (), {
                'salary_min': 100000,
                'salary_max': 150000,
                'salary_period': 'yearly',
                'posted_date': datetime.utcnow()
            })(),
            type('Row', (), {
                'salary_min': 120000,
                'salary_max': 180000,
                'salary_period': 'yearly',
                'posted_date': datetime.utcnow()
            })(),
            type('Row', (), {
                'salary_min': 80,
                'salary_max': 120,
                'salary_period': 'hourly',
                'posted_date': datetime.utcnow()
            })()
        ]
        
        # Test salary analysis
        result = await market_service._analyze_salary_trends(
            db=mock_db_session,
            role="Software Engineer",
            skills=["python"],
            location=None,
            experience_level=None,
            days=90
        )
        
        # Assertions
        assert result is not None
        assert 'avg_salary' in result
        assert 'salary_range' in result
        assert 'growth_trend' in result
        assert result['avg_salary'] > 0
        assert result['salary_range'][0] < result['salary_range'][1]
    
    @pytest.mark.asyncio
    async def test_skill_demand_calculation(self, market_service, mock_db_session):
        """Test skill demand calculation"""
        
        # Mock trend analyzer
        with patch.object(market_service.trend_analyzer, 'get_skill_market_data') as mock_skill_data:
            mock_skill_data.return_value = {
                'job_count': 200,
                'growth_trend': 0.15,
                'avg_salary': 140000,
                'market_competitiveness': 'medium'
            }
            
            # Test skill demand calculation
            result = await market_service._calculate_skill_demand(
                db=mock_db_session,
                skills=["python", "react", "aws"],
                days=90
            )
            
            # Assertions
            assert result is not None
            assert 'top_skills' in result
            assert 'skill_breakdown' in result
            assert len(result['skill_breakdown']) <= 3  # Should match input skills
            
            for skill_info in result['skill_breakdown']:
                assert 'skill_name' in skill_info
                assert 'demand_count' in skill_info
                assert 'growth_rate' in skill_info
    
    @pytest.mark.asyncio
    async def test_competition_level_calculation(self, market_service, mock_db_session):
        """Test competition level calculation"""
        
        # Mock market data for different scenarios
        scenarios = [
            ({'total_jobs': 1500, 'demand_score': 0.15}, 'Low'),
            ({'total_jobs': 800, 'demand_score': 0.08}, 'Medium'),
            ({'total_jobs': 200, 'demand_score': 0.02}, 'High'),
            ({'total_jobs': 50, 'demand_score': 0.005}, 'Very High')
        ]
        
        for market_data, expected_level in scenarios:
            with patch.object(market_service, '_gather_market_data') as mock_gather:
                mock_gather.return_value = market_data
                
                result = await market_service._calculate_competition_level(
                    db=mock_db_session,
                    role="Test Role",
                    skills=["test_skill"],
                    location=None,
                    days=90
                )
                
                assert result == expected_level
    
    def test_cache_functionality(self, market_service):
        """Test caching functionality"""
        
        # Test cache key generation
        cache_key = market_service._generate_cache_key(
            role="Software Engineer",
            skills=["python", "react"],
            location="San Francisco",
            experience_level="mid",
            days=90
        )
        
        assert cache_key is not None
        assert isinstance(cache_key, str)
        assert 'software_engineer' in cache_key
        assert 'python' in cache_key
        assert 'react' in cache_key
    
    def test_fallback_insights(self, market_service):
        """Test fallback insights generation"""
        
        result = market_service._get_fallback_insights(
            role="Software Engineer",
            skills=["python", "react"]
        )
        
        # Assertions
        assert result is not None
        assert result['demand_trend'] == 'Medium'
        assert result['competition_level'] == 'Medium'
        assert 'python' in result['top_skills']
        assert 'react' in result['top_skills']
        assert result['data_freshness'] == 'Fallback'


class TestMarketInsightsEndpoints:
    """Test cases for market insights API endpoints"""
    
    @pytest.mark.asyncio
    async def test_get_simple_market_insights(self, client, mock_current_user):
        """Test simple market insights endpoint"""
        
        with patch('app.api.v1.endpoints.market_insights.market_insights_service') as mock_service:
            mock_service.get_comprehensive_market_insights.return_value = {
                'demand_trend': 'High',
                'salary_growth': '+15% YoY',
                'top_skills': ['Python', 'React', 'AWS'],
                'competition_level': 'Medium'
            }
            
            response = client.get(
                "/api/v1/market-insights/simple",
                params={
                    'role': 'Software Engineer',
                    'skills': 'python,react,aws',
                    'location': 'San Francisco',
                    'days': 90
                }
            )
            
            assert response.status_code == 200
            data = response.json()
            
            assert data['demandTrend'] == 'High'
            assert data['salaryGrowth'] == '+15% YoY'
            assert 'Python' in data['topSkills']
            assert data['competitionLevel'] == 'Medium'
    
    @pytest.mark.asyncio
    async def test_get_comprehensive_market_insights_endpoint(self, client, mock_current_user):
        """Test comprehensive market insights endpoint"""
        
        with patch('app.api.v1.endpoints.market_insights.market_insights_service') as mock_service:
            mock_service.get_comprehensive_market_insights.return_value = {
                'demand_trend': 'High',
                'salary_growth': '+15% YoY',
                'top_skills': ['Python', 'React', 'AWS'],
                'competition_level': 'Medium',
                'market_overview': {
                    'total_jobs': 1500,
                    'avg_salary': 150000,
                    'salary_range': (120000, 200000),
                    'growth_rate': 0.15,
                    'demand_score': 0.12
                },
                'skill_analysis': [],
                'geographic_data': {'top_locations': [], 'remote_opportunities': 500},
                'industry_trends': {'emerging_skills': [], 'declining_skills': [], 'hot_technologies': []},
                'recommendations': [],
                'analysis_date': datetime.utcnow().isoformat(),
                'data_freshness': 'Real-time'
            }
            
            response = client.post(
                "/api/v1/market-insights/comprehensive",
                json={
                    'role': 'Software Engineer',
                    'skills': ['python', 'react', 'aws'],
                    'location': 'San Francisco',
                    'experience_level': 'mid',
                    'days': 90
                }
            )
            
            assert response.status_code == 200
            data = response.json()
            
            assert data['demand_trend'] == 'High'
            assert data['market_overview']['total_jobs'] == 1500
            assert data['geographic_data']['remote_opportunities'] == 500
    
    @pytest.mark.asyncio
    async def test_get_trending_skills_endpoint(self, client, mock_current_user):
        """Test trending skills endpoint"""
        
        with patch('app.api.v1.endpoints.market_insights.market_insights_service') as mock_service:
            mock_service.get_comprehensive_market_insights.return_value = {
                'skill_analysis': [
                    {
                        'skill_name': 'Python',
                        'demand_count': 500,
                        'growth_rate': 0.15,
                        'trend_direction': 'growing',
                        'avg_salary': 150000
                    },
                    {
                        'skill_name': 'React',
                        'demand_count': 400,
                        'growth_rate': 0.12,
                        'trend_direction': 'growing',
                        'avg_salary': 140000
                    }
                ]
            }
            
            response = client.get(
                "/api/v1/market-insights/trending-skills",
                params={'days': 30, 'limit': 10}
            )
            
            assert response.status_code == 200
            data = response.json()
            
            assert 'trending_skills' in data
            assert len(data['trending_skills']) == 2
            assert data['trending_skills'][0]['skill'] == 'Python'
            assert data['analysis_period_days'] == 30
    
    @pytest.mark.asyncio
    async def test_market_insights_error_handling(self, client, mock_current_user):
        """Test error handling in market insights endpoints"""
        
        with patch('app.api.v1.endpoints.market_insights.market_insights_service') as mock_service:
            mock_service.get_comprehensive_market_insights.side_effect = Exception("Database error")
            
            response = client.get("/api/v1/market-insights/simple")
            
            assert response.status_code == 500
            assert "Failed to generate market insights" in response.json()['detail']
    
    def test_health_check_endpoint(self, client):
        """Test market insights health check endpoint"""
        
        response = client.get("/api/v1/market-insights/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data['status'] == 'healthy'
        assert data['service'] == 'market_insights'