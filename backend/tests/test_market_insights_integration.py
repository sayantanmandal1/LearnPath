"""
Integration tests for market insights functionality
"""
import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, patch

from app.services.market_insights_service import MarketInsightsService


class TestMarketInsightsIntegration:
    """Integration tests for market insights service"""
    
    @pytest.fixture
    def market_service(self):
        """Create market insights service instance"""
        return MarketInsightsService()
    
    @pytest.fixture
    def mock_db_session(self):
        """Mock database session"""
        return AsyncMock()
    
    @pytest.mark.asyncio
    async def test_market_insights_service_basic_functionality(self, market_service, mock_db_session):
        """Test basic market insights service functionality"""
        
        # Mock database responses
        mock_db_session.execute.return_value.scalars.return_value.all.return_value = []
        mock_db_session.execute.return_value.scalar.return_value = 0
        mock_db_session.execute.return_value.fetchall.return_value = []
        
        # Mock trend analyzer methods
        with patch.object(market_service.trend_analyzer, 'analyze_skill_demand_trends') as mock_trends:
            mock_trends.return_value = []
            
            with patch.object(market_service.trend_analyzer, 'detect_emerging_skills') as mock_emerging:
                mock_emerging.return_value = []
                
                # Test the service
                result = await market_service.get_comprehensive_market_insights(
                    db=mock_db_session,
                    role="Software Engineer",
                    skills=["python", "react"],
                    location="San Francisco",
                    experience_level="mid",
                    days=90
                )
                
                # Basic structure validation
                assert result is not None
                assert isinstance(result, dict)
                
                # Check required fields
                required_fields = [
                    'demand_trend', 'salary_growth', 'top_skills', 'competition_level',
                    'market_overview', 'skill_analysis', 'geographic_data', 
                    'industry_trends', 'recommendations', 'analysis_date', 'data_freshness'
                ]
                
                for field in required_fields:
                    assert field in result, f"Missing required field: {field}"
                
                # Check data types
                assert isinstance(result['demand_trend'], str)
                assert isinstance(result['salary_growth'], str)
                assert isinstance(result['top_skills'], list)
                assert isinstance(result['competition_level'], str)
                assert isinstance(result['market_overview'], dict)
                assert isinstance(result['skill_analysis'], list)
                assert isinstance(result['geographic_data'], dict)
                assert isinstance(result['industry_trends'], dict)
                assert isinstance(result['recommendations'], list)
    
    @pytest.mark.asyncio
    async def test_fallback_functionality(self, market_service):
        """Test fallback functionality when service fails"""
        
        # Test fallback insights
        result = market_service._get_fallback_insights(
            role="Software Engineer",
            skills=["python", "react"]
        )
        
        # Validate fallback structure
        assert result is not None
        assert result['demand_trend'] == 'Medium'
        assert result['competition_level'] == 'Medium'
        assert 'python' in result['top_skills']
        assert 'react' in result['top_skills']
        assert result['data_freshness'] == 'Fallback'
    
    def test_cache_key_generation(self, market_service):
        """Test cache key generation"""
        
        cache_key = market_service._generate_cache_key(
            role="Software Engineer",
            skills=["python", "react"],
            location="San Francisco",
            experience_level="mid",
            days=90
        )
        
        assert cache_key is not None
        assert isinstance(cache_key, str)
        assert len(cache_key) > 0
    
    def test_growth_rate_estimation(self, market_service):
        """Test growth rate estimation logic"""
        
        # Test tech role with high-demand skills
        growth_rate = market_service._estimate_growth_rate(
            role="Software Engineer",
            skills=["python", "react", "aws"],
            avg_salary=150000
        )
        
        assert growth_rate > 8.0  # Should be higher than base rate
        assert growth_rate <= 25.0  # Should be capped
        
        # Test non-tech role
        growth_rate = market_service._estimate_growth_rate(
            role="Marketing Manager",
            skills=["marketing", "analytics"],
            avg_salary=80000
        )
        
        assert growth_rate >= 8.0  # Should at least be base rate
    
    @pytest.mark.asyncio
    async def test_error_handling(self, market_service, mock_db_session):
        """Test error handling in market insights service"""
        
        # Mock database error
        mock_db_session.execute.side_effect = Exception("Database connection error")
        
        # Should not raise exception, should return fallback
        result = await market_service.get_comprehensive_market_insights(
            db=mock_db_session,
            role="Software Engineer",
            skills=["python"],
            days=90
        )
        
        # Should return fallback data
        assert result is not None
        assert result['data_freshness'] == 'Fallback'


def test_market_insights_service_import():
    """Test that market insights service can be imported successfully"""
    from app.services.market_insights_service import MarketInsightsService
    
    service = MarketInsightsService()
    assert service is not None
    assert hasattr(service, 'get_comprehensive_market_insights')


def test_market_insights_endpoint_import():
    """Test that market insights endpoints can be imported successfully"""
    from app.api.v1.endpoints.market_insights import router
    
    assert router is not None
    assert hasattr(router, 'routes')
    assert len(router.routes) > 0