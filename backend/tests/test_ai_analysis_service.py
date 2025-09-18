"""
Tests for AI Analysis Service with Gemini integration
"""
import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime
from typing import Dict, Any

from app.services.ai_analysis_service import (
    AIAnalysisService, GeminiAPIClient, CompleteProfileData,
    SkillAssessment, CareerRecommendation, LearningPath, ProjectSuggestion
)
from app.models.analysis_result import AnalysisType
from app.schemas.profile import ProfileResponse
from app.schemas.resume import ParsedResumeData, ContactInfo, WorkExperience
from app.core.exceptions import ProcessingError


class TestGeminiAPIClient:
    """Test cases for Gemini API client"""
    
    @pytest.fixture
    def gemini_client(self):
        """Create Gemini API client for testing"""
        return GeminiAPIClient()
    
    @pytest.mark.asyncio
    async def test_generate_content_success(self, gemini_client):
        """Test successful content generation"""
        with patch('httpx.AsyncClient') as mock_client:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "candidates": [{
                    "content": {
                        "parts": [{"text": "Generated content"}]
                    }
                }]
            }
            mock_response.headers = {"x-request-id": "test-request-id"}
            
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(return_value=mock_response)
            
            # Mock API key
            with patch.dict('os.environ', {'GEMINI_API_KEY': 'test-key'}):
                client = GeminiAPIClient()
                content, request_id = await client.generate_content("Test prompt")
                
                assert content == "Generated content"
                assert request_id == "test-request-id"
    
    @pytest.mark.asyncio
    async def test_generate_content_no_api_key(self, gemini_client):
        """Test content generation without API key"""
        with patch.dict('os.environ', {}, clear=True):
            client = GeminiAPIClient()
            
            with pytest.raises(ProcessingError, match="Gemini API key not configured"):
                await client.generate_content("Test prompt")
    
    @pytest.mark.asyncio
    async def test_generate_content_api_error(self, gemini_client):
        """Test content generation with API error"""
        with patch('httpx.AsyncClient') as mock_client:
            mock_response = Mock()
            mock_response.status_code = 400
            mock_response.text = "Bad request"
            
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(return_value=mock_response)
            
            with patch.dict('os.environ', {'GEMINI_API_KEY': 'test-key'}):
                client = GeminiAPIClient()
                
                with pytest.raises(ProcessingError, match="Gemini API request failed"):
                    await client.generate_content("Test prompt")
    
    @pytest.mark.asyncio
    async def test_analyze_with_structured_output_success(self, gemini_client):
        """Test structured output analysis"""
        with patch('httpx.AsyncClient') as mock_client:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "candidates": [{
                    "content": {
                        "parts": [{"text": '{"test": "value"}'}]
                    }
                }]
            }
            mock_response.headers = {"x-request-id": "test-request-id"}
            
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(return_value=mock_response)
            
            with patch.dict('os.environ', {'GEMINI_API_KEY': 'test-key'}):
                client = GeminiAPIClient()
                result, request_id = await client.analyze_with_structured_output(
                    "Test prompt", "Test schema"
                )
                
                assert result == {"test": "value"}
                assert request_id == "test-request-id"
    
    @pytest.mark.asyncio
    async def test_analyze_with_structured_output_invalid_json(self, gemini_client):
        """Test structured output with invalid JSON"""
        with patch('httpx.AsyncClient') as mock_client:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "candidates": [{
                    "content": {
                        "parts": [{"text": "Invalid JSON"}]
                    }
                }]
            }
            
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(return_value=mock_response)
            
            with patch.dict('os.environ', {'GEMINI_API_KEY': 'test-key'}):
                client = GeminiAPIClient()
                
                with pytest.raises(ProcessingError, match="Invalid JSON response"):
                    await client.analyze_with_structured_output("Test prompt", "Test schema")


class TestCompleteProfileData:
    """Test cases for CompleteProfileData"""
    
    def test_to_analysis_context(self):
        """Test conversion to analysis context"""
        # Create test profile data
        profile = ProfileResponse(
            id="test-id",
            user_id="user-123",
            current_role="Software Developer",
            experience_years=3,
            location="Bangalore",
            dream_job="Senior Developer",
            industry="Technology",
            desired_role="Full Stack Developer",
            career_goals="Become a tech lead",
            skills={"Python": 0.8, "JavaScript": 0.7},
            platform_data={},
            resume_data={},
            career_interests={},
            skill_gaps={},
            profile_score=0.8,
            completeness_score=0.9,
            data_last_updated=datetime.utcnow(),
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
        resume_data = ParsedResumeData(
            contact_info=ContactInfo(name="John Doe", email="john@example.com"),
            summary="Experienced software developer",
            work_experience=[
                WorkExperience(
                    company="Tech Corp",
                    position="Developer",
                    technologies=["Python", "React"]
                )
            ]
        )
        
        platform_data = {
            "github": {"public_repos": 15, "followers": 50},
            "leetcode": {"problems_solved": 200}
        }
        
        complete_data = CompleteProfileData(
            user_id="user-123",
            basic_profile=profile,
            resume_data=resume_data,
            platform_data=platform_data,
            career_preferences={"desired_role": "Full Stack Developer"}
        )
        
        context = complete_data.to_analysis_context()
        
        assert "Current Role: Software Developer" in context
        assert "Experience: 3 years" in context
        assert "Professional Summary: Experienced software developer" in context
        assert "GitHub: 15 repositories, 50 followers" in context
        assert "LeetCode: 200 problems solved" in context


class TestAIAnalysisService:
    """Test cases for AI Analysis Service"""
    
    @pytest.fixture
    def ai_service(self):
        """Create AI analysis service for testing"""
        return AIAnalysisService()
    
    @pytest.fixture
    def mock_db(self):
        """Create mock database session"""
        return AsyncMock()
    
    @pytest.mark.asyncio
    async def test_analyze_complete_profile_success(self, ai_service, mock_db):
        """Test successful complete profile analysis"""
        # Mock profile data aggregation
        with patch.object(ai_service, '_aggregate_profile_data') as mock_aggregate:
            mock_profile_data = CompleteProfileData(
                user_id="user-123",
                basic_profile=None,
                resume_data=None,
                platform_data={},
                career_preferences={}
            )
            mock_aggregate.return_value = mock_profile_data
            
            # Mock AI analysis generation
            with patch.object(ai_service, '_generate_comprehensive_analysis') as mock_generate:
                mock_analysis = Mock()
                mock_analysis.user_id = "user-123"
                mock_analysis.skill_assessment = SkillAssessment(
                    technical_skills={"Python": 0.8},
                    soft_skills={"Communication": 0.7},
                    skill_strengths=["Programming"],
                    skill_gaps=["Leadership"],
                    improvement_areas=["System Design"],
                    market_relevance_score=0.8,
                    confidence_score=0.9
                )
                mock_analysis.career_recommendations = []
                mock_analysis.learning_paths = []
                mock_analysis.project_suggestions = []
                mock_analysis.market_insights = {}
                mock_analysis.analysis_timestamp = datetime.utcnow()
                mock_analysis.gemini_request_id = "test-request"
                
                mock_generate.return_value = mock_analysis
                
                # Mock storing results
                with patch.object(ai_service, '_store_analysis_results') as mock_store:
                    mock_store.return_value = None
                    
                    result = await ai_service.analyze_complete_profile("user-123", mock_db)
                    
                    assert result.user_id == "user-123"
                    assert result.skill_assessment.technical_skills == {"Python": 0.8}
                    mock_aggregate.assert_called_once_with("user-123", mock_db)
                    mock_generate.assert_called_once()
                    mock_store.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_analyze_complete_profile_fallback(self, ai_service, mock_db):
        """Test complete profile analysis with fallback"""
        # Mock profile data aggregation to raise exception
        with patch.object(ai_service, '_aggregate_profile_data') as mock_aggregate:
            mock_aggregate.side_effect = Exception("API error")
            
            # Mock fallback analysis
            with patch.object(ai_service, '_fallback_analysis') as mock_fallback:
                mock_fallback_result = Mock()
                mock_fallback_result.user_id = "user-123"
                mock_fallback.return_value = mock_fallback_result
                
                result = await ai_service.analyze_complete_profile("user-123", mock_db)
                
                assert result.user_id == "user-123"
                mock_fallback.assert_called_once_with("user-123", mock_db)
    
    @pytest.mark.asyncio
    async def test_generate_skill_assessment_success(self, ai_service, mock_db):
        """Test successful skill assessment generation"""
        # Mock profile data aggregation
        with patch.object(ai_service, '_aggregate_profile_data') as mock_aggregate:
            mock_profile_data = CompleteProfileData(
                user_id="user-123",
                basic_profile=None,
                resume_data=None,
                platform_data={},
                career_preferences={}
            )
            mock_aggregate.return_value = mock_profile_data
            
            # Mock skill assessment generation
            with patch.object(ai_service, '_generate_skill_assessment') as mock_generate:
                mock_assessment = SkillAssessment(
                    technical_skills={"Python": 0.8, "JavaScript": 0.7},
                    soft_skills={"Communication": 0.8},
                    skill_strengths=["Programming", "Problem Solving"],
                    skill_gaps=["System Design", "Leadership"],
                    improvement_areas=["Architecture", "Team Management"],
                    market_relevance_score=0.85,
                    confidence_score=0.9
                )
                mock_generate.return_value = mock_assessment
                
                result = await ai_service.generate_skill_assessment("user-123", mock_db)
                
                assert result.technical_skills == {"Python": 0.8, "JavaScript": 0.7}
                assert result.confidence_score == 0.9
                assert "Programming" in result.skill_strengths
                assert "System Design" in result.skill_gaps
    
    @pytest.mark.asyncio
    async def test_generate_career_recommendations_success(self, ai_service, mock_db):
        """Test successful career recommendations generation"""
        with patch.object(ai_service, '_aggregate_profile_data') as mock_aggregate:
            mock_profile_data = CompleteProfileData(
                user_id="user-123",
                basic_profile=None,
                resume_data=None,
                platform_data={},
                career_preferences={}
            )
            mock_aggregate.return_value = mock_profile_data
            
            with patch.object(ai_service, '_generate_skill_assessment') as mock_skill:
                mock_skill_assessment = SkillAssessment(
                    technical_skills={"Python": 0.8},
                    soft_skills={"Communication": 0.7},
                    skill_strengths=["Programming"],
                    skill_gaps=["Leadership"],
                    improvement_areas=["System Design"],
                    market_relevance_score=0.8,
                    confidence_score=0.9
                )
                mock_skill.return_value = mock_skill_assessment
                
                with patch.object(ai_service, '_generate_career_recommendations') as mock_career:
                    mock_recommendations = [
                        CareerRecommendation(
                            recommended_role="Senior Developer",
                            match_score=0.85,
                            reasoning="Strong technical skills",
                            required_skills=["Python", "System Design"],
                            skill_gaps=["Leadership"],
                            preparation_timeline="6-12 months",
                            salary_range="15-25 LPA",
                            market_demand="High"
                        )
                    ]
                    mock_career.return_value = mock_recommendations
                    
                    result = await ai_service.generate_career_recommendations("user-123", mock_db)
                    
                    assert len(result) == 1
                    assert result[0].recommended_role == "Senior Developer"
                    assert result[0].match_score == 0.85
    
    @pytest.mark.asyncio
    async def test_get_cached_analysis_success(self, ai_service, mock_db):
        """Test successful cached analysis retrieval"""
        # Mock database query
        mock_result = Mock()
        mock_analysis_result = Mock()
        mock_analysis_result.result_data = {"test": "data"}
        mock_result.scalar_one_or_none.return_value = mock_analysis_result
        mock_db.execute.return_value = mock_result
        
        result = await ai_service.get_cached_analysis("user-123", AnalysisType.SKILL_ASSESSMENT, mock_db)
        
        assert result == {"test": "data"}
    
    @pytest.mark.asyncio
    async def test_get_cached_analysis_not_found(self, ai_service, mock_db):
        """Test cached analysis retrieval when not found"""
        # Mock database query returning None
        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db.execute.return_value = mock_result
        
        result = await ai_service.get_cached_analysis("user-123", AnalysisType.SKILL_ASSESSMENT, mock_db)
        
        assert result is None
    
    @pytest.mark.asyncio
    async def test_is_analysis_stale_fresh(self, ai_service, mock_db):
        """Test analysis staleness check for fresh data"""
        # Mock database query
        mock_result = Mock()
        mock_analysis_result = Mock()
        mock_analysis_result.created_at = datetime.utcnow()  # Fresh data
        mock_result.scalar_one_or_none.return_value = mock_analysis_result
        mock_db.execute.return_value = mock_result
        
        is_stale = await ai_service.is_analysis_stale("user-123", AnalysisType.SKILL_ASSESSMENT, mock_db, max_age_hours=24)
        
        assert not is_stale
    
    def test_fallback_skill_assessment(self, ai_service):
        """Test fallback skill assessment"""
        result = ai_service._fallback_skill_assessment()
        
        assert isinstance(result, SkillAssessment)
        assert result.confidence_score == 0.3  # Low confidence for fallback
        assert len(result.technical_skills) > 0
        assert len(result.skill_gaps) > 0
    
    def test_fallback_career_recommendations(self, ai_service):
        """Test fallback career recommendations"""
        result = ai_service._fallback_career_recommendations()
        
        assert isinstance(result, list)
        assert len(result) > 0
        assert isinstance(result[0], CareerRecommendation)
        assert result[0].recommended_role == "Software Developer"
    
    def test_fallback_learning_paths(self, ai_service):
        """Test fallback learning paths"""
        result = ai_service._fallback_learning_paths()
        
        assert isinstance(result, list)
        assert len(result) > 0
        assert isinstance(result[0], LearningPath)
        assert result[0].title == "Full Stack Development Path"
    
    def test_fallback_project_suggestions(self, ai_service):
        """Test fallback project suggestions"""
        result = ai_service._fallback_project_suggestions()
        
        assert isinstance(result, list)
        assert len(result) > 0
        assert isinstance(result[0], ProjectSuggestion)
        assert result[0].title == "Personal Portfolio Website"
    
    def test_fallback_market_insights(self, ai_service):
        """Test fallback market insights"""
        result = ai_service._fallback_market_insights()
        
        assert isinstance(result, dict)
        assert "industry_trends" in result
        assert "in_demand_skills" in result
        assert "salary_trends" in result
        assert len(result["industry_trends"]) > 0


@pytest.mark.integration
class TestAIAnalysisIntegration:
    """Integration tests for AI Analysis Service"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_analysis_flow(self):
        """Test complete end-to-end analysis flow"""
        # This would require actual database and potentially API access
        # For now, we'll mock the major components
        
        service = AIAnalysisService()
        mock_db = AsyncMock()
        
        # Mock all the dependencies
        with patch.object(service, '_aggregate_profile_data') as mock_aggregate, \
             patch.object(service, '_generate_comprehensive_analysis') as mock_generate, \
             patch.object(service, '_store_analysis_results') as mock_store:
            
            # Setup mocks
            mock_profile_data = CompleteProfileData(
                user_id="test-user",
                basic_profile=None,
                resume_data=None,
                platform_data={},
                career_preferences={}
            )
            mock_aggregate.return_value = mock_profile_data
            
            mock_analysis = Mock()
            mock_analysis.user_id = "test-user"
            mock_analysis.skill_assessment = service._fallback_skill_assessment()
            mock_analysis.career_recommendations = service._fallback_career_recommendations()
            mock_analysis.learning_paths = service._fallback_learning_paths()
            mock_analysis.project_suggestions = service._fallback_project_suggestions()
            mock_analysis.market_insights = service._fallback_market_insights()
            mock_analysis.analysis_timestamp = datetime.utcnow()
            mock_analysis.gemini_request_id = None
            
            mock_generate.return_value = mock_analysis
            mock_store.return_value = None
            
            # Execute the analysis
            result = await service.analyze_complete_profile("test-user", mock_db)
            
            # Verify the result
            assert result.user_id == "test-user"
            assert result.skill_assessment is not None
            assert len(result.career_recommendations) > 0
            assert len(result.learning_paths) > 0
            assert len(result.project_suggestions) > 0
            assert result.market_insights is not None
            
            # Verify all methods were called
            mock_aggregate.assert_called_once()
            mock_generate.assert_called_once()
            mock_store.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__])