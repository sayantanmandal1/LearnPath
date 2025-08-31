"""
Tests for analytics API endpoints
"""
import pytest
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch
from fastapi.testclient import TestClient
from fastapi import status

from app.main import app
from app.schemas.analytics import (
    AnalyticsRequest, CareerAnalysisReport, SkillRadarChart,
    CareerRoadmapVisualization, SkillGapReport, JobCompatibilityReport,
    HistoricalProgressReport, ChartConfiguration, ChartType,
    VisualizationResponse, PDFReportRequest, PDFReportResponse
)
from app.models.user import User
from app.core.exceptions import AnalyticsError, VisualizationError, PDFGenerationError


@pytest.fixture
def client():
    """Test client"""
    return TestClient(app)


@pytest.fixture
def mock_user():
    """Mock authenticated user"""
    return User(
        id="test-user-123",
        email="test@example.com",
        is_active=True
    )


@pytest.fixture
def auth_headers():
    """Mock authentication headers"""
    return {"Authorization": "Bearer test-token"}


@pytest.fixture
def sample_analytics_request():
    """Sample analytics request"""
    return {
        "user_id": "test-user-123",
        "analysis_types": ["full_report"],
        "target_role": "Senior Developer",
        "include_job_matches": True,
        "include_progress_tracking": True,
        "tracking_period_days": 90
    }


@pytest.fixture
def sample_chart_config():
    """Sample chart configuration"""
    return {
        "chart_type": "radar",
        "title": "Skill Analysis",
        "width": 800,
        "height": 600,
        "color_scheme": "professional",
        "interactive": True,
        "export_format": "svg"
    }


class TestAnalyticsEndpoints:
    """Test cases for analytics API endpoints"""
    
    @patch('app.api.v1.endpoints.analytics.get_current_user')
    @patch('app.api.v1.endpoints.analytics.AnalyticsService')
    def test_generate_comprehensive_analytics_report_success(
        self, mock_analytics_service, mock_get_user, client, mock_user, sample_analytics_request
    ):
        """Test successful comprehensive analytics report generation"""
        # Mock authentication
        mock_get_user.return_value = mock_user
        
        # Mock analytics service
        mock_service_instance = Mock()
        mock_analytics_service.return_value = mock_service_instance
        
        mock_report = Mock(spec=CareerAnalysisReport)
        mock_report.user_id = "test-user-123"
        mock_report.recommendations = ["Focus on Python"]
        mock_report.next_steps = ["Take course"]
        
        mock_service_instance.generate_comprehensive_report = AsyncMock(return_value=mock_report)
        
        # Make request
        response = client.post(
            "/api/v1/analytics/generate-report",
            json=sample_analytics_request,
            headers={"Authorization": "Bearer test-token"}
        )
        
        assert response.status_code == status.HTTP_200_OK
        mock_service_instance.generate_comprehensive_report.assert_called_once()
    
    @patch('app.api.v1.endpoints.analytics.get_current_user')
    def test_generate_report_forbidden_different_user(
        self, mock_get_user, client, mock_user, sample_analytics_request
    ):
        """Test forbidden access when requesting different user's analytics"""
        mock_get_user.return_value = mock_user
        
        # Request analytics for different user
        sample_analytics_request["user_id"] = "different-user-456"
        
        response = client.post(
            "/api/v1/analytics/generate-report",
            json=sample_analytics_request,
            headers={"Authorization": "Bearer test-token"}
        )
        
        assert response.status_code == status.HTTP_403_FORBIDDEN
        assert "Can only request analytics for your own profile" in response.json()["detail"]
    
    @patch('app.api.v1.endpoints.analytics.get_current_user')
    @patch('app.api.v1.endpoints.analytics.AnalyticsService')
    def test_generate_report_analytics_error(
        self, mock_analytics_service, mock_get_user, client, mock_user, sample_analytics_request
    ):
        """Test analytics error handling"""
        mock_get_user.return_value = mock_user
        
        # Mock analytics service error
        mock_service_instance = Mock()
        mock_analytics_service.return_value = mock_service_instance
        mock_service_instance.generate_comprehensive_report = AsyncMock(
            side_effect=AnalyticsError("Analytics processing failed")
        )
        
        response = client.post(
            "/api/v1/analytics/generate-report",
            json=sample_analytics_request,
            headers={"Authorization": "Bearer test-token"}
        )
        
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "Analytics processing failed" in response.json()["detail"]
    
    @patch('app.api.v1.endpoints.analytics.get_current_user')
    @patch('app.api.v1.endpoints.analytics.AnalyticsService')
    def test_get_skill_radar_chart_success(
        self, mock_analytics_service, mock_get_user, client, mock_user
    ):
        """Test successful skill radar chart generation"""
        mock_get_user.return_value = mock_user
        
        # Mock analytics service
        mock_service_instance = Mock()
        mock_analytics_service.return_value = mock_service_instance
        
        mock_radar_chart = Mock(spec=SkillRadarChart)
        mock_radar_chart.user_id = "test-user-123"
        mock_radar_chart.categories = ["Programming", "Frameworks"]
        mock_radar_chart.user_scores = [85.0, 70.0]
        
        mock_service_instance.generate_skill_radar_chart = AsyncMock(return_value=mock_radar_chart)
        
        response = client.get(
            "/api/v1/analytics/skill-radar?target_role=Senior Developer",
            headers={"Authorization": "Bearer test-token"}
        )
        
        assert response.status_code == status.HTTP_200_OK
        mock_service_instance.generate_skill_radar_chart.assert_called_once_with(
            "test-user-123", "Senior Developer"
        )
    
    @patch('app.api.v1.endpoints.analytics.get_current_user')
    @patch('app.api.v1.endpoints.analytics.AnalyticsService')
    def test_get_career_roadmap_success(
        self, mock_analytics_service, mock_get_user, client, mock_user
    ):
        """Test successful career roadmap generation"""
        mock_get_user.return_value = mock_user
        
        # Mock analytics service
        mock_service_instance = Mock()
        mock_analytics_service.return_value = mock_service_instance
        
        mock_roadmap = Mock(spec=CareerRoadmapVisualization)
        mock_roadmap.user_id = "test-user-123"
        mock_roadmap.nodes = []
        mock_roadmap.edges = []
        
        mock_service_instance.generate_career_roadmap = AsyncMock(return_value=mock_roadmap)
        
        response = client.get(
            "/api/v1/analytics/career-roadmap?target_role=Senior Developer",
            headers={"Authorization": "Bearer test-token"}
        )
        
        assert response.status_code == status.HTTP_200_OK
        mock_service_instance.generate_career_roadmap.assert_called_once_with(
            "test-user-123", "Senior Developer"
        )
    
    @patch('app.api.v1.endpoints.analytics.get_current_user')
    @patch('app.api.v1.endpoints.analytics.AnalyticsService')
    def test_analyze_skill_gaps_success(
        self, mock_analytics_service, mock_get_user, client, mock_user
    ):
        """Test successful skill gap analysis"""
        mock_get_user.return_value = mock_user
        
        # Mock analytics service
        mock_service_instance = Mock()
        mock_analytics_service.return_value = mock_service_instance
        
        mock_gap_report = Mock(spec=SkillGapReport)
        mock_gap_report.user_id = "test-user-123"
        mock_gap_report.target_role = "Senior Developer"
        mock_gap_report.overall_match_score = 75.0
        
        mock_service_instance.analyze_skill_gaps = AsyncMock(return_value=mock_gap_report)
        
        response = client.get(
            "/api/v1/analytics/skill-gaps?target_role=Senior Developer",
            headers={"Authorization": "Bearer test-token"}
        )
        
        assert response.status_code == status.HTTP_200_OK
        mock_service_instance.analyze_skill_gaps.assert_called_once_with(
            "test-user-123", "Senior Developer"
        )
    
    @patch('app.api.v1.endpoints.analytics.get_current_user')
    @patch('app.api.v1.endpoints.analytics.AnalyticsService')
    def test_get_job_compatibility_scores_success(
        self, mock_analytics_service, mock_get_user, client, mock_user
    ):
        """Test successful job compatibility scoring"""
        mock_get_user.return_value = mock_user
        
        # Mock analytics service
        mock_service_instance = Mock()
        mock_analytics_service.return_value = mock_service_instance
        
        mock_compatibility_report = Mock(spec=JobCompatibilityReport)
        mock_compatibility_report.user_id = "test-user-123"
        mock_compatibility_report.job_matches = []
        mock_compatibility_report.total_jobs_analyzed = 10
        
        mock_service_instance.generate_job_compatibility_scores = AsyncMock(
            return_value=mock_compatibility_report
        )
        
        response = client.get(
            "/api/v1/analytics/job-compatibility?location=San Francisco&limit=20",
            headers={"Authorization": "Bearer test-token"}
        )
        
        assert response.status_code == status.HTTP_200_OK
        
        # Verify filters were passed correctly
        call_args = mock_service_instance.generate_job_compatibility_scores.call_args
        assert call_args[0][0] == "test-user-123"  # user_id
        assert call_args[0][1]["location"] == "San Francisco"  # filters
        assert call_args[0][2] == 20  # limit
    
    @patch('app.api.v1.endpoints.analytics.get_current_user')
    @patch('app.api.v1.endpoints.analytics.AnalyticsService')
    def test_get_progress_tracking_success(
        self, mock_analytics_service, mock_get_user, client, mock_user
    ):
        """Test successful progress tracking"""
        mock_get_user.return_value = mock_user
        
        # Mock analytics service
        mock_service_instance = Mock()
        mock_analytics_service.return_value = mock_service_instance
        
        mock_progress_report = Mock(spec=HistoricalProgressReport)
        mock_progress_report.user_id = "test-user-123"
        mock_progress_report.tracking_period_days = 90
        mock_progress_report.overall_improvement_score = 8.5
        
        mock_service_instance.track_historical_progress = AsyncMock(
            return_value=mock_progress_report
        )
        
        response = client.get(
            "/api/v1/analytics/progress-tracking?tracking_period_days=90",
            headers={"Authorization": "Bearer test-token"}
        )
        
        assert response.status_code == status.HTTP_200_OK
        mock_service_instance.track_historical_progress.assert_called_once_with(
            "test-user-123", 90
        )
    
    @patch('app.api.v1.endpoints.analytics.get_current_user')
    @patch('app.api.v1.endpoints.analytics.AnalyticsService')
    @patch('app.api.v1.endpoints.analytics.VisualizationService')
    def test_create_skill_radar_visualization_success(
        self, mock_viz_service, mock_analytics_service, mock_get_user, 
        client, mock_user, sample_chart_config
    ):
        """Test successful skill radar visualization creation"""
        mock_get_user.return_value = mock_user
        
        # Mock analytics service
        mock_analytics_instance = Mock()
        mock_analytics_service.return_value = mock_analytics_instance
        mock_radar_chart = Mock(spec=SkillRadarChart)
        mock_analytics_instance.generate_skill_radar_chart = AsyncMock(return_value=mock_radar_chart)
        
        # Mock visualization service
        mock_viz_instance = Mock()
        mock_viz_service.return_value = mock_viz_instance
        mock_chart_data = {"type": "radar", "data": {}, "options": {}}
        mock_viz_instance.generate_skill_radar_chart_data = Mock(return_value=mock_chart_data)
        
        mock_viz_response = Mock(spec=VisualizationResponse)
        mock_viz_response.chart_id = "chart-123"
        mock_viz_response.chart_type = ChartType.RADAR
        mock_viz_instance.create_visualization_response = Mock(return_value=mock_viz_response)
        
        response = client.post(
            "/api/v1/analytics/visualizations/skill-radar?target_role=Senior Developer",
            json=sample_chart_config,
            headers={"Authorization": "Bearer test-token"}
        )
        
        assert response.status_code == status.HTTP_200_OK
        mock_analytics_instance.generate_skill_radar_chart.assert_called_once()
        mock_viz_instance.generate_skill_radar_chart_data.assert_called_once()
        mock_viz_instance.create_visualization_response.assert_called_once()
    
    @patch('app.api.v1.endpoints.analytics.get_current_user')
    @patch('app.api.v1.endpoints.analytics.AnalyticsService')
    @patch('app.api.v1.endpoints.analytics.VisualizationService')
    def test_create_career_roadmap_visualization_success(
        self, mock_viz_service, mock_analytics_service, mock_get_user,
        client, mock_user, sample_chart_config
    ):
        """Test successful career roadmap visualization creation"""
        mock_get_user.return_value = mock_user
        
        # Mock analytics service
        mock_analytics_instance = Mock()
        mock_analytics_service.return_value = mock_analytics_instance
        mock_roadmap = Mock(spec=CareerRoadmapVisualization)
        mock_analytics_instance.generate_career_roadmap = AsyncMock(return_value=mock_roadmap)
        
        # Mock visualization service
        mock_viz_instance = Mock()
        mock_viz_service.return_value = mock_viz_instance
        mock_chart_data = {"nodes": [], "edges": [], "options": {}}
        mock_viz_instance.generate_career_roadmap_data = Mock(return_value=mock_chart_data)
        
        mock_viz_response = Mock(spec=VisualizationResponse)
        mock_viz_instance.create_visualization_response = Mock(return_value=mock_viz_response)
        
        response = client.post(
            "/api/v1/analytics/visualizations/career-roadmap?target_role=Senior Developer",
            json=sample_chart_config,
            headers={"Authorization": "Bearer test-token"}
        )
        
        assert response.status_code == status.HTTP_200_OK
        mock_analytics_instance.generate_career_roadmap.assert_called_once()
        mock_viz_instance.generate_career_roadmap_data.assert_called_once()
    
    @patch('app.api.v1.endpoints.analytics.get_current_user')
    @patch('app.api.v1.endpoints.analytics.AnalyticsService')
    @patch('app.api.v1.endpoints.analytics.VisualizationService')
    def test_create_skill_gap_visualization_success(
        self, mock_viz_service, mock_analytics_service, mock_get_user,
        client, mock_user, sample_chart_config
    ):
        """Test successful skill gap visualization creation"""
        mock_get_user.return_value = mock_user
        
        # Mock analytics service
        mock_analytics_instance = Mock()
        mock_analytics_service.return_value = mock_analytics_instance
        mock_gap_report = Mock(spec=SkillGapReport)
        mock_analytics_instance.analyze_skill_gaps = AsyncMock(return_value=mock_gap_report)
        
        # Mock visualization service
        mock_viz_instance = Mock()
        mock_viz_service.return_value = mock_viz_instance
        mock_chart_data = {"type": "bar", "data": {}, "options": {}}
        mock_viz_instance.generate_skill_gap_chart_data = Mock(return_value=mock_chart_data)
        
        mock_viz_response = Mock(spec=VisualizationResponse)
        mock_viz_instance.create_visualization_response = Mock(return_value=mock_viz_response)
        
        response = client.post(
            "/api/v1/analytics/visualizations/skill-gaps?target_role=Senior Developer",
            json=sample_chart_config,
            headers={"Authorization": "Bearer test-token"}
        )
        
        assert response.status_code == status.HTTP_200_OK
        mock_analytics_instance.analyze_skill_gaps.assert_called_once()
        mock_viz_instance.generate_skill_gap_chart_data.assert_called_once()
    
    @patch('app.api.v1.endpoints.analytics.get_current_user')
    @patch('app.api.v1.endpoints.analytics.AnalyticsService')
    @patch('app.api.v1.endpoints.analytics.PDFReportService')
    def test_generate_pdf_report_success(
        self, mock_pdf_service, mock_analytics_service, mock_get_user,
        client, mock_user
    ):
        """Test successful PDF report generation"""
        mock_get_user.return_value = mock_user
        
        # Mock analytics service
        mock_analytics_instance = Mock()
        mock_analytics_service.return_value = mock_analytics_instance
        mock_report_data = Mock(spec=CareerAnalysisReport)
        mock_analytics_instance.generate_comprehensive_report = AsyncMock(
            return_value=mock_report_data
        )
        
        # Mock PDF service
        mock_pdf_instance = Mock()
        mock_pdf_service.return_value = mock_pdf_instance
        mock_pdf_response = Mock(spec=PDFReportResponse)
        mock_pdf_response.report_id = "report-123"
        mock_pdf_response.file_url = "/reports/report.pdf"
        mock_pdf_instance.generate_comprehensive_report = AsyncMock(
            return_value=mock_pdf_response
        )
        
        pdf_request = {
            "user_id": "test-user-123",
            "report_type": "comprehensive",
            "include_charts": True,
            "include_recommendations": True
        }
        
        response = client.post(
            "/api/v1/analytics/reports/pdf",
            json=pdf_request,
            headers={"Authorization": "Bearer test-token"}
        )
        
        assert response.status_code == status.HTTP_200_OK
        mock_analytics_instance.generate_comprehensive_report.assert_called_once()
        mock_pdf_instance.generate_comprehensive_report.assert_called_once()
    
    @patch('app.api.v1.endpoints.analytics.get_current_user')
    def test_generate_pdf_report_forbidden_different_user(
        self, mock_get_user, client, mock_user
    ):
        """Test forbidden access when requesting PDF for different user"""
        mock_get_user.return_value = mock_user
        
        pdf_request = {
            "user_id": "different-user-456",
            "report_type": "comprehensive"
        }
        
        response = client.post(
            "/api/v1/analytics/reports/pdf",
            json=pdf_request,
            headers={"Authorization": "Bearer test-token"}
        )
        
        assert response.status_code == status.HTTP_403_FORBIDDEN
        assert "Can only request reports for your own profile" in response.json()["detail"]
    
    @patch('app.api.v1.endpoints.analytics.get_current_user')
    @patch('app.api.v1.endpoints.analytics.PDFReportService')
    def test_download_pdf_report_success(
        self, mock_pdf_service, mock_get_user, client, mock_user
    ):
        """Test successful PDF report download"""
        mock_get_user.return_value = mock_user
        
        # Mock PDF service
        mock_pdf_instance = Mock()
        mock_pdf_service.return_value = mock_pdf_instance
        
        # Mock file path
        from pathlib import Path
        import tempfile
        
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_file:
            temp_file.write(b"PDF content")
            temp_path = Path(temp_file.name)
        
        mock_pdf_instance.get_report_file = AsyncMock(return_value=temp_path)
        
        with patch('app.api.v1.endpoints.analytics.FileResponse') as mock_file_response:
            mock_file_response.return_value = Mock()
            
            response = client.get(
                "/api/v1/analytics/reports/report-123/download",
                headers={"Authorization": "Bearer test-token"}
            )
            
            mock_pdf_instance.get_report_file.assert_called_once_with("report-123")
            mock_file_response.assert_called_once()
        
        # Clean up
        temp_path.unlink()
    
    @patch('app.api.v1.endpoints.analytics.get_current_user')
    @patch('app.api.v1.endpoints.analytics.PDFReportService')
    def test_download_pdf_report_not_found(
        self, mock_pdf_service, mock_get_user, client, mock_user
    ):
        """Test PDF report download when file not found"""
        mock_get_user.return_value = mock_user
        
        # Mock PDF service
        mock_pdf_instance = Mock()
        mock_pdf_service.return_value = mock_pdf_instance
        mock_pdf_instance.get_report_file = AsyncMock(return_value=None)
        
        response = client.get(
            "/api/v1/analytics/reports/non-existent-report/download",
            headers={"Authorization": "Bearer test-token"}
        )
        
        assert response.status_code == status.HTTP_404_NOT_FOUND
        assert "Report not found or expired" in response.json()["detail"]
    
    def test_invalid_request_data(self, client):
        """Test endpoints with invalid request data"""
        # Test with invalid analytics request
        invalid_request = {
            "user_id": "test-user-123",
            "analysis_types": ["invalid_type"],  # Invalid analysis type
            "tracking_period_days": -1  # Invalid period
        }
        
        response = client.post(
            "/api/v1/analytics/generate-report",
            json=invalid_request,
            headers={"Authorization": "Bearer test-token"}
        )
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_missing_required_parameters(self, client):
        """Test endpoints with missing required parameters"""
        # Test career roadmap without target_role
        response = client.get(
            "/api/v1/analytics/career-roadmap",
            headers={"Authorization": "Bearer test-token"}
        )
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_invalid_query_parameters(self, client):
        """Test endpoints with invalid query parameters"""
        # Test progress tracking with invalid period
        response = client.get(
            "/api/v1/analytics/progress-tracking?tracking_period_days=500",  # Too large
            headers={"Authorization": "Bearer test-token"}
        )
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


class TestAnalyticsEndpointsErrorHandling:
    """Test error handling in analytics endpoints"""
    
    @patch('app.api.v1.endpoints.analytics.get_current_user')
    @patch('app.api.v1.endpoints.analytics.AnalyticsService')
    def test_visualization_error_handling(
        self, mock_analytics_service, mock_get_user, client, mock_user, sample_chart_config
    ):
        """Test visualization error handling"""
        mock_get_user.return_value = mock_user
        
        # Mock analytics service
        mock_analytics_instance = Mock()
        mock_analytics_service.return_value = mock_analytics_instance
        mock_analytics_instance.generate_skill_radar_chart = AsyncMock(
            side_effect=VisualizationError("Visualization failed")
        )
        
        response = client.post(
            "/api/v1/analytics/visualizations/skill-radar",
            json=sample_chart_config,
            headers={"Authorization": "Bearer test-token"}
        )
        
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "Visualization failed" in response.json()["detail"]
    
    @patch('app.api.v1.endpoints.analytics.get_current_user')
    @patch('app.api.v1.endpoints.analytics.AnalyticsService')
    @patch('app.api.v1.endpoints.analytics.PDFReportService')
    def test_pdf_generation_error_handling(
        self, mock_pdf_service, mock_analytics_service, mock_get_user, client, mock_user
    ):
        """Test PDF generation error handling"""
        mock_get_user.return_value = mock_user
        
        # Mock analytics service
        mock_analytics_instance = Mock()
        mock_analytics_service.return_value = mock_analytics_instance
        mock_analytics_instance.generate_comprehensive_report = AsyncMock(
            return_value=Mock(spec=CareerAnalysisReport)
        )
        
        # Mock PDF service error
        mock_pdf_instance = Mock()
        mock_pdf_service.return_value = mock_pdf_instance
        mock_pdf_instance.generate_comprehensive_report = AsyncMock(
            side_effect=PDFGenerationError("PDF generation failed")
        )
        
        pdf_request = {
            "user_id": "test-user-123",
            "report_type": "comprehensive"
        }
        
        response = client.post(
            "/api/v1/analytics/reports/pdf",
            json=pdf_request,
            headers={"Authorization": "Bearer test-token"}
        )
        
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "PDF generation failed" in response.json()["detail"]
    
    @patch('app.api.v1.endpoints.analytics.get_current_user')
    @patch('app.api.v1.endpoints.analytics.AnalyticsService')
    def test_general_exception_handling(
        self, mock_analytics_service, mock_get_user, client, mock_user, sample_analytics_request
    ):
        """Test general exception handling"""
        mock_get_user.return_value = mock_user
        
        # Mock unexpected error
        mock_service_instance = Mock()
        mock_analytics_service.return_value = mock_service_instance
        mock_service_instance.generate_comprehensive_report = AsyncMock(
            side_effect=Exception("Unexpected error")
        )
        
        response = client.post(
            "/api/v1/analytics/generate-report",
            json=sample_analytics_request,
            headers={"Authorization": "Bearer test-token"}
        )
        
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        assert "Failed to generate analytics report" in response.json()["detail"]