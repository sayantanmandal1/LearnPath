"""
Comprehensive tests for PDF report generation functionality
"""
import pytest
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock

from app.services.pdf_report_service import PDFReportService
from app.schemas.analytics import (
    CareerAnalysisReport, PDFReportRequest, PDFReportResponse,
    SkillRadarChart, CareerRoadmapVisualization, CareerRoadmapNode, CareerRoadmapEdge,
    SkillGapReport, SkillGapAnalysis, JobCompatibilityReport, JobCompatibilityScore,
    HistoricalProgressReport, ProgressTrackingEntry
)
from app.core.exceptions import PDFGenerationError


class TestPDFReportService:
    """Test PDF report service functionality"""
    
    @pytest.fixture
    def temp_storage(self):
        """Create temporary storage directory"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def pdf_service(self, temp_storage):
        """Create PDF service with temporary storage"""
        return PDFReportService(storage_path=temp_storage)
    
    @pytest.fixture
    def sample_career_report(self):
        """Create sample career analysis report"""
        # Sample skill radar chart
        skill_radar = SkillRadarChart(
            user_id="test-user-123",
            categories=["Programming", "Databases", "Cloud", "DevOps", "Soft Skills"],
            user_scores=[85.0, 70.0, 60.0, 55.0, 80.0],
            market_average=[75.0, 65.0, 70.0, 60.0, 75.0],
            target_scores=[90.0, 80.0, 85.0, 75.0, 85.0]
        )
        
        # Sample career roadmap
        nodes = [
            CareerRoadmapNode(
                id="current",
                title="Software Developer",
                description="Current position",
                position={"x": 0, "y": 0},
                node_type="current",
                timeline_months=0,
                completion_status="completed"
            ),
            CareerRoadmapNode(
                id="milestone_1",
                title="Senior Developer",
                description="Next career step",
                position={"x": 200, "y": 0},
                node_type="milestone",
                timeline_months=12,
                required_skills=["Advanced Python", "System Design"],
                completion_status="not_started"
            ),
            CareerRoadmapNode(
                id="target",
                title="Tech Lead",
                description="Target position",
                position={"x": 400, "y": 0},
                node_type="target",
                timeline_months=24,
                completion_status="not_started"
            )
        ]
        
        edges = [
            CareerRoadmapEdge(
                id="edge_1",
                source_id="current",
                target_id="milestone_1",
                edge_type="direct",
                difficulty=0.6,
                estimated_duration_months=12
            ),
            CareerRoadmapEdge(
                id="edge_2",
                source_id="milestone_1",
                target_id="target",
                edge_type="direct",
                difficulty=0.8,
                estimated_duration_months=12
            )
        ]
        
        career_roadmap = CareerRoadmapVisualization(
            user_id="test-user-123",
            nodes=nodes,
            edges=edges,
            metadata={"target_role": "Tech Lead", "total_timeline_months": 24}
        )
        
        # Sample skill gap report
        skill_gaps = [
            SkillGapAnalysis(
                skill_name="System Design",
                current_level=40.0,
                target_level=80.0,
                gap_size=40.0,
                priority="high",
                estimated_learning_hours=120,
                recommended_resources=[{"type": "course", "name": "System Design Course"}],
                market_demand=0.9,
                salary_impact=15000.0
            ),
            SkillGapAnalysis(
                skill_name="Kubernetes",
                current_level=30.0,
                target_level=70.0,
                gap_size=40.0,
                priority="medium",
                estimated_learning_hours=80,
                recommended_resources=[{"type": "tutorial", "name": "K8s Tutorial"}],
                market_demand=0.8,
                salary_impact=10000.0
            )
        ]
        
        skill_gap_report = SkillGapReport(
            user_id="test-user-123",
            target_role="Tech Lead",
            overall_match_score=72.5,
            skill_gaps=skill_gaps,
            strengths=["Python", "FastAPI", "PostgreSQL"],
            total_learning_hours=200,
            priority_skills=["System Design"]
        )
        
        # Sample job compatibility report
        job_matches = [
            JobCompatibilityScore(
                job_id="job-1",
                job_title="Senior Software Engineer",
                company="Tech Corp",
                overall_score=85.0,
                skill_match_score=80.0,
                experience_match_score=90.0,
                matched_skills=["Python", "FastAPI", "PostgreSQL"],
                missing_skills=["System Design", "Kubernetes"],
                recommendation="apply"
            ),
            JobCompatibilityScore(
                job_id="job-2",
                job_title="Tech Lead",
                company="Innovation Inc",
                overall_score=65.0,
                skill_match_score=60.0,
                experience_match_score=70.0,
                matched_skills=["Python", "PostgreSQL"],
                missing_skills=["System Design", "Team Leadership"],
                recommendation="improve_first"
            )
        ]
        
        job_compatibility_report = JobCompatibilityReport(
            user_id="test-user-123",
            job_matches=job_matches,
            total_jobs_analyzed=50
        )
        
        # Sample progress report
        skill_improvements = [
            ProgressTrackingEntry(
                user_id="test-user-123",
                skill_name="Python",
                previous_score=75.0,
                current_score=85.0,
                improvement=10.0,
                tracking_period_days=90,
                milestone_achieved="Advanced Python Certification"
            ),
            ProgressTrackingEntry(
                user_id="test-user-123",
                skill_name="FastAPI",
                previous_score=60.0,
                current_score=75.0,
                improvement=15.0,
                tracking_period_days=90
            )
        ]
        
        progress_report = HistoricalProgressReport(
            user_id="test-user-123",
            tracking_period_days=90,
            skill_improvements=skill_improvements,
            overall_improvement_score=12.5,
            milestones_achieved=["Advanced Python Certification"],
            trend_analysis={"trend": "improving", "velocity": "moderate"}
        )
        
        # Create comprehensive report
        return CareerAnalysisReport(
            user_id="test-user-123",
            profile_summary={
                "first_name": "John",
                "last_name": "Doe",
                "current_role": "Software Developer",
                "experience_level": "Mid-level",
                "location": "San Francisco, CA",
                "education": "BS Computer Science",
                "skills": ["Python", "FastAPI", "PostgreSQL", "React", "Docker"]
            },
            skill_radar_chart=skill_radar,
            career_roadmap=career_roadmap,
            skill_gap_report=skill_gap_report,
            job_compatibility_report=job_compatibility_report,
            progress_report=progress_report,
            recommendations=[
                "Focus on learning System Design fundamentals",
                "Gain hands-on experience with Kubernetes",
                "Develop leadership and mentoring skills",
                "Build a portfolio of complex projects"
            ],
            next_steps=[
                "Enroll in System Design course",
                "Set up personal Kubernetes cluster",
                "Start mentoring junior developers",
                "Lead a major project initiative"
            ]
        )
    
    @pytest.fixture
    def sample_pdf_request(self):
        """Create sample PDF request"""
        return PDFReportRequest(
            user_id="test-user-123",
            report_type="comprehensive",
            include_charts=True,
            include_recommendations=True
        )
    
    @pytest.mark.asyncio
    async def test_generate_comprehensive_report_success(self, pdf_service, sample_career_report, sample_pdf_request):
        """Test successful PDF report generation"""
        with patch('app.services.pdf_report_service.REPORTLAB_AVAILABLE', True):
            response = await pdf_service.generate_comprehensive_report(
                sample_career_report, sample_pdf_request
            )
            
            assert isinstance(response, PDFReportResponse)
            assert response.report_id is not None
            assert response.file_url.startswith("/api/v1/analytics/reports/")
            assert response.file_size_bytes > 0
            assert response.page_count > 0
            assert response.expires_at > datetime.utcnow()
    
    @pytest.mark.asyncio
    async def test_generate_report_without_reportlab(self, pdf_service, sample_career_report, sample_pdf_request):
        """Test PDF generation when ReportLab is not available"""
        with patch('app.services.pdf_report_service.REPORTLAB_AVAILABLE', False):
            with pytest.raises(PDFGenerationError, match="ReportLab library not available"):
                await pdf_service.generate_comprehensive_report(
                    sample_career_report, sample_pdf_request
                )
    
    @pytest.mark.asyncio
    async def test_generate_report_different_types(self, pdf_service, sample_career_report):
        """Test generating different types of reports"""
        report_types = ["comprehensive", "skills_only", "career_only", "progress_only"]
        
        with patch('app.services.pdf_report_service.REPORTLAB_AVAILABLE', True):
            for report_type in report_types:
                request = PDFReportRequest(
                    user_id="test-user-123",
                    report_type=report_type,
                    include_charts=True,
                    include_recommendations=True
                )
                
                response = await pdf_service.generate_comprehensive_report(
                    sample_career_report, request
                )
                
                assert isinstance(response, PDFReportResponse)
                assert response.report_id is not None
    
    @pytest.mark.asyncio
    async def test_generate_report_without_charts(self, pdf_service, sample_career_report):
        """Test generating report without charts"""
        request = PDFReportRequest(
            user_id="test-user-123",
            report_type="comprehensive",
            include_charts=False,
            include_recommendations=True
        )
        
        with patch('app.services.pdf_report_service.REPORTLAB_AVAILABLE', True):
            response = await pdf_service.generate_comprehensive_report(
                sample_career_report, request
            )
            
            assert isinstance(response, PDFReportResponse)
            assert response.report_id is not None
    
    @pytest.mark.asyncio
    async def test_generate_report_without_recommendations(self, pdf_service, sample_career_report):
        """Test generating report without recommendations"""
        request = PDFReportRequest(
            user_id="test-user-123",
            report_type="comprehensive",
            include_charts=True,
            include_recommendations=False
        )
        
        with patch('app.services.pdf_report_service.REPORTLAB_AVAILABLE', True):
            response = await pdf_service.generate_comprehensive_report(
                sample_career_report, request
            )
            
            assert isinstance(response, PDFReportResponse)
            assert response.report_id is not None
    
    @pytest.mark.asyncio
    async def test_get_report_file_success(self, pdf_service, sample_career_report, sample_pdf_request):
        """Test retrieving generated report file"""
        with patch('app.services.pdf_report_service.REPORTLAB_AVAILABLE', True):
            response = await pdf_service.generate_comprehensive_report(
                sample_career_report, sample_pdf_request
            )
            
            file_path = await pdf_service.get_report_file(response.report_id)
            assert file_path is not None
            assert file_path.exists()
            assert file_path.suffix == '.pdf'
    
    @pytest.mark.asyncio
    async def test_get_report_file_not_found(self, pdf_service):
        """Test retrieving non-existent report file"""
        file_path = await pdf_service.get_report_file("non-existent-id")
        assert file_path is None
    
    @pytest.mark.asyncio
    async def test_cleanup_expired_reports(self, pdf_service, sample_career_report, sample_pdf_request):
        """Test cleanup of expired reports"""
        with patch('app.services.pdf_report_service.REPORTLAB_AVAILABLE', True):
            # Generate a report
            response = await pdf_service.generate_comprehensive_report(
                sample_career_report, sample_pdf_request
            )
            
            file_path = await pdf_service.get_report_file(response.report_id)
            assert file_path.exists()
            
            # Mock file modification time to be old
            old_time = datetime.utcnow() - timedelta(days=8)
            with patch('pathlib.Path.stat') as mock_stat:
                mock_stat.return_value.st_mtime = old_time.timestamp()
                
                await pdf_service.cleanup_expired_reports()
    
    def test_setup_custom_styles(self, pdf_service):
        """Test custom style setup"""
        with patch('app.services.pdf_report_service.REPORTLAB_AVAILABLE', True):
            with patch('app.services.pdf_report_service.getSampleStyleSheet') as mock_styles:
                # Mock the style sheet
                mock_style_sheet = Mock()
                mock_style_sheet.__contains__ = Mock(return_value=True)
                mock_style_sheet.add = Mock()
                mock_styles.return_value = mock_style_sheet
                
                service = PDFReportService()
                
                # Verify that styles were set up
                assert service.styles is not None
                # Verify that custom styles were added
                assert mock_style_sheet.add.call_count >= 5  # Should add at least 5 custom styles
    
    def test_create_title_page(self, pdf_service, sample_career_report, sample_pdf_request):
        """Test title page creation"""
        with patch('app.services.pdf_report_service.REPORTLAB_AVAILABLE', True):
            service = PDFReportService()
            content = service._create_title_page(sample_career_report, sample_pdf_request)
            
            assert len(content) > 0
            # Check that content includes expected elements
            content_text = str(content)
            assert "Career Analysis Report" in content_text
    
    def test_create_executive_summary(self, pdf_service, sample_career_report):
        """Test executive summary creation"""
        with patch('app.services.pdf_report_service.REPORTLAB_AVAILABLE', True):
            service = PDFReportService()
            content = service._create_executive_summary(sample_career_report)
            
            assert len(content) > 0
            content_text = str(content)
            assert "Executive Summary" in content_text
    
    def test_create_skill_analysis_section(self, pdf_service, sample_career_report):
        """Test skill analysis section creation"""
        with patch('app.services.pdf_report_service.REPORTLAB_AVAILABLE', True):
            service = PDFReportService()
            content = service._create_skill_analysis_section(sample_career_report)
            
            assert len(content) > 0
            # Should include skill table
            assert any("Table" in str(type(item)) for item in content)
    
    def test_create_career_roadmap_section(self, pdf_service, sample_career_report):
        """Test career roadmap section creation"""
        with patch('app.services.pdf_report_service.REPORTLAB_AVAILABLE', True):
            service = PDFReportService()
            content = service._create_career_roadmap_section(sample_career_report)
            
            assert len(content) > 0
            content_text = str(content)
            assert "Career Roadmap" in content_text
    
    def test_create_skill_gap_section(self, pdf_service, sample_career_report):
        """Test skill gap section creation"""
        with patch('app.services.pdf_report_service.REPORTLAB_AVAILABLE', True):
            service = PDFReportService()
            content = service._create_skill_gap_section(sample_career_report)
            
            assert len(content) > 0
            content_text = str(content)
            assert "Skill Gap Analysis" in content_text
    
    def test_create_job_compatibility_section(self, pdf_service, sample_career_report):
        """Test job compatibility section creation"""
        with patch('app.services.pdf_report_service.REPORTLAB_AVAILABLE', True):
            service = PDFReportService()
            content = service._create_job_compatibility_section(sample_career_report)
            
            assert len(content) > 0
            content_text = str(content)
            assert "Job Compatibility Analysis" in content_text
    
    def test_create_progress_section(self, pdf_service, sample_career_report):
        """Test progress section creation"""
        with patch('app.services.pdf_report_service.REPORTLAB_AVAILABLE', True):
            service = PDFReportService()
            content = service._create_progress_section(sample_career_report)
            
            assert len(content) > 0
            content_text = str(content)
            assert "Progress Tracking" in content_text
    
    def test_create_recommendations_section(self, pdf_service, sample_career_report):
        """Test recommendations section creation"""
        with patch('app.services.pdf_report_service.REPORTLAB_AVAILABLE', True):
            service = PDFReportService()
            content = service._create_recommendations_section(sample_career_report)
            
            assert len(content) > 0
            content_text = str(content)
            assert "Recommendations" in content_text
    
    def test_create_next_steps_section(self, pdf_service, sample_career_report):
        """Test next steps section creation"""
        with patch('app.services.pdf_report_service.REPORTLAB_AVAILABLE', True):
            service = PDFReportService()
            content = service._create_next_steps_section(sample_career_report)
            
            assert len(content) > 0
            content_text = str(content)
            assert "Next Steps" in content_text
    
    @pytest.mark.asyncio
    async def test_error_handling_during_generation(self, pdf_service, sample_career_report, sample_pdf_request):
        """Test error handling during PDF generation"""
        with patch('app.services.pdf_report_service.REPORTLAB_AVAILABLE', True):
            with patch('app.services.pdf_report_service.SimpleDocTemplate') as mock_doc:
                mock_doc.side_effect = Exception("PDF generation failed")
                
                with pytest.raises(PDFGenerationError, match="Failed to generate PDF report"):
                    await pdf_service.generate_comprehensive_report(
                        sample_career_report, sample_pdf_request
                    )
    
    def test_matplotlib_chart_generation(self, pdf_service, sample_career_report):
        """Test matplotlib chart generation"""
        with patch('app.services.pdf_report_service.MATPLOTLIB_AVAILABLE', True):
            with patch('app.services.pdf_report_service.plt') as mock_plt:
                mock_fig = Mock()
                mock_ax = Mock()
                mock_plt.subplots.return_value = (mock_fig, mock_ax)
                
                service = PDFReportService()
                chart_image = service._create_skill_radar_chart_image(sample_career_report.skill_radar_chart)
                
                # Should attempt to create chart when matplotlib is available
                mock_plt.subplots.assert_called_once()
    
    def test_matplotlib_chart_generation_unavailable(self, pdf_service, sample_career_report):
        """Test chart generation when matplotlib is unavailable"""
        with patch('app.services.pdf_report_service.MATPLOTLIB_AVAILABLE', False):
            service = PDFReportService()
            chart_image = service._create_skill_radar_chart_image(sample_career_report.skill_radar_chart)
            
            assert chart_image is None


class TestPDFReportIntegration:
    """Integration tests for PDF report functionality"""
    
    @pytest.mark.asyncio
    async def test_full_pdf_generation_workflow(self):
        """Test complete PDF generation workflow"""
        # This would be an integration test that tests the full workflow
        # from API request to PDF file generation
        pass
    
    @pytest.mark.asyncio
    async def test_pdf_report_api_endpoints(self):
        """Test PDF report API endpoints"""
        # This would test the actual API endpoints
        pass