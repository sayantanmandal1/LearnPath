"""
Tests for PDF report service functionality
"""
import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile
import shutil

from app.services.pdf_report_service import PDFReportService, REPORTLAB_AVAILABLE
from app.schemas.analytics import (
    CareerAnalysisReport, PDFReportRequest, PDFReportResponse,
    SkillRadarChart, SkillGapReport, SkillGapAnalysis,
    JobCompatibilityReport, JobCompatibilityScore,
    HistoricalProgressReport, ProgressTrackingEntry
)
from app.core.exceptions import PDFGenerationError


@pytest.fixture
def temp_storage():
    """Create temporary storage directory"""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def pdf_service(temp_storage):
    """PDF report service with temporary storage"""
    return PDFReportService(storage_path=str(temp_storage))


@pytest.fixture
def sample_report_data():
    """Sample career analysis report data"""
    profile_summary = {
        "name": "John Doe",
        "current_role": "Software Developer",
        "experience_years": 3,
        "location": "San Francisco, CA",
        "skills": ["Python", "JavaScript", "React"],
        "skill_count": 3
    }
    
    skill_radar_chart = SkillRadarChart(
        user_id="test-user-123",
        categories=["Programming", "Frameworks", "Databases"],
        user_scores=[85.0, 70.0, 60.0],
        market_average=[65.0, 65.0, 65.0],
        target_scores=[90.0, 85.0, 75.0],
        max_score=100.0
    )
    
    skill_gap_report = SkillGapReport(
        user_id="test-user-123",
        target_role="Senior Developer",
        overall_match_score=75.0,
        skill_gaps=[
            SkillGapAnalysis(
                skill_name="Python",
                current_level=70.0,
                target_level=90.0,
                gap_size=20.0,
                priority="high",
                estimated_learning_hours=40
            ),
            SkillGapAnalysis(
                skill_name="Docker",
                current_level=30.0,
                target_level=70.0,
                gap_size=40.0,
                priority="medium",
                estimated_learning_hours=80
            )
        ],
        strengths=["JavaScript", "HTML"],
        total_learning_hours=120,
        priority_skills=["Python"]
    )
    
    job_compatibility_report = JobCompatibilityReport(
        user_id="test-user-123",
        job_matches=[
            JobCompatibilityScore(
                job_id="job-1",
                job_title="Senior Python Developer",
                company="Tech Corp",
                overall_score=85.0,
                skill_match_score=80.0,
                experience_match_score=90.0,
                matched_skills=["Python", "JavaScript"],
                missing_skills=["Docker"],
                recommendation="apply"
            ),
            JobCompatibilityScore(
                job_id="job-2",
                job_title="Full Stack Developer",
                company="Startup Inc",
                overall_score=70.0,
                skill_match_score=65.0,
                experience_match_score=75.0,
                matched_skills=["JavaScript"],
                missing_skills=["Python", "React"],
                recommendation="consider"
            )
        ],
        filters_applied={"location": "San Francisco"},
        total_jobs_analyzed=10
    )
    
    progress_report = HistoricalProgressReport(
        user_id="test-user-123",
        tracking_period_days=90,
        skill_improvements=[
            ProgressTrackingEntry(
                user_id="test-user-123",
                skill_name="Python",
                previous_score=70.0,
                current_score=80.0,
                improvement=10.0,
                tracking_period_days=90
            )
        ],
        overall_improvement_score=8.5,
        milestones_achieved=["Completed Python Course"],
        trend_analysis={"trend": "improving", "velocity": 8.5}
    )
    
    return CareerAnalysisReport(
        user_id="test-user-123",
        profile_summary=profile_summary,
        skill_radar_chart=skill_radar_chart,
        career_roadmap=None,  # Not included in this test
        skill_gap_report=skill_gap_report,
        job_compatibility_report=job_compatibility_report,
        progress_report=progress_report,
        recommendations=[
            "Focus on Python development",
            "Learn Docker containerization",
            "Build more portfolio projects"
        ],
        next_steps=[
            "Take advanced Python course",
            "Complete Docker tutorial",
            "Contribute to open source projects"
        ]
    )


@pytest.fixture
def sample_pdf_request():
    """Sample PDF report request"""
    return PDFReportRequest(
        user_id="test-user-123",
        report_type="comprehensive",
        include_charts=True,
        include_recommendations=True
    )


class TestPDFReportService:
    """Test cases for PDFReportService"""
    
    def test_init_with_storage_path(self, temp_storage):
        """Test service initialization with custom storage path"""
        service = PDFReportService(storage_path=str(temp_storage))
        assert service.storage_path == temp_storage
        assert temp_storage.exists()
    
    def test_init_creates_storage_directory(self):
        """Test that service creates storage directory if it doesn't exist"""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage_path = Path(temp_dir) / "reports"
            assert not storage_path.exists()
            
            service = PDFReportService(storage_path=str(storage_path))
            assert storage_path.exists()
    
    @pytest.mark.skipif(not REPORTLAB_AVAILABLE, reason="ReportLab not available")
    @pytest.mark.asyncio
    async def test_generate_comprehensive_report_success(
        self, pdf_service, sample_report_data, sample_pdf_request
    ):
        """Test successful PDF report generation"""
        result = await pdf_service.generate_comprehensive_report(
            sample_report_data, sample_pdf_request
        )
        
        assert isinstance(result, PDFReportResponse)
        assert result.report_id is not None
        assert result.file_url.endswith(".pdf")
        assert result.file_size_bytes > 0
        assert result.page_count > 0
        assert result.expires_at > datetime.utcnow()
        
        # Check that file was actually created
        report_file = await pdf_service.get_report_file(result.report_id)
        assert report_file is not None
        assert report_file.exists()
    
    @pytest.mark.skipif(not REPORTLAB_AVAILABLE, reason="ReportLab not available")
    @pytest.mark.asyncio
    async def test_generate_skills_only_report(
        self, pdf_service, sample_report_data, sample_pdf_request
    ):
        """Test skills-only PDF report generation"""
        sample_pdf_request.report_type = "skills_only"
        
        result = await pdf_service.generate_comprehensive_report(
            sample_report_data, sample_pdf_request
        )
        
        assert isinstance(result, PDFReportResponse)
        assert result.file_size_bytes > 0
        
        # File should be created
        report_file = await pdf_service.get_report_file(result.report_id)
        assert report_file is not None
        assert report_file.exists()
    
    @pytest.mark.skipif(not REPORTLAB_AVAILABLE, reason="ReportLab not available")
    @pytest.mark.asyncio
    async def test_generate_career_only_report(
        self, pdf_service, sample_report_data, sample_pdf_request
    ):
        """Test career-only PDF report generation"""
        sample_pdf_request.report_type = "career_only"
        
        result = await pdf_service.generate_comprehensive_report(
            sample_report_data, sample_pdf_request
        )
        
        assert isinstance(result, PDFReportResponse)
        assert result.file_size_bytes > 0
    
    @pytest.mark.skipif(not REPORTLAB_AVAILABLE, reason="ReportLab not available")
    @pytest.mark.asyncio
    async def test_generate_progress_only_report(
        self, pdf_service, sample_report_data, sample_pdf_request
    ):
        """Test progress-only PDF report generation"""
        sample_pdf_request.report_type = "progress_only"
        
        result = await pdf_service.generate_comprehensive_report(
            sample_report_data, sample_pdf_request
        )
        
        assert isinstance(result, PDFReportResponse)
        assert result.file_size_bytes > 0
    
    @pytest.mark.skipif(not REPORTLAB_AVAILABLE, reason="ReportLab not available")
    @pytest.mark.asyncio
    async def test_generate_report_without_recommendations(
        self, pdf_service, sample_report_data, sample_pdf_request
    ):
        """Test PDF report generation without recommendations"""
        sample_pdf_request.include_recommendations = False
        
        result = await pdf_service.generate_comprehensive_report(
            sample_report_data, sample_pdf_request
        )
        
        assert isinstance(result, PDFReportResponse)
        assert result.file_size_bytes > 0
    
    @pytest.mark.skipif(REPORTLAB_AVAILABLE, reason="Test requires ReportLab to be unavailable")
    @pytest.mark.asyncio
    async def test_generate_report_without_reportlab(
        self, pdf_service, sample_report_data, sample_pdf_request
    ):
        """Test PDF report generation when ReportLab is not available"""
        with pytest.raises(PDFGenerationError, match="ReportLab library not available"):
            await pdf_service.generate_comprehensive_report(
                sample_report_data, sample_pdf_request
            )
    
    @pytest.mark.skipif(not REPORTLAB_AVAILABLE, reason="ReportLab not available")
    @pytest.mark.asyncio
    async def test_generate_report_with_error(
        self, pdf_service, sample_report_data, sample_pdf_request
    ):
        """Test PDF report generation with error handling"""
        # Mock an error in PDF generation
        with patch('app.services.pdf_report_service.SimpleDocTemplate') as mock_doc:
            mock_doc.side_effect = Exception("PDF generation failed")
            
            with pytest.raises(PDFGenerationError, match="Failed to generate PDF report"):
                await pdf_service.generate_comprehensive_report(
                    sample_report_data, sample_pdf_request
                )
    
    @pytest.mark.asyncio
    async def test_get_report_file_exists(self, pdf_service, temp_storage):
        """Test getting existing report file"""
        # Create a test file
        test_file = temp_storage / "career_report_test-user_test-report-id.pdf"
        test_file.write_text("test content")
        
        result = await pdf_service.get_report_file("test-report-id")
        assert result == test_file
    
    @pytest.mark.asyncio
    async def test_get_report_file_not_exists(self, pdf_service):
        """Test getting non-existent report file"""
        result = await pdf_service.get_report_file("non-existent-id")
        assert result is None
    
    @pytest.mark.asyncio
    async def test_cleanup_expired_reports(self, pdf_service, temp_storage):
        """Test cleanup of expired report files"""
        # Create test files with different ages
        current_time = datetime.utcnow()
        
        # Recent file (should not be deleted)
        recent_file = temp_storage / "recent_report.pdf"
        recent_file.write_text("recent content")
        
        # Old file (should be deleted)
        old_file = temp_storage / "old_report.pdf"
        old_file.write_text("old content")
        
        # Modify file timestamps to simulate age
        import os
        import time
        
        # Make old file appear 8 days old
        old_timestamp = time.mktime((current_time - timedelta(days=8)).timetuple())
        os.utime(old_file, (old_timestamp, old_timestamp))
        
        # Make recent file appear 1 day old
        recent_timestamp = time.mktime((current_time - timedelta(days=1)).timetuple())
        os.utime(recent_file, (recent_timestamp, recent_timestamp))
        
        # Run cleanup
        await pdf_service.cleanup_expired_reports()
        
        # Check results
        assert recent_file.exists()  # Should still exist
        assert not old_file.exists()  # Should be deleted
    
    @pytest.mark.asyncio
    async def test_cleanup_expired_reports_with_error(self, pdf_service, temp_storage):
        """Test cleanup with file system errors"""
        # Create a file that will cause an error when trying to delete
        with patch('pathlib.Path.unlink') as mock_unlink:
            mock_unlink.side_effect = PermissionError("Permission denied")
            
            # Create a test file
            test_file = temp_storage / "test_report.pdf"
            test_file.write_text("test content")
            
            # Should not raise an exception, just log the error
            await pdf_service.cleanup_expired_reports()
    
    def test_estimate_page_count(self, pdf_service):
        """Test page count estimation"""
        # Test with different story lengths
        short_story = ["element1", "element2", "element3"]
        assert pdf_service._estimate_page_count(short_story) == 1
        
        long_story = ["element" + str(i) for i in range(25)]
        estimated_pages = pdf_service._estimate_page_count(long_story)
        assert estimated_pages >= 2
        
        empty_story = []
        assert pdf_service._estimate_page_count(empty_story) == 1  # Minimum 1 page


class TestPDFReportServiceIntegration:
    """Integration tests for PDF report service"""
    
    @pytest.mark.skipif(not REPORTLAB_AVAILABLE, reason="ReportLab not available")
    @pytest.mark.asyncio
    async def test_full_report_generation_workflow(
        self, pdf_service, sample_report_data, sample_pdf_request
    ):
        """Test complete workflow from generation to cleanup"""
        # Generate report
        result = await pdf_service.generate_comprehensive_report(
            sample_report_data, sample_pdf_request
        )
        
        # Verify file exists
        report_file = await pdf_service.get_report_file(result.report_id)
        assert report_file is not None
        assert report_file.exists()
        assert report_file.stat().st_size > 0
        
        # Verify file content is PDF
        with open(report_file, 'rb') as f:
            header = f.read(4)
            assert header == b'%PDF'  # PDF file signature
        
        # Test cleanup (won't delete recent file)
        await pdf_service.cleanup_expired_reports()
        assert report_file.exists()  # Should still exist
    
    @pytest.mark.skipif(not REPORTLAB_AVAILABLE, reason="ReportLab not available")
    @pytest.mark.asyncio
    async def test_multiple_report_generation(
        self, pdf_service, sample_report_data, sample_pdf_request
    ):
        """Test generating multiple reports"""
        results = []
        
        # Generate multiple reports
        for i in range(3):
            sample_pdf_request.user_id = f"test-user-{i}"
            sample_report_data.user_id = f"test-user-{i}"
            
            result = await pdf_service.generate_comprehensive_report(
                sample_report_data, sample_pdf_request
            )
            results.append(result)
        
        # Verify all reports were created
        assert len(results) == 3
        assert len(set(r.report_id for r in results)) == 3  # All unique IDs
        
        # Verify all files exist
        for result in results:
            report_file = await pdf_service.get_report_file(result.report_id)
            assert report_file is not None
            assert report_file.exists()
    
    @pytest.mark.skipif(not REPORTLAB_AVAILABLE, reason="ReportLab not available")
    @pytest.mark.asyncio
    async def test_report_with_minimal_data(self, pdf_service, sample_pdf_request):
        """Test report generation with minimal data"""
        # Create minimal report data
        minimal_data = CareerAnalysisReport(
            user_id="test-user-123",
            profile_summary={"name": "Test User"},
            skill_radar_chart=None,
            career_roadmap=None,
            skill_gap_report=None,
            job_compatibility_report=None,
            progress_report=None,
            recommendations=[],
            next_steps=[]
        )
        
        result = await pdf_service.generate_comprehensive_report(
            minimal_data, sample_pdf_request
        )
        
        assert isinstance(result, PDFReportResponse)
        assert result.file_size_bytes > 0  # Should still generate a basic report
        
        # Verify file exists
        report_file = await pdf_service.get_report_file(result.report_id)
        assert report_file is not None
        assert report_file.exists()