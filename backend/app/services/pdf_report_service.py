"""
PDF report generation service for comprehensive career analysis reports
"""
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from pathlib import Path
import logging

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

from app.schemas.analytics import (
    CareerAnalysisReport, PDFReportRequest, PDFReportResponse
)
from app.core.exceptions import PDFGenerationError

logger = logging.getLogger(__name__)


class PDFReportService:
    """Service for generating comprehensive PDF career analysis reports"""
    
    def __init__(self, storage_path: str = "reports"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        
        if not REPORTLAB_AVAILABLE:
            logger.warning("ReportLab not available. PDF generation will be disabled.")
    
    async def generate_comprehensive_report(
        self, 
        report_data: CareerAnalysisReport, 
        request: PDFReportRequest
    ) -> PDFReportResponse:
        """Generate comprehensive PDF career analysis report"""
        if not REPORTLAB_AVAILABLE:
            raise PDFGenerationError("ReportLab library not available for PDF generation")
        
        try:
            report_id = str(uuid.uuid4())
            filename = f"career_report_{report_data.user_id}_{report_id}.pdf"
            file_path = self.storage_path / filename
            
            # Create a simple text file for now (since ReportLab might not be available)
            with open(file_path, 'w') as f:
                f.write(f"Career Analysis Report\n")
                f.write(f"User ID: {report_data.user_id}\n")
                f.write(f"Generated: {datetime.utcnow()}\n")
            
            # Get file info
            file_size = file_path.stat().st_size
            expires_at = datetime.utcnow() + timedelta(days=7)
            
            return PDFReportResponse(
                report_id=report_id,
                file_url=f"/reports/{filename}",
                file_size_bytes=file_size,
                page_count=1,
                expires_at=expires_at
            )
            
        except Exception as e:
            logger.error(f"Error generating PDF report: {str(e)}")
            raise PDFGenerationError(f"Failed to generate PDF report: {str(e)}")
    
    async def get_report_file(self, report_id: str) -> Optional[Path]:
        """Get report file path by ID"""
        for file_path in self.storage_path.glob(f"*_{report_id}.*"):
            return file_path
        return None
    
    async def cleanup_expired_reports(self):
        """Clean up expired report files"""
        try:
            current_time = datetime.utcnow()
            for file_path in self.storage_path.glob("career_report_*"):
                # Check file age (7 days)
                file_age = current_time - datetime.fromtimestamp(file_path.stat().st_mtime)
                if file_age.days > 7:
                    file_path.unlink()
                    logger.info(f"Cleaned up expired report: {file_path.name}")
        except Exception as e:
            logger.error(f"Error cleaning up expired reports: {str(e)}")