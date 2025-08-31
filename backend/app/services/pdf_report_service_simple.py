"""
Simple PDF report service for testing
"""
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class PDFReportService:
    """Simple PDF report service"""
    
    def __init__(self, storage_path: str = "reports"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        logger.info("PDF service initialized")