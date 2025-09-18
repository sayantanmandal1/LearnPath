"""
Tests for resume processing service and endpoints
"""
import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
from fastapi import UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from app.services.resume_processing_service import ResumeProcessingService
from app.models.resume import ResumeData, ProcessingStatus
from app.schemas.resume import ParsedResumeData, ContactInfo, WorkExperience
from app.core.exceptions import ValidationError, ProcessingError


class TestResumeProcessingService:
    """Test cases for ResumeProcessingService"""
    
    @pytest.fixture
    def resume_service(self):
        """Create resume processing service instance"""
        return ResumeProcessingService()
    
    @pytest.fixture
    def mock_db(self):
        """Mock database session"""
        return Mock(spec=AsyncSession)
    
    @pytest.fixture
    def sample_pdf_content(self):
        """Sample PDF content for testing"""
        return b'%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\n'
    
    @pytest.fixture
    def sample_upload_file(self, sample_pdf_content):
        """Create sample upload file"""
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
            tmp.write(sample_pdf_content)
            tmp.flush()
            
            file = UploadFile(
                filename="test_resume.pdf",
                file=open(tmp.name, 'rb'),
                content_type="application/pdf"
            )
            yield file
            file.file.close()
            os.unlink(tmp.name)
    
    @pytest.mark.asyncio
    async def test_validate_file_success(self, resume_service, sample_upload_file):
        """Test successful file validation"""
        # Should not raise any exception
        await resume_service._validate_file(sample_upload_file)
    
    @pytest.mark.asyncio
    async def test_validate_file_unsupported_type(self, resume_service):
        """Test validation failure for unsupported file type"""
        with tempfile.NamedTemporaryFile(suffix='.txt') as tmp:
            tmp.write(b'This is a text file')
            tmp.flush()
            
            file = UploadFile(
                filename="test.txt",
                file=open(tmp.name, 'rb'),
                content_type="text/plain"
            )
            
            with pytest.raises(ValidationError, match="Unsupported file type"):
                await resume_service._validate_file(file)
            
            file.file.close()
    
    @pytest.mark.asyncio
    async def test_validate_file_too_large(self, resume_service):
        """Test validation failure for file too large"""
        # Create a file larger than the limit
        large_content = b'x' * (11 * 1024 * 1024)  # 11MB
        
        with tempfile.NamedTemporaryFile(suffix='.pdf') as tmp:
            tmp.write(b'%PDF-1.4\n')  # Valid PDF header
            tmp.write(large_content)
            tmp.flush()
            
            file = UploadFile(
                filename="large_resume.pdf",
                file=open(tmp.name, 'rb'),
                content_type="application/pdf"
            )
            
            with pytest.raises(ValidationError, match="File size exceeds maximum limit"):
                await resume_service._validate_file(file)
            
            file.file.close()
    
    @pytest.mark.asyncio
    async def test_validate_file_empty(self, resume_service):
        """Test validation failure for empty file"""
        with tempfile.NamedTemporaryFile(suffix='.pdf') as tmp:
            # Don't write anything to create empty file
            tmp.flush()
            
            file = UploadFile(
                filename="empty.pdf",
                file=open(tmp.name, 'rb'),
                content_type="application/pdf"
            )
            
            with pytest.raises(ValidationError, match="File is empty"):
                await resume_service._validate_file(file)
            
            file.file.close()
    
    @pytest.mark.asyncio
    async def test_extract_from_pdf_success(self, resume_service):
        """Test successful PDF text extraction"""
        # Create a simple PDF file for testing
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
            # Write minimal PDF content
            pdf_content = b'''%PDF-1.4
1 0 obj
<<
/Type /Catalog
/Pages 2 0 R
>>
endobj
2 0 obj
<<
/Type /Pages
/Kids [3 0 R]
/Count 1
>>
endobj
3 0 obj
<<
/Type /Page
/Parent 2 0 R
/MediaBox [0 0 612 792]
/Contents 4 0 R
>>
endobj
4 0 obj
<<
/Length 44
>>
stream
BT
/F1 12 Tf
100 700 Td
(John Doe Resume) Tj
ET
endstream
endobj
xref
0 5
0000000000 65535 f 
0000000009 00000 n 
0000000058 00000 n 
0000000115 00000 n 
0000000206 00000 n 
trailer
<<
/Size 5
/Root 1 0 R
>>
startxref
299
%%EOF'''
            tmp.write(pdf_content)
            tmp.flush()
            
            try:
                text, confidence = await resume_service._extract_from_pdf(tmp.name)
                assert isinstance(text, str)
                assert isinstance(confidence, float)
                assert 0.0 <= confidence <= 1.0
            finally:
                os.unlink(tmp.name)
    
    @pytest.mark.asyncio
    async def test_extract_from_docx_success(self, resume_service):
        """Test successful DOCX text extraction"""
        # Mock the Document class since creating a real DOCX is complex
        with patch('app.services.resume_processing_service.Document') as mock_doc:
            mock_paragraph = Mock()
            mock_paragraph.text = "John Doe\nSoftware Engineer"
            
            mock_doc_instance = Mock()
            mock_doc_instance.paragraphs = [mock_paragraph]
            mock_doc_instance.tables = []
            mock_doc.return_value = mock_doc_instance
            
            text, confidence = await resume_service._extract_from_docx("dummy_path.docx")
            
            assert "John Doe" in text
            assert "Software Engineer" in text
            assert confidence == 0.9
    
    @pytest.mark.asyncio
    async def test_fallback_parsing(self, resume_service):
        """Test fallback parsing when Gemini API is unavailable"""
        sample_text = """
        John Doe
        john.doe@email.com
        +1234567890
        Software Engineer with 5 years experience
        """
        
        result = await resume_service._fallback_parsing(sample_text)
        
        assert isinstance(result, ParsedResumeData)
        assert result.contact_info is not None
        assert result.contact_info.email == "john.doe@email.com"
        assert result.contact_info.phone == "+1234567890"
        assert "manual" in result.summary.lower()
    
    @pytest.mark.asyncio
    async def test_validate_parsed_data(self, resume_service):
        """Test validation of parsed resume data"""
        parsed_data = ParsedResumeData(
            contact_info=ContactInfo(
                name="John Doe",
                email="john.doe@email.com",
                phone="+1234567890"
            ),
            work_experience=[
                WorkExperience(
                    company="Tech Corp",
                    position="Software Engineer",
                    start_date="2020-01",
                    end_date="2023-12"
                )
            ]
        )
        
        validation_result = await resume_service._validate_parsed_data(parsed_data)
        
        assert validation_result.is_valid
        assert validation_result.confidence_score > 0.5
        assert len(validation_result.errors) == 0
    
    @pytest.mark.asyncio
    async def test_validate_parsed_data_missing_contact(self, resume_service):
        """Test validation with missing contact information"""
        parsed_data = ParsedResumeData(
            contact_info=None,
            work_experience=[]
        )
        
        validation_result = await resume_service._validate_parsed_data(parsed_data)
        
        assert not validation_result.is_valid
        assert len(validation_result.errors) > 0
        assert any("contact_info" in error.field for error in validation_result.errors)
    
    @pytest.mark.asyncio
    @patch('app.services.resume_processing_service.httpx.AsyncClient')
    async def test_parse_with_gemini_success(self, mock_client, resume_service):
        """Test successful Gemini API parsing"""
        # Mock successful Gemini API response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "candidates": [{
                "content": {
                    "parts": [{
                        "text": '{"contact_info": {"name": "John Doe", "email": "john@email.com"}, "summary": "Software Engineer"}'
                    }]
                }
            }]
        }
        
        mock_client_instance = Mock()
        mock_client_instance.post = AsyncMock(return_value=mock_response)
        mock_client.return_value.__aenter__.return_value = mock_client_instance
        
        # Set API key for test
        resume_service.gemini_api_key = "test_key"
        
        result = await resume_service._parse_with_gemini("Sample resume text")
        
        assert isinstance(result, ParsedResumeData)
        assert result.contact_info is not None
        assert result.contact_info.name == "John Doe"
        assert result.summary == "Software Engineer"
    
    @pytest.mark.asyncio
    @patch('app.services.resume_processing_service.httpx.AsyncClient')
    async def test_parse_with_gemini_api_error(self, mock_client, resume_service):
        """Test Gemini API error handling"""
        # Mock API error response
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        
        mock_client_instance = Mock()
        mock_client_instance.post = AsyncMock(return_value=mock_response)
        mock_client.return_value.__aenter__.return_value = mock_client_instance
        
        resume_service.gemini_api_key = "test_key"
        
        result = await resume_service._parse_with_gemini("Sample resume text")
        
        # Should fall back to basic parsing
        assert isinstance(result, ParsedResumeData)
        assert "manual" in result.summary.lower()
    
    @pytest.mark.asyncio
    async def test_parse_with_gemini_no_api_key(self, resume_service):
        """Test Gemini parsing without API key"""
        resume_service.gemini_api_key = None
        
        result = await resume_service._parse_with_gemini("Sample resume text")
        
        # Should fall back to basic parsing
        assert isinstance(result, ParsedResumeData)
        assert "manual" in result.summary.lower()
    
    @pytest.mark.asyncio
    async def test_upload_resume_success(self, resume_service, sample_upload_file, mock_db):
        """Test successful resume upload"""
        mock_db.add = Mock()
        mock_db.commit = AsyncMock()
        mock_db.refresh = AsyncMock()
        
        # Mock the created resume data
        mock_resume = Mock(spec=ResumeData)
        mock_resume.id = "test-id"
        mock_resume.user_id = "user-123"
        mock_resume.original_filename = "test_resume.pdf"
        mock_resume.file_size = 1024
        mock_resume.file_type = "application/pdf"
        mock_resume.processing_status = ProcessingStatus.PENDING
        mock_resume.created_at = "2023-01-01T00:00:00"
        
        with patch.object(resume_service, '_validate_file', return_value=None):
            result = await resume_service.upload_resume(sample_upload_file, "user-123", mock_db)
            
            assert mock_db.add.called
            assert mock_db.commit.called
            assert mock_db.refresh.called


class TestResumeEndpoints:
    """Test cases for resume API endpoints"""
    
    @pytest.fixture
    def mock_resume_service(self):
        """Mock resume processing service"""
        return Mock(spec=ResumeProcessingService)
    
    @pytest.mark.asyncio
    async def test_upload_resume_endpoint_success(self, client, mock_user, mock_resume_service):
        """Test successful resume upload endpoint"""
        # Mock service response
        mock_resume_data = Mock()
        mock_resume_data.id = "test-id"
        mock_resume_data.user_id = "user-123"
        mock_resume_data.original_filename = "test.pdf"
        mock_resume_data.file_size = 1024
        mock_resume_data.file_type = "application/pdf"
        mock_resume_data.processing_status = ProcessingStatus.PENDING
        mock_resume_data.created_at = "2023-01-01T00:00:00"
        
        mock_resume_service.upload_resume.return_value = mock_resume_data
        
        # Create test file
        with tempfile.NamedTemporaryFile(suffix='.pdf') as tmp:
            tmp.write(b'%PDF-1.4\ntest content')
            tmp.flush()
            
            with patch('app.api.v1.endpoints.resume.resume_service', mock_resume_service):
                with patch('app.api.v1.endpoints.resume.get_current_user', return_value=mock_user):
                    response = await client.post(
                        "/api/v1/resume/upload",
                        files={"file": ("test.pdf", open(tmp.name, 'rb'), "application/pdf")}
                    )
        
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "test-id"
        assert data["original_filename"] == "test.pdf"
        assert data["processing_status"] == "pending"
    
    @pytest.mark.asyncio
    async def test_get_resume_result_success(self, client, mock_user, mock_resume_service):
        """Test successful resume result retrieval"""
        # Mock service response
        mock_resume_data = Mock()
        mock_resume_data.id = "test-id"
        mock_resume_data.user_id = "user-123"
        mock_resume_data.original_filename = "test.pdf"
        mock_resume_data.processing_status = ProcessingStatus.COMPLETED
        mock_resume_data.extracted_text = "Sample extracted text"
        mock_resume_data.extraction_confidence = 0.95
        mock_resume_data.parsed_sections = {
            "contact_info": {"name": "John Doe", "email": "john@email.com"}
        }
        mock_resume_data.error_message = None
        mock_resume_data.processing_started_at = "2023-01-01T00:00:00"
        mock_resume_data.processing_completed_at = "2023-01-01T00:01:00"
        mock_resume_data.created_at = "2023-01-01T00:00:00"
        mock_resume_data.updated_at = "2023-01-01T00:01:00"
        
        mock_resume_service.get_resume_by_id.return_value = mock_resume_data
        
        with patch('app.api.v1.endpoints.resume.resume_service', mock_resume_service):
            with patch('app.api.v1.endpoints.resume.get_current_user', return_value=mock_user):
                response = await client.get("/api/v1/resume/test-id")
        
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "test-id"
        assert data["processing_status"] == "completed"
        assert data["extraction_confidence"] == 0.95
    
    @pytest.mark.asyncio
    async def test_get_resume_not_found(self, client, mock_user, mock_resume_service):
        """Test resume not found error"""
        mock_resume_service.get_resume_by_id.return_value = None
        
        with patch('app.api.v1.endpoints.resume.resume_service', mock_resume_service):
            with patch('app.api.v1.endpoints.resume.get_current_user', return_value=mock_user):
                response = await client.get("/api/v1/resume/nonexistent-id")
        
        assert response.status_code == 404
        assert "Resume not found" in response.json()["detail"]
    
    @pytest.mark.asyncio
    async def test_get_resume_access_denied(self, client, mock_user, mock_resume_service):
        """Test access denied for resume belonging to different user"""
        # Mock resume belonging to different user
        mock_resume_data = Mock()
        mock_resume_data.user_id = "different-user-id"
        mock_resume_service.get_resume_by_id.return_value = mock_resume_data
        
        with patch('app.api.v1.endpoints.resume.resume_service', mock_resume_service):
            with patch('app.api.v1.endpoints.resume.get_current_user', return_value=mock_user):
                response = await client.get("/api/v1/resume/test-id")
        
        assert response.status_code == 403
        assert "Access denied" in response.json()["detail"]


@pytest.fixture
def mock_user():
    """Mock user for testing"""
    user = Mock()
    user.id = "user-123"
    user.email = "test@example.com"
    return user


@pytest.fixture
async def client():
    """Create test client"""
    from fastapi.testclient import TestClient
    from app.main import app
    
    return TestClient(app)