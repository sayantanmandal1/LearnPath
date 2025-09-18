"""
Demo script for testing resume processing functionality
"""
import asyncio
import tempfile
import os
from pathlib import Path
import sys

# Add the backend directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from app.services.resume_processing_service import ResumeProcessingService
from app.schemas.resume import ParsedResumeData, ContactInfo, WorkExperience
from app.core.exceptions import ValidationError, ProcessingError


async def create_sample_pdf():
    """Create a sample PDF file for testing"""
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
/Length 200
>>
stream
BT
/F1 12 Tf
100 700 Td
(John Doe) Tj
0 -20 Td
(Software Engineer) Tj
0 -20 Td
(john.doe@email.com) Tj
0 -20 Td
(+1-555-123-4567) Tj
0 -20 Td
(5 years experience in Python and JavaScript) Tj
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
450
%%EOF'''
    
    with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
        tmp.write(pdf_content)
        tmp.flush()
        return tmp.name


async def test_text_extraction():
    """Test text extraction from PDF"""
    print("=== Testing Text Extraction ===")
    
    service = ResumeProcessingService()
    pdf_path = await create_sample_pdf()
    
    try:
        text, confidence = await service._extract_from_pdf(pdf_path)
        print(f"Extracted text: {text}")
        print(f"Confidence score: {confidence}")
        print("✓ Text extraction successful")
    except Exception as e:
        print(f"✗ Text extraction failed: {e}")
    finally:
        os.unlink(pdf_path)


async def test_fallback_parsing():
    """Test fallback parsing functionality"""
    print("\n=== Testing Fallback Parsing ===")
    
    service = ResumeProcessingService()
    sample_text = """
    John Doe
    Software Engineer
    john.doe@email.com
    +1-555-123-4567
    
    EXPERIENCE
    Senior Software Engineer at Tech Corp (2020-2023)
    - Developed web applications using Python and React
    - Led a team of 5 developers
    - Improved system performance by 40%
    
    EDUCATION
    Bachelor of Science in Computer Science
    University of Technology (2016-2020)
    GPA: 3.8/4.0
    
    SKILLS
    Programming Languages: Python, JavaScript, Java
    Frameworks: React, Django, Flask
    Databases: PostgreSQL, MongoDB
    """
    
    try:
        parsed_data = await service._fallback_parsing(sample_text)
        print(f"Contact info: {parsed_data.contact_info}")
        print(f"Summary: {parsed_data.summary}")
        print("✓ Fallback parsing successful")
    except Exception as e:
        print(f"✗ Fallback parsing failed: {e}")


async def test_data_validation():
    """Test resume data validation"""
    print("\n=== Testing Data Validation ===")
    
    service = ResumeProcessingService()
    
    # Test valid data
    valid_data = ParsedResumeData(
        contact_info=ContactInfo(
            name="John Doe",
            email="john.doe@email.com",
            phone="+1-555-123-4567"
        ),
        summary="Experienced software engineer with 5 years in web development",
        work_experience=[
            WorkExperience(
                company="Tech Corp",
                position="Senior Software Engineer",
                start_date="2020-01",
                end_date="2023-12",
                description="Led development of web applications"
            )
        ]
    )
    
    try:
        validation_result = await service._validate_parsed_data(valid_data)
        print(f"Validation result: {validation_result.is_valid}")
        print(f"Confidence score: {validation_result.confidence_score}")
        print(f"Errors: {len(validation_result.errors)}")
        print(f"Warnings: {len(validation_result.warnings)}")
        print("✓ Data validation successful")
    except Exception as e:
        print(f"✗ Data validation failed: {e}")
    
    # Test invalid data
    invalid_data = ParsedResumeData(
        contact_info=None,  # Missing contact info
        summary="No contact information provided"
    )
    
    try:
        validation_result = await service._validate_parsed_data(invalid_data)
        print(f"\nInvalid data validation: {validation_result.is_valid}")
        print(f"Errors: {[error.message for error in validation_result.errors]}")
        print("✓ Invalid data validation successful")
    except Exception as e:
        print(f"✗ Invalid data validation failed: {e}")


async def test_file_validation():
    """Test file validation functionality"""
    print("\n=== Testing File Validation ===")
    
    service = ResumeProcessingService()
    
    # Test valid PDF file
    pdf_path = await create_sample_pdf()
    
    try:
        from fastapi import UploadFile
        
        with open(pdf_path, 'rb') as f:
            upload_file = UploadFile(
                filename="test_resume.pdf",
                file=f,
                content_type="application/pdf"
            )
            
            await service._validate_file(upload_file)
            print("✓ Valid PDF file validation successful")
    except Exception as e:
        print(f"✗ Valid PDF file validation failed: {e}")
    finally:
        os.unlink(pdf_path)
    
    # Test invalid file type
    try:
        with tempfile.NamedTemporaryFile(suffix='.txt') as tmp:
            tmp.write(b'This is a text file')
            tmp.flush()
            
            with open(tmp.name, 'rb') as f:
                upload_file = UploadFile(
                    filename="test.txt",
                    file=f,
                    content_type="text/plain"
                )
                
                await service._validate_file(upload_file)
                print("✗ Invalid file type validation should have failed")
    except ValidationError as e:
        print(f"✓ Invalid file type validation successful: {e}")
    except Exception as e:
        print(f"✗ Invalid file type validation failed unexpectedly: {e}")


async def test_gemini_prompt_creation():
    """Test Gemini API prompt creation"""
    print("\n=== Testing Gemini Prompt Creation ===")
    
    service = ResumeProcessingService()
    sample_text = "John Doe\nSoftware Engineer\njohn@email.com"
    
    try:
        prompt = service._create_parsing_prompt(sample_text)
        print(f"Prompt length: {len(prompt)} characters")
        print("Prompt preview:")
        print(prompt[:200] + "..." if len(prompt) > 200 else prompt)
        print("✓ Gemini prompt creation successful")
    except Exception as e:
        print(f"✗ Gemini prompt creation failed: {e}")


async def main():
    """Run all demo tests"""
    print("Resume Processing Service Demo")
    print("=" * 50)
    
    await test_file_validation()
    await test_text_extraction()
    await test_fallback_parsing()
    await test_data_validation()
    await test_gemini_prompt_creation()
    
    print("\n" + "=" * 50)
    print("Demo completed!")


if __name__ == "__main__":
    asyncio.run(main())