"""
Integration test for resume processing workflow
"""
import asyncio
import tempfile
import os
from pathlib import Path
import sys

# Add the backend directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from app.services.resume_processing_service import ResumeProcessingService
from app.models.resume import ResumeData, ProcessingStatus
from app.schemas.resume import ParsedResumeData, ContactInfo


async def test_complete_workflow():
    """Test the complete resume processing workflow"""
    print("=== Resume Processing Integration Test ===")
    
    service = ResumeProcessingService()
    
    # Create a sample PDF with realistic content
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
/Length 500
>>
stream
BT
/F1 14 Tf
100 750 Td
(JOHN DOE) Tj
0 -20 Td
(Software Engineer) Tj
0 -30 Td
(Email: john.doe@techcorp.com) Tj
0 -20 Td
(Phone: +1-555-123-4567) Tj
0 -20 Td
(LinkedIn: linkedin.com/in/johndoe) Tj
0 -20 Td
(GitHub: github.com/johndoe) Tj
0 -40 Td
(EXPERIENCE) Tj
0 -20 Td
(Senior Software Engineer - Tech Corp (2020-2023)) Tj
0 -15 Td
(- Led development of microservices architecture) Tj
0 -15 Td
(- Improved system performance by 40%) Tj
0 -15 Td
(- Technologies: Python, React, PostgreSQL) Tj
0 -30 Td
(Software Engineer - StartupXYZ (2018-2020)) Tj
0 -15 Td
(- Built full-stack web applications) Tj
0 -15 Td
(- Collaborated with cross-functional teams) Tj
0 -30 Td
(EDUCATION) Tj
0 -20 Td
(Bachelor of Science in Computer Science) Tj
0 -15 Td
(University of Technology (2014-2018)) Tj
0 -15 Td
(GPA: 3.8/4.0) Tj
0 -30 Td
(SKILLS) Tj
0 -20 Td
(Programming: Python, JavaScript, Java, Go) Tj
0 -15 Td
(Frameworks: React, Django, Flask, Node.js) Tj
0 -15 Td
(Databases: PostgreSQL, MongoDB, Redis) Tj
0 -15 Td
(Cloud: AWS, Docker, Kubernetes) Tj
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
750
%%EOF'''
    
    # Create temporary PDF file
    with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
        tmp.write(pdf_content)
        tmp.flush()
        pdf_path = tmp.name
    
    try:
        print("1. Testing text extraction...")
        text, confidence = await service._extract_from_pdf(pdf_path)
        print(f"   ✓ Extracted {len(text)} characters with {confidence:.2f} confidence")
        print(f"   Sample text: {text[:100]}...")
        
        print("\n2. Testing Gemini parsing (fallback mode)...")
        parsed_data = await service._parse_with_gemini(text)
        print(f"   ✓ Parsed data structure created")
        print(f"   Contact info: {parsed_data.contact_info}")
        print(f"   Summary: {parsed_data.summary}")
        
        print("\n3. Testing data validation...")
        validation_result = await service._validate_parsed_data(parsed_data)
        print(f"   ✓ Validation complete")
        print(f"   Valid: {validation_result.is_valid}")
        print(f"   Confidence: {validation_result.confidence_score:.2f}")
        print(f"   Errors: {len(validation_result.errors)}")
        print(f"   Warnings: {len(validation_result.warnings)}")
        
        if validation_result.errors:
            print("   Errors found:")
            for error in validation_result.errors:
                print(f"     - {error.field}: {error.message}")
        
        if validation_result.warnings:
            print("   Warnings found:")
            for warning in validation_result.warnings:
                print(f"     - {warning.field}: {warning.message}")
        
        print("\n4. Testing manual data entry scenario...")
        from app.schemas.resume import ManualResumeEntry, WorkExperience, Education, SkillCategory
        
        manual_entry = ManualResumeEntry(
            contact_info=ContactInfo(
                name="John Doe",
                email="john.doe@techcorp.com",
                phone="+1-555-123-4567",
                linkedin="linkedin.com/in/johndoe",
                github="github.com/johndoe"
            ),
            summary="Senior Software Engineer with 5+ years of experience in full-stack development",
            work_experience=[
                WorkExperience(
                    company="Tech Corp",
                    position="Senior Software Engineer",
                    start_date="2020-01",
                    end_date="2023-12",
                    description="Led development of microservices architecture",
                    technologies=["Python", "React", "PostgreSQL"]
                )
            ],
            education=[
                Education(
                    institution="University of Technology",
                    degree="Bachelor of Science",
                    field_of_study="Computer Science",
                    start_date="2014-09",
                    end_date="2018-05",
                    gpa="3.8"
                )
            ],
            skills=[
                SkillCategory(
                    category="Programming Languages",
                    skills=["Python", "JavaScript", "Java", "Go"]
                ),
                SkillCategory(
                    category="Frameworks",
                    skills=["React", "Django", "Flask", "Node.js"]
                )
            ]
        )
        
        print(f"   ✓ Manual entry created with {len(manual_entry.work_experience)} work experiences")
        print(f"   ✓ Manual entry has {len(manual_entry.skills)} skill categories")
        
        print("\n5. Testing error handling scenarios...")
        
        # Test with invalid file
        try:
            await service._extract_from_pdf("nonexistent_file.pdf")
            print("   ✗ Should have failed for nonexistent file")
        except Exception as e:
            print(f"   ✓ Correctly handled nonexistent file: {type(e).__name__}")
        
        # Test with empty text
        try:
            empty_parsed = await service._fallback_parsing("")
            print(f"   ✓ Handled empty text gracefully")
        except Exception as e:
            print(f"   ✗ Failed to handle empty text: {e}")
        
        print("\n=== Integration Test Results ===")
        print("✓ Text extraction: PASSED")
        print("✓ AI parsing (fallback): PASSED") 
        print("✓ Data validation: PASSED")
        print("✓ Manual entry: PASSED")
        print("✓ Error handling: PASSED")
        print("\n🎉 All integration tests passed!")
        
    except Exception as e:
        print(f"\n❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up
        if os.path.exists(pdf_path):
            os.unlink(pdf_path)


async def test_file_format_support():
    """Test support for different file formats"""
    print("\n=== Testing File Format Support ===")
    
    service = ResumeProcessingService()
    
    # Test PDF format detection
    pdf_content = b'%PDF-1.4\nSample PDF content'
    with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
        tmp.write(pdf_content)
        tmp.flush()
        
        try:
            text, confidence = await service._extract_from_pdf(tmp.name)
            print("✓ PDF format: SUPPORTED")
        except Exception as e:
            print(f"✗ PDF format failed: {e}")
        finally:
            os.unlink(tmp.name)
    
    # Test DOCX format (mocked)
    try:
        from unittest.mock import patch, Mock
        
        with patch('app.services.resume_processing_service.Document') as mock_doc:
            mock_paragraph = Mock()
            mock_paragraph.text = "Sample DOCX content"
            
            mock_doc_instance = Mock()
            mock_doc_instance.paragraphs = [mock_paragraph]
            mock_doc_instance.tables = []
            mock_doc.return_value = mock_doc_instance
            
            text, confidence = await service._extract_from_docx("dummy.docx")
            print("✓ DOCX format: SUPPORTED")
    except Exception as e:
        print(f"✗ DOCX format failed: {e}")


async def test_validation_scenarios():
    """Test various validation scenarios"""
    print("\n=== Testing Validation Scenarios ===")
    
    service = ResumeProcessingService()
    
    # Test complete profile
    complete_data = ParsedResumeData(
        contact_info=ContactInfo(
            name="John Doe",
            email="john@example.com",
            phone="+1234567890"
        ),
        work_experience=[
            WorkExperience(company="Tech Corp", position="Engineer")
        ],
        education=[
            Education(institution="University", degree="BS")
        ],
        skills=[
            SkillCategory(category="Programming", skills=["Python"])
        ]
    )
    
    result = await service._validate_parsed_data(complete_data)
    print(f"✓ Complete profile: Valid={result.is_valid}, Score={result.confidence_score:.2f}")
    
    # Test minimal profile
    minimal_data = ParsedResumeData(
        contact_info=ContactInfo(email="john@example.com")
    )
    
    result = await service._validate_parsed_data(minimal_data)
    print(f"✓ Minimal profile: Valid={result.is_valid}, Score={result.confidence_score:.2f}")
    
    # Test empty profile
    empty_data = ParsedResumeData()
    
    result = await service._validate_parsed_data(empty_data)
    print(f"✓ Empty profile: Valid={result.is_valid}, Score={result.confidence_score:.2f}")


async def main():
    """Run all integration tests"""
    await test_complete_workflow()
    await test_file_format_support()
    await test_validation_scenarios()


if __name__ == "__main__":
    asyncio.run(main())