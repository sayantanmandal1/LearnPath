"""
Resume upload and processing API endpoints
"""
import logging
from typing import List
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.api.dependencies import get_current_user
from app.models.user import User
from app.schemas.resume import (
    ResumeUploadResponse, ResumeProcessingResult, ManualResumeEntry,
    ResumeProcessingStats, ProcessingStatus
)
from app.services.resume_processing_service import ResumeProcessingService
from app.core.exceptions import ValidationError, ProcessingError

logger = logging.getLogger(__name__)

router = APIRouter()
resume_service = ResumeProcessingService()


@router.post("/upload", response_model=ResumeUploadResponse)
async def upload_resume(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Upload resume file for processing
    
    Accepts PDF, DOC, and DOCX files up to 10MB.
    Processing happens in the background after upload.
    """
    try:
        # Upload file and create database record
        resume_data = await resume_service.upload_resume(file, current_user.id, db)
        
        # Schedule background processing
        background_tasks.add_task(
            process_resume_background,
            resume_data.id,
            db
        )
        
        return ResumeUploadResponse(
            id=resume_data.id,
            user_id=resume_data.user_id,
            original_filename=resume_data.original_filename,
            file_size=resume_data.file_size,
            file_type=resume_data.file_type,
            processing_status=resume_data.processing_status,
            message="Resume uploaded successfully. Processing will begin shortly.",
            created_at=resume_data.created_at
        )
        
    except ValidationError as e:
        logger.warning(f"Resume upload validation failed: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except ProcessingError as e:
        logger.error(f"Resume upload processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error during resume upload: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred")


@router.get("/{resume_id}", response_model=ResumeProcessingResult)
async def get_resume_processing_result(
    resume_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Get resume processing result by ID
    
    Returns the current processing status and extracted data if available.
    """
    try:
        resume_data = await resume_service.get_resume_by_id(resume_id, db)
        
        if not resume_data:
            raise HTTPException(status_code=404, detail="Resume not found")
        
        if resume_data.user_id != current_user.id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Convert parsed sections back to structured data
        parsed_data = None
        if resume_data.parsed_sections:
            from app.schemas.resume import ParsedResumeData
            parsed_data = ParsedResumeData(**resume_data.parsed_sections)
        
        return ResumeProcessingResult(
            id=resume_data.id,
            user_id=resume_data.user_id,
            original_filename=resume_data.original_filename,
            processing_status=resume_data.processing_status,
            extracted_text=resume_data.extracted_text,
            extraction_confidence=resume_data.extraction_confidence,
            parsed_data=parsed_data,
            error_message=resume_data.error_message,
            processing_started_at=resume_data.processing_started_at,
            processing_completed_at=resume_data.processing_completed_at,
            created_at=resume_data.created_at,
            updated_at=resume_data.updated_at
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving resume {resume_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve resume data")


@router.get("/", response_model=List[ResumeProcessingResult])
async def get_user_resumes(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Get all resumes for the current user
    
    Returns a list of all resume processing results for the authenticated user.
    """
    try:
        resumes = await resume_service.get_user_resumes(current_user.id, db)
        
        results = []
        for resume_data in resumes:
            # Convert parsed sections back to structured data
            parsed_data = None
            if resume_data.parsed_sections:
                from app.schemas.resume import ParsedResumeData
                parsed_data = ParsedResumeData(**resume_data.parsed_sections)
            
            results.append(ResumeProcessingResult(
                id=resume_data.id,
                user_id=resume_data.user_id,
                original_filename=resume_data.original_filename,
                processing_status=resume_data.processing_status,
                extracted_text=resume_data.extracted_text,
                extraction_confidence=resume_data.extraction_confidence,
                parsed_data=parsed_data,
                error_message=resume_data.error_message,
                processing_started_at=resume_data.processing_started_at,
                processing_completed_at=resume_data.processing_completed_at,
                created_at=resume_data.created_at,
                updated_at=resume_data.updated_at
            ))
        
        return results
        
    except Exception as e:
        logger.error(f"Error retrieving user resumes: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve resumes")


@router.post("/{resume_id}/reprocess")
async def reprocess_resume(
    resume_id: str,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Reprocess a resume that failed or needs updating
    
    Useful when Gemini API was unavailable or processing failed.
    """
    try:
        resume_data = await resume_service.get_resume_by_id(resume_id, db)
        
        if not resume_data:
            raise HTTPException(status_code=404, detail="Resume not found")
        
        if resume_data.user_id != current_user.id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        if resume_data.processing_status == ProcessingStatus.PROCESSING:
            raise HTTPException(status_code=400, detail="Resume is already being processed")
        
        # Schedule background reprocessing
        background_tasks.add_task(
            process_resume_background,
            resume_id,
            db
        )
        
        return {"message": "Resume reprocessing started", "resume_id": resume_id}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error reprocessing resume {resume_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to start reprocessing")


@router.post("/{resume_id}/manual-entry", response_model=ResumeProcessingResult)
async def submit_manual_resume_data(
    resume_id: str,
    manual_data: ManualResumeEntry,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Submit manual resume data when automatic processing fails
    
    Allows users to manually enter their resume information if automatic
    extraction and parsing fails.
    """
    try:
        resume_data = await resume_service.get_resume_by_id(resume_id, db)
        
        if not resume_data:
            raise HTTPException(status_code=404, detail="Resume not found")
        
        if resume_data.user_id != current_user.id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Update resume with manual data
        from app.schemas.resume import ParsedResumeData
        parsed_data = ParsedResumeData(
            contact_info=manual_data.contact_info,
            summary=manual_data.summary,
            work_experience=manual_data.work_experience,
            education=manual_data.education,
            skills=manual_data.skills,
            certifications=manual_data.certifications
        )
        
        # Create validation result for manual entry
        from app.schemas.resume import ResumeValidationResult
        validation_result = ResumeValidationResult(
            is_valid=True,
            confidence_score=1.0  # Manual entry gets full confidence
        )
        
        await resume_service._update_resume_with_results(
            db, resume_id, 
            extracted_text="Manual entry",
            confidence=1.0,
            parsed_data=parsed_data,
            validation_result=validation_result
        )
        
        # Update status to manual entry
        await resume_service._update_processing_status(
            db, resume_id, ProcessingStatus.MANUAL_ENTRY
        )
        
        # Return updated resume data
        updated_resume = await resume_service.get_resume_by_id(resume_id, db)
        
        return ResumeProcessingResult(
            id=updated_resume.id,
            user_id=updated_resume.user_id,
            original_filename=updated_resume.original_filename,
            processing_status=updated_resume.processing_status,
            extracted_text=updated_resume.extracted_text,
            extraction_confidence=updated_resume.extraction_confidence,
            parsed_data=parsed_data,
            error_message=updated_resume.error_message,
            processing_started_at=updated_resume.processing_started_at,
            processing_completed_at=updated_resume.processing_completed_at,
            created_at=updated_resume.created_at,
            updated_at=updated_resume.updated_at
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error submitting manual resume data for {resume_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to submit manual data")


@router.delete("/{resume_id}")
async def delete_resume(
    resume_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Delete a resume and its associated file
    
    Permanently removes the resume record and uploaded file.
    """
    try:
        resume_data = await resume_service.get_resume_by_id(resume_id, db)
        
        if not resume_data:
            raise HTTPException(status_code=404, detail="Resume not found")
        
        if resume_data.user_id != current_user.id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Delete file from disk
        import os
        if os.path.exists(resume_data.file_path):
            os.remove(resume_data.file_path)
        
        # Delete database record
        await db.delete(resume_data)
        await db.commit()
        
        return {"message": "Resume deleted successfully", "resume_id": resume_id}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting resume {resume_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to delete resume")


async def process_resume_background(resume_id: str, db: AsyncSession):
    """
    Background task for processing resume
    
    This function runs asynchronously after file upload to extract
    and parse resume content using Gemini API.
    """
    try:
        await resume_service.process_resume(resume_id, db)
        logger.info(f"Background processing completed for resume: {resume_id}")
    except Exception as e:
        logger.error(f"Background processing failed for resume {resume_id}: {str(e)}")


@router.get("/stats/processing", response_model=ResumeProcessingStats)
async def get_processing_stats(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Get resume processing statistics for the current user
    
    Returns statistics about resume processing success rates and performance.
    """
    try:
        from sqlalchemy import func, case
        from app.models.resume import ResumeData, ProcessingStatus
        
        # Get processing statistics
        result = await db.execute(
            db.query(
                func.count(ResumeData.id).label('total_processed'),
                func.sum(
                    case(
                        (ResumeData.processing_status == ProcessingStatus.COMPLETED, 1),
                        else_=0
                    )
                ).label('successful_extractions'),
                func.sum(
                    case(
                        (ResumeData.processing_status == ProcessingStatus.FAILED, 1),
                        else_=0
                    )
                ).label('failed_extractions'),
                func.sum(
                    case(
                        (ResumeData.processing_status == ProcessingStatus.MANUAL_ENTRY, 1),
                        else_=0
                    )
                ).label('manual_entries'),
                func.avg(ResumeData.extraction_confidence).label('avg_confidence')
            ).filter(ResumeData.user_id == current_user.id)
        )
        
        stats = result.first()
        
        return ResumeProcessingStats(
            total_processed=stats.total_processed or 0,
            successful_extractions=stats.successful_extractions or 0,
            failed_extractions=stats.failed_extractions or 0,
            manual_entries=stats.manual_entries or 0,
            average_processing_time=0.0,  # Would need to calculate from timestamps
            average_confidence_score=float(stats.avg_confidence or 0.0)
        )
        
    except Exception as e:
        logger.error(f"Error retrieving processing stats: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve statistics")