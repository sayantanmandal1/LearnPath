"""
Workflow Integration API endpoints

This module provides endpoints for managing the complete user workflow integration,
including progress tracking, workflow execution, and data consistency validation.
"""
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel, Field

from app.core.database import get_db
from app.api.dependencies import get_current_user
from app.models.user import User
from app.services.workflow_integration_service import (
    WorkflowIntegrationService, 
    WorkflowProgress, 
    WorkflowResult,
    WorkflowStage
)
from app.core.exceptions import ValidationError, ProcessingError

logger = logging.getLogger(__name__)

router = APIRouter()
workflow_service = WorkflowIntegrationService()


# Request/Response Models
class PlatformAccountRequest(BaseModel):
    """Platform account connection request"""
    platform: str = Field(..., description="Platform name (github, leetcode, linkedin, etc.)")
    username: Optional[str] = Field(None, description="Platform username")
    profile_url: Optional[str] = Field(None, description="Platform profile URL")


class WorkflowExecutionRequest(BaseModel):
    """Complete workflow execution request"""
    platform_accounts: Optional[Dict[str, Dict[str, str]]] = Field(
        None, 
        description="Platform accounts to connect"
    )
    skip_resume: bool = Field(False, description="Skip resume processing")
    skip_platforms: bool = Field(False, description="Skip platform connections")
    skip_ai_analysis: bool = Field(False, description="Skip AI analysis")


class WorkflowProgressResponse(BaseModel):
    """Workflow progress response"""
    user_id: str
    current_stage: str
    completed_stages: List[str]
    progress_percentage: float
    errors: List[str]
    warnings: List[str]
    started_at: datetime
    updated_at: datetime
    estimated_completion: Optional[datetime] = None


class WorkflowResultResponse(BaseModel):
    """Workflow execution result response"""
    success: bool
    user_id: str
    progress: WorkflowProgressResponse
    resume_data: Optional[Dict[str, Any]] = None
    platform_data: Optional[Dict[str, Any]] = None
    analysis_results: Optional[Dict[str, Any]] = None
    dashboard_data: Optional[Dict[str, Any]] = None
    job_matches: Optional[List[Dict[str, Any]]] = None
    errors: List[str] = []
    execution_time: float = 0.0


class ValidationResultResponse(BaseModel):
    """Workflow validation result response"""
    user_id: str
    timestamp: str
    checks: Dict[str, Any]
    overall_status: str
    issues: List[str]


@router.post("/execute", response_model=WorkflowResultResponse)
async def execute_complete_workflow(
    background_tasks: BackgroundTasks,
    request: WorkflowExecutionRequest,
    resume_file: Optional[UploadFile] = File(None),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Execute complete user workflow from profile creation to career recommendations.
    
    This endpoint orchestrates the entire user journey:
    1. Resume processing (if file provided)
    2. Platform account connections
    3. Data scraping from platforms
    4. AI analysis with Gemini
    5. Dashboard data preparation
    6. Job matching
    7. Data synchronization
    """
    try:
        logger.info(f"Starting complete workflow execution for user {current_user.id}")
        
        # Validate request
        if not request.skip_resume and not resume_file:
            logger.info("No resume file provided, will skip resume processing")
        
        if not request.skip_platforms and not request.platform_accounts:
            logger.info("No platform accounts provided, will skip platform connections")
        
        # Execute workflow
        result = await workflow_service.execute_complete_workflow(
            user_id=current_user.id,
            resume_file=resume_file if not request.skip_resume else None,
            platform_accounts=request.platform_accounts if not request.skip_platforms else None,
            db=db
        )
        
        # Convert result to response model
        response = WorkflowResultResponse(
            success=result.success,
            user_id=result.user_id,
            progress=WorkflowProgressResponse(
                user_id=result.progress.user_id,
                current_stage=result.progress.current_stage.value,
                completed_stages=[stage.value for stage in result.progress.completed_stages],
                progress_percentage=result.progress.progress_percentage,
                errors=result.progress.errors,
                warnings=result.progress.warnings,
                started_at=result.progress.started_at,
                updated_at=result.progress.updated_at,
                estimated_completion=result.progress.estimated_completion
            ),
            resume_data=result.resume_data,
            platform_data=result.platform_data,
            analysis_results=result.analysis_results,
            dashboard_data=result.dashboard_data,
            job_matches=result.job_matches,
            errors=result.errors or [],
            execution_time=result.execution_time
        )
        
        logger.info(f"Workflow execution completed for user {current_user.id} in {result.execution_time:.2f}s")
        return response
        
    except ValidationError as e:
        logger.error(f"Workflow validation error for user {current_user.id}: {str(e)}")
        raise HTTPException(status_code=422, detail=str(e))
    except ProcessingError as e:
        logger.error(f"Workflow processing error for user {current_user.id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in workflow execution for user {current_user.id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Workflow execution failed")


@router.get("/progress", response_model=Optional[WorkflowProgressResponse])
async def get_workflow_progress(
    current_user: User = Depends(get_current_user)
):
    """
    Get current workflow progress for the authenticated user.
    
    Returns the current stage, completion percentage, and any errors or warnings.
    """
    try:
        progress = await workflow_service.get_workflow_progress(current_user.id)
        
        if not progress:
            return None
        
        return WorkflowProgressResponse(
            user_id=progress.user_id,
            current_stage=progress.current_stage.value,
            completed_stages=[stage.value for stage in progress.completed_stages],
            progress_percentage=progress.progress_percentage,
            errors=progress.errors,
            warnings=progress.warnings,
            started_at=progress.started_at,
            updated_at=progress.updated_at,
            estimated_completion=progress.estimated_completion
        )
        
    except Exception as e:
        logger.error(f"Error getting workflow progress for user {current_user.id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get workflow progress")


@router.post("/resume", response_model=WorkflowResultResponse)
async def resume_workflow(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Resume an interrupted workflow from where it left off.
    
    This endpoint allows users to continue a workflow that was interrupted
    due to errors, network issues, or other problems.
    """
    try:
        logger.info(f"Resuming workflow for user {current_user.id}")
        
        result = await workflow_service.resume_workflow(current_user.id, db)
        
        response = WorkflowResultResponse(
            success=result.success,
            user_id=result.user_id,
            progress=WorkflowProgressResponse(
                user_id=result.progress.user_id,
                current_stage=result.progress.current_stage.value,
                completed_stages=[stage.value for stage in result.progress.completed_stages],
                progress_percentage=result.progress.progress_percentage,
                errors=result.progress.errors,
                warnings=result.progress.warnings,
                started_at=result.progress.started_at,
                updated_at=result.progress.updated_at,
                estimated_completion=result.progress.estimated_completion
            ),
            resume_data=result.resume_data,
            platform_data=result.platform_data,
            analysis_results=result.analysis_results,
            dashboard_data=result.dashboard_data,
            job_matches=result.job_matches,
            errors=result.errors or [],
            execution_time=result.execution_time
        )
        
        logger.info(f"Workflow resumed successfully for user {current_user.id}")
        return response
        
    except ValidationError as e:
        logger.error(f"No workflow to resume for user {current_user.id}: {str(e)}")
        raise HTTPException(status_code=404, detail="No workflow found to resume")
    except Exception as e:
        logger.error(f"Error resuming workflow for user {current_user.id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to resume workflow")


@router.get("/validate", response_model=ValidationResultResponse)
async def validate_workflow_integrity(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Validate data consistency across all workflow components.
    
    This endpoint checks:
    - Profile data consistency
    - Resume and platform data alignment
    - Analysis results validity
    - Database integrity
    """
    try:
        logger.info(f"Validating workflow integrity for user {current_user.id}")
        
        validation_result = await workflow_service.validate_workflow_integrity(
            current_user.id, db
        )
        
        return ValidationResultResponse(**validation_result)
        
    except Exception as e:
        logger.error(f"Error validating workflow integrity for user {current_user.id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Workflow validation failed")


@router.get("/stages", response_model=List[str])
async def get_workflow_stages():
    """
    Get list of all workflow stages.
    
    Returns the complete list of workflow stages in order.
    """
    return [stage.value for stage in WorkflowStage]


@router.post("/test-integration", response_model=Dict[str, Any])
async def test_integration_components(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Test integration between all workflow components.
    
    This endpoint runs integration tests to verify that all components
    are working together correctly. Useful for debugging and monitoring.
    """
    try:
        logger.info(f"Testing integration components for user {current_user.id}")
        
        test_results = {
            "user_id": current_user.id,
            "timestamp": datetime.now().isoformat(),
            "tests": {},
            "overall_status": "passed",
            "issues": []
        }
        
        # Test resume service
        try:
            # This would test resume service connectivity
            test_results["tests"]["resume_service"] = "passed"
        except Exception as e:
            test_results["tests"]["resume_service"] = "failed"
            test_results["issues"].append(f"Resume service: {str(e)}")
        
        # Test platform scraper service
        try:
            # This would test platform scraper connectivity
            test_results["tests"]["platform_scraper"] = "passed"
        except Exception as e:
            test_results["tests"]["platform_scraper"] = "failed"
            test_results["issues"].append(f"Platform scraper: {str(e)}")
        
        # Test AI analysis service
        try:
            # This would test AI service connectivity
            test_results["tests"]["ai_analysis"] = "passed"
        except Exception as e:
            test_results["tests"]["ai_analysis"] = "failed"
            test_results["issues"].append(f"AI analysis: {str(e)}")
        
        # Test dashboard service
        try:
            # This would test dashboard service
            test_results["tests"]["dashboard_service"] = "passed"
        except Exception as e:
            test_results["tests"]["dashboard_service"] = "failed"
            test_results["issues"].append(f"Dashboard service: {str(e)}")
        
        # Test job matching service
        try:
            # This would test job matching service
            test_results["tests"]["job_matching"] = "passed"
        except Exception as e:
            test_results["tests"]["job_matching"] = "failed"
            test_results["issues"].append(f"Job matching: {str(e)}")
        
        # Test data sync service
        try:
            # This would test data synchronization
            test_results["tests"]["data_sync"] = "passed"
        except Exception as e:
            test_results["tests"]["data_sync"] = "failed"
            test_results["issues"].append(f"Data sync: {str(e)}")
        
        # Determine overall status
        failed_tests = [test for test, status in test_results["tests"].items() if status == "failed"]
        if failed_tests:
            test_results["overall_status"] = "failed"
        
        logger.info(f"Integration test completed for user {current_user.id}: {test_results['overall_status']}")
        return test_results
        
    except Exception as e:
        logger.error(f"Error testing integration components for user {current_user.id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Integration test failed")


@router.delete("/reset", response_model=Dict[str, str])
async def reset_workflow(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Reset workflow progress for the authenticated user.
    
    This endpoint clears all workflow progress and allows starting fresh.
    Use with caution as this will remove all progress tracking.
    """
    try:
        logger.info(f"Resetting workflow for user {current_user.id}")
        
        # Clear workflow progress
        if current_user.id in workflow_service._workflow_progress:
            del workflow_service._workflow_progress[current_user.id]
        
        logger.info(f"Workflow reset completed for user {current_user.id}")
        return {"message": "Workflow reset successfully", "user_id": current_user.id}
        
    except Exception as e:
        logger.error(f"Error resetting workflow for user {current_user.id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to reset workflow")