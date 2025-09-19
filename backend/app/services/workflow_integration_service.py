"""
Workflow Integration Service

This service orchestrates the complete user workflow from profile creation to career recommendations.
It ensures all components work together seamlessly and handles the integration between:
1. Resume processing
2. Platform data scraping
3. AI analysis
4. Dashboard data aggregation
5. Real-time job matching
"""
import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update
from fastapi import HTTPException

from app.models.user import User
from app.models.profile import UserProfile
from app.models.resume import ResumeData
from app.models.platform_account import PlatformAccount
from app.models.analysis_result import AnalysisResult, AnalysisType
from app.services.resume_processing_service import ResumeProcessingService
from app.services.external_apis.multi_platform_scraper import MultiPlatformScraper
from app.services.ai_analysis_service import AIAnalysisService
from app.services.dashboard_service import DashboardService
from app.services.data_sync_service import DataSyncService
from app.services.real_time_job_service import RealTimeJobService
from app.core.exceptions import ProcessingError, ValidationError

logger = logging.getLogger(__name__)


class WorkflowStage(str, Enum):
    """Workflow stages for tracking progress"""
    PROFILE_CREATION = "profile_creation"
    RESUME_PROCESSING = "resume_processing"
    PLATFORM_CONNECTION = "platform_connection"
    DATA_SCRAPING = "data_scraping"
    AI_ANALYSIS = "ai_analysis"
    DASHBOARD_PREPARATION = "dashboard_preparation"
    JOB_MATCHING = "job_matching"
    COMPLETED = "completed"


@dataclass
class WorkflowProgress:
    """Track workflow progress and status"""
    user_id: str
    current_stage: WorkflowStage
    completed_stages: List[WorkflowStage]
    progress_percentage: float
    errors: List[str]
    warnings: List[str]
    started_at: datetime
    updated_at: datetime
    estimated_completion: Optional[datetime] = None


@dataclass
class WorkflowResult:
    """Complete workflow execution result"""
    success: bool
    user_id: str
    progress: WorkflowProgress
    resume_data: Optional[Dict[str, Any]] = None
    platform_data: Optional[Dict[str, Any]] = None
    analysis_results: Optional[Dict[str, Any]] = None
    dashboard_data: Optional[Dict[str, Any]] = None
    job_matches: Optional[List[Dict[str, Any]]] = None
    errors: List[str] = None
    execution_time: float = 0.0


class WorkflowIntegrationService:
    """Service for orchestrating complete user workflow"""
    
    def __init__(self):
        self.resume_service = ResumeProcessingService()
        self.scraper_service = MultiPlatformScraper()
        self.ai_service = AIAnalysisService()
        # DashboardService will be instantiated when needed with proper db session
        self.dashboard_service = None
        self.sync_service = DataSyncService()
        self.job_service = RealTimeJobService()
        
        # Workflow progress tracking
        self._workflow_progress: Dict[str, WorkflowProgress] = {}
    
    async def execute_complete_workflow(
        self,
        user_id: str,
        resume_file: Optional[Any] = None,
        platform_accounts: Optional[Dict[str, Dict[str, str]]] = None,
        db: AsyncSession = None
    ) -> WorkflowResult:
        """Execute complete user workflow from start to finish"""
        
        start_time = datetime.now()
        workflow_result = WorkflowResult(
            success=False,
            user_id=user_id,
            progress=self._initialize_workflow_progress(user_id),
            errors=[]
        )
        
        try:
            logger.info(f"Starting complete workflow for user {user_id}")
            
            # Stage 1: Profile Creation/Validation
            await self._update_workflow_stage(user_id, WorkflowStage.PROFILE_CREATION)
            profile_data = await self._ensure_user_profile(user_id, db)
            
            # Stage 2: Resume Processing (if provided)
            if resume_file:
                await self._update_workflow_stage(user_id, WorkflowStage.RESUME_PROCESSING)
                workflow_result.resume_data = await self._process_resume(
                    user_id, resume_file, db
                )
            
            # Stage 3: Platform Connection (if provided)
            if platform_accounts:
                await self._update_workflow_stage(user_id, WorkflowStage.PLATFORM_CONNECTION)
                await self._connect_platforms(user_id, platform_accounts, db)
            
            # Stage 4: Data Scraping
            await self._update_workflow_stage(user_id, WorkflowStage.DATA_SCRAPING)
            workflow_result.platform_data = await self._scrape_platform_data(user_id, db)
            
            # Stage 5: AI Analysis
            await self._update_workflow_stage(user_id, WorkflowStage.AI_ANALYSIS)
            workflow_result.analysis_results = await self._perform_ai_analysis(user_id, db)
            
            # Stage 6: Dashboard Preparation
            await self._update_workflow_stage(user_id, WorkflowStage.DASHBOARD_PREPARATION)
            workflow_result.dashboard_data = await self._prepare_dashboard_data(user_id, db)
            
            # Stage 7: Job Matching
            await self._update_workflow_stage(user_id, WorkflowStage.JOB_MATCHING)
            workflow_result.job_matches = await self._match_jobs(user_id, db)
            
            # Stage 8: Completion
            await self._update_workflow_stage(user_id, WorkflowStage.COMPLETED)
            
            # Data Synchronization
            await self._synchronize_data(user_id, db)
            
            workflow_result.success = True
            workflow_result.execution_time = (datetime.now() - start_time).total_seconds()
            
            logger.info(f"Workflow completed successfully for user {user_id} in {workflow_result.execution_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Workflow failed for user {user_id}: {str(e)}")
            workflow_result.errors.append(str(e))
            workflow_result.execution_time = (datetime.now() - start_time).total_seconds()
            
            # Update progress with error
            if user_id in self._workflow_progress:
                self._workflow_progress[user_id].errors.append(str(e))
        
        workflow_result.progress = self._workflow_progress.get(user_id)
        return workflow_result
    
    async def get_workflow_progress(self, user_id: str) -> Optional[WorkflowProgress]:
        """Get current workflow progress for a user"""
        return self._workflow_progress.get(user_id)
    
    async def resume_workflow(self, user_id: str, db: AsyncSession) -> WorkflowResult:
        """Resume an interrupted workflow"""
        
        progress = self._workflow_progress.get(user_id)
        if not progress:
            raise ValidationError("No workflow found to resume")
        
        logger.info(f"Resuming workflow for user {user_id} from stage {progress.current_stage}")
        
        # Determine next stage and continue from there
        remaining_stages = self._get_remaining_stages(progress.current_stage)
        
        # Execute remaining stages
        return await self._execute_stages(user_id, remaining_stages, db)
    
    def _initialize_workflow_progress(self, user_id: str) -> WorkflowProgress:
        """Initialize workflow progress tracking"""
        progress = WorkflowProgress(
            user_id=user_id,
            current_stage=WorkflowStage.PROFILE_CREATION,
            completed_stages=[],
            progress_percentage=0.0,
            errors=[],
            warnings=[],
            started_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        self._workflow_progress[user_id] = progress
        return progress
    
    async def _update_workflow_stage(self, user_id: str, stage: WorkflowStage):
        """Update workflow progress to next stage"""
        if user_id not in self._workflow_progress:
            self._initialize_workflow_progress(user_id)
        
        progress = self._workflow_progress[user_id]
        
        # Mark previous stage as completed
        if progress.current_stage not in progress.completed_stages:
            progress.completed_stages.append(progress.current_stage)
        
        # Update to new stage
        progress.current_stage = stage
        progress.updated_at = datetime.now()
        
        # Calculate progress percentage
        total_stages = len(WorkflowStage)
        completed_count = len(progress.completed_stages)
        progress.progress_percentage = (completed_count / total_stages) * 100
        
        logger.info(f"User {user_id} workflow updated to stage {stage.value} ({progress.progress_percentage:.1f}%)")
    
    async def _ensure_user_profile(self, user_id: str, db: AsyncSession) -> Dict[str, Any]:
        """Ensure user profile exists and is properly initialized"""
        try:
            # Check if profile exists
            result = await db.execute(
                select(UserProfile).where(UserProfile.user_id == user_id)
            )
            profile = result.scalar_one_or_none()
            
            if not profile:
                # Create basic profile
                profile = UserProfile(
                    user_id=user_id,
                    created_at=datetime.now(),
                    updated_at=datetime.now()
                )
                db.add(profile)
                await db.commit()
                logger.info(f"Created new profile for user {user_id}")
            
            return {"profile_id": profile.id, "user_id": user_id}
            
        except Exception as e:
            logger.error(f"Failed to ensure user profile for {user_id}: {str(e)}")
            raise ProcessingError(f"Profile creation failed: {str(e)}")
    
    async def _process_resume(
        self, 
        user_id: str, 
        resume_file: Any, 
        db: AsyncSession
    ) -> Dict[str, Any]:
        """Process resume upload and extraction"""
        try:
            result = await self.resume_service.upload_resume(resume_file, user_id, db)
            logger.info(f"Resume processed successfully for user {user_id}")
            return {
                "resume_id": result.id,
                "extraction_status": result.processing_status,
                "extracted_data": result.extracted_data
            }
        except Exception as e:
            logger.warning(f"Resume processing failed for user {user_id}: {str(e)}")
            # Add warning but don't fail the workflow
            if user_id in self._workflow_progress:
                self._workflow_progress[user_id].warnings.append(f"Resume processing failed: {str(e)}")
            return {"status": "failed", "error": str(e)}
    
    async def _connect_platforms(
        self,
        user_id: str,
        platform_accounts: Dict[str, Dict[str, str]],
        db: AsyncSession
    ) -> Dict[str, Any]:
        """Connect and validate platform accounts"""
        connected_platforms = {}
        
        for platform, account_data in platform_accounts.items():
            try:
                # Create platform account record
                platform_account = PlatformAccount(
                    user_id=user_id,
                    platform=platform,
                    username=account_data.get("username"),
                    profile_url=account_data.get("profile_url"),
                    is_active=True,
                    created_at=datetime.now()
                )
                
                db.add(platform_account)
                await db.commit()
                
                connected_platforms[platform] = {
                    "status": "connected",
                    "account_id": platform_account.id
                }
                
                logger.info(f"Connected {platform} account for user {user_id}")
                
            except Exception as e:
                logger.warning(f"Failed to connect {platform} for user {user_id}: {str(e)}")
                connected_platforms[platform] = {
                    "status": "failed",
                    "error": str(e)
                }
        
        return connected_platforms
    
    async def _scrape_platform_data(self, user_id: str, db: AsyncSession) -> Dict[str, Any]:
        """Scrape data from connected platforms"""
        try:
            # Get connected platforms
            result = await db.execute(
                select(PlatformAccount).where(
                    PlatformAccount.user_id == user_id,
                    PlatformAccount.is_active == True
                )
            )
            platforms = result.scalars().all()
            
            if not platforms:
                logger.info(f"No platforms connected for user {user_id}")
                return {"platforms_scraped": 0, "data": {}}
            
            # Scrape data from each platform
            scraped_data = {}
            for platform in platforms:
                try:
                    data = await self.scraper_service.scrape_platform_data(
                        platform.platform,
                        platform.username or platform.profile_url
                    )
                    scraped_data[platform.platform] = data
                    
                    # Update platform account with scraped data
                    platform.scraped_data = data
                    platform.last_scraped_at = datetime.now()
                    platform.scraping_status = "completed"
                    
                except Exception as e:
                    logger.warning(f"Failed to scrape {platform.platform} for user {user_id}: {str(e)}")
                    platform.scraping_status = "failed"
                    platform.last_error = str(e)
            
            await db.commit()
            
            return {
                "platforms_scraped": len(scraped_data),
                "data": scraped_data
            }
            
        except Exception as e:
            logger.error(f"Platform scraping failed for user {user_id}: {str(e)}")
            return {"status": "failed", "error": str(e)}
    
    async def _perform_ai_analysis(self, user_id: str, db: AsyncSession) -> Dict[str, Any]:
        """Perform AI analysis on collected data"""
        try:
            # Aggregate all user data
            profile_data = await self.ai_service.aggregate_profile_data(user_id, db)
            
            # Perform AI analysis
            analysis_result = await self.ai_service.analyze_complete_profile(profile_data)
            
            # Store analysis results
            analysis_record = AnalysisResult(
                user_id=user_id,
                analysis_type=AnalysisType.COMPREHENSIVE,
                results=analysis_result,
                created_at=datetime.now()
            )
            
            db.add(analysis_record)
            await db.commit()
            
            logger.info(f"AI analysis completed for user {user_id}")
            return analysis_result
            
        except Exception as e:
            logger.error(f"AI analysis failed for user {user_id}: {str(e)}")
            # Provide fallback basic analysis
            return await self._generate_fallback_analysis(user_id, db)
    
    async def _prepare_dashboard_data(self, user_id: str, db: AsyncSession) -> Dict[str, Any]:
        """Prepare comprehensive dashboard data"""
        try:
            # Create DashboardService with the provided db session
            from app.services.dashboard_service import DashboardService
            dashboard_service = DashboardService(db)
            dashboard_data = await dashboard_service.get_dashboard_data(user_id, db)
            logger.info(f"Dashboard data prepared for user {user_id}")
            return dashboard_data
        except Exception as e:
            logger.error(f"Dashboard preparation failed for user {user_id}: {str(e)}")
            return {"status": "failed", "error": str(e)}
    
    async def _match_jobs(self, user_id: str, db: AsyncSession) -> List[Dict[str, Any]]:
        """Match jobs based on user profile"""
        try:
            job_matches = await self.job_service.get_matched_jobs(user_id, db)
            logger.info(f"Found {len(job_matches)} job matches for user {user_id}")
            return job_matches
        except Exception as e:
            logger.warning(f"Job matching failed for user {user_id}: {str(e)}")
            return []
    
    async def _synchronize_data(self, user_id: str, db: AsyncSession):
        """Synchronize data between Supabase and PostgreSQL"""
        try:
            await self.sync_service.sync_user_profile(user_id, db)
            logger.info(f"Data synchronized for user {user_id}")
        except Exception as e:
            logger.warning(f"Data synchronization failed for user {user_id}: {str(e)}")
            # Don't fail workflow for sync issues
    
    async def _generate_fallback_analysis(self, user_id: str, db: AsyncSession) -> Dict[str, Any]:
        """Generate basic analysis when AI analysis fails"""
        return {
            "status": "fallback",
            "skill_assessment": {
                "overall_score": 70,
                "technical_skills": {}
            },
            "career_recommendations": [
                {
                    "role": "Software Developer",
                    "match_score": 75,
                    "salary_range": "$80,000 - $120,000",
                    "growth_potential": "Medium"
                }
            ],
            "skill_gaps": [],
            "learning_paths": [
                {
                    "title": "Complete your profile for better recommendations",
                    "duration": "1 week",
                    "priority": "High"
                }
            ]
        }
    
    def _get_remaining_stages(self, current_stage: WorkflowStage) -> List[WorkflowStage]:
        """Get remaining workflow stages"""
        all_stages = list(WorkflowStage)
        current_index = all_stages.index(current_stage)
        return all_stages[current_index + 1:]
    
    async def _execute_stages(
        self, 
        user_id: str, 
        stages: List[WorkflowStage], 
        db: AsyncSession
    ) -> WorkflowResult:
        """Execute specific workflow stages"""
        # This would implement stage-specific execution logic
        # For now, return a basic result
        return WorkflowResult(
            success=True,
            user_id=user_id,
            progress=self._workflow_progress.get(user_id),
            errors=[]
        )
    
    async def validate_workflow_integrity(self, user_id: str, db: AsyncSession) -> Dict[str, Any]:
        """Validate data consistency across all workflow components"""
        validation_results = {
            "user_id": user_id,
            "timestamp": datetime.now().isoformat(),
            "checks": {},
            "overall_status": "passed",
            "issues": []
        }
        
        try:
            # Check profile existence
            profile_result = await db.execute(
                select(UserProfile).where(UserProfile.user_id == user_id)
            )
            profile = profile_result.scalar_one_or_none()
            validation_results["checks"]["profile_exists"] = profile is not None
            
            if not profile:
                validation_results["issues"].append("User profile not found")
                validation_results["overall_status"] = "failed"
                return validation_results
            
            # Check resume data
            resume_result = await db.execute(
                select(ResumeData).where(ResumeData.user_id == user_id)
            )
            resume = resume_result.scalar_one_or_none()
            validation_results["checks"]["resume_processed"] = resume is not None
            
            # Check platform connections
            platform_result = await db.execute(
                select(PlatformAccount).where(PlatformAccount.user_id == user_id)
            )
            platforms = platform_result.scalars().all()
            validation_results["checks"]["platforms_connected"] = len(platforms)
            
            # Check analysis results
            analysis_result = await db.execute(
                select(AnalysisResult).where(AnalysisResult.user_id == user_id)
            )
            analyses = analysis_result.scalars().all()
            validation_results["checks"]["analysis_completed"] = len(analyses) > 0
            
            # Validate data consistency
            if profile and resume:
                # Check if profile data matches resume data
                if hasattr(profile, 'email') and hasattr(resume, 'extracted_data'):
                    resume_email = resume.extracted_data.get('personal_info', {}).get('email')
                    if resume_email and profile.email != resume_email:
                        validation_results["issues"].append("Email mismatch between profile and resume")
            
            if validation_results["issues"]:
                validation_results["overall_status"] = "warning"
            
            logger.info(f"Workflow integrity validation completed for user {user_id}: {validation_results['overall_status']}")
            
        except Exception as e:
            logger.error(f"Workflow integrity validation failed for user {user_id}: {str(e)}")
            validation_results["overall_status"] = "error"
            validation_results["issues"].append(f"Validation error: {str(e)}")
        
        return validation_results