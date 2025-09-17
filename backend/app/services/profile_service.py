"""
User Profile Service for multi-source data integration and aggregation.

This service handles:
- Profile creation with multi-source data integration
- Skill profile merging with confidence scoring
- Unified user profile generation from multiple data sources
- Profile update mechanisms and change tracking
- Data consistency validation and conflict resolution
"""

import logging
import asyncio
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from uuid import uuid4

from sqlalchemy.ext.asyncio import AsyncSession
from fastapi import UploadFile

from app.models.profile import UserProfile
from app.schemas.profile import (
    ProfileCreate, ProfileUpdate, ProfileResponse, 
    ResumeUpload, SkillExtraction, PlatformDataUpdate
)
from app.repositories.profile import ProfileRepository
from app.services.external_apis.integration_service import (
    ExternalAPIIntegrationService, ProfileExtractionRequest, ProfileExtractionResult
)
from app.core.exceptions import ValidationError, NotFoundError, ConflictError

# Import NLP engine for resume processing
import sys
import os
# Check if ML dependencies should be disabled
DISABLE_ML = os.getenv('DISABLE_ML', 'false').lower() == 'true'

if not DISABLE_ML:
    sys.path.append(os.path.join(os.path.dirname(__file__), '../../..', 'machinelearningmodel'))
    
    try:
        from nlp_engine import NLPEngine
        from models import ResumeData, SkillExtraction as MLSkillExtraction
    except ImportError as e:
        # Fallback if ML models not available
        print(f"Warning: ML models not available: {e}")
        NLPEngine = None
        ResumeData = None
        MLSkillExtraction = None
else:
    print("ML dependencies disabled via environment variable")
    NLPEngine = None
    ResumeData = None
    MLSkillExtraction = None

logger = logging.getLogger(__name__)


class SkillMergeResult:
    """Result of skill merging operation."""
    
    def __init__(self):
        self.merged_skills: Dict[str, float] = {}
        self.conflicts: List[Dict[str, Any]] = []
        self.sources: Dict[str, List[str]] = {}
        self.confidence_scores: Dict[str, float] = {}


class ProfileChangeTracker:
    """Tracks changes to user profiles for audit and rollback purposes."""
    
    def __init__(self):
        self.changes: List[Dict[str, Any]] = []
    
    def track_change(self, field: str, old_value: Any, new_value: Any, source: str):
        """Track a field change."""
        self.changes.append({
            'field': field,
            'old_value': old_value,
            'new_value': new_value,
            'source': source,
            'timestamp': datetime.utcnow()
        })
    
    def get_changes(self) -> List[Dict[str, Any]]:
        """Get all tracked changes."""
        return self.changes.copy()


class UserProfileService:
    """Service for comprehensive user profile management and data aggregation."""
    
    def __init__(self, github_token: Optional[str] = None):
        self.profile_repo = ProfileRepository()
        self.external_api_service = ExternalAPIIntegrationService(github_token=github_token)
        self.nlp_engine = NLPEngine() if NLPEngine else None
        
        # Skill merging configuration
        self.skill_confidence_weights = {
            'resume': 0.9,
            'github': 0.8,
            'leetcode': 0.7,
            'linkedin': 0.6,
            'manual': 1.0
        }
        
        # Conflict resolution strategies
        self.conflict_resolution_strategies = {
            'skills': 'merge_with_confidence',
            'experience_years': 'take_highest',
            'current_role': 'take_most_recent',
            'location': 'take_most_recent'
        }
        
        logger.info("UserProfileService initialized")
    
    async def create_profile_with_integration(
        self,
        db: AsyncSession,
        user_id: str,
        profile_data: ProfileCreate,
        resume_file: Optional[UploadFile] = None
    ) -> ProfileResponse:
        """
        Create user profile with multi-source data integration.
        
        Args:
            db: Database session
            user_id: User ID
            profile_data: Basic profile data
            resume_file: Optional resume file for parsing
            
        Returns:
            Complete ProfileResponse with integrated data
        """
        logger.info(f"Creating profile for user {user_id} with multi-source integration")
        
        try:
            # Check if profile already exists
            existing_profile = await self.profile_repo.get_by_user_id(db, user_id)
            if existing_profile:
                raise ConflictError(f"Profile already exists for user {user_id}")
            
            # Initialize change tracker
            change_tracker = ProfileChangeTracker()
            
            # Step 1: Create basic profile
            basic_profile = await self.profile_repo.create(db, {
                **profile_data.dict(exclude_unset=True),
                'user_id': user_id,
                'skills': profile_data.skills or {},
                'platform_data': {},
                'resume_data': {},
                'career_interests': profile_data.career_interests or {},
                'data_last_updated': datetime.utcnow()
            })
            
            # Step 2: Process resume if provided
            resume_skills = {}
            if resume_file and self.nlp_engine:
                try:
                    resume_data = await self._process_resume_file(resume_file)
                    resume_skills = self._extract_skills_from_resume(resume_data)
                    
                    # Update profile with resume data
                    await self.profile_repo.update(db, basic_profile.id, {
                        'resume_data': resume_data.dict() if hasattr(resume_data, 'dict') else resume_data
                    })
                    
                    change_tracker.track_change(
                        'resume_data', None, 'processed', 'resume_upload'
                    )
                    
                except Exception as e:
                    logger.warning(f"Resume processing failed: {str(e)}")
                    resume_skills = {}
            
            # Step 3: Extract data from external platforms
            platform_skills = {}
            platform_data = {}
            
            if any([profile_data.github_username, profile_data.leetcode_id, profile_data.linkedin_url]):
                extraction_result = await self._extract_external_platform_data(
                    profile_data.github_username,
                    profile_data.leetcode_id,
                    profile_data.linkedin_url
                )
                
                if extraction_result.success:
                    platform_data = {
                        'github': extraction_result.github_profile,
                        'leetcode': extraction_result.leetcode_profile,
                        'linkedin': extraction_result.linkedin_profile
                    }
                    
                    platform_skills = self._extract_skills_from_platforms(platform_data)
                    
                    # Update profile with platform data
                    await self.profile_repo.update(db, basic_profile.id, {
                        'platform_data': platform_data
                    })
                    
                    change_tracker.track_change(
                        'platform_data', {}, platform_data, 'external_apis'
                    )
            
            # Step 4: Merge all skill sources
            skill_merge_result = self._merge_skill_profiles({
                'manual': profile_data.skills or {},
                'resume': resume_skills,
                **platform_skills
            })
            
            # Step 5: Generate unified profile
            unified_profile = await self._generate_unified_profile(
                basic_profile,
                skill_merge_result,
                platform_data,
                change_tracker
            )
            
            # Step 6: Update profile with unified data
            updated_profile = await self.profile_repo.update(db, basic_profile.id, {
                'skills': unified_profile['skills'],
                'career_interests': unified_profile['career_interests'],
                'skill_gaps': unified_profile.get('skill_gaps', {}),
                'data_last_updated': datetime.utcnow()
            })
            
            logger.info(f"Successfully created integrated profile for user {user_id}")
            return ProfileResponse.from_orm(updated_profile)
            
        except Exception as e:
            logger.error(f"Error creating profile for user {user_id}: {str(e)}")
            raise
    
    async def update_profile_with_validation(
        self,
        db: AsyncSession,
        user_id: str,
        update_data: ProfileUpdate
    ) -> ProfileResponse:
        """
        Update user profile with data consistency validation and conflict resolution.
        
        Args:
            db: Database session
            user_id: User ID
            update_data: Profile update data
            
        Returns:
            Updated ProfileResponse
        """
        logger.info(f"Updating profile for user {user_id} with validation")
        
        try:
            # Get existing profile
            existing_profile = await self.profile_repo.get_by_user_id(db, user_id)
            if not existing_profile:
                raise NotFoundError(f"Profile not found for user {user_id}")
            
            # Initialize change tracker
            change_tracker = ProfileChangeTracker()
            
            # Track changes
            update_dict = update_data.dict(exclude_unset=True)
            for field, new_value in update_dict.items():
                old_value = getattr(existing_profile, field, None)
                if old_value != new_value:
                    change_tracker.track_change(field, old_value, new_value, 'manual_update')
            
            # Validate data consistency
            validation_result = await self._validate_profile_consistency(
                existing_profile, update_data
            )
            
            if not validation_result['is_valid']:
                raise ValidationError(f"Profile validation failed: {validation_result['errors']}")
            
            # Resolve conflicts if any
            resolved_data = await self._resolve_profile_conflicts(
                existing_profile, update_dict, change_tracker
            )
            
            # Update profile
            updated_profile = await self.profile_repo.update(
                db, existing_profile.id, resolved_data
            )
            
            logger.info(f"Successfully updated profile for user {user_id}")
            return ProfileResponse.from_orm(updated_profile)
            
        except Exception as e:
            logger.error(f"Error updating profile for user {user_id}: {str(e)}")
            raise
    
    async def refresh_external_data(
        self,
        db: AsyncSession,
        user_id: str,
        force_refresh: bool = False
    ) -> ProfileResponse:
        """
        Refresh external platform data for user profile.
        
        Args:
            db: Database session
            user_id: User ID
            force_refresh: Force refresh even if data is recent
            
        Returns:
            Updated ProfileResponse
        """
        logger.info(f"Refreshing external data for user {user_id}")
        
        try:
            # Get existing profile
            profile = await self.profile_repo.get_by_user_id(db, user_id)
            if not profile:
                raise NotFoundError(f"Profile not found for user {user_id}")
            
            # Check if refresh is needed
            if not force_refresh and profile.data_last_updated:
                time_since_update = datetime.utcnow() - profile.data_last_updated
                if time_since_update < timedelta(hours=24):
                    logger.info(f"Data is recent for user {user_id}, skipping refresh")
                    return ProfileResponse.from_orm(profile)
            
            # Extract fresh data from external platforms
            extraction_result = await self._extract_external_platform_data(
                profile.github_username,
                profile.leetcode_id,
                profile.linkedin_url
            )
            
            if extraction_result.success:
                # Update platform data
                new_platform_data = {
                    'github': extraction_result.github_profile,
                    'leetcode': extraction_result.leetcode_profile,
                    'linkedin': extraction_result.linkedin_profile
                }
                
                # Extract skills from new platform data
                platform_skills = self._extract_skills_from_platforms(new_platform_data)
                
                # Merge with existing skills
                existing_skills = {
                    'manual': profile.skills or {},
                    'resume': self._extract_skills_from_resume_data(profile.resume_data or {})
                }
                existing_skills.update(platform_skills)
                
                skill_merge_result = self._merge_skill_profiles(existing_skills)
                
                # Update profile
                updated_profile = await self.profile_repo.update(db, profile.id, {
                    'platform_data': new_platform_data,
                    'skills': skill_merge_result.merged_skills,
                    'data_last_updated': datetime.utcnow()
                })
                
                logger.info(f"Successfully refreshed external data for user {user_id}")
                return ProfileResponse.from_orm(updated_profile)
            else:
                logger.warning(f"External data refresh failed for user {user_id}: {extraction_result.errors}")
                return ProfileResponse.from_orm(profile)
                
        except Exception as e:
            logger.error(f"Error refreshing external data for user {user_id}: {str(e)}")
            raise
    
    async def _process_resume_file(self, resume_file: UploadFile) -> Any:
        """Process uploaded resume file using NLP engine."""
        if not self.nlp_engine:
            raise ValidationError("Resume processing not available - NLP engine not initialized")
        
        # Save file temporarily
        temp_file_path = f"/tmp/resume_{uuid4()}.{resume_file.filename.split('.')[-1]}"
        
        try:
            with open(temp_file_path, "wb") as temp_file:
                content = await resume_file.read()
                temp_file.write(content)
            
            # Process with NLP engine
            resume_data = await self.nlp_engine.parse_resume(temp_file_path)
            return resume_data
            
        finally:
            # Clean up temp file
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
    
    def _extract_skills_from_resume(self, resume_data: Any) -> Dict[str, float]:
        """Extract skills from processed resume data."""
        if not resume_data or not hasattr(resume_data, 'skills'):
            return {}
        
        skills = {}
        for skill_extraction in resume_data.skills:
            if hasattr(skill_extraction, 'skill_name') and hasattr(skill_extraction, 'confidence_score'):
                skills[skill_extraction.skill_name] = skill_extraction.confidence_score
        
        return skills
    
    def _extract_skills_from_resume_data(self, resume_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract skills from resume data dictionary."""
        if not resume_data or 'skills' not in resume_data:
            return {}
        
        skills = {}
        for skill_data in resume_data['skills']:
            if isinstance(skill_data, dict) and 'skill_name' in skill_data and 'confidence_score' in skill_data:
                skills[skill_data['skill_name']] = skill_data['confidence_score']
        
        return skills
    
    async def _extract_external_platform_data(
        self,
        github_username: Optional[str],
        leetcode_id: Optional[str],
        linkedin_url: Optional[str]
    ) -> ProfileExtractionResult:
        """Extract data from external platforms."""
        request = ProfileExtractionRequest(
            github_username=github_username,
            leetcode_username=leetcode_id,
            linkedin_url=linkedin_url,
            timeout_seconds=60,
            enable_validation=True,
            enable_graceful_degradation=True
        )
        
        return await self.external_api_service.extract_comprehensive_profile(request)
    
    def _extract_skills_from_platforms(self, platform_data: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        """Extract skills from platform data."""
        platform_skills = {}
        
        # GitHub skills
        if platform_data.get('github'):
            github_skills = {}
            github_data = platform_data['github']
            
            # Extract from languages
            if 'languages' in github_data:
                total_bytes = sum(github_data['languages'].values())
                for lang, bytes_count in github_data['languages'].items():
                    confidence = min(0.9, bytes_count / total_bytes * 2)  # Scale to reasonable confidence
                    github_skills[lang] = confidence
            
            # Extract from repository topics
            if 'repositories' in github_data:
                topic_counts = {}
                for repo in github_data['repositories']:
                    if 'topics' in repo:
                        for topic in repo['topics']:
                            topic_counts[topic] = topic_counts.get(topic, 0) + 1
                
                # Convert topic counts to skills
                max_count = max(topic_counts.values()) if topic_counts else 1
                for topic, count in topic_counts.items():
                    confidence = min(0.8, count / max_count)
                    github_skills[topic] = confidence
            
            platform_skills['github'] = github_skills
        
        # LeetCode skills
        if platform_data.get('leetcode'):
            leetcode_skills = {}
            leetcode_data = platform_data['leetcode']
            
            # Extract from skill tags
            if 'skill_tags' in leetcode_data:
                total_problems = sum(leetcode_data['skill_tags'].values())
                for skill, count in leetcode_data['skill_tags'].items():
                    confidence = min(0.8, count / total_problems * 3)  # Scale to reasonable confidence
                    leetcode_skills[skill] = confidence
            
            # Extract from languages used
            if 'languages_used' in leetcode_data:
                total_usage = sum(leetcode_data['languages_used'].values())
                for lang, usage in leetcode_data['languages_used'].items():
                    confidence = min(0.7, usage / total_usage * 2)
                    leetcode_skills[lang] = confidence
            
            platform_skills['leetcode'] = leetcode_skills
        
        # LinkedIn skills
        if platform_data.get('linkedin'):
            linkedin_skills = {}
            linkedin_data = platform_data['linkedin']
            
            # Extract from skills section
            if 'skills' in linkedin_data:
                for skill_data in linkedin_data['skills']:
                    if isinstance(skill_data, dict) and 'name' in skill_data:
                        # Use endorsements to determine confidence
                        endorsements = skill_data.get('endorsements', 0)
                        confidence = min(0.8, 0.3 + (endorsements / 50) * 0.5)  # Base 0.3, up to 0.8
                        linkedin_skills[skill_data['name']] = confidence
            
            platform_skills['linkedin'] = linkedin_skills
        
        return platform_skills
    
    def _merge_skill_profiles(self, skill_sources: Dict[str, Dict[str, float]]) -> SkillMergeResult:
        """
        Merge skill profiles from multiple sources with confidence scoring.
        
        Args:
            skill_sources: Dictionary mapping source names to skill dictionaries
            
        Returns:
            SkillMergeResult with merged skills and metadata
        """
        result = SkillMergeResult()
        
        # Collect all unique skills
        all_skills = set()
        for skills in skill_sources.values():
            all_skills.update(skills.keys())
        
        # Merge each skill
        for skill in all_skills:
            skill_data = []
            sources_for_skill = []
            
            # Collect data from all sources that have this skill
            for source, skills in skill_sources.items():
                if skill in skills:
                    confidence = skills[skill]
                    weight = self.skill_confidence_weights.get(source, 0.5)
                    weighted_confidence = confidence * weight
                    
                    skill_data.append({
                        'source': source,
                        'confidence': confidence,
                        'weighted_confidence': weighted_confidence
                    })
                    sources_for_skill.append(source)
            
            if skill_data:
                # Calculate merged confidence using weighted average
                total_weight = sum(data['weighted_confidence'] for data in skill_data)
                source_count = len(skill_data)
                
                # Boost confidence for skills found in multiple sources
                multi_source_boost = min(0.2, (source_count - 1) * 0.1)
                final_confidence = min(1.0, total_weight / source_count + multi_source_boost)
                
                result.merged_skills[skill] = final_confidence
                result.sources[skill] = sources_for_skill
                result.confidence_scores[skill] = final_confidence
                
                # Check for conflicts (significant confidence differences)
                confidences = [data['confidence'] for data in skill_data]
                if len(confidences) > 1:
                    confidence_range = max(confidences) - min(confidences)
                    if confidence_range > 0.4:  # Significant difference
                        result.conflicts.append({
                            'skill': skill,
                            'sources': sources_for_skill,
                            'confidences': confidences,
                            'range': confidence_range,
                            'resolved_confidence': final_confidence
                        })
        
        logger.info(f"Merged {len(result.merged_skills)} skills from {len(skill_sources)} sources")
        if result.conflicts:
            logger.info(f"Resolved {len(result.conflicts)} skill confidence conflicts")
        
        return result
    
    async def _generate_unified_profile(
        self,
        base_profile: UserProfile,
        skill_merge_result: SkillMergeResult,
        platform_data: Dict[str, Any],
        change_tracker: ProfileChangeTracker
    ) -> Dict[str, Any]:
        """Generate unified profile from all data sources."""
        unified_profile = {
            'skills': skill_merge_result.merged_skills,
            'career_interests': base_profile.career_interests or {},
            'skill_gaps': {}
        }
        
        # Enhance career interests based on platform data
        if platform_data.get('github'):
            github_data = platform_data['github']
            if 'repositories' in github_data:
                # Infer interests from repository topics and languages
                topics = []
                for repo in github_data['repositories']:
                    if 'topics' in repo:
                        topics.extend(repo['topics'])
                
                # Count topic frequencies
                topic_counts = {}
                for topic in topics:
                    topic_counts[topic] = topic_counts.get(topic, 0) + 1
                
                # Add top topics as career interests
                sorted_topics = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)
                for topic, count in sorted_topics[:5]:  # Top 5 topics
                    unified_profile['career_interests'][f"github_{topic}"] = count / len(topics)
        
        # Calculate skill gaps if dream job is specified
        if base_profile.dream_job:
            skill_gaps = await self._calculate_skill_gaps(
                unified_profile['skills'],
                base_profile.dream_job
            )
            unified_profile['skill_gaps'] = skill_gaps
        
        return unified_profile
    
    async def _calculate_skill_gaps(
        self,
        current_skills: Dict[str, float],
        dream_job: str
    ) -> Dict[str, float]:
        """Calculate skill gaps for target job role."""
        # This is a simplified implementation
        # In a real system, this would use job market data and ML models
        
        # Common skills for different job roles
        job_skill_requirements = {
            'software engineer': {
                'Python': 0.8, 'JavaScript': 0.7, 'Git': 0.9, 'SQL': 0.6,
                'React': 0.6, 'Docker': 0.5, 'AWS': 0.4
            },
            'data scientist': {
                'Python': 0.9, 'SQL': 0.8, 'Machine Learning': 0.9,
                'Pandas': 0.8, 'NumPy': 0.7, 'TensorFlow': 0.6, 'R': 0.5
            },
            'devops engineer': {
                'Docker': 0.9, 'Kubernetes': 0.8, 'AWS': 0.9, 'Linux': 0.8,
                'Jenkins': 0.7, 'Terraform': 0.6, 'Python': 0.6
            }
        }
        
        # Find matching job requirements
        dream_job_lower = dream_job.lower()
        required_skills = {}
        
        for job_type, skills in job_skill_requirements.items():
            if job_type in dream_job_lower:
                required_skills = skills
                break
        
        # Calculate gaps
        skill_gaps = {}
        for required_skill, required_level in required_skills.items():
            current_level = current_skills.get(required_skill, 0.0)
            if current_level < required_level:
                skill_gaps[required_skill] = required_level - current_level
        
        return skill_gaps
    
    async def get_profile_analytics(
        self,
        db: AsyncSession,
        user_id: str
    ) -> Dict[str, Any]:
        """
        Get comprehensive profile analytics and insights.
        
        Returns:
        - Profile completeness score
        - Skill distribution analysis
        - Data freshness metrics
        - Platform coverage analysis
        - Skill gaps summary
        - Personalized recommendations
        """
        try:
            # Get user profile
            profile = await self.profile_repo.get_by_user_id(db, user_id)
            if not profile:
                raise NotFoundError(f"Profile not found for user {user_id}")
            
            # Calculate profile completeness
            completeness_score = self._calculate_profile_completeness(profile)
            
            # Calculate overall profile score
            profile_score = self._calculate_profile_score(profile)
            
            # Analyze skill distribution
            skill_analysis = self._analyze_skill_distribution(profile.skills or {})
            
            # Check data freshness
            data_freshness = self._analyze_data_freshness(profile)
            
            # Analyze platform coverage
            platform_coverage = self._analyze_platform_coverage(profile)
            
            # Generate recommendations
            recommendations = self._generate_profile_recommendations(profile, completeness_score)
            
            # Update profile with calculated scores
            await self.profile_repo.update(db, profile.id, {
                'profile_score': profile_score,
                'completeness_score': completeness_score
            })
            
            return {
                'profile_score': profile_score,
                'completeness_score': completeness_score,
                'skill_analysis': skill_analysis,
                'data_freshness': data_freshness,
                'platform_coverage': platform_coverage,
                'recommendations': recommendations,
                'summary': {
                    'total_skills': len(profile.skills or {}),
                    'platforms_connected': len([p for p in [profile.github_username, profile.leetcode_id, profile.linkedin_url] if p]),
                    'profile_strength': 'Strong' if profile_score >= 80 else 'Good' if profile_score >= 60 else 'Needs Improvement'
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting profile analytics for user {user_id}: {str(e)}")
            raise
    
    def _calculate_profile_completeness(self, profile: UserProfile) -> float:
        """Calculate profile completeness score (0-100)."""
        total_fields = 0
        completed_fields = 0
        
        # Core fields (weighted more heavily)
        core_fields = [
            ('current_role', 3), ('experience_years', 2), ('location', 2),
            ('industry', 2), ('desired_role', 3), ('career_goals', 2),
            ('education', 2), ('skills', 4)
        ]
        
        for field_name, weight in core_fields:
            total_fields += weight
            field_value = getattr(profile, field_name, None)
            if field_value:
                if field_name == 'skills' and isinstance(field_value, dict) and len(field_value) > 0:
                    completed_fields += weight
                elif field_name != 'skills' and str(field_value).strip():
                    completed_fields += weight
        
        # Optional fields (lower weight)
        optional_fields = [
            'timeframe', 'salary_expectation', 'certifications', 'languages',
            'work_type', 'company_size', 'work_culture', 'benefits'
        ]
        
        for field_name in optional_fields:
            total_fields += 1
            field_value = getattr(profile, field_name, None)
            if field_value:
                if field_name == 'benefits' and isinstance(field_value, list) and len(field_value) > 0:
                    completed_fields += 1
                elif field_name != 'benefits' and str(field_value).strip():
                    completed_fields += 1
        
        # Platform connections
        platform_fields = ['github_username', 'leetcode_id', 'linkedin_url']
        for field_name in platform_fields:
            total_fields += 1
            field_value = getattr(profile, field_name, None)
            if field_value and str(field_value).strip():
                completed_fields += 1
        
        return (completed_fields / total_fields) * 100 if total_fields > 0 else 0
    
    def _calculate_profile_score(self, profile: UserProfile) -> float:
        """Calculate overall profile score (0-100)."""
        score = 0
        
        # Completeness contributes 40%
        completeness_score = self._calculate_profile_completeness(profile)
        score += completeness_score * 0.4
        
        # Skills quality contributes 30%
        skills_score = self._calculate_skills_score(profile.skills or {})
        score += skills_score * 0.3
        
        # Platform integration contributes 20%
        platform_score = self._calculate_platform_integration_score(profile)
        score += platform_score * 0.2
        
        # Data freshness contributes 10%
        freshness_score = self._calculate_data_freshness_score(profile)
        score += freshness_score * 0.1
        
        return min(100, max(0, score))
    
    def _calculate_skills_score(self, skills: Dict[str, float]) -> float:
        """Calculate skills quality score."""
        if not skills:
            return 0
        
        # Base score from number of skills
        skill_count_score = min(50, len(skills) * 5)  # Up to 50 points for 10+ skills
        
        # Quality score from confidence levels
        avg_confidence = sum(skills.values()) / len(skills)
        confidence_score = avg_confidence * 50  # Up to 50 points for high confidence
        
        return skill_count_score + confidence_score
    
    def _calculate_platform_integration_score(self, profile: UserProfile) -> float:
        """Calculate platform integration score."""
        platforms = [profile.github_username, profile.leetcode_id, profile.linkedin_url]
        connected_platforms = len([p for p in platforms if p])
        
        base_score = (connected_platforms / len(platforms)) * 70  # Up to 70 points
        
        # Bonus for having platform data
        platform_data_bonus = 0
        if profile.platform_data:
            for platform in ['github', 'leetcode', 'linkedin']:
                if platform in profile.platform_data and profile.platform_data[platform]:
                    platform_data_bonus += 10
        
        return min(100, base_score + platform_data_bonus)
    
    def _calculate_data_freshness_score(self, profile: UserProfile) -> float:
        """Calculate data freshness score."""
        if not profile.data_last_updated:
            return 50  # Neutral score for new profiles
        
        days_since_update = (datetime.utcnow() - profile.data_last_updated).days
        
        if days_since_update <= 7:
            return 100
        elif days_since_update <= 30:
            return 80
        elif days_since_update <= 90:
            return 60
        else:
            return 30
    
    def _analyze_skill_distribution(self, skills: Dict[str, float]) -> Dict[str, Any]:
        """Analyze skill distribution and categorization."""
        if not skills:
            return {
                'total_skills': 0,
                'categories': {},
                'confidence_distribution': {},
                'top_skills': []
            }
        
        # Categorize skills (simplified categorization)
        categories = {
            'programming_languages': [],
            'frameworks': [],
            'tools': [],
            'soft_skills': [],
            'other': []
        }
        
        programming_languages = ['python', 'javascript', 'java', 'c++', 'c#', 'go', 'rust', 'typescript', 'php', 'ruby']
        frameworks = ['react', 'angular', 'vue', 'django', 'flask', 'spring', 'express', 'laravel']
        tools = ['git', 'docker', 'kubernetes', 'aws', 'azure', 'jenkins', 'terraform']
        soft_skills = ['leadership', 'communication', 'teamwork', 'problem-solving']
        
        for skill, confidence in skills.items():
            skill_lower = skill.lower()
            if any(lang in skill_lower for lang in programming_languages):
                categories['programming_languages'].append({'skill': skill, 'confidence': confidence})
            elif any(fw in skill_lower for fw in frameworks):
                categories['frameworks'].append({'skill': skill, 'confidence': confidence})
            elif any(tool in skill_lower for tool in tools):
                categories['tools'].append({'skill': skill, 'confidence': confidence})
            elif any(soft in skill_lower for soft in soft_skills):
                categories['soft_skills'].append({'skill': skill, 'confidence': confidence})
            else:
                categories['other'].append({'skill': skill, 'confidence': confidence})
        
        # Confidence distribution
        confidence_ranges = {'high': 0, 'medium': 0, 'low': 0}
        for confidence in skills.values():
            if confidence >= 0.7:
                confidence_ranges['high'] += 1
            elif confidence >= 0.4:
                confidence_ranges['medium'] += 1
            else:
                confidence_ranges['low'] += 1
        
        # Top skills
        top_skills = sorted(skills.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return {
            'total_skills': len(skills),
            'categories': {k: len(v) for k, v in categories.items()},
            'confidence_distribution': confidence_ranges,
            'top_skills': [{'skill': skill, 'confidence': conf} for skill, conf in top_skills]
        }
    
    def _analyze_data_freshness(self, profile: UserProfile) -> Dict[str, Any]:
        """Analyze data freshness metrics."""
        now = datetime.utcnow()
        
        freshness_info = {
            'last_updated': profile.data_last_updated.isoformat() if profile.data_last_updated else None,
            'days_since_update': None,
            'freshness_status': 'unknown'
        }
        
        if profile.data_last_updated:
            days_since = (now - profile.data_last_updated).days
            freshness_info['days_since_update'] = days_since
            
            if days_since <= 7:
                freshness_info['freshness_status'] = 'fresh'
            elif days_since <= 30:
                freshness_info['freshness_status'] = 'recent'
            elif days_since <= 90:
                freshness_info['freshness_status'] = 'stale'
            else:
                freshness_info['freshness_status'] = 'outdated'
        
        return freshness_info
    
    def _analyze_platform_coverage(self, profile: UserProfile) -> Dict[str, Any]:
        """Analyze platform coverage and integration."""
        platforms = {
            'github': {
                'connected': bool(profile.github_username),
                'username': profile.github_username,
                'has_data': bool(profile.platform_data and profile.platform_data.get('github'))
            },
            'leetcode': {
                'connected': bool(profile.leetcode_id),
                'username': profile.leetcode_id,
                'has_data': bool(profile.platform_data and profile.platform_data.get('leetcode'))
            },
            'linkedin': {
                'connected': bool(profile.linkedin_url),
                'url': profile.linkedin_url,
                'has_data': bool(profile.platform_data and profile.platform_data.get('linkedin'))
            }
        }
        
        connected_count = sum(1 for p in platforms.values() if p['connected'])
        data_count = sum(1 for p in platforms.values() if p['has_data'])
        
        return {
            'platforms': platforms,
            'connected_count': connected_count,
            'data_count': data_count,
            'coverage_percentage': (connected_count / len(platforms)) * 100
        }
    
    def _generate_profile_recommendations(self, profile: UserProfile, completeness_score: float) -> List[Dict[str, Any]]:
        """Generate personalized profile improvement recommendations."""
        recommendations = []
        
        # Completeness recommendations
        if completeness_score < 70:
            missing_fields = []
            if not profile.current_role:
                missing_fields.append('current role')
            if not profile.industry:
                missing_fields.append('industry')
            if not profile.desired_role:
                missing_fields.append('desired role')
            if not profile.career_goals:
                missing_fields.append('career goals')
            if not profile.education:
                missing_fields.append('education')
            
            if missing_fields:
                recommendations.append({
                    'type': 'completeness',
                    'priority': 'high',
                    'title': 'Complete Your Profile',
                    'description': f"Add missing information: {', '.join(missing_fields)}",
                    'action': 'Fill in missing profile fields to improve your score'
                })
        
        # Skills recommendations
        skills_count = len(profile.skills or {})
        if skills_count < 5:
            recommendations.append({
                'type': 'skills',
                'priority': 'high',
                'title': 'Add More Skills',
                'description': f"You have {skills_count} skills listed. Add more to improve your profile.",
                'action': 'List your technical and soft skills'
            })
        
        # Platform integration recommendations
        platforms = [profile.github_username, profile.leetcode_id, profile.linkedin_url]
        connected_platforms = len([p for p in platforms if p])
        
        if connected_platforms < 2:
            recommendations.append({
                'type': 'integration',
                'priority': 'medium',
                'title': 'Connect More Platforms',
                'description': 'Connect your GitHub, LeetCode, and LinkedIn profiles for better insights.',
                'action': 'Add your platform usernames/URLs'
            })
        
        # Data freshness recommendations
        if profile.data_last_updated:
            days_since_update = (datetime.utcnow() - profile.data_last_updated).days
            if days_since_update > 30:
                recommendations.append({
                    'type': 'freshness',
                    'priority': 'low',
                    'title': 'Refresh Your Data',
                    'description': f"Your profile data was last updated {days_since_update} days ago.",
                    'action': 'Refresh your external platform data'
                })
        
        return recommendations
    
    async def _validate_profile_consistency(
        self,
        existing_profile: UserProfile,
        update_data: ProfileUpdate
    ) -> Dict[str, Any]:
        """Validate profile data consistency."""
        validation_result = {
            'is_valid': True,
            'errors': [],
            'warnings': []
        }
        
        # Validate experience years consistency
        if update_data.experience_years is not None:
            if update_data.experience_years < 0:
                validation_result['errors'].append("Experience years cannot be negative")
                validation_result['is_valid'] = False
            
            if update_data.experience_years > 50:
                validation_result['warnings'].append("Experience years seems unusually high")
        
        # Validate platform usernames format
        if update_data.github_username is not None:
            if not update_data.github_username.replace('-', '').replace('_', '').isalnum():
                validation_result['errors'].append("Invalid GitHub username format")
                validation_result['is_valid'] = False
        
        # Validate skills format
        if update_data.skills is not None:
            for skill, confidence in update_data.skills.items():
                if not isinstance(confidence, (int, float)) or confidence < 0 or confidence > 1:
                    validation_result['errors'].append(f"Invalid confidence score for skill '{skill}'")
                    validation_result['is_valid'] = False
        
        # Validate timeframe values
        if update_data.timeframe is not None:
            valid_timeframes = ['immediate', 'short', 'medium', 'long']
            if update_data.timeframe not in valid_timeframes:
                validation_result['warnings'].append(f"Unusual timeframe value: {update_data.timeframe}")
        
        # Validate work type values
        if update_data.work_type is not None:
            valid_work_types = ['remote', 'hybrid', 'onsite', 'flexible']
            if update_data.work_type not in valid_work_types:
                validation_result['warnings'].append(f"Unusual work type value: {update_data.work_type}")
        
        # Validate company size values
        if update_data.company_size is not None:
            valid_sizes = ['startup', 'small', 'medium', 'large']
            if update_data.company_size not in valid_sizes:
                validation_result['warnings'].append(f"Unusual company size value: {update_data.company_size}")
        
        # Validate benefits format
        if update_data.benefits is not None:
            if not isinstance(update_data.benefits, list):
                validation_result['errors'].append("Benefits must be a list of strings")
                validation_result['is_valid'] = False
            else:
                for benefit in update_data.benefits:
                    if not isinstance(benefit, str):
                        validation_result['errors'].append("All benefits must be strings")
                        validation_result['is_valid'] = False
                        break
        
        return validation_result
    
    async def _resolve_profile_conflicts(
        self,
        existing_profile: UserProfile,
        update_data: Dict[str, Any],
        change_tracker: ProfileChangeTracker
    ) -> Dict[str, Any]:
        """Resolve conflicts in profile data using configured strategies."""
        resolved_data = update_data.copy()
        
        # Resolve skill conflicts
        if 'skills' in update_data and existing_profile.skills:
            existing_skills = existing_profile.skills
            new_skills = update_data['skills']
            
            # Merge skills using confidence-based strategy
            merged_skills = existing_skills.copy()
            for skill, confidence in new_skills.items():
                if skill in merged_skills:
                    # Take higher confidence
                    if confidence > merged_skills[skill]:
                        change_tracker.track_change(
                            f'skills.{skill}', merged_skills[skill], confidence, 'conflict_resolution'
                        )
                        merged_skills[skill] = confidence
                else:
                    merged_skills[skill] = confidence
            
            resolved_data['skills'] = merged_skills
        
        # Resolve experience years conflicts
        if 'experience_years' in update_data and existing_profile.experience_years:
            strategy = self.conflict_resolution_strategies.get('experience_years', 'take_highest')
            if strategy == 'take_highest':
                resolved_value = max(existing_profile.experience_years, update_data['experience_years'])
                if resolved_value != update_data['experience_years']:
                    change_tracker.track_change(
                        'experience_years', update_data['experience_years'], resolved_value, 'conflict_resolution'
                    )
                    resolved_data['experience_years'] = resolved_value
        
        return resolved_data
    
    async def get_profile_analytics(
        self,
        db: AsyncSession,
        user_id: str
    ) -> Dict[str, Any]:
        """Get comprehensive profile analytics and insights."""
        profile = await self.profile_repo.get_by_user_id(db, user_id)
        if not profile:
            raise NotFoundError(f"Profile not found for user {user_id}")
        
        analytics = {
            'profile_completeness': self._calculate_profile_completeness(profile),
            'skill_distribution': self._analyze_skill_distribution(profile.skills or {}),
            'data_freshness': self._analyze_data_freshness(profile),
            'platform_coverage': self._analyze_platform_coverage(profile),
            'skill_gaps_summary': self._summarize_skill_gaps(profile.skill_gaps or {}),
            'recommendations': await self._generate_profile_recommendations(profile)
        }
        
        return analytics
    
    def _calculate_profile_completeness(self, profile: UserProfile) -> Dict[str, Any]:
        """Calculate profile completeness score."""
        total_fields = 10
        completed_fields = 0
        
        if profile.dream_job:
            completed_fields += 1
        if profile.experience_years is not None:
            completed_fields += 1
        if profile.current_role:
            completed_fields += 1
        if profile.location:
            completed_fields += 1
        if profile.github_username:
            completed_fields += 1
        if profile.leetcode_id:
            completed_fields += 1
        if profile.linkedin_url:
            completed_fields += 1
        if profile.skills:
            completed_fields += 1
        if profile.resume_data:
            completed_fields += 1
        if profile.career_interests:
            completed_fields += 1
        
        completeness_score = completed_fields / total_fields
        
        return {
            'score': completeness_score,
            'completed_fields': completed_fields,
            'total_fields': total_fields,
            'missing_fields': self._identify_missing_fields(profile)
        }
    
    def _identify_missing_fields(self, profile: UserProfile) -> List[str]:
        """Identify missing profile fields."""
        missing = []
        
        if not profile.dream_job:
            missing.append('dream_job')
        if profile.experience_years is None:
            missing.append('experience_years')
        if not profile.current_role:
            missing.append('current_role')
        if not profile.location:
            missing.append('location')
        if not profile.github_username:
            missing.append('github_username')
        if not profile.leetcode_id:
            missing.append('leetcode_id')
        if not profile.linkedin_url:
            missing.append('linkedin_url')
        if not profile.skills:
            missing.append('skills')
        if not profile.resume_data:
            missing.append('resume_data')
        if not profile.career_interests:
            missing.append('career_interests')
        
        return missing
    
    def _analyze_skill_distribution(self, skills: Dict[str, float]) -> Dict[str, Any]:
        """Analyze skill distribution and categorization."""
        if not skills:
            return {'total_skills': 0, 'categories': {}, 'top_skills': []}
        
        # Simple skill categorization
        categories = {
            'programming_languages': [],
            'frameworks': [],
            'tools': [],
            'soft_skills': []
        }
        
        # Categorize skills (simplified)
        programming_languages = ['python', 'javascript', 'java', 'c++', 'go', 'rust', 'typescript']
        frameworks = ['react', 'angular', 'vue', 'django', 'flask', 'spring', 'express']
        tools = ['git', 'docker', 'kubernetes', 'aws', 'jenkins', 'terraform']
        
        for skill, confidence in skills.items():
            skill_lower = skill.lower()
            if any(lang in skill_lower for lang in programming_languages):
                categories['programming_languages'].append({'skill': skill, 'confidence': confidence})
            elif any(fw in skill_lower for fw in frameworks):
                categories['frameworks'].append({'skill': skill, 'confidence': confidence})
            elif any(tool in skill_lower for tool in tools):
                categories['tools'].append({'skill': skill, 'confidence': confidence})
            else:
                categories['soft_skills'].append({'skill': skill, 'confidence': confidence})
        
        # Get top skills
        top_skills = sorted(skills.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return {
            'total_skills': len(skills),
            'categories': categories,
            'top_skills': [{'skill': skill, 'confidence': conf} for skill, conf in top_skills],
            'average_confidence': sum(skills.values()) / len(skills)
        }
    
    def _analyze_data_freshness(self, profile: UserProfile) -> Dict[str, Any]:
        """Analyze how fresh the profile data is."""
        now = datetime.utcnow()
        
        freshness = {
            'last_updated': profile.data_last_updated,
            'days_since_update': None,
            'freshness_score': 1.0,
            'needs_refresh': False
        }
        
        if profile.data_last_updated:
            days_since_update = (now - profile.data_last_updated).days
            freshness['days_since_update'] = days_since_update
            
            # Calculate freshness score (1.0 = fresh, 0.0 = stale)
            if days_since_update <= 1:
                freshness['freshness_score'] = 1.0
            elif days_since_update <= 7:
                freshness['freshness_score'] = 0.8
            elif days_since_update <= 30:
                freshness['freshness_score'] = 0.6
            else:
                freshness['freshness_score'] = 0.3
            
            freshness['needs_refresh'] = days_since_update > 7
        
        return freshness
    
    def _analyze_platform_coverage(self, profile: UserProfile) -> Dict[str, Any]:
        """Analyze coverage of external platforms."""
        platforms = {
            'github': bool(profile.github_username),
            'leetcode': bool(profile.leetcode_id),
            'linkedin': bool(profile.linkedin_url),
            'codeforces': bool(profile.codeforces_id)
        }
        
        connected_count = sum(platforms.values())
        total_platforms = len(platforms)
        
        return {
            'platforms': platforms,
            'connected_count': connected_count,
            'total_platforms': total_platforms,
            'coverage_score': connected_count / total_platforms,
            'missing_platforms': [name for name, connected in platforms.items() if not connected]
        }
    
    def _summarize_skill_gaps(self, skill_gaps: Dict[str, float]) -> Dict[str, Any]:
        """Summarize skill gaps analysis."""
        if not skill_gaps:
            return {'total_gaps': 0, 'critical_gaps': [], 'moderate_gaps': [], 'minor_gaps': []}
        
        critical_gaps = []
        moderate_gaps = []
        minor_gaps = []
        
        for skill, gap in skill_gaps.items():
            if gap >= 0.7:
                critical_gaps.append({'skill': skill, 'gap': gap})
            elif gap >= 0.4:
                moderate_gaps.append({'skill': skill, 'gap': gap})
            else:
                minor_gaps.append({'skill': skill, 'gap': gap})
        
        return {
            'total_gaps': len(skill_gaps),
            'critical_gaps': critical_gaps,
            'moderate_gaps': moderate_gaps,
            'minor_gaps': minor_gaps,
            'average_gap': sum(skill_gaps.values()) / len(skill_gaps)
        }
    
    async def _generate_profile_recommendations(self, profile: UserProfile) -> List[Dict[str, Any]]:
        """Generate recommendations for profile improvement."""
        recommendations = []
        
        # Completeness recommendations
        if not profile.dream_job:
            recommendations.append({
                'type': 'completeness',
                'priority': 'high',
                'title': 'Add Dream Job',
                'description': 'Specify your target role to get personalized recommendations'
            })
        
        if not profile.resume_data:
            recommendations.append({
                'type': 'completeness',
                'priority': 'high',
                'title': 'Upload Resume',
                'description': 'Upload your resume to extract skills and experience automatically'
            })
        
        # Platform connection recommendations
        if not profile.github_username:
            recommendations.append({
                'type': 'platform',
                'priority': 'medium',
                'title': 'Connect GitHub',
                'description': 'Connect your GitHub to showcase your coding projects and skills'
            })
        
        # Data freshness recommendations
        if profile.data_last_updated:
            days_since_update = (datetime.utcnow() - profile.data_last_updated).days
            if days_since_update > 30:
                recommendations.append({
                    'type': 'freshness',
                    'priority': 'medium',
                    'title': 'Refresh Profile Data',
                    'description': f'Your profile data is {days_since_update} days old. Consider refreshing it.'
                })
        
        return recommendations 
   
    def calculate_profile_completeness(self, profile: 'UserProfile') -> float:
        """Calculate profile completeness score (0.0 to 1.0)."""
        try:
            score = 0.0
            max_score = 10.0  # Total possible points
            
            # Basic information (2 points)
            if profile.current_role:
                score += 0.5
            if profile.dream_job:
                score += 0.5
            if profile.experience_years is not None:
                score += 0.5
            if profile.location:
                score += 0.5
            
            # Skills (3 points)
            if profile.skills:
                skill_count = len(profile.skills)
                if skill_count >= 10:
                    score += 3.0
                elif skill_count >= 5:
                    score += 2.0
                elif skill_count >= 1:
                    score += 1.0
            
            # Resume data (2 points)
            if profile.resume_data:
                score += 2.0
            
            # External platform data (3 points)
            platform_data = profile.platform_data or {}
            if platform_data.get('github'):
                score += 1.0
            if platform_data.get('leetcode'):
                score += 1.0
            if platform_data.get('linkedin'):
                score += 1.0
            
            return min(score / max_score, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating profile completeness: {e}")
            return 0.0
    
    def get_missing_profile_elements(self, profile: 'UserProfile') -> List[str]:
        """Get list of missing profile elements."""
        try:
            missing = []
            
            if not profile.current_role:
                missing.append("current_role")
            if not profile.dream_job:
                missing.append("dream_job")
            if profile.experience_years is None:
                missing.append("experience_years")
            if not profile.location:
                missing.append("location")
            if not profile.skills or len(profile.skills) < 5:
                missing.append("skills")
            if not profile.resume_data:
                missing.append("resume")
            
            platform_data = profile.platform_data or {}
            if not platform_data.get('github'):
                missing.append("github_profile")
            if not platform_data.get('leetcode'):
                missing.append("leetcode_profile")
            if not platform_data.get('linkedin'):
                missing.append("linkedin_profile")
            
            return missing
            
        except Exception as e:
            logger.error(f"Error getting missing profile elements: {e}")
            return []
    
    async def get_profile_analytics(self, db: AsyncSession, user_id: str) -> Dict[str, Any]:
        """Get comprehensive profile analytics."""
        try:
            from app.repositories.profile import ProfileRepository
            profile_repo = ProfileRepository()
            
            profile = await profile_repo.get_by_user_id(db, user_id)
            if not profile:
                raise Exception("Profile not found")
            
            # Calculate completeness
            completeness_score = self.calculate_profile_completeness(profile)
            missing_elements = self.get_missing_profile_elements(profile)
            
            # Skill analysis
            skills = profile.skills or {}
            skill_analysis = {
                'total_skills': len(skills),
                'top_skills': sorted(skills.items(), key=lambda x: x[1], reverse=True)[:5],
                'skill_categories': self._categorize_skills(list(skills.keys())),
                'average_confidence': sum(skills.values()) / len(skills) if skills else 0
            }
            
            # Data freshness
            data_freshness = {
                'last_updated': profile.data_last_updated.isoformat() if profile.data_last_updated else None,
                'days_since_update': (datetime.utcnow() - profile.data_last_updated).days if profile.data_last_updated else None,
                'needs_refresh': (
                    not profile.data_last_updated or 
                    (datetime.utcnow() - profile.data_last_updated).days > 30
                )
            }
            
            # Platform coverage
            platform_data = profile.platform_data or {}
            platform_coverage = {
                'connected_platforms': list(platform_data.keys()),
                'platform_count': len(platform_data),
                'missing_platforms': [
                    platform for platform in ['github', 'leetcode', 'linkedin']
                    if platform not in platform_data
                ]
            }
            
            return {
                'profile_completeness': {
                    'score': completeness_score,
                    'percentage': round(completeness_score * 100, 1),
                    'missing_elements': missing_elements
                },
                'skill_analysis': skill_analysis,
                'data_freshness': data_freshness,
                'platform_coverage': platform_coverage,
                'recommendations': self._get_profile_improvement_recommendations(
                    completeness_score, missing_elements, skill_analysis
                )
            }
            
        except Exception as e:
            logger.error(f"Error getting profile analytics: {e}")
            raise Exception(f"Failed to get profile analytics: {str(e)}")
    
    def _categorize_skills(self, skills: List[str]) -> Dict[str, List[str]]:
        """Categorize skills into different categories."""
        categories = {
            'programming_languages': [],
            'frameworks_libraries': [],
            'databases': [],
            'cloud_platforms': [],
            'tools': [],
            'soft_skills': [],
            'other': []
        }
        
        # Simple categorization based on skill names
        for skill in skills:
            skill_lower = skill.lower()
            
            if any(lang in skill_lower for lang in ['python', 'java', 'javascript', 'c++', 'c#', 'go', 'rust', 'php', 'ruby']):
                categories['programming_languages'].append(skill)
            elif any(fw in skill_lower for fw in ['react', 'angular', 'vue', 'django', 'flask', 'spring', 'express']):
                categories['frameworks_libraries'].append(skill)
            elif any(db in skill_lower for db in ['sql', 'mysql', 'postgresql', 'mongodb', 'redis', 'elasticsearch']):
                categories['databases'].append(skill)
            elif any(cloud in skill_lower for cloud in ['aws', 'azure', 'gcp', 'docker', 'kubernetes']):
                categories['cloud_platforms'].append(skill)
            elif any(tool in skill_lower for tool in ['git', 'jenkins', 'terraform', 'ansible']):
                categories['tools'].append(skill)
            elif any(soft in skill_lower for soft in ['communication', 'leadership', 'teamwork', 'problem solving']):
                categories['soft_skills'].append(skill)
            else:
                categories['other'].append(skill)
        
        return {k: v for k, v in categories.items() if v}  # Remove empty categories
    
    def _get_profile_improvement_recommendations(self, completeness_score: float,
                                               missing_elements: List[str],
                                               skill_analysis: Dict[str, Any]) -> List[str]:
        """Get recommendations for improving profile."""
        recommendations = []
        
        if completeness_score < 0.5:
            recommendations.append("Your profile is incomplete. Focus on adding basic information first.")
        
        if 'current_role' in missing_elements:
            recommendations.append("Add your current job role to help with better recommendations.")
        
        if 'dream_job' in missing_elements:
            recommendations.append("Specify your dream job to get targeted career recommendations.")
        
        if 'skills' in missing_elements or skill_analysis['total_skills'] < 5:
            recommendations.append("Add more skills to your profile. Aim for at least 10 relevant skills.")
        
        if 'resume' in missing_elements:
            recommendations.append("Upload your resume to automatically extract skills and experience.")
        
        if 'github_profile' in missing_elements:
            recommendations.append("Connect your GitHub profile to showcase your coding projects.")
        
        if 'linkedin_profile' in missing_elements:
            recommendations.append("Connect your LinkedIn profile for professional network insights.")
        
        if skill_analysis['average_confidence'] < 0.6:
            recommendations.append("Consider updating your skill confidence levels or adding more evidence.")
        
        return recommendations