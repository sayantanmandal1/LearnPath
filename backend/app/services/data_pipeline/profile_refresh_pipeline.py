"""
User Profile Refresh Pipeline
Refreshes user profiles with updated data from external platforms.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import json

from app.core.logging import get_logger
from app.services.data_pipeline.pipeline_monitor import get_pipeline_monitor
from app.services.external_apis.integration_service import ExternalAPIIntegrationService
from app.repositories.profile import ProfileRepository
from app.repositories.user import UserRepository
from app.core.database import get_db_session

logger = get_logger(__name__)


class ProfileRefreshPipeline:
    """
    Pipeline for refreshing user profiles with updated external data
    """
    
    def __init__(self):
        self.integration_service = ExternalAPIIntegrationService()
        self.monitor = None
        
    async def execute(self, metadata: Dict[str, Any] = None):
        """Execute the profile refresh pipeline"""
        execution_id = metadata.get('execution_id', f"profile_refresh_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}")
        
        self.monitor = await get_pipeline_monitor()
        
        try:
            logger.info(f"Starting profile refresh pipeline: {execution_id}")
            
            # Initialize metrics
            await self.monitor.update_job_metrics(
                execution_id,
                status="running",
                records_processed=0,
                records_failed=0
            )
            
            # Get refresh parameters
            refresh_mode = metadata.get('refresh_mode', 'incremental')  # 'full' or 'incremental'
            max_profiles = metadata.get('max_profiles', 1000)
            platforms = metadata.get('platforms', ['github', 'linkedin', 'leetcode'])
            
            # Get profiles to refresh
            profiles_to_refresh = await self._get_profiles_to_refresh(refresh_mode, max_profiles)
            
            total_processed = 0
            total_failed = 0
            refresh_results = {
                'successful_refreshes': [],
                'failed_refreshes': [],
                'platform_stats': {platform: {'success': 0, 'failed': 0} for platform in platforms}
            }
            
            # Process profiles in batches
            batch_size = 50
            for i in range(0, len(profiles_to_refresh), batch_size):
                batch = profiles_to_refresh[i:i + batch_size]
                
                # Process batch concurrently
                batch_tasks = [
                    self._refresh_profile(profile, platforms)
                    for profile in batch
                ]
                
                batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
                
                # Process batch results
                for profile, result in zip(batch, batch_results):
                    if isinstance(result, Exception):
                        logger.error(f"Failed to refresh profile {profile.id}: {result}")
                        total_failed += 1
                        refresh_results['failed_refreshes'].append({
                            'profile_id': profile.id,
                            'user_id': profile.user_id,
                            'error': str(result)
                        })
                    else:
                        total_processed += 1
                        refresh_results['successful_refreshes'].append(result)
                        
                        # Update platform stats
                        for platform, success in result.get('platform_results', {}).items():
                            if success:
                                refresh_results['platform_stats'][platform]['success'] += 1
                            else:
                                refresh_results['platform_stats'][platform]['failed'] += 1
                
                # Update metrics after each batch
                await self.monitor.update_job_metrics(
                    execution_id,
                    records_processed=total_processed,
                    records_failed=total_failed
                )
                
                # Small delay between batches to avoid overwhelming external APIs
                await asyncio.sleep(1)
            
            # Calculate success rate for data quality score
            total_attempts = total_processed + total_failed
            success_rate = total_processed / total_attempts if total_attempts > 0 else 0.0
            
            # Store refresh summary
            refresh_summary = {
                'execution_id': execution_id,
                'timestamp': datetime.utcnow().isoformat(),
                'refresh_mode': refresh_mode,
                'total_profiles': len(profiles_to_refresh),
                'successful_refreshes': total_processed,
                'failed_refreshes': total_failed,
                'success_rate': success_rate,
                'platform_stats': refresh_results['platform_stats'],
                'results': refresh_results
            }
            
            await self._store_refresh_summary(refresh_summary)
            
            # Update final metrics
            await self.monitor.update_job_metrics(
                execution_id,
                status="completed",
                records_processed=total_processed,
                records_failed=total_failed,
                data_quality_score=success_rate
            )
            
            logger.info(f"Profile refresh pipeline completed: {total_processed} profiles refreshed")
            
        except Exception as e:
            logger.error(f"Profile refresh pipeline failed: {e}")
            
            await self.monitor.update_job_metrics(
                execution_id,
                status="failed",
                error_count=1
            )
            raise
    
    async def _get_profiles_to_refresh(self, refresh_mode: str, max_profiles: int) -> List[Any]:
        """Get list of profiles that need refreshing"""
        try:
            async with get_db_session() as db:
                profile_repo = ProfileRepository(db)
                
                if refresh_mode == 'full':
                    # Refresh all active profiles
                    profiles = await profile_repo.get_active_profiles(limit=max_profiles)
                else:
                    # Incremental refresh - only profiles that haven't been updated recently
                    cutoff_date = datetime.utcnow() - timedelta(hours=24)  # Refresh if older than 24 hours
                    profiles = await profile_repo.get_profiles_updated_before(
                        cutoff_date, 
                        limit=max_profiles
                    )
                
                logger.info(f"Found {len(profiles)} profiles to refresh")
                return profiles
                
        except Exception as e:
            logger.error(f"Failed to get profiles to refresh: {e}")
            return []
    
    async def _refresh_profile(self, profile: Any, platforms: List[str]) -> Dict[str, Any]:
        """Refresh a single user profile"""
        try:
            refresh_result = {
                'profile_id': profile.id,
                'user_id': profile.user_id,
                'platform_results': {},
                'updated_fields': [],
                'new_skills': [],
                'refresh_timestamp': datetime.utcnow().isoformat()
            }
            
            # Get current profile data
            current_data = {
                'github_username': profile.github_username,
                'linkedin_url': profile.linkedin_url,
                'leetcode_id': profile.leetcode_id,
                'skills': profile.skills or {}
            }
            
            updated_data = {}
            
            # Refresh data from each platform
            for platform in platforms:
                try:
                    platform_data = await self._refresh_platform_data(profile, platform)
                    
                    if platform_data:
                        refresh_result['platform_results'][platform] = True
                        
                        # Merge platform data
                        if platform == 'github' and platform_data.get('skills'):
                            updated_data['github_skills'] = platform_data['skills']
                            updated_data['github_languages'] = platform_data.get('languages', [])
                            updated_data['github_updated_at'] = datetime.utcnow()
                            
                        elif platform == 'linkedin' and platform_data.get('skills'):
                            updated_data['linkedin_skills'] = platform_data['skills']
                            updated_data['linkedin_experience'] = platform_data.get('experience', [])
                            updated_data['linkedin_updated_at'] = datetime.utcnow()
                            
                        elif platform == 'leetcode' and platform_data.get('skills'):
                            updated_data['leetcode_skills'] = platform_data['skills']
                            updated_data['leetcode_stats'] = platform_data.get('stats', {})
                            updated_data['leetcode_updated_at'] = datetime.utcnow()
                    else:
                        refresh_result['platform_results'][platform] = False
                        
                except Exception as e:
                    logger.error(f"Failed to refresh {platform} data for profile {profile.id}: {e}")
                    refresh_result['platform_results'][platform] = False
            
            # Update unified skills if we got new data
            if any(updated_data.get(f'{platform}_skills') for platform in platforms):
                unified_skills = await self._merge_skills(current_data['skills'], updated_data)
                updated_data['skills'] = unified_skills
                
                # Track new skills
                old_skills = set(current_data['skills'].keys()) if current_data['skills'] else set()
                new_skills = set(unified_skills.keys()) - old_skills
                refresh_result['new_skills'] = list(new_skills)
            
            # Update profile in database
            if updated_data:
                async with get_db_session() as db:
                    profile_repo = ProfileRepository(db)
                    await profile_repo.update_profile(profile.id, updated_data)
                    
                    refresh_result['updated_fields'] = list(updated_data.keys())
            
            return refresh_result
            
        except Exception as e:
            logger.error(f"Failed to refresh profile {profile.id}: {e}")
            raise
    
    async def _refresh_platform_data(self, profile: Any, platform: str) -> Optional[Dict[str, Any]]:
        """Refresh data from a specific platform"""
        try:
            if platform == 'github' and profile.github_username:
                return await self.integration_service.fetch_github_data(profile.github_username)
                
            elif platform == 'linkedin' and profile.linkedin_url:
                return await self.integration_service.fetch_linkedin_data(profile.linkedin_url)
                
            elif platform == 'leetcode' and profile.leetcode_id:
                return await self.integration_service.fetch_leetcode_data(profile.leetcode_id)
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to refresh {platform} data: {e}")
            return None
    
    async def _merge_skills(self, current_skills: Dict[str, float], updated_data: Dict[str, Any]) -> Dict[str, float]:
        """Merge skills from different platforms with confidence scoring"""
        try:
            merged_skills = current_skills.copy() if current_skills else {}
            
            # Platform-specific skill weights
            platform_weights = {
                'github': 0.8,
                'linkedin': 0.6,
                'leetcode': 0.7,
                'resume': 0.9  # Resume skills have highest weight
            }
            
            # Merge skills from each platform
            for platform in ['github', 'linkedin', 'leetcode']:
                platform_skills = updated_data.get(f'{platform}_skills', {})
                weight = platform_weights.get(platform, 0.5)
                
                for skill, confidence in platform_skills.items():
                    skill_lower = skill.lower()
                    
                    if skill_lower in merged_skills:
                        # Update existing skill with weighted average
                        current_confidence = merged_skills[skill_lower]
                        new_confidence = (current_confidence + confidence * weight) / 2
                        merged_skills[skill_lower] = min(new_confidence, 1.0)
                    else:
                        # Add new skill
                        merged_skills[skill_lower] = confidence * weight
            
            # Normalize confidence scores
            if merged_skills:
                max_confidence = max(merged_skills.values())
                if max_confidence > 1.0:
                    for skill in merged_skills:
                        merged_skills[skill] = merged_skills[skill] / max_confidence
            
            return merged_skills
            
        except Exception as e:
            logger.error(f"Failed to merge skills: {e}")
            return current_skills or {}
    
    async def _store_refresh_summary(self, summary: Dict[str, Any]):
        """Store profile refresh summary for reporting"""
        try:
            from app.core.redis import get_redis_client
            
            redis_client = await get_redis_client()
            
            # Store detailed summary
            summary_key = f"profile_refresh_summary:{summary['execution_id']}"
            await redis_client.set(
                summary_key,
                json.dumps(summary),
                ex=86400 * 30  # Keep for 30 days
            )
            
            # Update daily stats
            today = datetime.utcnow().strftime('%Y-%m-%d')
            daily_stats_key = f"profile_refresh_daily:{today}"
            
            await redis_client.hincrby(daily_stats_key, 'total_refreshed', summary['successful_refreshes'])
            await redis_client.hincrby(daily_stats_key, 'total_failed', summary['failed_refreshes'])
            await redis_client.expire(daily_stats_key, 86400 * 90)  # Keep for 90 days
            
            # Update platform-specific stats
            for platform, stats in summary['platform_stats'].items():
                platform_key = f"profile_refresh_platform:{platform}:{today}"
                await redis_client.hincrby(platform_key, 'success', stats['success'])
                await redis_client.hincrby(platform_key, 'failed', stats['failed'])
                await redis_client.expire(platform_key, 86400 * 90)
            
        except Exception as e:
            logger.error(f"Failed to store refresh summary: {e}")
    
    async def get_refresh_stats(self, days: int = 7) -> Dict[str, Any]:
        """Get profile refresh statistics for the last N days"""
        try:
            from app.core.redis import get_redis_client
            
            redis_client = await get_redis_client()
            stats = {
                'daily_stats': {},
                'platform_stats': {},
                'total_refreshed': 0,
                'total_failed': 0,
                'success_rate': 0.0
            }
            
            # Get daily stats
            for i in range(days):
                date = (datetime.utcnow() - timedelta(days=i)).strftime('%Y-%m-%d')
                daily_key = f"profile_refresh_daily:{date}"
                
                daily_data = await redis_client.hgetall(daily_key)
                if daily_data:
                    refreshed = int(daily_data.get('total_refreshed', 0))
                    failed = int(daily_data.get('total_failed', 0))
                    
                    stats['daily_stats'][date] = {
                        'refreshed': refreshed,
                        'failed': failed,
                        'success_rate': refreshed / (refreshed + failed) if (refreshed + failed) > 0 else 0.0
                    }
                    
                    stats['total_refreshed'] += refreshed
                    stats['total_failed'] += failed
            
            # Calculate overall success rate
            total_attempts = stats['total_refreshed'] + stats['total_failed']
            if total_attempts > 0:
                stats['success_rate'] = stats['total_refreshed'] / total_attempts
            
            # Get platform stats
            platforms = ['github', 'linkedin', 'leetcode']
            for platform in platforms:
                platform_stats = {'success': 0, 'failed': 0}
                
                for i in range(days):
                    date = (datetime.utcnow() - timedelta(days=i)).strftime('%Y-%m-%d')
                    platform_key = f"profile_refresh_platform:{platform}:{date}"
                    
                    platform_data = await redis_client.hgetall(platform_key)
                    if platform_data:
                        platform_stats['success'] += int(platform_data.get('success', 0))
                        platform_stats['failed'] += int(platform_data.get('failed', 0))
                
                # Calculate platform success rate
                platform_total = platform_stats['success'] + platform_stats['failed']
                platform_stats['success_rate'] = (
                    platform_stats['success'] / platform_total if platform_total > 0 else 0.0
                )
                
                stats['platform_stats'][platform] = platform_stats
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get refresh stats: {e}")
            return {}
    
    async def refresh_single_profile(self, profile_id: str, platforms: List[str] = None) -> Dict[str, Any]:
        """Refresh a single profile on demand"""
        try:
            if platforms is None:
                platforms = ['github', 'linkedin', 'leetcode']
            
            async with get_db_session() as db:
                profile_repo = ProfileRepository(db)
                profile = await profile_repo.get_profile(profile_id)
                
                if not profile:
                    raise ValueError(f"Profile {profile_id} not found")
                
                result = await self._refresh_profile(profile, platforms)
                
                logger.info(f"Single profile refresh completed for {profile_id}")
                return result
                
        except Exception as e:
            logger.error(f"Failed to refresh single profile {profile_id}: {e}")
            raise
    
    async def get_profile_freshness(self, profile_id: str) -> Dict[str, Any]:
        """Get freshness information for a profile"""
        try:
            async with get_db_session() as db:
                profile_repo = ProfileRepository(db)
                profile = await profile_repo.get_profile(profile_id)
                
                if not profile:
                    return {}
                
                now = datetime.utcnow()
                freshness = {
                    'profile_id': profile_id,
                    'last_updated': profile.updated_at.isoformat() if profile.updated_at else None,
                    'platform_freshness': {}
                }
                
                # Check freshness for each platform
                platforms = {
                    'github': profile.github_updated_at,
                    'linkedin': profile.linkedin_updated_at,
                    'leetcode': profile.leetcode_updated_at
                }
                
                for platform, last_update in platforms.items():
                    if last_update:
                        hours_since_update = (now - last_update).total_seconds() / 3600
                        freshness['platform_freshness'][platform] = {
                            'last_updated': last_update.isoformat(),
                            'hours_since_update': hours_since_update,
                            'is_stale': hours_since_update > 24  # Consider stale after 24 hours
                        }
                    else:
                        freshness['platform_freshness'][platform] = {
                            'last_updated': None,
                            'hours_since_update': None,
                            'is_stale': True
                        }
                
                return freshness
                
        except Exception as e:
            logger.error(f"Failed to get profile freshness for {profile_id}: {e}")
            return {}