"""
Data Quality Monitoring and Validation
Validates data quality across all pipeline operations.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import json
import re
from collections import Counter, defaultdict

from app.core.logging import get_logger
from app.core.redis import get_redis
from app.repositories.job import JobRepository
from app.repositories.profile import ProfileRepository
from app.repositories.skill import SkillRepository
from app.core.database import get_db

logger = get_logger(__name__)


class DataQualityValidator:
    """
    Validates and monitors data quality across all pipeline operations
    """
    
    def __init__(self):
        self.quality_thresholds = {
            'job_posting_completeness': 0.8,  # 80% of required fields must be present
            'skill_extraction_accuracy': 0.85,  # 85% accuracy threshold
            'profile_data_freshness': 24,  # Hours before data is considered stale
            'duplicate_rate_threshold': 0.1,  # Max 10% duplicates allowed
            'data_consistency_threshold': 0.9  # 90% consistency required
        }
        
    async def validate_job_posting(self, job_data: Dict[str, Any]) -> bool:
        """Validate a single job posting for quality"""
        try:
            required_fields = ['title', 'company', 'description', 'location']
            optional_fields = ['salary_range', 'experience_level', 'required_skills', 'posted_date']
            
            # Check required fields
            missing_required = [field for field in required_fields if not job_data.get(field)]
            if missing_required:
                logger.warning(f"Job posting missing required fields: {missing_required}")
                return False
            
            # Validate field content quality
            if not self._validate_job_title(job_data['title']):
                return False
            
            if not self._validate_job_description(job_data['description']):
                return False
            
            if not self._validate_company_name(job_data['company']):
                return False
            
            # Check for suspicious patterns (spam, duplicates, etc.)
            if self._detect_spam_patterns(job_data):
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to validate job posting: {e}")
            return False
    
    async def validate_profile_data(self, profile_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate profile data quality and return validation results"""
        try:
            validation_result = {
                'is_valid': True,
                'quality_score': 1.0,
                'issues': [],
                'warnings': []
            }
            
            # Check required fields
            required_fields = ['user_id']
            missing_required = [field for field in required_fields if not profile_data.get(field)]
            
            if missing_required:
                validation_result['is_valid'] = False
                validation_result['issues'].append(f"Missing required fields: {missing_required}")
                validation_result['quality_score'] *= 0.5
            
            # Validate skills data
            skills_validation = await self._validate_skills_data(profile_data.get('skills', {}))
            validation_result['quality_score'] *= skills_validation['quality_score']
            validation_result['issues'].extend(skills_validation['issues'])
            validation_result['warnings'].extend(skills_validation['warnings'])
            
            # Check data freshness
            freshness_validation = self._validate_data_freshness(profile_data)
            validation_result['quality_score'] *= freshness_validation['quality_score']
            validation_result['warnings'].extend(freshness_validation['warnings'])
            
            # Validate external platform data
            platform_validation = await self._validate_platform_data(profile_data)
            validation_result['quality_score'] *= platform_validation['quality_score']
            validation_result['issues'].extend(platform_validation['issues'])
            
            return validation_result
            
        except Exception as e:
            logger.error(f"Failed to validate profile data: {e}")
            return {
                'is_valid': False,
                'quality_score': 0.0,
                'issues': [f"Validation error: {str(e)}"],
                'warnings': []
            }
    
    async def validate_skill_taxonomy(self) -> Dict[str, Any]:
        """Validate the overall skill taxonomy quality"""
        try:
            validation_result = {
                'quality_score': 1.0,
                'total_skills': 0,
                'issues': [],
                'warnings': [],
                'statistics': {}
            }
            
            async with get_db_session() as db:
                skill_repo = SkillRepository(db)
                all_skills = await skill_repo.get_all_skills()
                
                validation_result['total_skills'] = len(all_skills)
                
                # Check for duplicate skills
                skill_names = [skill.name.lower() for skill in all_skills]
                duplicates = [name for name, count in Counter(skill_names).items() if count > 1]
                
                if duplicates:
                    validation_result['issues'].append(f"Duplicate skills found: {len(duplicates)}")
                    validation_result['quality_score'] *= 0.9
                
                # Check for skills without categories
                uncategorized = [skill for skill in all_skills if not skill.category]
                if uncategorized:
                    validation_result['warnings'].append(f"Uncategorized skills: {len(uncategorized)}")
                    validation_result['quality_score'] *= 0.95
                
                # Check for obsolete skills
                obsolete_threshold = datetime.utcnow() - timedelta(days=180)
                potentially_obsolete = [
                    skill for skill in all_skills 
                    if skill.last_updated and skill.last_updated < obsolete_threshold
                    and skill.demand_score and skill.demand_score < 0.01
                ]
                
                if potentially_obsolete:
                    validation_result['warnings'].append(f"Potentially obsolete skills: {len(potentially_obsolete)}")
                
                # Generate statistics
                validation_result['statistics'] = {
                    'total_skills': len(all_skills),
                    'categorized_skills': len([s for s in all_skills if s.category]),
                    'skills_with_demand_scores': len([s for s in all_skills if s.demand_score]),
                    'emerging_skills': len([s for s in all_skills if getattr(s, 'is_emerging', False)]),
                    'obsolete_skills': len([s for s in all_skills if getattr(s, 'is_obsolete', False)]),
                    'duplicate_count': len(duplicates),
                    'categories': len(set(skill.category for skill in all_skills if skill.category))
                }
            
            return validation_result
            
        except Exception as e:
            logger.error(f"Failed to validate skill taxonomy: {e}")
            return {
                'quality_score': 0.0,
                'issues': [f"Validation error: {str(e)}"],
                'warnings': [],
                'statistics': {}
            }
    
    async def run_data_quality_checks(self) -> Dict[str, Any]:
        """Run comprehensive data quality checks across all data"""
        try:
            logger.info("Running comprehensive data quality checks")
            
            quality_report = {
                'timestamp': datetime.utcnow().isoformat(),
                'overall_quality_score': 1.0,
                'checks': {},
                'issues': [],
                'warnings': [],
                'recommendations': []
            }
            
            # Check job postings quality
            job_quality = await self._check_job_postings_quality()
            quality_report['checks']['job_postings'] = job_quality
            quality_report['overall_quality_score'] *= job_quality['quality_score']
            
            # Check profile data quality
            profile_quality = await self._check_profiles_quality()
            quality_report['checks']['profiles'] = profile_quality
            quality_report['overall_quality_score'] *= profile_quality['quality_score']
            
            # Check skill taxonomy quality
            skill_quality = await self.validate_skill_taxonomy()
            quality_report['checks']['skill_taxonomy'] = skill_quality
            quality_report['overall_quality_score'] *= skill_quality['quality_score']
            
            # Check data consistency
            consistency_check = await self._check_data_consistency()
            quality_report['checks']['data_consistency'] = consistency_check
            quality_report['overall_quality_score'] *= consistency_check['quality_score']
            
            # Check data freshness
            freshness_check = await self._check_data_freshness()
            quality_report['checks']['data_freshness'] = freshness_check
            quality_report['overall_quality_score'] *= freshness_check['quality_score']
            
            # Aggregate issues and warnings
            for check_name, check_result in quality_report['checks'].items():
                quality_report['issues'].extend([
                    f"{check_name}: {issue}" for issue in check_result.get('issues', [])
                ])
                quality_report['warnings'].extend([
                    f"{check_name}: {warning}" for warning in check_result.get('warnings', [])
                ])
            
            # Generate recommendations
            quality_report['recommendations'] = self._generate_quality_recommendations(quality_report)
            
            # Store quality report
            await self._store_quality_report(quality_report)
            
            logger.info(f"Data quality check completed. Overall score: {quality_report['overall_quality_score']:.2f}")
            
            return quality_report
            
        except Exception as e:
            logger.error(f"Failed to run data quality checks: {e}")
            return {
                'timestamp': datetime.utcnow().isoformat(),
                'overall_quality_score': 0.0,
                'error': str(e)
            }
    
    def _validate_job_title(self, title: str) -> bool:
        """Validate job title quality"""
        if not title or len(title.strip()) < 3:
            return False
        
        # Check for suspicious patterns
        suspicious_patterns = [
            r'^\$+',  # Starts with dollar signs
            r'[!@#$%^&*]{3,}',  # Multiple special characters
            r'(?i)(urgent|immediate|asap){2,}',  # Repeated urgent words
            r'(?i)(work from home|wfh|remote).*(easy|simple|no experience)',  # Suspicious remote work
        ]
        
        for pattern in suspicious_patterns:
            if re.search(pattern, title):
                return False
        
        return True
    
    def _validate_job_description(self, description: str) -> bool:
        """Validate job description quality"""
        if not description or len(description.strip()) < 50:
            return False
        
        # Check for minimum content requirements
        description_lower = description.lower()
        
        # Should contain some job-related keywords
        job_keywords = ['responsibilities', 'requirements', 'experience', 'skills', 'qualifications']
        if not any(keyword in description_lower for keyword in job_keywords):
            return False
        
        # Check for spam patterns
        spam_patterns = [
            r'(?i)(make money|earn \$|guaranteed income)',
            r'(?i)(no experience required).*(high pay|excellent salary)',
            r'[A-Z]{10,}',  # Too many consecutive capitals
            r'(.)\1{5,}',  # Repeated characters
        ]
        
        for pattern in spam_patterns:
            if re.search(pattern, description):
                return False
        
        return True
    
    def _validate_company_name(self, company: str) -> bool:
        """Validate company name quality"""
        if not company or len(company.strip()) < 2:
            return False
        
        # Check for suspicious patterns
        suspicious_patterns = [
            r'(?i)(confidential|undisclosed|private)',
            r'^[^a-zA-Z]*$',  # No letters at all
            r'[!@#$%^&*]{2,}',  # Multiple special characters
        ]
        
        for pattern in suspicious_patterns:
            if re.search(pattern, company):
                return False
        
        return True
    
    def _detect_spam_patterns(self, job_data: Dict[str, Any]) -> bool:
        """Detect spam patterns in job posting"""
        # Combine all text fields
        text_content = ' '.join([
            str(job_data.get('title', '')),
            str(job_data.get('description', '')),
            str(job_data.get('company', ''))
        ]).lower()
        
        # Spam indicators
        spam_indicators = [
            r'(?i)(work from home).*(no experience).*(high pay)',
            r'(?i)(make \$\d+).*(per day|per week).*(guaranteed)',
            r'(?i)(urgent|immediate).*(hiring|needed).*(apply now)',
            r'[!]{3,}',  # Multiple exclamation marks
            r'(?i)(100% remote).*(no interview|no experience)',
        ]
        
        spam_score = 0
        for pattern in spam_indicators:
            if re.search(pattern, text_content):
                spam_score += 1
        
        return spam_score >= 2  # Consider spam if 2+ indicators
    
    async def _validate_skills_data(self, skills: Dict[str, float]) -> Dict[str, Any]:
        """Validate skills data quality"""
        validation_result = {
            'quality_score': 1.0,
            'issues': [],
            'warnings': []
        }
        
        if not skills:
            validation_result['warnings'].append("No skills data provided")
            validation_result['quality_score'] *= 0.8
            return validation_result
        
        # Check confidence scores
        invalid_scores = [skill for skill, score in skills.items() if not (0 <= score <= 1)]
        if invalid_scores:
            validation_result['issues'].append(f"Invalid confidence scores for skills: {invalid_scores}")
            validation_result['quality_score'] *= 0.7
        
        # Check for suspicious skill names
        suspicious_skills = []
        for skill in skills.keys():
            if len(skill) < 2 or len(skill) > 50:
                suspicious_skills.append(skill)
            elif re.search(r'[!@#$%^&*()]{2,}', skill):
                suspicious_skills.append(skill)
        
        if suspicious_skills:
            validation_result['warnings'].append(f"Suspicious skill names: {suspicious_skills}")
            validation_result['quality_score'] *= 0.9
        
        return validation_result
    
    def _validate_data_freshness(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate data freshness"""
        validation_result = {
            'quality_score': 1.0,
            'warnings': []
        }
        
        now = datetime.utcnow()
        freshness_threshold = timedelta(hours=self.quality_thresholds['profile_data_freshness'])
        
        # Check various timestamp fields
        timestamp_fields = ['updated_at', 'github_updated_at', 'linkedin_updated_at', 'leetcode_updated_at']
        
        stale_fields = []
        for field in timestamp_fields:
            if field in data and data[field]:
                try:
                    if isinstance(data[field], str):
                        timestamp = datetime.fromisoformat(data[field].replace('Z', '+00:00'))
                    else:
                        timestamp = data[field]
                    
                    if now - timestamp > freshness_threshold:
                        stale_fields.append(field)
                except Exception:
                    validation_result['warnings'].append(f"Invalid timestamp format for {field}")
        
        if stale_fields:
            validation_result['warnings'].append(f"Stale data fields: {stale_fields}")
            validation_result['quality_score'] *= 0.9
        
        return validation_result
    
    async def _validate_platform_data(self, profile_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate external platform data"""
        validation_result = {
            'quality_score': 1.0,
            'issues': []
        }
        
        # Validate GitHub data
        if 'github_username' in profile_data and profile_data['github_username']:
            if not re.match(r'^[a-zA-Z0-9]([a-zA-Z0-9-])*[a-zA-Z0-9]$', profile_data['github_username']):
                validation_result['issues'].append("Invalid GitHub username format")
                validation_result['quality_score'] *= 0.9
        
        # Validate LinkedIn URL
        if 'linkedin_url' in profile_data and profile_data['linkedin_url']:
            linkedin_pattern = r'https?://(www\.)?linkedin\.com/in/[a-zA-Z0-9-]+'
            if not re.match(linkedin_pattern, profile_data['linkedin_url']):
                validation_result['issues'].append("Invalid LinkedIn URL format")
                validation_result['quality_score'] *= 0.9
        
        # Validate LeetCode ID
        if 'leetcode_id' in profile_data and profile_data['leetcode_id']:
            if not re.match(r'^[a-zA-Z0-9_-]+$', profile_data['leetcode_id']):
                validation_result['issues'].append("Invalid LeetCode ID format")
                validation_result['quality_score'] *= 0.9
        
        return validation_result
    
    async def _check_job_postings_quality(self) -> Dict[str, Any]:
        """Check overall job postings data quality"""
        try:
            async with get_db_session() as db:
                job_repo = JobRepository(db)
                
                # Get recent job postings
                recent_jobs = await job_repo.get_recent_jobs(days=7)
                
                if not recent_jobs:
                    return {
                        'quality_score': 0.0,
                        'issues': ['No recent job postings found'],
                        'warnings': []
                    }
                
                # Check completeness
                complete_jobs = 0
                for job in recent_jobs:
                    required_fields = ['title', 'company', 'description', 'location']
                    if all(getattr(job, field, None) for field in required_fields):
                        complete_jobs += 1
                
                completeness_rate = complete_jobs / len(recent_jobs)
                
                # Check for duplicates
                job_signatures = []
                for job in recent_jobs:
                    signature = f"{job.title}_{job.company}_{job.location}"
                    job_signatures.append(signature)
                
                unique_jobs = len(set(job_signatures))
                duplicate_rate = 1 - (unique_jobs / len(recent_jobs))
                
                quality_score = completeness_rate * (1 - min(duplicate_rate, 0.5))
                
                result = {
                    'quality_score': quality_score,
                    'issues': [],
                    'warnings': [],
                    'statistics': {
                        'total_jobs': len(recent_jobs),
                        'complete_jobs': complete_jobs,
                        'completeness_rate': completeness_rate,
                        'duplicate_rate': duplicate_rate,
                        'unique_jobs': unique_jobs
                    }
                }
                
                if completeness_rate < self.quality_thresholds['job_posting_completeness']:
                    result['issues'].append(f"Low job posting completeness: {completeness_rate:.1%}")
                
                if duplicate_rate > self.quality_thresholds['duplicate_rate_threshold']:
                    result['warnings'].append(f"High duplicate rate: {duplicate_rate:.1%}")
                
                return result
                
        except Exception as e:
            logger.error(f"Failed to check job postings quality: {e}")
            return {
                'quality_score': 0.0,
                'issues': [f"Quality check error: {str(e)}"],
                'warnings': []
            }
    
    async def _check_profiles_quality(self) -> Dict[str, Any]:
        """Check overall profile data quality"""
        try:
            async with get_db_session() as db:
                profile_repo = ProfileRepository(db)
                
                # Get recent profiles
                recent_profiles = await profile_repo.get_recent_profiles(days=30)
                
                if not recent_profiles:
                    return {
                        'quality_score': 0.0,
                        'issues': ['No recent profiles found'],
                        'warnings': []
                    }
                
                # Check profile completeness
                complete_profiles = 0
                profiles_with_skills = 0
                profiles_with_external_data = 0
                
                for profile in recent_profiles:
                    # Check basic completeness
                    if profile.user_id and profile.skills:
                        complete_profiles += 1
                    
                    # Check skills data
                    if profile.skills:
                        profiles_with_skills += 1
                    
                    # Check external platform data
                    if (profile.github_username or profile.linkedin_url or profile.leetcode_id):
                        profiles_with_external_data += 1
                
                completeness_rate = complete_profiles / len(recent_profiles)
                skills_rate = profiles_with_skills / len(recent_profiles)
                external_data_rate = profiles_with_external_data / len(recent_profiles)
                
                quality_score = (completeness_rate + skills_rate + external_data_rate) / 3
                
                result = {
                    'quality_score': quality_score,
                    'issues': [],
                    'warnings': [],
                    'statistics': {
                        'total_profiles': len(recent_profiles),
                        'complete_profiles': complete_profiles,
                        'completeness_rate': completeness_rate,
                        'profiles_with_skills': profiles_with_skills,
                        'skills_rate': skills_rate,
                        'profiles_with_external_data': profiles_with_external_data,
                        'external_data_rate': external_data_rate
                    }
                }
                
                if completeness_rate < 0.7:
                    result['issues'].append(f"Low profile completeness: {completeness_rate:.1%}")
                
                if skills_rate < 0.8:
                    result['warnings'].append(f"Many profiles lack skills data: {skills_rate:.1%}")
                
                return result
                
        except Exception as e:
            logger.error(f"Failed to check profiles quality: {e}")
            return {
                'quality_score': 0.0,
                'issues': [f"Quality check error: {str(e)}"],
                'warnings': []
            }
    
    async def _check_data_consistency(self) -> Dict[str, Any]:
        """Check data consistency across different data sources"""
        try:
            # This would check for consistency between different data sources
            # For example, skills mentioned in job postings vs. skills in taxonomy
            
            consistency_score = 1.0
            issues = []
            warnings = []
            
            # Check skill consistency between jobs and taxonomy
            async with get_db_session() as db:
                job_repo = JobRepository(db)
                skill_repo = SkillRepository(db)
                
                recent_jobs = await job_repo.get_recent_jobs(days=30)
                all_skills = await skill_repo.get_all_skills()
                
                # Extract skills mentioned in job postings
                job_skills = set()
                for job in recent_jobs:
                    if job.required_skills:
                        job_skills.update(skill.lower() for skill in job.required_skills)
                
                # Get skills in taxonomy
                taxonomy_skills = set(skill.name.lower() for skill in all_skills)
                
                # Find skills mentioned in jobs but not in taxonomy
                missing_skills = job_skills - taxonomy_skills
                if missing_skills and len(missing_skills) > len(job_skills) * 0.1:  # More than 10% missing
                    warnings.append(f"Many job skills not in taxonomy: {len(missing_skills)} skills")
                    consistency_score *= 0.9
            
            return {
                'quality_score': consistency_score,
                'issues': issues,
                'warnings': warnings,
                'statistics': {
                    'job_skills_count': len(job_skills) if 'job_skills' in locals() else 0,
                    'taxonomy_skills_count': len(taxonomy_skills) if 'taxonomy_skills' in locals() else 0,
                    'missing_skills_count': len(missing_skills) if 'missing_skills' in locals() else 0
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to check data consistency: {e}")
            return {
                'quality_score': 0.0,
                'issues': [f"Consistency check error: {str(e)}"],
                'warnings': []
            }
    
    async def _check_data_freshness(self) -> Dict[str, Any]:
        """Check overall data freshness"""
        try:
            freshness_score = 1.0
            issues = []
            warnings = []
            
            now = datetime.utcnow()
            stale_threshold = timedelta(hours=48)  # 48 hours
            
            async with get_db_session() as db:
                # Check job postings freshness
                job_repo = JobRepository(db)
                recent_jobs = await job_repo.get_jobs_created_since(now - stale_threshold)
                total_jobs = await job_repo.count_all_jobs()
                
                if total_jobs > 0:
                    fresh_job_rate = len(recent_jobs) / min(total_jobs, 1000)  # Sample of 1000
                    if fresh_job_rate < 0.1:  # Less than 10% fresh jobs
                        warnings.append(f"Low fresh job posting rate: {fresh_job_rate:.1%}")
                        freshness_score *= 0.9
                
                # Check profile freshness
                profile_repo = ProfileRepository(db)
                recent_profiles = await profile_repo.get_profiles_updated_since(now - stale_threshold)
                total_profiles = await profile_repo.count_all_profiles()
                
                if total_profiles > 0:
                    fresh_profile_rate = len(recent_profiles) / min(total_profiles, 1000)
                    if fresh_profile_rate < 0.05:  # Less than 5% fresh profiles
                        warnings.append(f"Low fresh profile rate: {fresh_profile_rate:.1%}")
                        freshness_score *= 0.9
            
            return {
                'quality_score': freshness_score,
                'issues': issues,
                'warnings': warnings,
                'statistics': {
                    'fresh_jobs': len(recent_jobs) if 'recent_jobs' in locals() else 0,
                    'fresh_profiles': len(recent_profiles) if 'recent_profiles' in locals() else 0,
                    'fresh_job_rate': fresh_job_rate if 'fresh_job_rate' in locals() else 0,
                    'fresh_profile_rate': fresh_profile_rate if 'fresh_profile_rate' in locals() else 0
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to check data freshness: {e}")
            return {
                'quality_score': 0.0,
                'issues': [f"Freshness check error: {str(e)}"],
                'warnings': []
            }
    
    def _generate_quality_recommendations(self, quality_report: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on quality report"""
        recommendations = []
        
        overall_score = quality_report['overall_quality_score']
        
        if overall_score < 0.7:
            recommendations.append("Overall data quality is low - consider comprehensive data cleanup")
        
        # Check specific areas
        checks = quality_report.get('checks', {})
        
        if checks.get('job_postings', {}).get('quality_score', 1.0) < 0.8:
            recommendations.append("Improve job posting data collection and validation")
        
        if checks.get('profiles', {}).get('quality_score', 1.0) < 0.8:
            recommendations.append("Enhance profile data completeness and validation")
        
        if checks.get('skill_taxonomy', {}).get('quality_score', 1.0) < 0.9:
            recommendations.append("Update and clean skill taxonomy")
        
        if checks.get('data_consistency', {}).get('quality_score', 1.0) < 0.9:
            recommendations.append("Address data consistency issues between sources")
        
        if checks.get('data_freshness', {}).get('quality_score', 1.0) < 0.8:
            recommendations.append("Increase data refresh frequency")
        
        return recommendations
    
    async def _store_quality_report(self, quality_report: Dict[str, Any]):
        """Store quality report in Redis"""
        try:
            redis_manager = await get_redis()
            redis_client = redis_manager.redis
            
            # Store detailed report
            report_key = f"data_quality_report:{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            await redis_client.set(
                report_key,
                json.dumps(quality_report),
                ex=86400 * 30  # Keep for 30 days
            )
            
            # Store summary metrics
            today = datetime.utcnow().strftime('%Y-%m-%d')
            metrics_key = f"data_quality_metrics:{today}"
            
            await redis_client.hset(metrics_key, mapping={
                'overall_score': quality_report['overall_quality_score'],
                'issues_count': len(quality_report['issues']),
                'warnings_count': len(quality_report['warnings']),
                'timestamp': quality_report['timestamp']
            })
            await redis_client.expire(metrics_key, 86400 * 90)  # Keep for 90 days
            
        except Exception as e:
            logger.error(f"Failed to store quality report: {e}")
    
    async def get_quality_history(self, days: int = 30) -> Dict[str, Any]:
        """Get data quality history for the last N days"""
        try:
            redis_manager = await get_redis()
            redis_client = redis_manager.redis
            history = {
                'daily_scores': {},
                'average_score': 0.0,
                'trend': 'stable'
            }
            
            scores = []
            for i in range(days):
                date = (datetime.utcnow() - timedelta(days=i)).strftime('%Y-%m-%d')
                metrics_key = f"data_quality_metrics:{date}"
                
                metrics = await redis_client.hgetall(metrics_key)
                if metrics:
                    score = float(metrics.get('overall_score', 0))
                    history['daily_scores'][date] = {
                        'score': score,
                        'issues_count': int(metrics.get('issues_count', 0)),
                        'warnings_count': int(metrics.get('warnings_count', 0))
                    }
                    scores.append(score)
            
            if scores:
                history['average_score'] = sum(scores) / len(scores)
                
                # Determine trend
                if len(scores) >= 7:
                    recent_avg = sum(scores[:7]) / 7
                    older_avg = sum(scores[7:14]) / min(7, len(scores) - 7) if len(scores) > 7 else recent_avg
                    
                    if recent_avg > older_avg + 0.05:
                        history['trend'] = 'improving'
                    elif recent_avg < older_avg - 0.05:
                        history['trend'] = 'declining'
            
            return history
            
        except Exception as e:
            logger.error(f"Failed to get quality history: {e}")
            return {}