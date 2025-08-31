"""
Job analysis service for parsing job descriptions and extracting skills
"""
import asyncio
import logging
import re
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime, timedelta

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from app.repositories.job import JobRepository, JobSkillRepository
from app.repositories.skill import SkillRepository
from app.models.job import JobPosting, JobSkill
from app.models.skill import Skill
from app.core.exceptions import ServiceException as ProcessingError

# Import NLP engine
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..', 'machinelearningmodel'))

try:
    from nlp_engine import NLPEngine
    from models import SkillExtraction, SkillCategory
except ImportError:
    # Fallback for testing
    NLPEngine = None
    SkillExtraction = None
    SkillCategory = None

logger = logging.getLogger(__name__)


class JobAnalysisService:
    """Service for analyzing job postings and extracting skills"""
    
    def __init__(self):
        self.job_repository = JobRepository()
        self.job_skill_repository = JobSkillRepository()
        self.skill_repository = SkillRepository()
        self.nlp_engine = NLPEngine() if NLPEngine else None
        
        # Skill importance keywords
        self.importance_keywords = {
            'required': [
                'required', 'must have', 'essential', 'mandatory', 'necessary',
                'need', 'should have', 'minimum', 'prerequisite'
            ],
            'preferred': [
                'preferred', 'desired', 'nice to have', 'bonus', 'plus',
                'advantage', 'beneficial', 'ideal', 'would be great'
            ],
            'nice-to-have': [
                'nice to have', 'optional', 'additional', 'extra',
                'would be nice', 'helpful', 'a plus'
            ]
        }
        
        # Experience level keywords
        self.experience_keywords = {
            'entry': ['entry', 'junior', 'graduate', 'intern', '0-2 years', 'new grad'],
            'mid': ['mid', 'intermediate', '2-5 years', '3-7 years', 'experienced'],
            'senior': ['senior', 'sr', 'lead', '5+ years', '7+ years', 'expert'],
            'executive': ['principal', 'staff', 'director', 'manager', 'head', 'chief']
        }
    
    async def process_unprocessed_jobs(
        self,
        db: AsyncSession,
        batch_size: int = 50,
        max_jobs: Optional[int] = None
    ) -> Dict[str, int]:
        """
        Process unprocessed job postings to extract skills
        
        Args:
            db: Database session
            batch_size: Number of jobs to process in each batch
            max_jobs: Maximum number of jobs to process (None for all)
            
        Returns:
            Processing statistics
        """
        if not self.nlp_engine:
            raise ProcessingError("NLP engine not available")
        
        stats = {
            'processed': 0,
            'failed': 0,
            'skills_extracted': 0
        }
        
        processed_count = 0
        
        while max_jobs is None or processed_count < max_jobs:
            # Get batch of unprocessed jobs
            remaining = max_jobs - processed_count if max_jobs else batch_size
            limit = min(batch_size, remaining)
            
            unprocessed_jobs = await self.job_repository.get_unprocessed_jobs(db, limit)
            
            if not unprocessed_jobs:
                break
            
            logger.info(f"Processing batch of {len(unprocessed_jobs)} jobs")
            
            # Process jobs concurrently
            tasks = [
                self._process_single_job(db, job)
                for job in unprocessed_jobs
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Update statistics
            for result in results:
                if isinstance(result, Exception):
                    stats['failed'] += 1
                    logger.error(f"Job processing failed: {str(result)}")
                else:
                    stats['processed'] += 1
                    stats['skills_extracted'] += result
            
            processed_count += len(unprocessed_jobs)
            
            # Commit batch
            await db.commit()
        
        logger.info(f"Job processing completed: {stats}")
        return stats
    
    async def _process_single_job(self, db: AsyncSession, job: JobPosting) -> int:
        """
        Process a single job posting to extract skills
        
        Args:
            db: Database session
            job: Job posting to process
            
        Returns:
            Number of skills extracted
        """
        try:
            # Combine job text for analysis
            job_text = f"{job.title}\n{job.description}"
            if job.requirements:
                job_text += f"\n{job.requirements}"
            
            # Extract skills using NLP
            skill_extractions = await self.nlp_engine.extract_skills(job_text)
            
            # Process extracted skills
            skills_count = 0
            processed_skills = {}
            
            for extraction in skill_extractions:
                # Find or create skill in database
                skill = await self._find_or_create_skill(db, extraction)
                
                # Determine importance level
                importance = self._determine_skill_importance(
                    extraction.skill_name, job_text
                )
                
                # Determine experience requirements
                years_required = self._extract_years_required(
                    extraction.skill_name, job_text
                )
                
                # Create job-skill relationship
                job_skill_data = {
                    'job_posting_id': job.id,
                    'skill_id': skill.id,
                    'importance': importance,
                    'confidence_score': extraction.confidence_score,
                    'years_required': years_required,
                    'proficiency_level': self._determine_proficiency_level(
                        extraction.skill_name, job_text
                    ),
                    'context': extraction.evidence[0] if extraction.evidence else None
                }
                
                await self.job_skill_repository.create(db, job_skill_data)
                
                # Add to processed skills
                processed_skills[extraction.skill_name] = extraction.confidence_score
                skills_count += 1
            
            # Mark job as processed
            await self.job_repository.mark_as_processed(
                db, job.id, processed_skills
            )
            
            logger.debug(f"Processed job {job.id}: extracted {skills_count} skills")
            return skills_count
            
        except Exception as e:
            logger.error(f"Failed to process job {job.id}: {str(e)}")
            raise
    
    async def _find_or_create_skill(
        self,
        db: AsyncSession,
        extraction: Any
    ) -> Skill:
        """
        Find existing skill or create new one
        
        Args:
            db: Database session
            extraction: Skill extraction object
            
        Returns:
            Skill instance
        """
        # Try to find existing skill
        existing_skill = await self.skill_repository.get_by_name(
            db, extraction.skill_name
        )
        
        if existing_skill:
            return existing_skill
        
        # Create new skill
        skill_data = {
            'name': extraction.skill_name,
            'category': extraction.category.value if hasattr(extraction.category, 'value') else 'technical',
            'description': f"Skill extracted from job postings",
            'is_verified': False
        }
        
        return await self.skill_repository.create(db, skill_data)
    
    def _determine_skill_importance(self, skill_name: str, job_text: str) -> str:
        """
        Determine skill importance level from job text
        
        Args:
            skill_name: Name of the skill
            job_text: Full job posting text
            
        Returns:
            Importance level: 'required', 'preferred', or 'nice-to-have'
        """
        job_text_lower = job_text.lower()
        skill_lower = skill_name.lower()
        
        # Find skill mentions in context
        skill_contexts = []
        for match in re.finditer(re.escape(skill_lower), job_text_lower):
            start = max(0, match.start() - 100)
            end = min(len(job_text_lower), match.end() + 100)
            context = job_text_lower[start:end]
            skill_contexts.append(context)
        
        # Check for importance keywords in context
        importance_scores = {'required': 0, 'preferred': 0, 'nice-to-have': 0}
        
        for context in skill_contexts:
            for importance, keywords in self.importance_keywords.items():
                for keyword in keywords:
                    if keyword in context:
                        importance_scores[importance] += 1
        
        # Return highest scoring importance level
        if importance_scores['required'] > 0:
            return 'required'
        elif importance_scores['preferred'] > 0:
            return 'preferred'
        else:
            return 'nice-to-have'
    
    def _extract_years_required(self, skill_name: str, job_text: str) -> Optional[int]:
        """
        Extract years of experience required for a skill
        
        Args:
            skill_name: Name of the skill
            job_text: Full job posting text
            
        Returns:
            Years required or None
        """
        job_text_lower = job_text.lower()
        skill_lower = skill_name.lower()
        
        # Find skill mentions and look for year patterns nearby
        for match in re.finditer(re.escape(skill_lower), job_text_lower):
            start = max(0, match.start() - 50)
            end = min(len(job_text_lower), match.end() + 50)
            context = job_text_lower[start:end]
            
            # Look for year patterns
            year_patterns = [
                r'(\d+)\+?\s*years?',
                r'(\d+)-\d+\s*years?',
                r'minimum\s+(\d+)\s*years?',
                r'at least\s+(\d+)\s*years?'
            ]
            
            for pattern in year_patterns:
                match_years = re.search(pattern, context)
                if match_years:
                    return int(match_years.group(1))
        
        return None
    
    def _determine_proficiency_level(self, skill_name: str, job_text: str) -> Optional[str]:
        """
        Determine required proficiency level for a skill
        
        Args:
            skill_name: Name of the skill
            job_text: Full job posting text
            
        Returns:
            Proficiency level or None
        """
        job_text_lower = job_text.lower()
        skill_lower = skill_name.lower()
        
        proficiency_keywords = {
            'beginner': ['beginner', 'basic', 'fundamental', 'introductory'],
            'intermediate': ['intermediate', 'working knowledge', 'proficient'],
            'advanced': ['advanced', 'expert', 'deep', 'extensive', 'mastery'],
            'expert': ['expert', 'guru', 'ninja', 'rockstar', 'wizard']
        }
        
        # Find skill mentions and check for proficiency keywords
        for match in re.finditer(re.escape(skill_lower), job_text_lower):
            start = max(0, match.start() - 100)
            end = min(len(job_text_lower), match.end() + 100)
            context = job_text_lower[start:end]
            
            for level, keywords in proficiency_keywords.items():
                if any(keyword in context for keyword in keywords):
                    return level
        
        return None
    
    async def analyze_market_trends(
        self,
        db: AsyncSession,
        days: int = 90,
        min_job_count: int = 10
    ) -> Dict[str, Any]:
        """
        Analyze job market trends from processed job data
        
        Args:
            db: Database session
            days: Number of days to analyze
            min_job_count: Minimum job count for trend analysis
            
        Returns:
            Market trend analysis
        """
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        # Get skill demand statistics
        query = select(
            Skill.name,
            Skill.category,
            func.count(JobSkill.id).label('job_count'),
            func.avg(JobSkill.confidence_score).label('avg_confidence'),
            func.count(
                func.case([(JobSkill.importance == 'required', 1)])
            ).label('required_count'),
            func.count(
                func.case([(JobSkill.importance == 'preferred', 1)])
            ).label('preferred_count')
        ).select_from(
            JobSkill.__table__.join(Skill.__table__).join(JobPosting.__table__)
        ).where(
            JobPosting.posted_date >= cutoff_date
        ).group_by(
            Skill.id, Skill.name, Skill.category
        ).having(
            func.count(JobSkill.id) >= min_job_count
        ).order_by(
            func.count(JobSkill.id).desc()
        )
        
        result = await db.execute(query)
        skill_trends = result.fetchall()
        
        # Calculate trend metrics
        trends = []
        for row in skill_trends:
            demand_score = row.job_count * row.avg_confidence
            required_ratio = row.required_count / row.job_count if row.job_count > 0 else 0
            
            trends.append({
                'skill_name': row.name,
                'category': row.category,
                'job_count': row.job_count,
                'demand_score': demand_score,
                'avg_confidence': float(row.avg_confidence),
                'required_ratio': required_ratio,
                'required_count': row.required_count,
                'preferred_count': row.preferred_count
            })
        
        # Get salary trends for top skills
        top_skills = trends[:20]
        for skill_trend in top_skills:
            salary_data = await self._get_skill_salary_trend(
                db, skill_trend['skill_name'], days
            )
            skill_trend['salary_trend'] = salary_data
        
        return {
            'analysis_period_days': days,
            'total_skills_analyzed': len(trends),
            'skill_trends': trends,
            'top_emerging_skills': self._identify_emerging_skills(trends),
            'top_declining_skills': self._identify_declining_skills(trends)
        }
    
    async def _get_skill_salary_trend(
        self,
        db: AsyncSession,
        skill_name: str,
        days: int
    ) -> Dict[str, Any]:
        """Get salary trend data for a specific skill"""
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        # Query jobs that require this skill and have salary data
        query = select(
            JobPosting.salary_min,
            JobPosting.salary_max,
            JobPosting.salary_period,
            JobPosting.posted_date
        ).select_from(
            JobPosting.__table__.join(JobSkill.__table__).join(Skill.__table__)
        ).where(
            Skill.name == skill_name,
            JobPosting.posted_date >= cutoff_date,
            JobPosting.salary_min.isnot(None)
        )
        
        result = await db.execute(query)
        salary_data = result.fetchall()
        
        if not salary_data:
            return {'sample_size': 0}
        
        # Normalize salaries to yearly
        yearly_salaries = []
        for row in salary_data:
            if row.salary_min and row.salary_max:
                avg_salary = (row.salary_min + row.salary_max) / 2
            else:
                avg_salary = row.salary_min or row.salary_max
            
            # Convert to yearly
            if row.salary_period == 'hourly':
                avg_salary *= 2080
            elif row.salary_period == 'monthly':
                avg_salary *= 12
            
            yearly_salaries.append(avg_salary)
        
        if not yearly_salaries:
            return {'sample_size': 0}
        
        yearly_salaries.sort()
        n = len(yearly_salaries)
        
        return {
            'sample_size': n,
            'min_salary': min(yearly_salaries),
            'max_salary': max(yearly_salaries),
            'median_salary': yearly_salaries[n // 2],
            'mean_salary': sum(yearly_salaries) / n,
            'percentile_25': yearly_salaries[n // 4],
            'percentile_75': yearly_salaries[3 * n // 4]
        }
    
    def _identify_emerging_skills(self, trends: List[Dict]) -> List[Dict]:
        """Identify emerging skills based on demand patterns"""
        # Simple heuristic: high demand score with high confidence
        emerging = [
            skill for skill in trends
            if skill['demand_score'] > 50 and skill['avg_confidence'] > 0.8
        ]
        
        # Sort by demand score
        emerging.sort(key=lambda x: x['demand_score'], reverse=True)
        return emerging[:10]
    
    def _identify_declining_skills(self, trends: List[Dict]) -> List[Dict]:
        """Identify potentially declining skills"""
        # Simple heuristic: low demand with low required ratio
        declining = [
            skill for skill in trends
            if skill['job_count'] < 20 and skill['required_ratio'] < 0.3
        ]
        
        # Sort by job count (ascending)
        declining.sort(key=lambda x: x['job_count'])
        return declining[:10]
    
    async def generate_skill_recommendations(
        self,
        db: AsyncSession,
        user_skills: List[str],
        target_role: Optional[str] = None,
        location: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate skill recommendations based on market demand
        
        Args:
            db: Database session
            user_skills: List of user's current skills
            target_role: Target job role
            location: Target location
            
        Returns:
            Skill recommendations
        """
        # Get market trends
        trends = await self.analyze_market_trends(db)
        
        # Find skill gaps
        user_skills_lower = [skill.lower() for skill in user_skills]
        
        skill_gaps = []
        complementary_skills = []
        
        for trend in trends['skill_trends']:
            skill_name_lower = trend['skill_name'].lower()
            
            if skill_name_lower not in user_skills_lower:
                if trend['demand_score'] > 30:
                    skill_gaps.append({
                        'skill': trend['skill_name'],
                        'demand_score': trend['demand_score'],
                        'job_count': trend['job_count'],
                        'required_ratio': trend['required_ratio'],
                        'salary_trend': trend.get('salary_trend', {})
                    })
            else:
                # Find complementary skills
                complementary = await self._find_complementary_skills(
                    db, trend['skill_name']
                )
                complementary_skills.extend(complementary)
        
        # Remove duplicates and sort
        skill_gaps.sort(key=lambda x: x['demand_score'], reverse=True)
        
        return {
            'user_skills': user_skills,
            'target_role': target_role,
            'location': location,
            'recommended_skills': skill_gaps[:20],
            'complementary_skills': complementary_skills[:10],
            'market_summary': {
                'total_jobs_analyzed': sum(
                    trend['job_count'] for trend in trends['skill_trends']
                ),
                'analysis_period': trends['analysis_period_days']
            }
        }
    
    async def _find_complementary_skills(
        self,
        db: AsyncSession,
        skill_name: str,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Find skills that commonly appear with the given skill"""
        # Query for jobs that require this skill
        query = select(JobSkill.job_posting_id).where(
            JobSkill.skill_id.in_(
                select(Skill.id).where(Skill.name == skill_name)
            )
        )
        
        result = await db.execute(query)
        job_ids = [row[0] for row in result.fetchall()]
        
        if not job_ids:
            return []
        
        # Find other skills in these jobs
        complementary_query = select(
            Skill.name,
            func.count(JobSkill.id).label('co_occurrence_count')
        ).select_from(
            JobSkill.__table__.join(Skill.__table__)
        ).where(
            JobSkill.job_posting_id.in_(job_ids),
            Skill.name != skill_name
        ).group_by(
            Skill.id, Skill.name
        ).order_by(
            func.count(JobSkill.id).desc()
        ).limit(limit)
        
        result = await db.execute(complementary_query)
        complementary = result.fetchall()
        
        return [
            {
                'skill': row.name,
                'co_occurrence_count': row.co_occurrence_count
            }
            for row in complementary
        ]