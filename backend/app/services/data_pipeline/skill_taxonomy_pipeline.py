"""
Skill Taxonomy Update Pipeline
Updates skill taxonomy based on market trend analysis and emerging skills detection.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Set, Tuple
import json
from collections import Counter, defaultdict

from app.core.logging import get_logger
from app.services.data_pipeline.pipeline_monitor import get_pipeline_monitor
from app.services.market_trend_analyzer import MarketTrendAnalyzer
from app.repositories.skill import SkillRepository
from app.repositories.job import JobRepository
from app.core.database import get_db_session
from machinelearningmodel.nlp_engine import NLPEngine

logger = get_logger(__name__)


class SkillTaxonomyPipeline:
    """
    Pipeline for updating skill taxonomy based on market trends and job postings
    """
    
    def __init__(self):
        self.trend_analyzer = MarketTrendAnalyzer()
        self.nlp_engine = NLPEngine()
        self.monitor = None
        
    async def execute(self, metadata: Dict[str, Any] = None):
        """Execute the skill taxonomy update pipeline"""
        execution_id = metadata.get('execution_id', f"skill_taxonomy_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}")
        
        self.monitor = await get_pipeline_monitor()
        
        try:
            logger.info(f"Starting skill taxonomy update pipeline: {execution_id}")
            
            # Initialize metrics
            await self.monitor.update_job_metrics(
                execution_id,
                status="running",
                records_processed=0,
                records_failed=0
            )
            
            # Step 1: Analyze recent job postings for skill trends
            logger.info("Analyzing job postings for skill trends")
            job_analysis_results = await self._analyze_job_postings()
            
            # Step 2: Detect emerging skills
            logger.info("Detecting emerging skills")
            emerging_skills = await self._detect_emerging_skills(job_analysis_results)
            
            # Step 3: Update skill categories and relationships
            logger.info("Updating skill categories")
            category_updates = await self._update_skill_categories(job_analysis_results)
            
            # Step 4: Update skill demand scores
            logger.info("Updating skill demand scores")
            demand_updates = await self._update_skill_demand_scores(job_analysis_results)
            
            # Step 5: Clean up obsolete skills
            logger.info("Cleaning up obsolete skills")
            cleanup_results = await self._cleanup_obsolete_skills()
            
            # Step 6: Update skill relationships and synonyms
            logger.info("Updating skill relationships")
            relationship_updates = await self._update_skill_relationships(job_analysis_results)
            
            # Calculate final metrics
            total_processed = (
                len(emerging_skills) + 
                len(category_updates) + 
                len(demand_updates) + 
                len(relationship_updates)
            )
            
            # Store update summary
            update_summary = {
                'execution_id': execution_id,
                'timestamp': datetime.utcnow().isoformat(),
                'emerging_skills': emerging_skills,
                'category_updates': category_updates,
                'demand_updates': demand_updates,
                'cleanup_results': cleanup_results,
                'relationship_updates': relationship_updates,
                'total_processed': total_processed
            }
            
            await self._store_update_summary(update_summary)
            
            # Update final metrics
            await self.monitor.update_job_metrics(
                execution_id,
                status="completed",
                records_processed=total_processed,
                data_quality_score=1.0  # Assume high quality for taxonomy updates
            )
            
            logger.info(f"Skill taxonomy update completed: {total_processed} updates processed")
            
        except Exception as e:
            logger.error(f"Skill taxonomy update pipeline failed: {e}")
            
            await self.monitor.update_job_metrics(
                execution_id,
                status="failed",
                error_count=1
            )
            raise
    
    async def _analyze_job_postings(self) -> Dict[str, Any]:
        """Analyze recent job postings to extract skill trends"""
        try:
            async with get_db_session() as db:
                job_repo = JobRepository(db)
                
                # Get job postings from the last 30 days
                recent_jobs = await job_repo.get_recent_jobs(days=30)
                
                skill_mentions = Counter()
                skill_cooccurrences = defaultdict(Counter)
                job_categories = defaultdict(list)
                
                for job in recent_jobs:
                    # Extract skills from job description
                    extracted_skills = await self.nlp_engine.extract_skills(job.description)
                    
                    # Count skill mentions
                    for skill in extracted_skills:
                        skill_mentions[skill.lower()] += 1
                    
                    # Track skill co-occurrences
                    for i, skill1 in enumerate(extracted_skills):
                        for skill2 in extracted_skills[i+1:]:
                            skill_cooccurrences[skill1.lower()][skill2.lower()] += 1
                            skill_cooccurrences[skill2.lower()][skill1.lower()] += 1
                    
                    # Categorize by job title
                    job_category = await self._categorize_job(job.title)
                    job_categories[job_category].extend([s.lower() for s in extracted_skills])
                
                return {
                    'total_jobs_analyzed': len(recent_jobs),
                    'skill_mentions': dict(skill_mentions),
                    'skill_cooccurrences': dict(skill_cooccurrences),
                    'job_categories': dict(job_categories),
                    'analysis_date': datetime.utcnow().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Failed to analyze job postings: {e}")
            return {}
    
    async def _detect_emerging_skills(self, analysis_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect emerging skills based on recent trends"""
        emerging_skills = []
        
        try:
            skill_mentions = analysis_results.get('skill_mentions', {})
            
            # Get historical skill data for comparison
            async with get_db_session() as db:
                skill_repo = SkillRepository(db)
                existing_skills = await skill_repo.get_all_skills()
                existing_skill_names = {skill.name.lower() for skill in existing_skills}
                
                # Find skills that appear frequently but aren't in our taxonomy
                min_mentions = max(10, len(analysis_results.get('total_jobs_analyzed', 0)) * 0.05)  # 5% threshold
                
                for skill_name, mention_count in skill_mentions.items():
                    if mention_count >= min_mentions and skill_name not in existing_skill_names:
                        # Validate that this is actually a skill
                        if await self._validate_skill(skill_name):
                            # Determine skill category
                            category = await self._determine_skill_category(skill_name, analysis_results)
                            
                            emerging_skill = {
                                'name': skill_name,
                                'mention_count': mention_count,
                                'category': category,
                                'confidence_score': min(mention_count / min_mentions, 1.0),
                                'first_detected': datetime.utcnow().isoformat()
                            }
                            
                            emerging_skills.append(emerging_skill)
                            
                            # Add to database
                            await skill_repo.create_skill({
                                'name': skill_name,
                                'category': category,
                                'demand_score': min(mention_count / min_mentions, 1.0),
                                'is_emerging': True,
                                'created_at': datetime.utcnow()
                            })
            
            logger.info(f"Detected {len(emerging_skills)} emerging skills")
            return emerging_skills
            
        except Exception as e:
            logger.error(f"Failed to detect emerging skills: {e}")
            return []
    
    async def _update_skill_categories(self, analysis_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Update skill categories based on job posting analysis"""
        category_updates = []
        
        try:
            job_categories = analysis_results.get('job_categories', {})
            
            async with get_db_session() as db:
                skill_repo = SkillRepository(db)
                
                # Analyze which skills appear most frequently in which job categories
                skill_category_scores = defaultdict(lambda: defaultdict(int))
                
                for job_category, skills in job_categories.items():
                    skill_counts = Counter(skills)
                    for skill, count in skill_counts.items():
                        skill_category_scores[skill][job_category] += count
                
                # Update skill categories based on strongest associations
                for skill_name, category_scores in skill_category_scores.items():
                    if not category_scores:
                        continue
                    
                    # Find the most common category for this skill
                    best_category = max(category_scores.items(), key=lambda x: x[1])
                    category_name, score = best_category
                    
                    # Get existing skill
                    skill = await skill_repo.get_by_name(skill_name)
                    if skill and skill.category != category_name:
                        # Update category if confidence is high enough
                        total_mentions = sum(category_scores.values())
                        confidence = score / total_mentions
                        
                        if confidence > 0.6:  # 60% confidence threshold
                            await skill_repo.update_skill(skill.id, {'category': category_name})
                            
                            category_updates.append({
                                'skill_name': skill_name,
                                'old_category': skill.category,
                                'new_category': category_name,
                                'confidence': confidence,
                                'mention_count': total_mentions
                            })
            
            logger.info(f"Updated categories for {len(category_updates)} skills")
            return category_updates
            
        except Exception as e:
            logger.error(f"Failed to update skill categories: {e}")
            return []
    
    async def _update_skill_demand_scores(self, analysis_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Update skill demand scores based on job posting frequency"""
        demand_updates = []
        
        try:
            skill_mentions = analysis_results.get('skill_mentions', {})
            total_jobs = analysis_results.get('total_jobs_analyzed', 1)
            
            async with get_db_session() as db:
                skill_repo = SkillRepository(db)
                
                for skill_name, mention_count in skill_mentions.items():
                    skill = await skill_repo.get_by_name(skill_name)
                    if skill:
                        # Calculate demand score (0-1 scale)
                        demand_score = min(mention_count / (total_jobs * 0.1), 1.0)  # 10% = max score
                        
                        # Apply exponential smoothing with previous score
                        if skill.demand_score:
                            smoothing_factor = 0.3  # Weight for new data
                            demand_score = (smoothing_factor * demand_score + 
                                          (1 - smoothing_factor) * skill.demand_score)
                        
                        # Update if score changed significantly
                        if abs(demand_score - (skill.demand_score or 0)) > 0.05:
                            await skill_repo.update_skill(skill.id, {
                                'demand_score': demand_score,
                                'last_updated': datetime.utcnow()
                            })
                            
                            demand_updates.append({
                                'skill_name': skill_name,
                                'old_score': skill.demand_score,
                                'new_score': demand_score,
                                'mention_count': mention_count
                            })
            
            logger.info(f"Updated demand scores for {len(demand_updates)} skills")
            return demand_updates
            
        except Exception as e:
            logger.error(f"Failed to update skill demand scores: {e}")
            return []
    
    async def _cleanup_obsolete_skills(self) -> Dict[str, Any]:
        """Clean up skills that are no longer relevant"""
        cleanup_results = {
            'obsolete_skills': [],
            'updated_skills': [],
            'total_cleaned': 0
        }
        
        try:
            async with get_db_session() as db:
                skill_repo = SkillRepository(db)
                
                # Get skills that haven't been mentioned in recent job postings
                cutoff_date = datetime.utcnow() - timedelta(days=90)  # 3 months
                old_skills = await skill_repo.get_skills_not_updated_since(cutoff_date)
                
                for skill in old_skills:
                    # Check if skill has very low demand score
                    if skill.demand_score and skill.demand_score < 0.01:
                        # Mark as obsolete instead of deleting
                        await skill_repo.update_skill(skill.id, {
                            'is_obsolete': True,
                            'obsolete_date': datetime.utcnow()
                        })
                        
                        cleanup_results['obsolete_skills'].append({
                            'skill_name': skill.name,
                            'last_updated': skill.last_updated.isoformat() if skill.last_updated else None,
                            'demand_score': skill.demand_score
                        })
                    
                    elif skill.demand_score and skill.demand_score < 0.1:
                        # Reduce demand score for low-demand skills
                        new_score = skill.demand_score * 0.9  # 10% reduction
                        await skill_repo.update_skill(skill.id, {
                            'demand_score': new_score,
                            'last_updated': datetime.utcnow()
                        })
                        
                        cleanup_results['updated_skills'].append({
                            'skill_name': skill.name,
                            'old_score': skill.demand_score,
                            'new_score': new_score
                        })
                
                cleanup_results['total_cleaned'] = (
                    len(cleanup_results['obsolete_skills']) + 
                    len(cleanup_results['updated_skills'])
                )
            
            logger.info(f"Cleaned up {cleanup_results['total_cleaned']} obsolete skills")
            return cleanup_results
            
        except Exception as e:
            logger.error(f"Failed to cleanup obsolete skills: {e}")
            return cleanup_results
    
    async def _update_skill_relationships(self, analysis_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Update skill relationships and synonyms based on co-occurrence patterns"""
        relationship_updates = []
        
        try:
            skill_cooccurrences = analysis_results.get('skill_cooccurrences', {})
            
            async with get_db_session() as db:
                skill_repo = SkillRepository(db)
                
                for skill1, related_skills in skill_cooccurrences.items():
                    skill1_obj = await skill_repo.get_by_name(skill1)
                    if not skill1_obj:
                        continue
                    
                    # Find strongly related skills (co-occur frequently)
                    total_mentions = sum(related_skills.values())
                    
                    for skill2, cooccurrence_count in related_skills.items():
                        if cooccurrence_count < 5:  # Minimum co-occurrence threshold
                            continue
                        
                        skill2_obj = await skill_repo.get_by_name(skill2)
                        if not skill2_obj:
                            continue
                        
                        # Calculate relationship strength
                        relationship_strength = cooccurrence_count / total_mentions
                        
                        if relationship_strength > 0.3:  # 30% co-occurrence threshold
                            # Create or update skill relationship
                            await skill_repo.create_or_update_relationship(
                                skill1_obj.id,
                                skill2_obj.id,
                                relationship_strength,
                                'frequently_paired'
                            )
                            
                            relationship_updates.append({
                                'skill1': skill1,
                                'skill2': skill2,
                                'strength': relationship_strength,
                                'cooccurrence_count': cooccurrence_count
                            })
            
            logger.info(f"Updated {len(relationship_updates)} skill relationships")
            return relationship_updates
            
        except Exception as e:
            logger.error(f"Failed to update skill relationships: {e}")
            return []
    
    async def _validate_skill(self, skill_name: str) -> bool:
        """Validate that a detected term is actually a skill"""
        try:
            # Use NLP to validate if this is a legitimate skill
            # This is a simplified validation - in practice, you might use
            # more sophisticated NLP models or external APIs
            
            # Basic validation rules
            if len(skill_name) < 2 or len(skill_name) > 50:
                return False
            
            # Check if it's a common non-skill word
            non_skills = {
                'experience', 'years', 'work', 'team', 'project', 'company',
                'position', 'role', 'job', 'career', 'opportunity', 'candidate'
            }
            
            if skill_name.lower() in non_skills:
                return False
            
            # Use NLP engine for more sophisticated validation
            is_skill = await self.nlp_engine.classify_as_skill(skill_name)
            return is_skill
            
        except Exception as e:
            logger.error(f"Failed to validate skill '{skill_name}': {e}")
            return False
    
    async def _determine_skill_category(self, skill_name: str, analysis_results: Dict[str, Any]) -> str:
        """Determine the category for a new skill"""
        try:
            # Use job categories where this skill appears most frequently
            job_categories = analysis_results.get('job_categories', {})
            
            skill_category_counts = defaultdict(int)
            for job_category, skills in job_categories.items():
                if skill_name in skills:
                    skill_category_counts[job_category] += skills.count(skill_name)
            
            if skill_category_counts:
                most_common_job_category = max(skill_category_counts.items(), key=lambda x: x[1])[0]
                
                # Map job categories to skill categories
                category_mapping = {
                    'software_engineering': 'programming',
                    'data_science': 'data_analysis',
                    'product_management': 'business',
                    'design': 'design',
                    'marketing': 'marketing',
                    'sales': 'business',
                    'operations': 'operations'
                }
                
                return category_mapping.get(most_common_job_category, 'technical')
            
            # Fallback: use NLP to classify skill category
            return await self.nlp_engine.classify_skill_category(skill_name)
            
        except Exception as e:
            logger.error(f"Failed to determine category for skill '{skill_name}': {e}")
            return 'general'
    
    async def _categorize_job(self, job_title: str) -> str:
        """Categorize a job based on its title"""
        try:
            job_title_lower = job_title.lower()
            
            # Simple keyword-based categorization
            if any(keyword in job_title_lower for keyword in ['engineer', 'developer', 'programmer']):
                return 'software_engineering'
            elif any(keyword in job_title_lower for keyword in ['data scientist', 'analyst', 'data engineer']):
                return 'data_science'
            elif any(keyword in job_title_lower for keyword in ['product manager', 'pm']):
                return 'product_management'
            elif any(keyword in job_title_lower for keyword in ['designer', 'ux', 'ui']):
                return 'design'
            elif any(keyword in job_title_lower for keyword in ['marketing', 'growth']):
                return 'marketing'
            elif any(keyword in job_title_lower for keyword in ['sales', 'account']):
                return 'sales'
            else:
                return 'general'
                
        except Exception as e:
            logger.error(f"Failed to categorize job '{job_title}': {e}")
            return 'general'
    
    async def _store_update_summary(self, summary: Dict[str, Any]):
        """Store taxonomy update summary for reporting"""
        try:
            from app.core.redis import get_redis_client
            
            redis_client = await get_redis_client()
            
            # Store detailed summary
            summary_key = f"skill_taxonomy_update:{summary['execution_id']}"
            await redis_client.set(
                summary_key,
                json.dumps(summary),
                ex=86400 * 30  # Keep for 30 days
            )
            
            # Update monthly stats
            month = datetime.utcnow().strftime('%Y-%m')
            monthly_stats_key = f"skill_taxonomy_monthly:{month}"
            
            await redis_client.hincrby(monthly_stats_key, 'total_updates', summary['total_processed'])
            await redis_client.hincrby(monthly_stats_key, 'emerging_skills', len(summary['emerging_skills']))
            await redis_client.expire(monthly_stats_key, 86400 * 365)  # Keep for 1 year
            
        except Exception as e:
            logger.error(f"Failed to store update summary: {e}")
    
    async def get_taxonomy_stats(self, months: int = 6) -> Dict[str, Any]:
        """Get skill taxonomy update statistics"""
        try:
            from app.core.redis import get_redis_client
            
            redis_client = await get_redis_client()
            stats = {
                'monthly_stats': {},
                'total_updates': 0,
                'total_emerging_skills': 0
            }
            
            # Get monthly stats
            for i in range(months):
                date = (datetime.utcnow() - timedelta(days=i*30)).strftime('%Y-%m')
                monthly_key = f"skill_taxonomy_monthly:{date}"
                
                monthly_data = await redis_client.hgetall(monthly_key)
                if monthly_data:
                    updates = int(monthly_data.get('total_updates', 0))
                    emerging = int(monthly_data.get('emerging_skills', 0))
                    
                    stats['monthly_stats'][date] = {
                        'total_updates': updates,
                        'emerging_skills': emerging
                    }
                    
                    stats['total_updates'] += updates
                    stats['total_emerging_skills'] += emerging
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get taxonomy stats: {e}")
            return {}