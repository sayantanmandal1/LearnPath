"""
Job matching service with AI-powered compatibility scoring.
Matches user profiles with job opportunities using advanced algorithms.
"""

import asyncio
import logging
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime
import json
import re
from dataclasses import dataclass

from ..schemas.job import JobPosting, JobMatch, SkillMatch, SkillGap
from ..models.profile import UserProfile
from ..services.ai_analysis_service import AIAnalysisService
from ..core.exceptions import MatchingError

logger = logging.getLogger(__name__)

@dataclass
class MatchingCriteria:
    """Criteria for job matching."""
    role_weight: float = 0.3
    skills_weight: float = 0.4
    experience_weight: float = 0.2
    location_weight: float = 0.1
    min_match_score: float = 0.6

class JobMatchingService:
    """Service for matching jobs with user profiles using AI analysis."""
    
    def __init__(self, ai_service: AIAnalysisService):
        self.ai_service = ai_service
        self.criteria = MatchingCriteria()
        
        # Indian tech cities for location scoring
        self.indian_tech_cities = {
            'bangalore': ['bangalore', 'bengaluru', 'karnataka'],
            'hyderabad': ['hyderabad', 'telangana'],
            'pune': ['pune', 'maharashtra'],
            'chennai': ['chennai', 'tamil nadu'],
            'mumbai': ['mumbai', 'maharashtra'],
            'delhi_ncr': ['delhi', 'gurgaon', 'noida', 'faridabad', 'ghaziabad'],
            'kolkata': ['kolkata', 'west bengal'],
            'ahmedabad': ['ahmedabad', 'gujarat'],
            'kochi': ['kochi', 'cochin', 'kerala']
        }

    async def match_jobs_to_profile(
        self, 
        jobs: List[JobPosting], 
        profile: UserProfile,
        preferred_locations: List[str] = None,
        target_role: str = None
    ) -> List[JobMatch]:
        """
        Match jobs to user profile with AI-powered scoring.
        
        Args:
            jobs: List of job postings to match
            profile: User profile to match against
            preferred_locations: User's preferred locations
            target_role: User's target role
            
        Returns:
            List of JobMatch objects sorted by match score
        """
        try:
            matches = []
            
            for job in jobs:
                try:
                    match = await self._calculate_job_match(
                        job, profile, preferred_locations, target_role
                    )
                    
                    if match and match.match_score >= self.criteria.min_match_score:
                        matches.append(match)
                        
                except Exception as e:
                    logger.warning(f"Error matching job {job.job_id}: {str(e)}")
                    continue
            
            # Sort by match score (highest first)
            matches.sort(key=lambda x: x.match_score, reverse=True)
            
            return matches
            
        except Exception as e:
            logger.error(f"Error in job matching: {str(e)}")
            raise MatchingError(f"Failed to match jobs: {str(e)}")

    async def _calculate_job_match(
        self,
        job: JobPosting,
        profile: UserProfile,
        preferred_locations: List[str] = None,
        target_role: str = None
    ) -> Optional[JobMatch]:
        """Calculate match score for a single job."""
        try:
            # Calculate individual scores
            role_score = self._calculate_role_match(job, profile, target_role)
            skills_score, skill_matches, skill_gaps = await self._calculate_skills_match(job, profile)
            experience_score = self._calculate_experience_match(job, profile)
            location_score = self._calculate_location_match(job, preferred_locations)
            
            # Calculate weighted overall score
            overall_score = (
                role_score * self.criteria.role_weight +
                skills_score * self.criteria.skills_weight +
                experience_score * self.criteria.experience_weight +
                location_score * self.criteria.location_weight
            )
            
            # Generate AI-powered recommendation reason
            recommendation_reason = await self._generate_recommendation_reason(
                job, profile, overall_score, skill_matches, skill_gaps
            )
            
            return JobMatch(
                job_posting=job,
                match_score=round(overall_score, 2),
                skill_matches=skill_matches,
                skill_gaps=skill_gaps,
                recommendation_reason=recommendation_reason
            )
            
        except Exception as e:
            logger.error(f"Error calculating match for job {job.job_id}: {str(e)}")
            return None

    def _calculate_role_match(
        self, 
        job: JobPosting, 
        profile: UserProfile, 
        target_role: str = None
    ) -> float:
        """Calculate role/title match score."""
        try:
            job_title = job.title.lower()
            
            # Use target role if provided, otherwise infer from profile
            if target_role:
                target_title = target_role.lower()
            else:
                # Extract target role from profile experience or skills
                target_title = self._infer_target_role(profile)
            
            # Direct title match
            if target_title in job_title or job_title in target_title:
                return 1.0
            
            # Keyword-based matching
            role_keywords = self._extract_role_keywords(target_title)
            job_keywords = self._extract_role_keywords(job_title)
            
            if not role_keywords or not job_keywords:
                return 0.5  # Neutral score if no keywords
            
            # Calculate keyword overlap
            common_keywords = set(role_keywords) & set(job_keywords)
            total_keywords = set(role_keywords) | set(job_keywords)
            
            if total_keywords:
                return len(common_keywords) / len(total_keywords)
            
            return 0.5
            
        except Exception as e:
            logger.warning(f"Error calculating role match: {str(e)}")
            return 0.5

    async def _calculate_skills_match(
        self, 
        job: JobPosting, 
        profile: UserProfile
    ) -> Tuple[float, List[SkillMatch], List[SkillGap]]:
        """Calculate skills match with detailed breakdown."""
        try:
            # Extract skills from job posting
            job_skills = self._extract_job_skills(job)
            
            # Get user skills from profile
            user_skills = [skill.name.lower() for skill in profile.skills] if profile.skills else []
            
            skill_matches = []
            skill_gaps = []
            
            if not job_skills:
                return 0.5, skill_matches, skill_gaps
            
            # Calculate matches and gaps
            for job_skill in job_skills:
                job_skill_lower = job_skill.lower()
                
                # Find exact or partial matches
                match_found = False
                for user_skill in user_skills:
                    if (job_skill_lower in user_skill or 
                        user_skill in job_skill_lower or
                        self._are_similar_skills(job_skill_lower, user_skill)):
                        
                        skill_matches.append(SkillMatch(
                            skill_name=job_skill,
                            user_skill_level=self._get_skill_level(profile, user_skill),
                            required_level="intermediate",  # Default assumption
                            match_strength=self._calculate_skill_similarity(job_skill_lower, user_skill)
                        ))
                        match_found = True
                        break
                
                if not match_found:
                    skill_gaps.append(SkillGap(
                        skill_name=job_skill,
                        required_level="intermediate",
                        gap_severity="medium",
                        learning_resources=[]  # Will be populated by AI service
                    ))
            
            # Calculate overall skills score
            if job_skills:
                skills_score = len(skill_matches) / len(job_skills)
            else:
                skills_score = 0.5
            
            return skills_score, skill_matches, skill_gaps
            
        except Exception as e:
            logger.warning(f"Error calculating skills match: {str(e)}")
            return 0.5, [], []

    def _calculate_experience_match(self, job: JobPosting, profile: UserProfile) -> float:
        """Calculate experience level match."""
        try:
            # Extract required experience from job
            job_exp_years = self._extract_experience_years(job)
            
            # Calculate user's total experience
            user_exp_years = 0
            if profile.experience:
                for exp in profile.experience:
                    # Estimate years from experience entries
                    user_exp_years += self._estimate_experience_duration(exp)
            
            if job_exp_years is None:
                return 0.7  # Neutral score if no experience requirement
            
            # Score based on experience match
            if user_exp_years >= job_exp_years:
                # User meets or exceeds requirement
                if user_exp_years <= job_exp_years * 1.5:
                    return 1.0  # Perfect match
                else:
                    return 0.8  # Overqualified but still good
            else:
                # User has less experience
                ratio = user_exp_years / job_exp_years if job_exp_years > 0 else 0
                return max(0.3, ratio)  # Minimum score of 0.3
                
        except Exception as e:
            logger.warning(f"Error calculating experience match: {str(e)}")
            return 0.5

    def _calculate_location_match(
        self, 
        job: JobPosting, 
        preferred_locations: List[str] = None
    ) -> float:
        """Calculate location preference match."""
        try:
            if not preferred_locations:
                return 0.7  # Neutral score if no preference
            
            job_location = job.location.lower() if job.location else ""
            
            # Check for direct location matches
            for pref_loc in preferred_locations:
                pref_loc_lower = pref_loc.lower()
                
                # Direct match
                if pref_loc_lower in job_location or job_location in pref_loc_lower:
                    return 1.0
                
                # Check city variations
                for city, variations in self.indian_tech_cities.items():
                    if pref_loc_lower in variations:
                        for variation in variations:
                            if variation in job_location:
                                return 1.0
            
            # Check for remote work
            if 'remote' in job_location or 'work from home' in job_location:
                return 0.9
            
            return 0.3  # Low score for non-matching locations
            
        except Exception as e:
            logger.warning(f"Error calculating location match: {str(e)}")
            return 0.5

    async def _generate_recommendation_reason(
        self,
        job: JobPosting,
        profile: UserProfile,
        match_score: float,
        skill_matches: List[SkillMatch],
        skill_gaps: List[SkillGap]
    ) -> str:
        """Generate AI-powered recommendation reason."""
        try:
            # Prepare context for AI analysis
            context = {
                "job_title": job.title,
                "company": job.company,
                "location": job.location,
                "match_score": match_score,
                "matched_skills": [sm.skill_name for sm in skill_matches],
                "skill_gaps": [sg.skill_name for sg in skill_gaps],
                "user_experience": len(profile.experience) if profile.experience else 0
            }
            
            # Use AI service to generate personalized reason
            reason = await self.ai_service.generate_job_recommendation_reason(context)
            
            if reason:
                return reason
            
            # Fallback to rule-based reason
            return self._generate_fallback_reason(match_score, skill_matches, skill_gaps)
            
        except Exception as e:
            logger.warning(f"Error generating recommendation reason: {str(e)}")
            return self._generate_fallback_reason(match_score, skill_matches, skill_gaps)

    def _generate_fallback_reason(
        self,
        match_score: float,
        skill_matches: List[SkillMatch],
        skill_gaps: List[SkillGap]
    ) -> str:
        """Generate fallback recommendation reason."""
        if match_score >= 0.9:
            return f"Excellent match! You have {len(skill_matches)} matching skills and minimal gaps."
        elif match_score >= 0.8:
            return f"Strong match with {len(skill_matches)} relevant skills. Consider developing: {', '.join([sg.skill_name for sg in skill_gaps[:2]])}."
        elif match_score >= 0.7:
            return f"Good potential match. You have {len(skill_matches)} matching skills but may need to develop {len(skill_gaps)} additional skills."
        else:
            return f"Moderate match. Focus on developing key skills: {', '.join([sg.skill_name for sg in skill_gaps[:3]])}."

    def _extract_job_skills(self, job: JobPosting) -> List[str]:
        """Extract skills from job posting."""
        skills = set()
        
        # Add explicitly listed skills
        if job.required_skills:
            skills.update([skill.strip() for skill in job.required_skills])
        
        # Extract skills from description
        text = f"{job.title} {job.description}".lower()
        
        # Common tech skills to look for
        tech_skills = [
            'python', 'java', 'javascript', 'typescript', 'react', 'angular', 'vue',
            'node.js', 'express', 'django', 'flask', 'spring', 'hibernate',
            'sql', 'mysql', 'postgresql', 'mongodb', 'redis', 'elasticsearch',
            'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'jenkins',
            'git', 'github', 'gitlab', 'jira', 'confluence',
            'html', 'css', 'sass', 'less', 'bootstrap', 'tailwind',
            'rest api', 'graphql', 'microservices', 'devops', 'ci/cd',
            'machine learning', 'data science', 'tensorflow', 'pytorch',
            'android', 'ios', 'react native', 'flutter', 'swift', 'kotlin'
        ]
        
        for skill in tech_skills:
            if skill in text:
                skills.add(skill)
        
        return list(skills)

    def _extract_role_keywords(self, title: str) -> List[str]:
        """Extract keywords from role title."""
        # Remove common words and extract meaningful keywords
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        
        # Split and clean
        words = re.findall(r'\b\w+\b', title.lower())
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        
        return keywords

    def _infer_target_role(self, profile: UserProfile) -> str:
        """Infer target role from user profile."""
        # Use most recent job title or skills to infer target role
        if profile.experience and profile.experience:
            return profile.experience[0].title  # Most recent experience
        
        # Fallback to skills-based inference
        if profile.skills:
            skill_names = [skill.name.lower() for skill in profile.skills]
            
            # Map skills to common roles
            if any(skill in skill_names for skill in ['python', 'django', 'flask']):
                return "python developer"
            elif any(skill in skill_names for skill in ['java', 'spring', 'hibernate']):
                return "java developer"
            elif any(skill in skill_names for skill in ['javascript', 'react', 'angular']):
                return "frontend developer"
            elif any(skill in skill_names for skill in ['node.js', 'express']):
                return "backend developer"
            elif any(skill in skill_names for skill in ['machine learning', 'data science']):
                return "data scientist"
        
        return "software developer"  # Default

    def _are_similar_skills(self, skill1: str, skill2: str) -> bool:
        """Check if two skills are similar."""
        # Define skill synonyms
        synonyms = {
            'javascript': ['js', 'ecmascript'],
            'typescript': ['ts'],
            'python': ['py'],
            'machine learning': ['ml', 'artificial intelligence', 'ai'],
            'database': ['db', 'sql'],
            'frontend': ['front-end', 'ui', 'user interface'],
            'backend': ['back-end', 'server-side'],
            'devops': ['dev-ops', 'deployment', 'infrastructure']
        }
        
        for main_skill, skill_synonyms in synonyms.items():
            if ((skill1 == main_skill and skill2 in skill_synonyms) or
                (skill2 == main_skill and skill1 in skill_synonyms)):
                return True
        
        return False

    def _calculate_skill_similarity(self, skill1: str, skill2: str) -> float:
        """Calculate similarity score between two skills."""
        if skill1 == skill2:
            return 1.0
        elif skill1 in skill2 or skill2 in skill1:
            return 0.8
        elif self._are_similar_skills(skill1, skill2):
            return 0.7
        else:
            return 0.5

    def _get_skill_level(self, profile: UserProfile, skill_name: str) -> str:
        """Get user's skill level for a specific skill."""
        if profile.skills:
            for skill in profile.skills:
                if skill.name.lower() == skill_name.lower():
                    return skill.level if hasattr(skill, 'level') else "intermediate"
        return "beginner"

    def _extract_experience_years(self, job: JobPosting) -> Optional[int]:
        """Extract required experience years from job posting."""
        text = f"{job.title} {job.description} {job.experience_level}".lower()
        
        # Look for experience patterns
        patterns = [
            r'(\d+)\+?\s*years?\s*(?:of\s*)?experience',
            r'(\d+)\+?\s*yrs?\s*(?:of\s*)?experience',
            r'minimum\s*(\d+)\s*years?',
            r'at least\s*(\d+)\s*years?'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return int(match.group(1))
        
        # Check experience level keywords
        if 'senior' in text or 'lead' in text:
            return 5
        elif 'mid' in text or 'intermediate' in text:
            return 3
        elif 'junior' in text or 'entry' in text:
            return 1
        
        return None

    def _estimate_experience_duration(self, experience) -> float:
        """Estimate duration of an experience entry in years."""
        # This would need to be implemented based on your Experience model
        # For now, return a default estimate
        return 2.0  # Default 2 years per experience entry