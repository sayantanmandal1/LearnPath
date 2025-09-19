"""
Job application tracking and feedback service.
"""

import logging
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import and_, func, desc

from ..models.job_application import JobApplication, JobApplicationFeedback, JobRecommendationFeedback
from ..models.profile import UserProfile
from ..schemas.job_application import (
    JobApplicationCreate, JobApplicationUpdate, JobApplicationResponse,
    JobApplicationFeedbackCreate, JobRecommendationFeedbackCreate,
    JobApplicationStats, EnhancedJobMatch
)
from ..core.exceptions import NotFoundError, ValidationError
from ..core.database import get_db

logger = logging.getLogger(__name__)


class JobApplicationService:
    """Service for managing job applications and feedback."""
    
    def __init__(self, db: Session):
        self.db = db
    
    async def create_application(
        self, 
        user_id: str, 
        application_data: JobApplicationCreate
    ) -> JobApplicationResponse:
        """Create a new job application."""
        try:
            # Check if application already exists
            existing = self.db.query(JobApplication).filter(
                and_(
                    JobApplication.user_id == user_id,
                    JobApplication.job_posting_id == application_data.job_posting_id
                )
            ).first()
            
            if existing:
                raise ValidationError("Application for this job already exists")
            
            # Create new application
            application = JobApplication(
                user_id=user_id,
                job_posting_id=application_data.job_posting_id,
                job_title=application_data.job_title,
                company_name=application_data.company_name,
                job_url=application_data.job_url,
                match_score=application_data.match_score,
                skill_matches=application_data.skill_matches,
                skill_gaps=application_data.skill_gaps,
                application_method=application_data.application_method,
                cover_letter=application_data.cover_letter,
                notes=application_data.notes,
                status="interested"
            )
            
            self.db.add(application)
            self.db.commit()
            self.db.refresh(application)
            
            logger.info(f"Created job application {application.id} for user {user_id}")
            return JobApplicationResponse.from_orm(application)
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error creating job application: {str(e)}")
            raise
    
    async def update_application(
        self, 
        application_id: str, 
        user_id: str, 
        update_data: JobApplicationUpdate
    ) -> JobApplicationResponse:
        """Update an existing job application."""
        try:
            application = self.db.query(JobApplication).filter(
                and_(
                    JobApplication.id == application_id,
                    JobApplication.user_id == user_id
                )
            ).first()
            
            if not application:
                raise NotFoundError("Job application not found")
            
            # Update fields
            for field, value in update_data.dict(exclude_unset=True).items():
                setattr(application, field, value)
            
            application.last_updated = datetime.utcnow()
            
            self.db.commit()
            self.db.refresh(application)
            
            logger.info(f"Updated job application {application_id}")
            return JobApplicationResponse.from_orm(application)
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error updating job application: {str(e)}")
            raise
    
    async def get_user_applications(
        self, 
        user_id: str, 
        status: Optional[str] = None,
        limit: int = 50,
        offset: int = 0
    ) -> List[JobApplicationResponse]:
        """Get user's job applications."""
        try:
            query = self.db.query(JobApplication).filter(
                JobApplication.user_id == user_id
            )
            
            if status:
                query = query.filter(JobApplication.status == status)
            
            applications = query.order_by(desc(JobApplication.created_at)).offset(offset).limit(limit).all()
            
            return [JobApplicationResponse.from_orm(app) for app in applications]
            
        except Exception as e:
            logger.error(f"Error getting user applications: {str(e)}")
            raise
    
    async def get_application_stats(self, user_id: str) -> JobApplicationStats:
        """Get user's job application statistics."""
        try:
            # Total applications
            total_applications = self.db.query(JobApplication).filter(
                JobApplication.user_id == user_id
            ).count()
            
            # Status breakdown
            status_query = self.db.query(
                JobApplication.status,
                func.count(JobApplication.id).label('count')
            ).filter(
                JobApplication.user_id == user_id
            ).group_by(JobApplication.status).all()
            
            status_breakdown = {status: count for status, count in status_query}
            
            # Average match score
            avg_match_score = self.db.query(
                func.avg(JobApplication.match_score)
            ).filter(
                and_(
                    JobApplication.user_id == user_id,
                    JobApplication.match_score.isnot(None)
                )
            ).scalar()
            
            # Applications this month
            month_start = datetime.utcnow().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            applications_this_month = self.db.query(JobApplication).filter(
                and_(
                    JobApplication.user_id == user_id,
                    JobApplication.created_at >= month_start
                )
            ).count()
            
            # Interviews scheduled
            interviews_scheduled = self.db.query(JobApplication).filter(
                and_(
                    JobApplication.user_id == user_id,
                    JobApplication.interview_scheduled == True
                )
            ).count()
            
            # Success rate (accepted / total applied)
            total_applied = self.db.query(JobApplication).filter(
                and_(
                    JobApplication.user_id == user_id,
                    JobApplication.status.in_(["applied", "interviewing", "rejected", "accepted"])
                )
            ).count()
            
            accepted_count = status_breakdown.get("accepted", 0)
            success_rate = (accepted_count / total_applied * 100) if total_applied > 0 else 0.0
            
            # Top companies
            company_query = self.db.query(
                JobApplication.company_name,
                func.count(JobApplication.id).label('count')
            ).filter(
                JobApplication.user_id == user_id
            ).group_by(JobApplication.company_name).order_by(desc('count')).limit(5).all()
            
            top_companies = [{"company": company, "applications": count} for company, count in company_query]
            
            # Application timeline (last 6 months)
            six_months_ago = datetime.utcnow() - timedelta(days=180)
            timeline_query = self.db.query(
                func.date_trunc('month', JobApplication.created_at).label('month'),
                func.count(JobApplication.id).label('count')
            ).filter(
                and_(
                    JobApplication.user_id == user_id,
                    JobApplication.created_at >= six_months_ago
                )
            ).group_by('month').order_by('month').all()
            
            application_timeline = [
                {"month": month.strftime("%Y-%m"), "applications": count} 
                for month, count in timeline_query
            ]
            
            return JobApplicationStats(
                total_applications=total_applications,
                status_breakdown=status_breakdown,
                average_match_score=float(avg_match_score) if avg_match_score else None,
                applications_this_month=applications_this_month,
                interviews_scheduled=interviews_scheduled,
                success_rate=round(success_rate, 2),
                top_companies=top_companies,
                application_timeline=application_timeline
            )
            
        except Exception as e:
            logger.error(f"Error getting application stats: {str(e)}")
            raise
    
    async def add_application_feedback(
        self, 
        application_id: str, 
        user_id: str, 
        feedback_data: JobApplicationFeedbackCreate
    ) -> str:
        """Add feedback for a job application."""
        try:
            # Verify application belongs to user
            application = self.db.query(JobApplication).filter(
                and_(
                    JobApplication.id == application_id,
                    JobApplication.user_id == user_id
                )
            ).first()
            
            if not application:
                raise NotFoundError("Job application not found")
            
            # Create feedback
            feedback = JobApplicationFeedback(
                application_id=application_id,
                feedback_type=feedback_data.feedback_type,
                rating=feedback_data.rating,
                feedback_text=feedback_data.feedback_text,
                match_accuracy_rating=feedback_data.match_accuracy_rating,
                recommendation_helpfulness=feedback_data.recommendation_helpfulness,
                gap_analysis_accuracy=feedback_data.gap_analysis_accuracy,
                suggested_improvements=feedback_data.suggested_improvements
            )
            
            self.db.add(feedback)
            self.db.commit()
            self.db.refresh(feedback)
            
            logger.info(f"Added feedback for application {application_id}")
            return str(feedback.id)
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error adding application feedback: {str(e)}")
            raise
    
    async def add_recommendation_feedback(
        self, 
        user_id: str, 
        feedback_data: JobRecommendationFeedbackCreate
    ) -> str:
        """Add feedback for a job recommendation."""
        try:
            feedback = JobRecommendationFeedback(
                user_id=user_id,
                job_posting_id=feedback_data.job_posting_id,
                user_interested=feedback_data.user_interested,
                user_applied=feedback_data.user_applied,
                match_score_feedback=feedback_data.match_score_feedback,
                skill_match_feedback=feedback_data.skill_match_feedback,
                location_feedback=feedback_data.location_feedback,
                feedback_text=feedback_data.feedback_text,
                improvement_suggestions=feedback_data.improvement_suggestions
            )
            
            self.db.add(feedback)
            self.db.commit()
            self.db.refresh(feedback)
            
            logger.info(f"Added recommendation feedback for job {feedback_data.job_posting_id}")
            return str(feedback.id)
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error adding recommendation feedback: {str(e)}")
            raise
    
    async def get_enhanced_job_matches(
        self, 
        user_id: str, 
        job_matches: List[Any],
        preferred_cities: List[str] = None
    ) -> List[EnhancedJobMatch]:
        """Enhance job matches with application tracking and location scoring."""
        try:
            enhanced_matches = []
            
            # Get user's existing applications
            user_applications = self.db.query(JobApplication).filter(
                JobApplication.user_id == user_id
            ).all()
            
            application_map = {app.job_posting_id: app for app in user_applications}
            
            # Indian tech cities for location scoring
            indian_tech_cities = [
                'bangalore', 'bengaluru', 'hyderabad', 'pune', 'chennai', 
                'mumbai', 'delhi', 'gurgaon', 'noida', 'kolkata', 'ahmedabad'
            ]
            
            for match in job_matches:
                # Check if user has applied
                application = application_map.get(match.job_posting.job_id)
                
                # Calculate location score
                location_score = self._calculate_location_score(
                    match.job_posting.location, 
                    preferred_cities, 
                    indian_tech_cities
                )
                
                # Check if it's an Indian tech city
                is_indian_tech_city = any(
                    city in match.job_posting.location.lower() 
                    for city in indian_tech_cities
                )
                
                # Enhanced gap analysis
                gap_analysis = self._create_enhanced_gap_analysis(match)
                
                enhanced_match = EnhancedJobMatch(
                    job_posting_id=match.job_posting.job_id,
                    job_title=match.job_posting.title,
                    company_name=match.job_posting.company,
                    location=match.job_posting.location,
                    job_url=match.job_posting.url,
                    match_score=match.match_score,
                    skill_matches=[sm.skill for sm in match.skill_matches] if hasattr(match, 'skill_matches') else [],
                    skill_gaps=[sg.skill for sg in match.skill_gaps] if hasattr(match, 'skill_gaps') else [],
                    salary_range=self._format_salary_range(match.job_posting.salary_range),
                    experience_level=match.job_posting.experience_level,
                    posted_date=match.job_posting.posted_date,
                    source=match.job_posting.source,
                    application_status=application.status if application else None,
                    application_id=str(application.id) if application else None,
                    gap_analysis=gap_analysis,
                    recommendation_reason=getattr(match, 'recommendation_reason', ''),
                    location_score=location_score,
                    is_indian_tech_city=is_indian_tech_city,
                    market_demand=self._assess_market_demand(match.job_posting.title),
                    competition_level=self._assess_competition_level(match.match_score)
                )
                
                enhanced_matches.append(enhanced_match)
            
            return enhanced_matches
            
        except Exception as e:
            logger.error(f"Error enhancing job matches: {str(e)}")
            raise
    
    def _calculate_location_score(
        self, 
        job_location: str, 
        preferred_cities: List[str], 
        indian_tech_cities: List[str]
    ) -> float:
        """Calculate location preference score."""
        if not job_location:
            return 0.5
        
        job_location_lower = job_location.lower()
        
        # Check preferred cities first
        if preferred_cities:
            for pref_city in preferred_cities:
                if pref_city.lower() in job_location_lower:
                    return 1.0
        
        # Check if it's an Indian tech city
        for city in indian_tech_cities:
            if city in job_location_lower:
                return 0.8
        
        # Check for remote work
        if any(term in job_location_lower for term in ['remote', 'work from home', 'wfh']):
            return 0.9
        
        return 0.3
    
    def _create_enhanced_gap_analysis(self, match) -> Dict[str, Any]:
        """Create enhanced gap analysis."""
        skill_matches = getattr(match, 'skill_matches', [])
        skill_gaps = getattr(match, 'skill_gaps', [])
        
        total_skills = len(skill_matches) + len(skill_gaps)
        skill_strength = len(skill_matches) / total_skills if total_skills > 0 else 0.5
        
        return {
            "skill_strength": skill_strength,
            "total_skills_required": total_skills,
            "skills_matched": len(skill_matches),
            "skills_missing": len(skill_gaps),
            "experience_gap": max(0, 1 - match.match_score),
            "improvement_priority": skill_gaps[:3] if skill_gaps else [],
            "strength_areas": skill_matches[:3] if skill_matches else []
        }
    
    def _format_salary_range(self, salary_range) -> Optional[str]:
        """Format salary range for display."""
        if not salary_range:
            return None
        
        if hasattr(salary_range, 'min_amount') and hasattr(salary_range, 'max_amount'):
            if salary_range.min_amount and salary_range.max_amount:
                currency = getattr(salary_range, 'currency', 'INR')
                if currency == 'INR':
                    min_lakh = salary_range.min_amount / 100000
                    max_lakh = salary_range.max_amount / 100000
                    return f"₹{min_lakh:.1f}L - ₹{max_lakh:.1f}L"
                else:
                    return f"{currency} {salary_range.min_amount:,} - {salary_range.max_amount:,}"
        
        return str(salary_range)
    
    def _assess_market_demand(self, job_title: str) -> str:
        """Assess market demand for the role."""
        high_demand_roles = [
            'python developer', 'java developer', 'react developer', 
            'data scientist', 'machine learning', 'devops', 'cloud engineer'
        ]
        
        job_title_lower = job_title.lower()
        
        if any(role in job_title_lower for role in high_demand_roles):
            return "high"
        elif any(term in job_title_lower for term in ['developer', 'engineer', 'analyst']):
            return "medium"
        else:
            return "low"
    
    def _assess_competition_level(self, match_score: float) -> str:
        """Assess competition level based on match score."""
        if match_score >= 0.8:
            return "low"  # High match means less competition for this candidate
        elif match_score >= 0.6:
            return "medium"
        else:
            return "high"
    
    async def mark_job_as_applied(
        self, 
        user_id: str, 
        job_posting_id: str, 
        application_method: str = "external"
    ) -> str:
        """Mark a job as applied when user applies externally."""
        try:
            # Check if application already exists
            existing = self.db.query(JobApplication).filter(
                and_(
                    JobApplication.user_id == user_id,
                    JobApplication.job_posting_id == job_posting_id
                )
            ).first()
            
            if existing:
                # Update existing application
                existing.status = "applied"
                existing.applied_date = datetime.utcnow()
                existing.application_method = application_method
                existing.last_updated = datetime.utcnow()
                
                self.db.commit()
                return str(existing.id)
            else:
                # Create new application record
                application = JobApplication(
                    user_id=user_id,
                    job_posting_id=job_posting_id,
                    job_title="Applied Job",  # Will be updated with actual data
                    company_name="Unknown Company",  # Will be updated with actual data
                    status="applied",
                    applied_date=datetime.utcnow(),
                    application_method=application_method
                )
                
                self.db.add(application)
                self.db.commit()
                self.db.refresh(application)
                
                return str(application.id)
                
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error marking job as applied: {str(e)}")
            raise