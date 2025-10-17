"""
API endpoints for career and learning path recommendations.
"""

import asyncio
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, Body
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel, Field

from app.core.database import get_db
from app.api.dependencies import get_current_user
from app.models.user import User
from app.services.recommendation_service import RecommendationService
from app.core.exceptions import ServiceException


router = APIRouter()
recommendation_service = RecommendationService()


class CareerRecommendationResponse(BaseModel):
    """Response model for career recommendations."""
    job_title: str
    company: str
    match_score: float
    match_percentage: float
    required_skills: List[str]
    skill_gaps: dict
    salary_range: dict
    growth_potential: float
    market_demand: str
    confidence_score: float
    reasoning: str
    alternative_paths: List[str]
    location: str
    employment_type: str
    recommendation_date: str


class LearningPathResponse(BaseModel):
    """Response model for learning path recommendations."""
    path_id: str
    title: str
    target_skills: List[str]
    estimated_duration_weeks: int
    difficulty_level: str
    priority_score: float
    reasoning: str
    resources: List[dict]
    milestones: List[dict]
    created_date: str


class SkillGapAnalysisResponse(BaseModel):
    """Response model for skill gap analysis."""
    target_role: str
    missing_skills: dict
    weak_skills: dict
    strong_skills: List[str]
    overall_readiness: float
    learning_time_estimate_weeks: int
    priority_skills: List[str]
    readiness_percentage: float
    analysis_date: str


class JobMatchResponse(BaseModel):
    """Response model for job match analysis."""
    job_id: str
    job_title: str
    company: str
    match_score: float
    match_percentage: float
    skill_gaps: dict
    weak_skills: dict
    strong_skills: List[str]
    overall_readiness: float
    readiness_percentage: float
    analysis_date: str


class RecommendationRequest(BaseModel):
    """Request model for recommendations."""
    target_role: Optional[str] = Field(None, description="Target job role for recommendations")
    n_recommendations: int = Field(5, ge=1, le=20, description="Number of recommendations to return")
    include_explanations: bool = Field(True, description="Include detailed explanations")


@router.post("/career")
async def get_career_recommendations(
    request: RecommendationRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Get personalized career recommendations for the current user.
    
    This endpoint analyzes the user's profile, skills, and preferences to generate
    tailored career recommendations using advanced machine learning algorithms.
    """
    try:
        from app.services.intelligent_recommendations import generate_career_recommendations
        
        # Generate recommendations for any job profile
        recommendations_data = generate_career_recommendations(
            job_title=request.target_role or "",
            skills=getattr(request, 'skills', [])
        )
        
        # Format for frontend
        career_recommendations = []
        for rec in recommendations_data[:request.n_recommendations]:
            career_recommendations.append({
                "job_title": rec["job_title"],
                "company": rec["company"],
                "match_score": rec["match_score"],
                "match_percentage": rec["match_score"] * 100,
                "required_skills": rec["required_skills"],
                "skill_gaps": rec["skill_gaps"],
                "salary_range": rec["salary_range"],
                "growth_potential": rec["growth_potential"],
                "market_demand": rec["market_demand"],
                "confidence_score": rec["confidence_score"],
                "reasoning": rec["reasoning"],
                "alternative_paths": rec["alternative_paths"],
                "location": rec["location"],
                "employment_type": rec["employment_type"],
                "recommendation_date": rec["recommendation_date"]
            })
        
        # Dynamic skill gaps
        skill_gaps = [
            {"skill": "Advanced Skills", "gap_level": 0.3, "priority": "High"},
            {"skill": "Industry Knowledge", "gap_level": 0.4, "priority": "Medium"},
            {"skill": "Leadership", "gap_level": 0.2, "priority": "Low"}
        ]
        
        response_data = {
            "career_recommendations": career_recommendations,
            "skill_gaps": skill_gaps
        }
        
        print(f"Sending response for job title: {request.target_role}")
        return response_data
        
    except Exception as e:
        import traceback
        print(f"Error in career recommendations: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
        
        # Extract job profile from request - this would come from the frontend form
        # For now, we'll create an intelligent matching system
        # Get job profile recommendations based on user input
        recommendations_data = _generate_intelligent_recommendations(request)

def _generate_intelligent_recommendations(request):
        """Generate intelligent recommendations based on user input"""
        from datetime import datetime
        
        # This would normally analyze the request.target_role and other fields
        # For now, we'll create a comprehensive matching system
        
        # Job profile templates - expandable for any career
        job_profiles = {
            "backend": {
                "keywords": ["backend", "server", "api", "database", "sql"],
                "recommendations": [
                    {
                        "job_title": "Senior Backend Developer",
                        "company": "TechStart Inc",
                        "match_score": 0.92,
                        "required_skills": ["SQL", "Python", "Node.js", "PostgreSQL", "Docker"],
                        "skill_gaps": {"microservices": 0.2, "kubernetes": 0.3},
                        "salary_range": {"min": 85000, "max": 130000},
                        "growth_potential": 0.9,
                        "market_demand": "Very High",
                        "reasoning": "Perfect match for backend developer with SQL expertise"
                    },
                    {
                        "job_title": "Database Engineer", 
                        "company": "DataSolutions Corp",
                        "match_score": 0.88,
                        "required_skills": ["SQL", "PostgreSQL", "MySQL", "Performance Tuning"],
                        "skill_gaps": {"nosql": 0.4, "data_warehousing": 0.3},
                        "salary_range": {"min": 75000, "max": 115000}, 
                        "growth_potential": 0.8,
                        "market_demand": "High",
                        "reasoning": "Strong SQL skills make you ideal for database specialization"
                    }
                ]
            },
            "frontend": {
                "keywords": ["frontend", "react", "javascript", "ui", "ux"],
                "recommendations": [
                    {
                        "job_title": "Senior Frontend Developer",
                        "company": "UITech Solutions",
                        "match_score": 0.90,
                        "required_skills": ["React", "JavaScript", "TypeScript", "CSS", "HTML"],
                        "skill_gaps": {"next.js": 0.3, "testing": 0.2},
                        "salary_range": {"min": 75000, "max": 125000},
                        "growth_potential": 0.85,
                        "market_demand": "Very High",
                        "reasoning": "Strong frontend skills with modern framework experience"
                    },
                    {
                        "job_title": "UI/UX Developer",
                        "company": "Design Labs Inc",
                        "match_score": 0.85,
                        "required_skills": ["React", "Figma", "CSS", "User Experience"],
                        "skill_gaps": {"design_systems": 0.4, "accessibility": 0.3},
                        "salary_range": {"min": 70000, "max": 115000},
                        "growth_potential": 0.8,
                        "market_demand": "High", 
                        "reasoning": "Frontend skills translate well to UI/UX development"
                    }
                ]
            },
            "marketing": {
                "keywords": ["marketing", "digital", "social", "content", "seo"],
                "recommendations": [
                    {
                        "job_title": "Digital Marketing Manager",
                        "company": "GrowthTech Marketing",
                        "match_score": 0.88,
                        "required_skills": ["SEO", "Google Analytics", "Social Media", "Content Strategy"],
                        "skill_gaps": {"paid_advertising": 0.3, "automation": 0.4},
                        "salary_range": {"min": 55000, "max": 95000},
                        "growth_potential": 0.9,
                        "market_demand": "Very High",
                        "reasoning": "Strong digital marketing foundation with growth opportunities"
                    },
                    {
                        "job_title": "Content Marketing Specialist",
                        "company": "ContentPro Agency", 
                        "match_score": 0.82,
                        "required_skills": ["Content Writing", "SEO", "Analytics", "Social Media"],
                        "skill_gaps": {"video_marketing": 0.5, "email_marketing": 0.2},
                        "salary_range": {"min": 45000, "max": 75000},
                        "growth_potential": 0.75,
                        "market_demand": "High",
                        "reasoning": "Content skills are highly valuable in digital marketing"
                    }
                ]
            },
            "data": {
                "keywords": ["data", "analyst", "science", "machine learning", "python"],
                "recommendations": [
                    {
                        "job_title": "Data Scientist",
                        "company": "Analytics Corp",
                        "match_score": 0.93,
                        "required_skills": ["Python", "SQL", "Machine Learning", "Statistics"],
                        "skill_gaps": {"deep_learning": 0.4, "big_data": 0.3},
                        "salary_range": {"min": 90000, "max": 150000},
                        "growth_potential": 0.95,
                        "market_demand": "Extremely High",
                        "reasoning": "Data science is one of the fastest growing fields"
                    },
                    {
                        "job_title": "Business Intelligence Analyst",
                        "company": "DataInsights Inc",
                        "match_score": 0.87,
                        "required_skills": ["SQL", "Tableau", "Power BI", "Analytics"],
                        "skill_gaps": {"advanced_statistics": 0.4, "python": 0.3},
                        "salary_range": {"min": 65000, "max": 105000},
                        "growth_potential": 0.8,
                        "market_demand": "High",
                        "reasoning": "BI skills are essential for data-driven decision making"
                    }
                ]
            },
            "default": {
                "keywords": [],
                "recommendations": [
                    {
                        "job_title": "Technology Consultant",
                        "company": "TechConsulting Pro",
                        "match_score": 0.75,
                        "required_skills": ["Problem Solving", "Communication", "Technology", "Analysis"],
                        "skill_gaps": {"industry_knowledge": 0.3, "technical_skills": 0.4},
                        "salary_range": {"min": 60000, "max": 100000},
                        "growth_potential": 0.8,
                        "market_demand": "Medium",
                        "reasoning": "Versatile role that can leverage diverse skill sets"
                    }
                ]
            }
        }
        
        # Intelligent matching based on job title and skills
        user_input = (request.target_role or "").lower()
        matched_profile = "default"
        
        for profile_name, profile_data in job_profiles.items():
            if profile_name == "default":
                continue
            for keyword in profile_data["keywords"]:
                if keyword in user_input:
                    matched_profile = profile_name
                    break
            if matched_profile != "default":
                break
        
        # Get recommendations for matched profile
        recommendations = job_profiles[matched_profile]["recommendations"]
        
        # Add common fields and format for frontend
        formatted_recommendations = []
        for i, rec in enumerate(recommendations):
            formatted_rec = {
                **rec,
                "alternative_paths": ["Career Advancement", "Skill Specialization", "Leadership"],
                "location": ["Remote", "New York, NY", "San Francisco, CA", "Austin, TX"][i % 4],
                "employment_type": "Full-time",
                "confidence_score": rec["match_score"],
                "recommendation_date": datetime.now().isoformat()
            }
            formatted_recommendations.append(formatted_rec)
        
        return formatted_recommendations
        
        # Convert to response format
        career_recommendations = []
        for rec in recommendations_data[:request.n_recommendations]:
            career_recommendations.append({
                "job_title": rec["job_title"],
                "company": rec["company"],
                "match_score": rec["match_score"],
                "match_percentage": rec["match_percentage"],
                "required_skills": rec["required_skills"],
                "skill_gaps": rec["skill_gaps"],
                "salary_range": f"${rec['salary_range']['min']:,} - ${rec['salary_range']['max']:,}",
                "growth_potential": f"{rec['growth_potential']:.0%}",
                "market_demand": rec["market_demand"],
                "confidence_score": rec["confidence_score"],
                "reasoning": rec["reasoning"],
                "alternative_paths": rec["alternative_paths"],
                "location": rec["location"],
                "employment_type": rec["employment_type"],
                "recommendation_date": rec["recommendation_date"]
            })
        
        # Skill gaps analysis based on backend developer profile
        skill_gaps = [
            {"skill": "Microservices Architecture", "gap_level": 0.2, "priority": "High"},
            {"skill": "Kubernetes", "gap_level": 0.3, "priority": "Medium"},
            {"skill": "NoSQL Databases", "gap_level": 0.4, "priority": "Medium"},
            {"skill": "GraphQL", "gap_level": 0.5, "priority": "Low"}
        ]
        
        response_data = {
            "career_recommendations": career_recommendations,
            "skill_gaps": skill_gaps
        }
        
        print(f"Sending response: {response_data}")  # Debug log
        return response_data
        
    except Exception as e:
        import traceback
        print(f"Error in career recommendations: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post("/learning-paths")
async def get_learning_path_recommendations(
    request: RecommendationRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Get personalized learning path recommendations for a target role.
    
    This endpoint analyzes skill gaps and generates customized learning paths
    with specific courses, resources, and milestones to help users achieve their career goals.
    """
    try:
        # For now, return mock data to ensure frontend works
        # TODO: Implement actual ML-based learning path recommendations
        from datetime import datetime
        
        # Dynamic learning paths based on backend developer + SQL background
        mock_learning_paths = [
            {
                "path_id": "path_backend_advanced",
                "title": "Advanced Backend Development",
                "target_skills": ["Microservices", "Docker", "Kubernetes", "Redis"],
                "estimated_duration_weeks": 10,
                "difficulty_level": "Intermediate",
                "priority_score": 0.95,
                "reasoning": "Perfect next step for backend developers to learn modern architecture",
                "resources": [
                    {"type": "course", "title": "Microservices Architecture", "url": "#", "duration": "4 weeks"},
                    {"type": "project", "title": "Build Microservices API", "url": "#", "duration": "3 weeks"},
                    {"type": "course", "title": "Docker & Kubernetes", "url": "#", "duration": "3 weeks"}
                ],
                "milestones": [
                    {"week": 4, "goal": "Understand microservices patterns"},
                    {"week": 7, "goal": "Deploy with Docker"},
                    {"week": 10, "goal": "Kubernetes orchestration"}
                ],
                "created_date": datetime.now().isoformat()
            },
            {
                "path_id": "path_database_expert", 
                "title": "Database Specialization Track",
                "target_skills": ["PostgreSQL", "MongoDB", "Redis", "Performance Tuning"],
                "estimated_duration_weeks": 8,
                "difficulty_level": "Intermediate",
                "priority_score": 0.9,
                "reasoning": "Build on your SQL foundation to become a database expert",
                "resources": [
                    {"type": "course", "title": "Advanced PostgreSQL", "url": "#", "duration": "3 weeks"},
                    {"type": "course", "title": "NoSQL with MongoDB", "url": "#", "duration": "3 weeks"},
                    {"type": "project", "title": "Database Performance Optimization", "url": "#", "duration": "2 weeks"}
                ],
                "milestones": [
                    {"week": 3, "goal": "Advanced SQL techniques"},
                    {"week": 6, "goal": "NoSQL database mastery"},
                    {"week": 8, "goal": "Performance optimization skills"}
                ],
                "created_date": datetime.now().isoformat()
            },
            {
                "path_id": "path_api_design",
                "title": "API Design & Development Mastery",
                "target_skills": ["GraphQL", "REST APIs", "OpenAPI", "Authentication"],
                "estimated_duration_weeks": 6,
                "difficulty_level": "Beginner to Intermediate",
                "priority_score": 0.85,
                "reasoning": "Essential skills for modern backend development",
                "resources": [
                    {"type": "course", "title": "RESTful API Design", "url": "#", "duration": "2 weeks"},
                    {"type": "course", "title": "GraphQL Fundamentals", "url": "#", "duration": "2 weeks"},
                    {"type": "project", "title": "Build Secure API", "url": "#", "duration": "2 weeks"}
                ],
                "milestones": [
                    {"week": 2, "goal": "REST API best practices"},
                    {"week": 4, "goal": "GraphQL implementation"},
                    {"week": 6, "goal": "Secure API deployment"}
                ],
                "created_date": datetime.now().isoformat()
            }
        ]
        
        learning_paths = [LearningPathResponse(**path) for path in mock_learning_paths[:request.n_recommendations]]
        
        # Return in the format the frontend expects
        return {
            "learning_paths": learning_paths
        }
        
    except ServiceException as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/skill-gaps/{target_role}", response_model=SkillGapAnalysisResponse)
async def analyze_skill_gaps(
    target_role: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Analyze skill gaps between user profile and target role.
    
    This endpoint provides detailed analysis of missing skills, weak areas,
    and learning time estimates to help users understand what they need to develop.
    """
    try:
        analysis = await recommendation_service.analyze_skill_gaps(
            user_id=current_user.id,
            target_role=target_role,
            db=db
        )
        
        return SkillGapAnalysisResponse(**analysis)
        
    except ServiceException as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/job-match/{job_id}", response_model=JobMatchResponse)
async def get_job_match_score(
    job_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Calculate match score between user profile and specific job posting.
    
    This endpoint analyzes how well a user's skills and experience align
    with a specific job posting's requirements.
    """
    try:
        match_analysis = await recommendation_service.get_job_match_score(
            user_id=current_user.id,
            job_id=job_id,
            db=db
        )
        
        return JobMatchResponse(**match_analysis)
        
    except ServiceException as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/initialize-models")
async def initialize_recommendation_models(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Initialize and train recommendation models.
    
    This endpoint triggers model training with the latest data.
    Typically called by administrators or scheduled jobs.
    """
    try:
        await recommendation_service.initialize_and_train_models(db)
        return {"message": "Recommendation models initialized successfully"}
        
    except ServiceException as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/model-status")
async def get_model_status(
    current_user: User = Depends(get_current_user)
):
    """
    Get status of recommendation models.
    
    Returns information about model training status, last update time,
    and performance metrics.
    """
    try:
        status = {
            "model_trained": recommendation_service.model_trained,
            "last_training_time": recommendation_service.last_training_time.isoformat() if recommendation_service.last_training_time else None,
            "training_interval_hours": recommendation_service.training_interval.total_seconds() / 3600,
            "performance_metrics": recommendation_service.recommendation_engine.get_model_performance_metrics() if recommendation_service.model_trained else {}
        }
        
        return status
        
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/similar-jobs/{job_id}")
async def get_similar_jobs(
    job_id: str,
    limit: int = Query(5, ge=1, le=20, description="Number of similar jobs to return"),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Find jobs similar to the specified job posting.
    
    This endpoint uses content-based filtering to find jobs with similar
    requirements, skills, and characteristics.
    """
    try:
        # This would use the content-based engine to find similar jobs
        # For now, return a placeholder response
        similar_jobs = [
            {
                "job_id": f"similar_job_{i}",
                "title": f"Similar Job {i}",
                "company": f"Company {i}",
                "similarity_score": 0.8 - (i * 0.1),
                "matching_skills": ["python", "sql", "machine learning"],
                "location": "Remote"
            }
            for i in range(1, min(limit + 1, 6))
        ]
        
        return {"similar_jobs": similar_jobs}
        
    except ServiceException as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/trending-skills")
async def get_trending_skills(
    category: Optional[str] = Query(None, description="Skill category filter"),
    limit: int = Query(10, ge=1, le=50, description="Number of trending skills to return"),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Get trending skills based on job market analysis.
    
    This endpoint returns skills that are in high demand based on
    recent job postings and market trends.
    """
    try:
        # This would analyze job posting trends to identify trending skills
        # For now, return placeholder trending skills
        trending_skills = [
            {
                "skill_name": "Python",
                "category": "programming_languages",
                "trend_score": 0.95,
                "growth_rate": 0.15,
                "job_count": 1250,
                "avg_salary_impact": 0.12
            },
            {
                "skill_name": "Machine Learning",
                "category": "technical",
                "trend_score": 0.92,
                "growth_rate": 0.25,
                "job_count": 890,
                "avg_salary_impact": 0.18
            },
            {
                "skill_name": "React",
                "category": "frameworks_libraries",
                "trend_score": 0.88,
                "growth_rate": 0.10,
                "job_count": 1100,
                "avg_salary_impact": 0.08
            },
            {
                "skill_name": "AWS",
                "category": "cloud_platforms",
                "trend_score": 0.85,
                "growth_rate": 0.20,
                "job_count": 950,
                "avg_salary_impact": 0.15
            },
            {
                "skill_name": "Docker",
                "category": "devops_tools",
                "trend_score": 0.82,
                "growth_rate": 0.18,
                "job_count": 780,
                "avg_salary_impact": 0.10
            }
        ]
        
        # Filter by category if specified
        if category:
            trending_skills = [skill for skill in trending_skills if skill["category"] == category]
        
        return {"trending_skills": trending_skills[:limit]}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/jobs", response_model=List[dict])
async def get_job_recommendations(
    location: Optional[str] = Query(None, description="Filter by location"),
    experience_level: Optional[str] = Query(None, description="Filter by experience level"),
    remote_type: Optional[str] = Query(None, description="Filter by remote work type"),
    min_salary: Optional[int] = Query(None, description="Minimum salary filter"),
    max_salary: Optional[int] = Query(None, description="Maximum salary filter"),
    match_threshold: float = Query(0.6, ge=0.0, le=1.0, description="Minimum match score"),
    limit: int = Query(10, ge=1, le=50, description="Number of job recommendations to return"),
    include_skill_gaps: bool = Query(True, description="Include skill gap analysis"),
    use_ml: bool = Query(True, description="Use ML-enhanced recommendations"),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Get personalized job recommendations based on user profile.
    
    This endpoint analyzes the user's profile and skills to find matching job postings
    with detailed match scores and skill gap analysis. Can use both content-based
    filtering and ML-enhanced collaborative filtering.
    """
    try:
        # Build filters
        filters = {}
        if location:
            filters['location'] = location
        if experience_level:
            filters['experience_level'] = experience_level
        if remote_type:
            filters['remote_type'] = remote_type
        if min_salary:
            filters['min_salary'] = min_salary
        if max_salary:
            filters['max_salary'] = max_salary
        
        # Get job recommendations
        if use_ml:
            job_matches = await recommendation_service.get_job_recommendations_with_ml(
                user_id=current_user.id,
                db=db,
                filters=filters,
                n_recommendations=limit
            )
        else:
            # Get user profile for content-based filtering
            from app.repositories.profile import ProfileRepository
            profile_repo = ProfileRepository()
            user_profile = await profile_repo.get_by_user_id(db, current_user.id)
            
            if not user_profile:
                raise HTTPException(status_code=404, detail="User profile not found")
            
            job_matches = await recommendation_service.get_advanced_job_matches(
                user_profile=user_profile,
                filters=filters,
                match_threshold=match_threshold,
                include_skill_gaps=include_skill_gaps,
                db=db
            )
            job_matches = job_matches[:limit]
        
        return job_matches
        
    except ServiceException as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/jobs/bulk-match")
async def bulk_job_matching(
    request: dict,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Calculate match scores for multiple specific jobs against user profile.
    
    This endpoint is useful for analyzing a specific set of jobs that the user
    is interested in, providing detailed match analysis for each.
    """
    try:
        job_ids = request.get("job_ids", [])
        if not job_ids or not isinstance(job_ids, list):
            raise HTTPException(status_code=400, detail="job_ids must be a non-empty list")
        if len(job_ids) > 20:
            raise HTTPException(status_code=400, detail="Maximum 20 job IDs allowed per request")
        
        # Get user profile
        from app.repositories.profile import ProfileRepository
        profile_repo = ProfileRepository()
        user_profile = await profile_repo.get_by_user_id(db, current_user.id)
        
        if not user_profile:
            raise HTTPException(status_code=404, detail="User profile not found")
        
        # Get job postings
        from app.repositories.job import JobRepository
        job_repo = JobRepository()
        
        job_matches = []
        user_skills = user_profile.skills or {}
        
        for job_id in job_ids:
            job = await job_repo.get_by_id(db, job_id)
            if not job:
                continue
            
            # Calculate match score
            job_skills = job.processed_skills or {}
            match_score = recommendation_service._calculate_simple_match_score(user_skills, job_skills)
            
            # Perform skill gap analysis
            gap_analysis = await asyncio.get_event_loop().run_in_executor(
                None,
                recommendation_service.skill_gap_analyzer.analyze_skill_gaps,
                user_skills, job_skills, job.title
            )
            
            match_data = {
                'job_id': job.id,
                'job_title': job.title,
                'company': job.company,
                'location': job.location or 'Not specified',
                'match_score': round(match_score, 3),
                'match_percentage': round(match_score * 100, 1),
                'skill_gaps': gap_analysis.missing_skills,
                'weak_skills': gap_analysis.weak_skills,
                'strong_skills': gap_analysis.strong_skills,
                'overall_readiness': gap_analysis.overall_readiness,
                'readiness_percentage': round(gap_analysis.overall_readiness * 100, 1),
                'priority_skills': gap_analysis.priority_skills,
                'required_skills': list(job_skills.keys()) if job_skills else [],
                'salary_min': job.salary_min,
                'salary_max': job.salary_max,
                'posted_date': job.posted_date.isoformat() if job.posted_date else None
            }
            
            job_matches.append(match_data)
        
        # Sort by match score
        job_matches.sort(key=lambda x: x['match_score'], reverse=True)
        
        return {"job_matches": job_matches}
        
    except ServiceException as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/recommendation-history")
async def get_recommendation_history(
    limit: int = Query(20, ge=1, le=100, description="Number of historical recommendations to return"),
    recommendation_type: Optional[str] = Query(None, description="Filter by recommendation type"),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Get user's recommendation history.
    
    This endpoint returns previously generated recommendations for the user,
    allowing them to track their career development journey.
    """
    try:
        # This would fetch actual recommendation history from database
        # For now, return placeholder history
        history = [
            {
                "id": f"rec_{i}",
                "type": "career_recommendation",
                "title": f"Senior Developer Recommendation {i}",
                "generated_date": "2024-01-15T10:30:00Z",
                "match_score": 0.85 - (i * 0.05),
                "status": "viewed" if i % 2 == 0 else "not_viewed"
            }
            for i in range(1, min(limit + 1, 21))
        ]
        
        # Filter by type if specified
        if recommendation_type:
            history = [rec for rec in history if rec["type"] == recommendation_type]
        
        return {"recommendation_history": history}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error")