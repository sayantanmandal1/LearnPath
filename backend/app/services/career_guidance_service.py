"""
Career Guidance Service
Provides comprehensive career guidance including focus areas, project specifications,
preparation roadmaps, and curated resource recommendations.
"""
import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import json
import logging
from sqlalchemy.ext.asyncio import AsyncSession

from app.schemas.career_guidance import (
    CareerGuidanceRequest, CareerGuidanceResponse, FocusArea, ProjectSpecification,
    PreparationRoadmap, CuratedResource, Milestone, LearningOutcome, ResourceRating,
    DifficultyLevel, ResourceType
)
from app.core.database import get_db
from app.services.ai_analysis_service import AIAnalysisService

logger = logging.getLogger(__name__)


class CareerGuidanceService:
    """Service for generating comprehensive career guidance"""
    
    def __init__(self):
        self.ai_service = AIAnalysisService()
        
    async def generate_career_guidance(
        self, 
        request: CareerGuidanceRequest,
        db: AsyncSession
    ) -> CareerGuidanceResponse:
        """Generate comprehensive career guidance for a user"""
        try:
            logger.info(f"Generating career guidance for user {request.user_id}, target role: {request.target_role}")
            
            # Get user's current profile and skills
            user_profile = await self._get_user_profile(request.user_id, db)
            
            # Generate focus areas based on target role and current skills
            focus_areas = await self._generate_focus_areas(request, user_profile)
            
            # Create project specifications with learning outcomes
            project_specs = await self._generate_project_specifications(request, focus_areas)
            
            # Build preparation roadmap with timelines and milestones
            roadmap = await self._generate_preparation_roadmap(request, focus_areas, project_specs)
            
            # Curate resources with quality ratings
            resources = await self._curate_learning_resources(request, focus_areas)
            
            # Calculate confidence score based on data quality
            confidence_score = self._calculate_confidence_score(user_profile, focus_areas, resources)
            
            response = CareerGuidanceResponse(
                user_id=request.user_id,
                target_role=request.target_role,
                generated_at=datetime.utcnow(),
                focus_areas=focus_areas,
                project_specifications=project_specs,
                preparation_roadmap=roadmap,
                curated_resources=resources,
                personalization_factors=self._extract_personalization_factors(request, user_profile),
                confidence_score=confidence_score
            )
            
            # Store guidance for future reference
            await self._store_career_guidance(response, db)
            
            logger.info(f"Successfully generated career guidance for user {request.user_id}")
            return response
            
        except Exception as e:
            logger.error(f"Error generating career guidance: {str(e)}")
            raise
    
    async def _get_user_profile(self, user_id: str, db: AsyncSession) -> Dict[str, Any]:
        """Get user's current profile and skills"""
        # This would typically fetch from database
        # For now, return mock data structure
        return {
            "skills": ["Python", "JavaScript", "SQL", "React"],
            "experience_years": 2,
            "education": "Computer Science",
            "current_role": "Junior Developer",
            "platform_data": {
                "github": {"repositories": 15, "languages": ["Python", "JavaScript"]},
                "leetcode": {"problems_solved": 150, "rating": 1400}
            }
        }
    
    async def _generate_focus_areas(
        self, 
        request: CareerGuidanceRequest, 
        user_profile: Dict[str, Any]
    ) -> List[FocusArea]:
        """Generate detailed focus area recommendations based on target role"""
        
        # Define focus areas for different target roles
        role_focus_mapping = {
            "Senior Software Engineer": [
                {
                    "name": "System Design & Architecture",
                    "description": "Master large-scale system design patterns and architectural principles",
                    "importance_score": 9.5,
                    "skills_required": ["Microservices", "Load Balancing", "Database Design", "Caching"],
                    "estimated_time_weeks": 12
                },
                {
                    "name": "Advanced Programming Patterns",
                    "description": "Deep understanding of design patterns and clean code principles",
                    "importance_score": 8.5,
                    "skills_required": ["Design Patterns", "SOLID Principles", "Code Review", "Testing"],
                    "estimated_time_weeks": 8
                },
                {
                    "name": "Leadership & Mentoring",
                    "description": "Develop technical leadership and team mentoring capabilities",
                    "importance_score": 8.0,
                    "skills_required": ["Code Review", "Technical Writing", "Mentoring", "Project Planning"],
                    "estimated_time_weeks": 10
                }
            ],
            "Full Stack Developer": [
                {
                    "name": "Frontend Frameworks Mastery",
                    "description": "Advanced proficiency in modern frontend frameworks and state management",
                    "importance_score": 9.0,
                    "skills_required": ["React", "Vue.js", "State Management", "Component Design"],
                    "estimated_time_weeks": 10
                },
                {
                    "name": "Backend API Development",
                    "description": "Build robust, scalable backend APIs and microservices",
                    "importance_score": 9.0,
                    "skills_required": ["REST APIs", "GraphQL", "Database Design", "Authentication"],
                    "estimated_time_weeks": 12
                },
                {
                    "name": "DevOps & Deployment",
                    "description": "Master deployment pipelines and cloud infrastructure",
                    "importance_score": 7.5,
                    "skills_required": ["Docker", "CI/CD", "Cloud Platforms", "Monitoring"],
                    "estimated_time_weeks": 8
                }
            ]
        }
        
        base_areas = role_focus_mapping.get(request.target_role, role_focus_mapping["Senior Software Engineer"])
        
        focus_areas = []
        for i, area_data in enumerate(base_areas):
            # Determine current and target levels based on user profile
            current_level = self._assess_current_level(area_data["skills_required"], user_profile)
            target_level = DifficultyLevel.ADVANCED if request.current_experience_years >= 3 else DifficultyLevel.INTERMEDIATE
            
            focus_area = FocusArea(
                id=f"focus_{i+1}",
                name=area_data["name"],
                description=area_data["description"],
                importance_score=area_data["importance_score"],
                current_level=current_level,
                target_level=target_level,
                skills_required=area_data["skills_required"],
                estimated_time_weeks=area_data["estimated_time_weeks"],
                priority_rank=i+1
            )
            focus_areas.append(focus_area)
        
        return focus_areas
    
    def _assess_current_level(self, required_skills: List[str], user_profile: Dict[str, Any]) -> DifficultyLevel:
        """Assess user's current level in a focus area based on their skills"""
        user_skills = set(user_profile.get("skills", []))
        required_skills_set = set(required_skills)
        
        overlap = len(user_skills.intersection(required_skills_set))
        coverage = overlap / len(required_skills_set) if required_skills_set else 0
        
        if coverage >= 0.8:
            return DifficultyLevel.ADVANCED
        elif coverage >= 0.5:
            return DifficultyLevel.INTERMEDIATE
        elif coverage >= 0.2:
            return DifficultyLevel.BEGINNER
        else:
            return DifficultyLevel.BEGINNER
    
    async def _generate_project_specifications(
        self, 
        request: CareerGuidanceRequest, 
        focus_areas: List[FocusArea]
    ) -> List[ProjectSpecification]:
        """Generate project specifications with detailed learning outcomes"""
        
        project_templates = {
            "System Design & Architecture": {
                "title": "Scalable E-commerce Microservices Platform",
                "description": "Build a complete e-commerce platform using microservices architecture with proper service communication, data consistency, and fault tolerance.",
                "technologies": ["Node.js", "Docker", "Kubernetes", "Redis", "PostgreSQL", "RabbitMQ"],
                "deliverables": [
                    "User service with authentication",
                    "Product catalog service",
                    "Order processing service",
                    "Payment gateway integration",
                    "API Gateway implementation",
                    "Monitoring and logging setup"
                ]
            },
            "Frontend Frameworks Mastery": {
                "title": "Real-time Collaborative Dashboard",
                "description": "Create a real-time collaborative dashboard with complex state management, WebSocket integration, and responsive design.",
                "technologies": ["React", "Redux Toolkit", "WebSocket", "Chart.js", "Material-UI", "TypeScript"],
                "deliverables": [
                    "Real-time data visualization",
                    "User collaboration features",
                    "Responsive design implementation",
                    "State management optimization",
                    "Performance monitoring",
                    "Accessibility compliance"
                ]
            },
            "Backend API Development": {
                "title": "Multi-tenant SaaS API Platform",
                "description": "Develop a multi-tenant SaaS API with proper data isolation, rate limiting, and comprehensive testing.",
                "technologies": ["Python", "FastAPI", "PostgreSQL", "Redis", "JWT", "Docker"],
                "deliverables": [
                    "Multi-tenant data architecture",
                    "RESTful API endpoints",
                    "Authentication & authorization",
                    "Rate limiting implementation",
                    "Comprehensive test suite",
                    "API documentation"
                ]
            }
        }
        
        project_specs = []
        for i, focus_area in enumerate(focus_areas[:3]):  # Limit to top 3 focus areas
            template_key = next((key for key in project_templates.keys() if key in focus_area.name), 
                              list(project_templates.keys())[0])
            template = project_templates[template_key]
            
            # Generate learning outcomes based on focus area skills
            learning_outcomes = []
            for j, skill in enumerate(focus_area.skills_required[:4]):  # Top 4 skills
                outcome = LearningOutcome(
                    id=f"outcome_{i+1}_{j+1}",
                    description=f"Gain practical experience in {skill} through hands-on implementation",
                    skills_gained=[skill],
                    competency_level=focus_area.target_level,
                    measurable_criteria=[
                        f"Successfully implement {skill} in project",
                        f"Demonstrate understanding through code review",
                        f"Write comprehensive tests for {skill} functionality"
                    ]
                )
                learning_outcomes.append(outcome)
            
            project_spec = ProjectSpecification(
                id=f"project_{i+1}",
                title=template["title"],
                description=template["description"],
                difficulty_level=focus_area.target_level,
                estimated_duration_weeks=max(4, focus_area.estimated_time_weeks // 2),
                technologies=template["technologies"],
                learning_outcomes=learning_outcomes,
                prerequisites=focus_area.skills_required[:2],  # First 2 skills as prerequisites
                deliverables=template["deliverables"],
                success_metrics=[
                    "All deliverables completed and tested",
                    "Code quality meets industry standards",
                    "Performance benchmarks achieved",
                    "Documentation is comprehensive"
                ],
                github_template_url=f"https://github.com/templates/{template['title'].lower().replace(' ', '-')}"
            )
            project_specs.append(project_spec)
        
        return project_specs
    
    async def _generate_preparation_roadmap(
        self, 
        request: CareerGuidanceRequest, 
        focus_areas: List[FocusArea], 
        project_specs: List[ProjectSpecification]
    ) -> PreparationRoadmap:
        """Generate preparation roadmap with timelines and milestones"""
        
        total_weeks = request.career_timeline_months * 4
        buffer_weeks = max(2, total_weeks // 10)  # 10% buffer time
        
        # Create phases based on focus areas
        phases = []
        current_week = 0
        
        for i, focus_area in enumerate(focus_areas):
            phase_duration = min(focus_area.estimated_time_weeks, total_weeks // len(focus_areas))
            
            phase = {
                "id": f"phase_{i+1}",
                "name": f"Phase {i+1}: {focus_area.name}",
                "description": focus_area.description,
                "start_week": current_week + 1,
                "duration_weeks": phase_duration,
                "focus_area_id": focus_area.id,
                "objectives": focus_area.skills_required
            }
            phases.append(phase)
            current_week += phase_duration
        
        # Generate milestones
        milestones = []
        milestone_count = 0
        
        for i, phase in enumerate(phases):
            # Create milestones for each phase
            phase_milestones = self._create_phase_milestones(
                phase, focus_areas[i], project_specs[i] if i < len(project_specs) else None
            )
            
            for milestone_data in phase_milestones:
                milestone_count += 1
                milestone = Milestone(
                    id=f"milestone_{milestone_count}",
                    title=milestone_data["title"],
                    description=milestone_data["description"],
                    target_date=datetime.utcnow() + timedelta(weeks=milestone_data["week"]),
                    completion_criteria=milestone_data["criteria"],
                    dependencies=milestone_data.get("dependencies", []),
                    estimated_effort_hours=milestone_data["effort_hours"],
                    resources_needed=milestone_data["resources"]
                )
                milestones.append(milestone)
        
        # Calculate success probability based on user profile and timeline
        success_probability = self._calculate_success_probability(request, focus_areas, total_weeks)
        
        roadmap = PreparationRoadmap(
            id=f"roadmap_{request.user_id}_{int(datetime.utcnow().timestamp())}",
            target_role=request.target_role,
            total_duration_weeks=total_weeks,
            phases=phases,
            milestones=milestones,
            critical_path=[f"phase_{i+1}" for i in range(len(phases))],
            buffer_time_weeks=buffer_weeks,
            success_probability=success_probability
        )
        
        return roadmap
    
    def _create_phase_milestones(
        self, 
        phase: Dict[str, Any], 
        focus_area: FocusArea, 
        project_spec: Optional[ProjectSpecification]
    ) -> List[Dict[str, Any]]:
        """Create milestones for a specific phase"""
        
        milestones = []
        phase_start = phase["start_week"]
        phase_duration = phase["duration_weeks"]
        
        # Learning milestone (25% through phase)
        learning_week = phase_start + max(1, phase_duration // 4)
        milestones.append({
            "title": f"Complete {focus_area.name} Learning Foundation",
            "description": f"Master fundamental concepts in {focus_area.name}",
            "week": learning_week,
            "criteria": [
                f"Complete theoretical learning for {skill}" for skill in focus_area.skills_required[:2]
            ],
            "effort_hours": 20,
            "resources": ["Online courses", "Documentation", "Practice exercises"]
        })
        
        # Practice milestone (50% through phase)
        practice_week = phase_start + max(2, phase_duration // 2)
        milestones.append({
            "title": f"Hands-on Practice in {focus_area.name}",
            "description": f"Apply learned concepts through practical exercises",
            "week": practice_week,
            "criteria": [
                "Complete practice exercises",
                "Build small proof-of-concept projects",
                "Demonstrate skill application"
            ],
            "effort_hours": 30,
            "resources": ["Practice platforms", "Code repositories", "Tutorials"]
        })
        
        # Project milestone (75% through phase)
        if project_spec:
            project_week = phase_start + max(3, (phase_duration * 3) // 4)
            milestones.append({
                "title": f"Complete {project_spec.title}",
                "description": project_spec.description,
                "week": project_week,
                "criteria": [f"Deliver {deliverable}" for deliverable in project_spec.deliverables[:3]],
                "effort_hours": 40,
                "resources": ["Development environment", "Project template", "Documentation"]
            })
        
        # Assessment milestone (end of phase)
        assessment_week = phase_start + phase_duration
        milestones.append({
            "title": f"Assessment and Review for {focus_area.name}",
            "description": f"Evaluate progress and mastery in {focus_area.name}",
            "week": assessment_week,
            "criteria": [
                "Pass skill assessment",
                "Complete peer review",
                "Document learning outcomes"
            ],
            "effort_hours": 10,
            "resources": ["Assessment tools", "Peer review platform", "Portfolio documentation"]
        })
        
        return milestones
    
    def _calculate_success_probability(
        self, 
        request: CareerGuidanceRequest, 
        focus_areas: List[FocusArea], 
        total_weeks: int
    ) -> float:
        """Calculate probability of success based on various factors"""
        
        # Base probability
        base_prob = 0.7
        
        # Adjust based on time commitment
        time_factor = min(1.0, request.time_commitment_hours_per_week / 15)  # 15 hours/week is optimal
        
        # Adjust based on experience level
        exp_factor = min(1.0, request.current_experience_years / 5)  # 5 years is considered experienced
        
        # Adjust based on timeline realism
        total_estimated_weeks = sum(fa.estimated_time_weeks for fa in focus_areas)
        timeline_factor = min(1.0, total_weeks / total_estimated_weeks) if total_estimated_weeks > 0 else 1.0
        
        # Adjust based on focus area difficulty
        avg_importance = sum(fa.importance_score for fa in focus_areas) / len(focus_areas) if focus_areas else 5.0
        difficulty_factor = max(0.5, 1.0 - (avg_importance - 5.0) / 10.0)
        
        # Calculate final probability
        success_prob = base_prob * time_factor * exp_factor * timeline_factor * difficulty_factor
        
        return min(0.95, max(0.1, success_prob))  # Clamp between 10% and 95%
    
    async def _curate_learning_resources(
        self, 
        request: CareerGuidanceRequest, 
        focus_areas: List[FocusArea]
    ) -> List[CuratedResource]:
        """Curate learning resources with quality ratings"""
        
        # Resource database (in production, this would be a proper database)
        resource_db = {
            "System Design": [
                {
                    "title": "Designing Data-Intensive Applications",
                    "description": "Comprehensive guide to building scalable, reliable systems",
                    "type": ResourceType.BOOK,
                    "url": "https://dataintensive.net/",
                    "provider": "O'Reilly",
                    "difficulty": DifficultyLevel.ADVANCED,
                    "time_hours": 40,
                    "cost": 45.0,
                    "rating": {"overall": 4.8, "content": 4.9, "difficulty": 4.7, "relevance": 4.8, "community": 4.7},
                    "tags": ["System Design", "Scalability", "Databases", "Distributed Systems"]
                },
                {
                    "title": "System Design Interview Course",
                    "description": "Interactive course covering system design fundamentals",
                    "type": ResourceType.COURSE,
                    "url": "https://www.educative.io/courses/grokking-the-system-design-interview",
                    "provider": "Educative",
                    "difficulty": DifficultyLevel.INTERMEDIATE,
                    "time_hours": 25,
                    "cost": 59.0,
                    "rating": {"overall": 4.6, "content": 4.5, "difficulty": 4.6, "relevance": 4.7, "community": 4.5},
                    "tags": ["System Design", "Interview Prep", "Architecture"]
                }
            ],
            "React": [
                {
                    "title": "React - The Complete Guide",
                    "description": "Comprehensive React course with hooks, context, and advanced patterns",
                    "type": ResourceType.COURSE,
                    "url": "https://www.udemy.com/course/react-the-complete-guide-incl-redux/",
                    "provider": "Udemy",
                    "difficulty": DifficultyLevel.INTERMEDIATE,
                    "time_hours": 48,
                    "cost": 89.99,
                    "rating": {"overall": 4.7, "content": 4.8, "difficulty": 4.6, "relevance": 4.8, "community": 4.6},
                    "tags": ["React", "JavaScript", "Frontend", "Hooks"]
                },
                {
                    "title": "React Official Documentation",
                    "description": "Official React documentation with examples and best practices",
                    "type": ResourceType.DOCUMENTATION,
                    "url": "https://react.dev/",
                    "provider": "Meta",
                    "difficulty": DifficultyLevel.BEGINNER,
                    "time_hours": 15,
                    "cost": 0.0,
                    "rating": {"overall": 4.9, "content": 4.9, "difficulty": 4.8, "relevance": 4.9, "community": 4.8},
                    "tags": ["React", "Documentation", "Official"]
                }
            ],
            "Python": [
                {
                    "title": "Python Crash Course",
                    "description": "Hands-on introduction to Python programming",
                    "type": ResourceType.BOOK,
                    "url": "https://nostarch.com/pythoncrashcourse2e",
                    "provider": "No Starch Press",
                    "difficulty": DifficultyLevel.BEGINNER,
                    "time_hours": 30,
                    "cost": 35.0,
                    "rating": {"overall": 4.5, "content": 4.6, "difficulty": 4.4, "relevance": 4.5, "community": 4.4},
                    "tags": ["Python", "Programming", "Beginner"]
                }
            ]
        }
        
        curated_resources = []
        resource_id = 1
        
        for focus_area in focus_areas:
            # Find relevant resources for each skill in the focus area
            for skill in focus_area.skills_required:
                # Find matching resources
                matching_resources = []
                for key, resources in resource_db.items():
                    if key.lower() in skill.lower() or skill.lower() in key.lower():
                        matching_resources.extend(resources)
                
                # If no direct match, use general resources or broader matching
                if not matching_resources:
                    # Try broader matching for common skills
                    if any(common_skill in skill.lower() for common_skill in ["python", "javascript", "react", "system", "design"]):
                        for key, resources in resource_db.items():
                            if any(common_skill in key.lower() for common_skill in ["python", "javascript", "react", "system"]):
                                matching_resources.extend(resources)
                                break
                    
                    # If still no match, use the first available resource type
                    if not matching_resources and resource_db:
                        matching_resources = list(resource_db.values())[0]
                
                # Convert to CuratedResource objects
                for resource_data in matching_resources[:2]:  # Limit to 2 resources per skill
                    rating = ResourceRating(
                        overall_score=resource_data["rating"]["overall"],
                        content_quality=resource_data["rating"]["content"],
                        difficulty_accuracy=resource_data["rating"]["difficulty"],
                        practical_relevance=resource_data["rating"]["relevance"],
                        community_rating=resource_data["rating"]["community"],
                        last_updated=datetime.utcnow() - timedelta(days=30)
                    )
                    
                    resource = CuratedResource(
                        id=f"resource_{resource_id}",
                        title=resource_data["title"],
                        description=resource_data["description"],
                        resource_type=resource_data["type"],
                        url=resource_data["url"],
                        provider=resource_data["provider"],
                        difficulty_level=resource_data["difficulty"],
                        estimated_time_hours=resource_data["time_hours"],
                        cost=resource_data["cost"],
                        currency="USD",
                        rating=rating,
                        tags=resource_data["tags"],
                        prerequisites=[],
                        learning_outcomes=[f"Master {skill}", f"Apply {skill} in projects"],
                        is_free=resource_data["cost"] == 0.0,
                        certification_available=resource_data["type"] == ResourceType.COURSE
                    )
                    
                    curated_resources.append(resource)
                    resource_id += 1
        
        # Remove duplicates and sort by rating
        seen_titles = set()
        unique_resources = []
        for resource in curated_resources:
            if resource.title not in seen_titles:
                seen_titles.add(resource.title)
                unique_resources.append(resource)
        
        # Sort by overall rating (descending)
        unique_resources.sort(key=lambda r: r.rating.overall_score, reverse=True)
        
        return unique_resources[:10]  # Return top 10 resources
    
    def _extract_personalization_factors(
        self, 
        request: CareerGuidanceRequest, 
        user_profile: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract factors used for personalization"""
        return {
            "current_experience_years": request.current_experience_years,
            "time_commitment_hours_per_week": request.time_commitment_hours_per_week,
            "budget_limit": request.budget_limit,
            "preferred_learning_style": request.preferred_learning_style,
            "specific_interests": request.specific_interests,
            "current_skills": user_profile.get("skills", []),
            "platform_activity": user_profile.get("platform_data", {}),
            "career_timeline_months": request.career_timeline_months
        }
    
    def _calculate_confidence_score(
        self, 
        user_profile: Dict[str, Any], 
        focus_areas: List[FocusArea], 
        resources: List[CuratedResource]
    ) -> float:
        """Calculate confidence score for the guidance"""
        
        # Base confidence
        base_confidence = 0.8
        
        # Adjust based on user profile completeness
        profile_completeness = len(user_profile.get("skills", [])) / 10  # Assume 10 skills is complete
        profile_factor = min(1.0, profile_completeness)
        
        # Adjust based on focus area coverage
        focus_coverage = len(focus_areas) / 5  # Assume 5 focus areas is comprehensive
        focus_factor = min(1.0, focus_coverage)
        
        # Adjust based on resource quality
        avg_resource_rating = sum(r.rating.overall_score for r in resources) / len(resources) if resources else 3.0
        resource_factor = avg_resource_rating / 5.0  # Normalize to 0-1
        
        confidence = base_confidence * profile_factor * focus_factor * resource_factor
        
        return min(0.95, max(0.5, confidence))
    
    async def _store_career_guidance(self, guidance: CareerGuidanceResponse, db: AsyncSession):
        """Store career guidance for future reference"""
        # In production, this would store to database
        logger.info(f"Storing career guidance for user {guidance.user_id}")
        pass
    
    async def get_career_guidance_history(
        self, 
        user_id: str, 
        db: AsyncSession
    ) -> List[CareerGuidanceResponse]:
        """Get user's career guidance history"""
        # In production, this would fetch from database
        return []
    
    async def update_guidance_feedback(
        self, 
        guidance_id: str, 
        feedback: Dict[str, Any], 
        db: AsyncSession
    ) -> bool:
        """Update guidance based on user feedback"""
        # In production, this would update database and retrain models
        logger.info(f"Updating guidance {guidance_id} with feedback")
        return True