"""
Learning Path Service for the AI Career Recommender System.

This service implements personalized learning path generation including:
- Skill gap identification and prioritization
- Learning resource integration from multiple platforms
- Project recommendations from GitHub
- Learning path sequencing and timeline estimation
- Resource quality scoring and filtering
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Any
from collections import defaultdict
import httpx
import json

from ..schemas.learning_path import (
    LearningPath, LearningResource, Milestone, SkillGap, ProjectRecommendation,
    LearningPathRequest, LearningPathResponse, DifficultyLevel, ResourceType,
    ResourceProvider, LearningProgress
)
from ..schemas.profile import ProfileResponse
from ..core.exceptions import ServiceException
from ..core.redis import get_redis


logger = logging.getLogger(__name__)


class LearningPathService:
    """Service for generating and managing personalized learning paths."""
    
    def __init__(self):
        self.redis_client = None  # Will be initialized when needed
        self.skill_taxonomy = self._load_skill_taxonomy()
        self.resource_cache_ttl = 3600  # 1 hour
        
    def _load_skill_taxonomy(self) -> Dict[str, Any]:
        """Load skill taxonomy and relationships."""
        # This would typically load from database or config file
        return {
            "programming_languages": {
                "python": {"difficulty": "beginner", "category": "backend", "prerequisites": []},
                "javascript": {"difficulty": "beginner", "category": "frontend", "prerequisites": []},
                "java": {"difficulty": "intermediate", "category": "backend", "prerequisites": ["programming_basics"]},
                "react": {"difficulty": "intermediate", "category": "frontend", "prerequisites": ["javascript", "html", "css"]},
                "machine_learning": {"difficulty": "advanced", "category": "ai", "prerequisites": ["python", "statistics", "linear_algebra"]},
                "docker": {"difficulty": "intermediate", "category": "devops", "prerequisites": ["linux", "command_line"]},
                "kubernetes": {"difficulty": "advanced", "category": "devops", "prerequisites": ["docker", "networking"]},
            },
            "skill_relationships": {
                "python": ["data_science", "machine_learning", "web_development", "automation"],
                "javascript": ["react", "node.js", "vue.js", "angular"],
                "machine_learning": ["deep_learning", "nlp", "computer_vision", "data_science"],
            }
        }

    async def generate_learning_paths(self, request: LearningPathRequest) -> LearningPathResponse:
        """
        Generate personalized learning paths based on user requirements.
        
        Implements requirements 4.1, 4.4, 4.7
        """
        try:
            logger.info(f"Generating learning paths for user {request.user_id}")
            
            # Step 1: Identify and prioritize skill gaps
            skill_gaps = await self._identify_skill_gaps(request)
            
            # Step 2: Generate multiple learning path options
            learning_paths = []
            
            # Generate primary path (most direct)
            primary_path = await self._generate_primary_path(request, skill_gaps)
            learning_paths.append(primary_path)
            
            # Generate alternative paths (different approaches)
            alternative_paths = await self._generate_alternative_paths(request, skill_gaps)
            learning_paths.extend(alternative_paths)
            
            # Step 3: Rank and filter paths
            ranked_paths = await self._rank_learning_paths(learning_paths, request)
            
            return LearningPathResponse(
                learning_paths=ranked_paths[:5],  # Return top 5 paths
                total_paths=len(ranked_paths),
                skill_gaps_identified=skill_gaps,
                recommendations_metadata={
                    "generation_method": "hybrid_recommendation",
                    "personalization_factors": ["current_skills", "target_role", "time_commitment"],
                    "data_sources": ["coursera", "udemy", "edx", "github", "freecodecamp"]
                }
            )
            
        except Exception as e:
            logger.error(f"Error generating learning paths: {str(e)}")
            raise ServiceException(f"Failed to generate learning paths: {str(e)}")

    async def _identify_skill_gaps(self, request: LearningPathRequest) -> List[SkillGap]:
        """
        Identify and prioritize skill gaps based on current skills and target role.
        
        Implements requirement 4.1
        """
        skill_gaps = []
        
        # Get target skills for the role
        target_skills = await self._get_target_skills_for_role(request.target_role)
        target_skills.extend(request.target_skills)
        
        # Calculate gaps for each target skill
        for skill in set(target_skills):
            current_level = request.current_skills.get(skill, 0.0)
            target_level = 0.8  # Default target level
            
            if current_level < target_level:
                gap_size = target_level - current_level
                priority = await self._calculate_skill_priority(skill, gap_size, request.target_role)
                market_demand = await self._get_skill_market_demand(skill)
                
                skill_gap = SkillGap(
                    skill_name=skill,
                    current_level=current_level,
                    target_level=target_level,
                    gap_size=gap_size,
                    priority=priority,
                    market_demand=market_demand,
                    estimated_learning_hours=self._estimate_learning_hours(skill, gap_size),
                    difficulty=self._get_skill_difficulty(skill)
                )
                skill_gaps.append(skill_gap)
        
        # Sort by priority (highest first)
        skill_gaps.sort(key=lambda x: x.priority, reverse=True)
        
        return skill_gaps

    async def _generate_primary_path(self, request: LearningPathRequest, skill_gaps: List[SkillGap]) -> LearningPath:
        """Generate the primary (most direct) learning path."""
        
        # Select top priority skills
        priority_skills = [gap.skill_name for gap in skill_gaps[:8]]  # Top 8 skills
        
        # Create learning path
        path = LearningPath(
            title=f"Primary Path to {request.target_role or 'Target Skills'}",
            description="The most direct path to achieve your learning goals",
            target_role=request.target_role,
            target_skills=priority_skills,
            skill_gaps=skill_gaps,
            difficulty_level=self._determine_overall_difficulty(skill_gaps),
            estimated_duration_weeks=self._calculate_total_duration_weeks(skill_gaps, request.time_commitment_hours_per_week),
            estimated_duration_hours=int(sum(gap.estimated_learning_hours for gap in skill_gaps))
        )
        
        # Generate milestones and resources
        path.milestones = await self._create_milestones(priority_skills, skill_gaps)
        path.resources = await self._gather_learning_resources(priority_skills, request)
        
        # Calculate confidence score
        path.confidence_score = self._calculate_path_confidence(path, request)
        
        return path

    async def _generate_alternative_paths(self, request: LearningPathRequest, skill_gaps: List[SkillGap]) -> List[LearningPath]:
        """Generate alternative learning paths with different approaches."""
        
        alternative_paths = []
        
        # Path 1: Project-focused path
        project_path = await self._generate_project_focused_path(request, skill_gaps)
        alternative_paths.append(project_path)
        
        # Path 2: Certification-focused path
        cert_path = await self._generate_certification_path(request, skill_gaps)
        alternative_paths.append(cert_path)
        
        # Path 3: Free resources only path (if budget is a concern)
        if request.include_free_only or request.budget_limit and request.budget_limit < 100:
            free_path = await self._generate_free_resources_path(request, skill_gaps)
            alternative_paths.append(free_path)
        
        return alternative_paths

    async def _create_milestones(self, skills: List[str], skill_gaps: List[SkillGap]) -> List[Milestone]:
        """
        Create learning milestones with proper sequencing.
        
        Implements requirement 4.4
        """
        milestones = []
        
        # Group skills by difficulty and dependencies
        skill_levels = self._group_skills_by_level(skills)
        
        milestone_order = 0
        for level, level_skills in skill_levels.items():
            if not level_skills:
                continue
                
            milestone = Milestone(
                title=f"{level.title()} Level Skills",
                description=f"Master {level} level skills: {', '.join(level_skills)}",
                order=milestone_order,
                skills_to_acquire=level_skills,
                estimated_duration_hours=int(sum(
                    gap.estimated_learning_hours for gap in skill_gaps 
                    if gap.skill_name in level_skills
                )),
                completion_criteria=[
                    f"Complete courses for {skill}" for skill in level_skills
                ] + [
                    f"Build project demonstrating {skill}" for skill in level_skills[:2]  # Projects for first 2 skills
                ]
            )
            
            milestones.append(milestone)
            milestone_order += 1
        
        return milestones

    async def _gather_learning_resources(self, skills: List[str], request: LearningPathRequest) -> List[LearningResource]:
        """
        Gather learning resources from multiple platforms.
        
        Implements requirement 4.2
        """
        all_resources = []
        
        # Gather resources from different platforms concurrently
        tasks = []
        
        if not request.preferred_providers or ResourceProvider.COURSERA in request.preferred_providers:
            tasks.append(self._get_coursera_resources(skills, request))
        
        if not request.preferred_providers or ResourceProvider.UDEMY in request.preferred_providers:
            tasks.append(self._get_udemy_resources(skills, request))
        
        if not request.preferred_providers or ResourceProvider.EDX in request.preferred_providers:
            tasks.append(self._get_edx_resources(skills, request))
        
        if not request.preferred_providers or ResourceProvider.FREECODECAMP in request.preferred_providers:
            tasks.append(self._get_freecodecamp_resources(skills, request))
        
        # Execute all tasks concurrently
        resource_lists = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Combine results and handle exceptions
        for resource_list in resource_lists:
            if isinstance(resource_list, Exception):
                logger.warning(f"Failed to fetch resources: {resource_list}")
                continue
            all_resources.extend(resource_list)
        
        # Filter and score resources
        filtered_resources = await self._filter_and_score_resources(all_resources, request)
        
        return filtered_resources

    async def _get_coursera_resources(self, skills: List[str], request: LearningPathRequest) -> List[LearningResource]:
        """Fetch learning resources from Coursera API."""
        resources = []
        
        try:
            # This would integrate with actual Coursera API
            # For now, using mock data based on common courses
            coursera_courses = {
                "python": {
                    "title": "Python for Everybody Specialization",
                    "url": "https://www.coursera.org/specializations/python",
                    "duration_hours": 120,
                    "cost": 49.0,
                    "rating": 4.8,
                    "difficulty": DifficultyLevel.BEGINNER
                },
                "machine_learning": {
                    "title": "Machine Learning Course by Andrew Ng",
                    "url": "https://www.coursera.org/learn/machine-learning",
                    "duration_hours": 60,
                    "cost": 79.0,
                    "rating": 4.9,
                    "difficulty": DifficultyLevel.INTERMEDIATE
                },
                "data_science": {
                    "title": "IBM Data Science Professional Certificate",
                    "url": "https://www.coursera.org/professional-certificates/ibm-data-science",
                    "duration_hours": 180,
                    "cost": 39.0,
                    "rating": 4.6,
                    "difficulty": DifficultyLevel.INTERMEDIATE
                }
            }
            
            for skill in skills:
                if skill.lower() in coursera_courses:
                    course_data = coursera_courses[skill.lower()]
                    
                    # Skip if over budget
                    if request.budget_limit and course_data["cost"] > request.budget_limit:
                        continue
                    
                    # Skip if free only requested
                    if request.include_free_only and course_data["cost"] > 0:
                        continue
                    
                    resource = LearningResource(
                        title=course_data["title"],
                        type=ResourceType.COURSE,
                        provider=ResourceProvider.COURSERA,
                        url=course_data["url"],
                        rating=course_data["rating"],
                        duration_hours=course_data["duration_hours"],
                        cost=course_data["cost"],
                        difficulty_level=course_data["difficulty"],
                        skills_taught=[skill],
                        certificate_available=True,
                        quality_score=0.9,  # High quality for Coursera
                        popularity_score=0.8
                    )
                    resources.append(resource)
            
        except Exception as e:
            logger.error(f"Error fetching Coursera resources: {e}")
        
        return resources

    async def _get_udemy_resources(self, skills: List[str], request: LearningPathRequest) -> List[LearningResource]:
        """Fetch learning resources from Udemy API."""
        resources = []
        
        try:
            # Mock Udemy courses data
            udemy_courses = {
                "javascript": {
                    "title": "The Complete JavaScript Course 2024",
                    "url": "https://www.udemy.com/course/the-complete-javascript-course/",
                    "duration_hours": 69,
                    "cost": 84.99,
                    "rating": 4.7,
                    "difficulty": DifficultyLevel.BEGINNER
                },
                "react": {
                    "title": "React - The Complete Guide",
                    "url": "https://www.udemy.com/course/react-the-complete-guide-incl-redux/",
                    "duration_hours": 48,
                    "cost": 94.99,
                    "rating": 4.6,
                    "difficulty": DifficultyLevel.INTERMEDIATE
                },
                "docker": {
                    "title": "Docker Mastery: with Kubernetes +Swarm",
                    "url": "https://www.udemy.com/course/docker-mastery/",
                    "duration_hours": 19,
                    "cost": 89.99,
                    "rating": 4.6,
                    "difficulty": DifficultyLevel.INTERMEDIATE
                }
            }
            
            for skill in skills:
                if skill.lower() in udemy_courses:
                    course_data = udemy_courses[skill.lower()]
                    
                    if request.budget_limit and course_data["cost"] > request.budget_limit:
                        continue
                    
                    if request.include_free_only and course_data["cost"] > 0:
                        continue
                    
                    resource = LearningResource(
                        title=course_data["title"],
                        type=ResourceType.COURSE,
                        provider=ResourceProvider.UDEMY,
                        url=course_data["url"],
                        rating=course_data["rating"],
                        duration_hours=course_data["duration_hours"],
                        cost=course_data["cost"],
                        difficulty_level=course_data["difficulty"],
                        skills_taught=[skill],
                        certificate_available=True,
                        hands_on_projects=True,
                        quality_score=0.8,
                        popularity_score=0.9
                    )
                    resources.append(resource)
            
        except Exception as e:
            logger.error(f"Error fetching Udemy resources: {e}")
        
        return resources

    async def _get_edx_resources(self, skills: List[str], request: LearningPathRequest) -> List[LearningResource]:
        """Fetch learning resources from edX API."""
        resources = []
        
        try:
            # Mock edX courses data
            edx_courses = {
                "computer_science": {
                    "title": "CS50's Introduction to Computer Science",
                    "url": "https://www.edx.org/course/introduction-computer-science-harvardx-cs50x",
                    "duration_hours": 100,
                    "cost": 0.0,  # Free
                    "rating": 4.8,
                    "difficulty": DifficultyLevel.BEGINNER
                },
                "artificial_intelligence": {
                    "title": "Artificial Intelligence (AI)",
                    "url": "https://www.edx.org/course/artificial-intelligence-ai",
                    "duration_hours": 80,
                    "cost": 0.0,
                    "rating": 4.5,
                    "difficulty": DifficultyLevel.ADVANCED
                }
            }
            
            for skill in skills:
                # Map skills to edX course categories
                course_key = self._map_skill_to_edx_category(skill)
                if course_key in edx_courses:
                    course_data = edx_courses[course_key]
                    
                    resource = LearningResource(
                        title=course_data["title"],
                        type=ResourceType.COURSE,
                        provider=ResourceProvider.EDX,
                        url=course_data["url"],
                        rating=course_data["rating"],
                        duration_hours=course_data["duration_hours"],
                        cost=course_data["cost"],
                        difficulty_level=course_data["difficulty"],
                        skills_taught=[skill],
                        certificate_available=True,
                        quality_score=0.9,  # High quality for edX
                        popularity_score=0.7
                    )
                    resources.append(resource)
            
        except Exception as e:
            logger.error(f"Error fetching edX resources: {e}")
        
        return resources

    async def _get_freecodecamp_resources(self, skills: List[str], request: LearningPathRequest) -> List[LearningResource]:
        """Fetch learning resources from freeCodeCamp."""
        resources = []
        
        try:
            # freeCodeCamp curriculum mapping
            fcc_curriculum = {
                "html": {
                    "title": "Responsive Web Design Certification",
                    "url": "https://www.freecodecamp.org/learn/responsive-web-design/",
                    "duration_hours": 300,
                    "cost": 0.0,
                    "difficulty": DifficultyLevel.BEGINNER
                },
                "javascript": {
                    "title": "JavaScript Algorithms and Data Structures",
                    "url": "https://www.freecodecamp.org/learn/javascript-algorithms-and-data-structures/",
                    "duration_hours": 300,
                    "cost": 0.0,
                    "difficulty": DifficultyLevel.INTERMEDIATE
                },
                "python": {
                    "title": "Scientific Computing with Python",
                    "url": "https://www.freecodecamp.org/learn/scientific-computing-with-python/",
                    "duration_hours": 300,
                    "cost": 0.0,
                    "difficulty": DifficultyLevel.INTERMEDIATE
                }
            }
            
            for skill in skills:
                if skill.lower() in fcc_curriculum:
                    course_data = fcc_curriculum[skill.lower()]
                    
                    resource = LearningResource(
                        title=course_data["title"],
                        type=ResourceType.COURSE,
                        provider=ResourceProvider.FREECODECAMP,
                        url=course_data["url"],
                        rating=4.7,  # Generally high rating
                        duration_hours=course_data["duration_hours"],
                        cost=course_data["cost"],
                        difficulty_level=course_data["difficulty"],
                        skills_taught=[skill],
                        certificate_available=True,
                        hands_on_projects=True,
                        quality_score=0.8,
                        popularity_score=0.9
                    )
                    resources.append(resource)
            
        except Exception as e:
            logger.error(f"Error fetching freeCodeCamp resources: {e}")
        
        return resources

    async def get_project_recommendations(self, skills: List[str], difficulty: DifficultyLevel = DifficultyLevel.INTERMEDIATE) -> List[ProjectRecommendation]:
        """
        Get project recommendations from GitHub trending repositories.
        
        Implements requirement 4.3
        """
        try:
            projects = []
            
            # This would integrate with GitHub API to get trending repositories
            # For now, using curated project recommendations
            project_templates = {
                "python": [
                    {
                        "title": "Personal Finance Tracker",
                        "description": "Build a web app to track personal expenses with data visualization",
                        "repository_url": "https://github.com/example/finance-tracker",
                        "skills_practiced": ["python", "flask", "sqlite", "data_visualization"],
                        "technologies": ["Python", "Flask", "SQLite", "Chart.js"],
                        "estimated_duration_hours": 40
                    },
                    {
                        "title": "Machine Learning Stock Predictor",
                        "description": "Create a ML model to predict stock prices using historical data",
                        "repository_url": "https://github.com/example/stock-predictor",
                        "skills_practiced": ["python", "machine_learning", "pandas", "scikit-learn"],
                        "technologies": ["Python", "Pandas", "Scikit-learn", "Matplotlib"],
                        "estimated_duration_hours": 60
                    }
                ],
                "javascript": [
                    {
                        "title": "Task Management App",
                        "description": "Build a full-stack task management application with React and Node.js",
                        "repository_url": "https://github.com/example/task-manager",
                        "skills_practiced": ["javascript", "react", "node.js", "mongodb"],
                        "technologies": ["React", "Node.js", "Express", "MongoDB"],
                        "estimated_duration_hours": 50
                    }
                ],
                "react": [
                    {
                        "title": "E-commerce Dashboard",
                        "description": "Create an admin dashboard for an e-commerce platform",
                        "repository_url": "https://github.com/example/ecommerce-dashboard",
                        "skills_practiced": ["react", "typescript", "state_management", "api_integration"],
                        "technologies": ["React", "TypeScript", "Redux", "Material-UI"],
                        "estimated_duration_hours": 70
                    }
                ]
            }
            
            for skill in skills:
                if skill.lower() in project_templates:
                    for project_data in project_templates[skill.lower()]:
                        project = ProjectRecommendation(
                            title=project_data["title"],
                            description=project_data["description"],
                            repository_url=project_data["repository_url"],
                            difficulty_level=difficulty,
                            skills_practiced=project_data["skills_practiced"],
                            technologies=project_data["technologies"],
                            estimated_duration_hours=project_data["estimated_duration_hours"],
                            stars=1250,  # Mock data
                            forks=340,
                            trending_score=0.8,
                            learning_value=0.9,
                            market_relevance=0.85
                        )
                        projects.append(project)
            
            # Sort by learning value and market relevance
            projects.sort(key=lambda x: (x.learning_value or 0) * (x.market_relevance or 0), reverse=True)
            
            return projects[:10]  # Return top 10 projects
            
        except Exception as e:
            logger.error(f"Error getting project recommendations: {e}")
            raise ServiceException(f"Failed to get project recommendations: {str(e)}")

    async def _filter_and_score_resources(self, resources: List[LearningResource], request: LearningPathRequest) -> List[LearningResource]:
        """
        Filter and score learning resources based on quality and relevance.
        
        Implements requirement 4.6
        """
        filtered_resources = []
        
        for resource in resources:
            # Apply filters
            if request.include_free_only and resource.cost and resource.cost > 0:
                continue
            
            if request.budget_limit and resource.cost and resource.cost > request.budget_limit:
                continue
            
            if request.difficulty_preference and resource.difficulty_level != request.difficulty_preference:
                continue
            
            # Calculate overall quality score
            quality_factors = []
            
            # Rating factor (0-1)
            if resource.rating:
                quality_factors.append(resource.rating / 5.0)
            
            # Provider reputation factor
            provider_scores = {
                ResourceProvider.COURSERA: 0.9,
                ResourceProvider.EDX: 0.9,
                ResourceProvider.UDEMY: 0.8,
                ResourceProvider.FREECODECAMP: 0.8,
                ResourceProvider.PLURALSIGHT: 0.85,
                ResourceProvider.UDACITY: 0.85
            }
            quality_factors.append(provider_scores.get(resource.provider, 0.7))
            
            # Certificate availability factor
            if resource.certificate_available:
                quality_factors.append(0.8)
            
            # Hands-on projects factor
            if resource.hands_on_projects:
                quality_factors.append(0.9)
            
            # Calculate weighted average
            resource.quality_score = sum(quality_factors) / len(quality_factors) if quality_factors else 0.5
            
            filtered_resources.append(resource)
        
        # Sort by quality score (highest first)
        filtered_resources.sort(key=lambda x: x.quality_score or 0, reverse=True)
        
        return filtered_resources

    # Helper methods
    
    async def _get_target_skills_for_role(self, target_role: Optional[str]) -> List[str]:
        """Get required skills for a target role."""
        if not target_role:
            return []
        
        # This would typically query a database of job requirements
        role_skills = {
            "software_engineer": ["python", "javascript", "git", "sql", "algorithms", "system_design"],
            "data_scientist": ["python", "machine_learning", "statistics", "sql", "data_visualization", "pandas"],
            "frontend_developer": ["javascript", "react", "html", "css", "typescript", "responsive_design"],
            "backend_developer": ["python", "java", "sql", "api_design", "microservices", "docker"],
            "devops_engineer": ["docker", "kubernetes", "aws", "linux", "ci_cd", "monitoring"],
            "machine_learning_engineer": ["python", "machine_learning", "tensorflow", "docker", "mlops", "statistics"]
        }
        
        return role_skills.get(target_role.lower().replace(" ", "_"), [])

    async def _calculate_skill_priority(self, skill: str, gap_size: float, target_role: Optional[str]) -> float:
        """Calculate priority score for a skill gap."""
        priority = gap_size  # Base priority on gap size
        
        # Boost priority for role-critical skills
        if target_role:
            role_critical_skills = await self._get_target_skills_for_role(target_role)
            if skill in role_critical_skills:
                priority *= 1.5
        
        # Boost priority for high-demand skills
        market_demand = await self._get_skill_market_demand(skill)
        priority *= (1 + market_demand)
        
        return min(priority, 1.0)  # Cap at 1.0

    async def _get_skill_market_demand(self, skill: str) -> float:
        """Get market demand score for a skill."""
        # This would typically query job market data
        demand_scores = {
            "python": 0.9,
            "javascript": 0.9,
            "react": 0.8,
            "machine_learning": 0.85,
            "docker": 0.7,
            "kubernetes": 0.75,
            "aws": 0.8,
            "sql": 0.8
        }
        
        return demand_scores.get(skill.lower(), 0.5)

    def _estimate_learning_hours(self, skill: str, gap_size: float) -> int:
        """Estimate learning hours needed to bridge a skill gap."""
        base_hours = {
            "python": 80,
            "javascript": 70,
            "react": 60,
            "machine_learning": 120,
            "docker": 40,
            "kubernetes": 60,
            "sql": 50
        }
        
        skill_hours = base_hours.get(skill.lower(), 60)
        return int(skill_hours * gap_size)

    def _get_skill_difficulty(self, skill: str) -> DifficultyLevel:
        """Get difficulty level for a skill."""
        skill_info = self.skill_taxonomy.get("programming_languages", {}).get(skill.lower(), {})
        difficulty = skill_info.get("difficulty", "intermediate")
        
        return DifficultyLevel(difficulty)

    def _determine_overall_difficulty(self, skill_gaps: List[SkillGap]) -> DifficultyLevel:
        """Determine overall difficulty level for a learning path."""
        if not skill_gaps:
            return DifficultyLevel.BEGINNER
        
        difficulty_counts = defaultdict(int)
        for gap in skill_gaps:
            difficulty_counts[gap.difficulty] += 1
        
        # Return the most common difficulty level
        return max(difficulty_counts.items(), key=lambda x: x[1])[0]

    def _calculate_total_duration_weeks(self, skill_gaps: List[SkillGap], hours_per_week: int) -> int:
        """Calculate total duration in weeks."""
        total_hours = sum(gap.estimated_learning_hours for gap in skill_gaps)
        return max(1, int(total_hours / hours_per_week))

    def _group_skills_by_level(self, skills: List[str]) -> Dict[str, List[str]]:
        """Group skills by difficulty level for milestone creation."""
        levels = {
            "beginner": [],
            "intermediate": [],
            "advanced": []
        }
        
        for skill in skills:
            difficulty = self._get_skill_difficulty(skill)
            levels[difficulty.value].append(skill)
        
        return levels

    def _calculate_path_confidence(self, path: LearningPath, request: LearningPathRequest) -> float:
        """Calculate confidence score for a learning path."""
        factors = []
        
        # Resource availability factor
        if path.resources:
            avg_quality = sum(r.quality_score or 0.5 for r in path.resources) / len(path.resources)
            factors.append(avg_quality)
        
        # Skill coverage factor
        target_skills_covered = len(set(path.target_skills) & set(request.target_skills))
        if request.target_skills:
            coverage = target_skills_covered / len(request.target_skills)
            factors.append(coverage)
        
        # Time commitment feasibility
        if request.time_commitment_hours_per_week > 0:
            weekly_load = path.estimated_duration_hours / path.estimated_duration_weeks
            feasibility = min(1.0, request.time_commitment_hours_per_week / weekly_load)
            factors.append(feasibility)
        
        return sum(factors) / len(factors) if factors else 0.7

    async def _generate_project_focused_path(self, request: LearningPathRequest, skill_gaps: List[SkillGap]) -> LearningPath:
        """Generate a project-focused learning path."""
        skills = [gap.skill_name for gap in skill_gaps[:6]]
        
        path = LearningPath(
            title="Project-Based Learning Path",
            description="Learn through hands-on projects and practical applications",
            target_role=request.target_role,
            target_skills=skills,
            skill_gaps=skill_gaps,
            difficulty_level=DifficultyLevel.INTERMEDIATE,
            estimated_duration_weeks=self._calculate_total_duration_weeks(skill_gaps, request.time_commitment_hours_per_week),
            estimated_duration_hours=int(sum(gap.estimated_learning_hours for gap in skill_gaps) * 1.2)  # 20% more for projects
        )
        
        # Focus on resources with hands-on projects
        all_resources = await self._gather_learning_resources(skills, request)
        project_resources = [r for r in all_resources if r.hands_on_projects]
        path.resources = project_resources[:10]  # Top 10 project-based resources
        
        path.milestones = await self._create_project_milestones(skills)
        path.confidence_score = 0.8
        
        return path

    async def _generate_certification_path(self, request: LearningPathRequest, skill_gaps: List[SkillGap]) -> LearningPath:
        """Generate a certification-focused learning path."""
        skills = [gap.skill_name for gap in skill_gaps[:5]]
        
        path = LearningPath(
            title="Certification-Focused Path",
            description="Structured learning path leading to industry certifications",
            target_role=request.target_role,
            target_skills=skills,
            skill_gaps=skill_gaps,
            difficulty_level=DifficultyLevel.INTERMEDIATE,
            estimated_duration_weeks=self._calculate_total_duration_weeks(skill_gaps, request.time_commitment_hours_per_week),
            estimated_duration_hours=int(sum(gap.estimated_learning_hours for gap in skill_gaps))
        )
        
        # Focus on resources with certifications
        all_resources = await self._gather_learning_resources(skills, request)
        cert_resources = [r for r in all_resources if r.certificate_available]
        path.resources = cert_resources[:8]  # Top 8 certification resources
        
        path.milestones = await self._create_certification_milestones(skills)
        path.confidence_score = 0.85
        
        return path

    async def _generate_free_resources_path(self, request: LearningPathRequest, skill_gaps: List[SkillGap]) -> LearningPath:
        """Generate a path using only free resources."""
        skills = [gap.skill_name for gap in skill_gaps[:8]]
        
        # Override request to include only free resources
        free_request = request.copy()
        free_request.include_free_only = True
        
        path = LearningPath(
            title="Free Learning Path",
            description="Complete learning path using only free resources",
            target_role=request.target_role,
            target_skills=skills,
            skill_gaps=skill_gaps,
            difficulty_level=DifficultyLevel.BEGINNER,
            estimated_duration_weeks=int(self._calculate_total_duration_weeks(skill_gaps, request.time_commitment_hours_per_week) * 1.3),  # 30% longer
            estimated_duration_hours=int(sum(gap.estimated_learning_hours for gap in skill_gaps) * 1.3)
        )
        
        path.resources = await self._gather_learning_resources(skills, free_request)
        path.milestones = await self._create_milestones(skills, skill_gaps)
        path.confidence_score = 0.75
        
        return path

    async def _create_project_milestones(self, skills: List[str]) -> List[Milestone]:
        """Create project-focused milestones."""
        milestones = []
        
        for i, skill in enumerate(skills[:4]):  # Max 4 project milestones
            milestone = Milestone(
                title=f"Build {skill.title()} Project",
                description=f"Complete a hands-on project demonstrating {skill} skills",
                order=i,
                skills_to_acquire=[skill],
                estimated_duration_hours=40,
                completion_criteria=[
                    f"Complete project using {skill}",
                    "Deploy project to production",
                    "Document project and learnings"
                ],
                projects=[f"{skill}_project_{i+1}"]
            )
            milestones.append(milestone)
        
        return milestones

    async def _create_certification_milestones(self, skills: List[str]) -> List[Milestone]:
        """Create certification-focused milestones."""
        milestones = []
        
        # Group skills into certification tracks
        cert_tracks = self._group_skills_for_certifications(skills)
        
        for i, (track_name, track_skills) in enumerate(cert_tracks.items()):
            milestone = Milestone(
                title=f"{track_name} Certification",
                description=f"Complete certification in {track_name}",
                order=i,
                skills_to_acquire=track_skills,
                estimated_duration_hours=80,
                completion_criteria=[
                    f"Complete all courses in {track_name} track",
                    "Pass certification exam",
                    "Earn industry-recognized certificate"
                ]
            )
            milestones.append(milestone)
        
        return milestones

    def _group_skills_for_certifications(self, skills: List[str]) -> Dict[str, List[str]]:
        """Group skills into certification tracks."""
        tracks = {}
        
        # Define certification tracks
        cert_mappings = {
            "AWS Cloud": ["aws", "cloud_computing", "devops"],
            "Data Science": ["python", "machine_learning", "data_science", "statistics"],
            "Web Development": ["javascript", "react", "html", "css", "node.js"],
            "DevOps": ["docker", "kubernetes", "ci_cd", "linux"]
        }
        
        for track, track_skills in cert_mappings.items():
            matching_skills = [s for s in skills if s.lower() in [ts.lower() for ts in track_skills]]
            if matching_skills:
                tracks[track] = matching_skills
        
        return tracks

    def _map_skill_to_edx_category(self, skill: str) -> str:
        """Map a skill to edX course category."""
        mappings = {
            "python": "computer_science",
            "javascript": "computer_science",
            "machine_learning": "artificial_intelligence",
            "data_science": "artificial_intelligence",
            "algorithms": "computer_science"
        }
        
        return mappings.get(skill.lower(), "computer_science")

    async def _rank_learning_paths(self, paths: List[LearningPath], request: LearningPathRequest) -> List[LearningPath]:
        """Rank learning paths based on user preferences and path quality."""
        
        def calculate_ranking_score(path: LearningPath) -> float:
            score = 0.0
            
            # Confidence score weight (40%)
            score += (path.confidence_score or 0.7) * 0.4
            
            # Resource quality weight (30%)
            if path.resources:
                avg_quality = sum(r.quality_score or 0.5 for r in path.resources) / len(path.resources)
                score += avg_quality * 0.3
            
            # Time feasibility weight (20%)
            if request.time_commitment_hours_per_week > 0:
                weekly_hours = path.estimated_duration_hours / max(1, path.estimated_duration_weeks)
                feasibility = min(1.0, request.time_commitment_hours_per_week / max(1, weekly_hours))
                score += feasibility * 0.2
            
            # Skill coverage weight (10%)
            if request.target_skills:
                coverage = len(set(path.target_skills) & set(request.target_skills)) / len(request.target_skills)
                score += coverage * 0.1
            
            return score
        
        # Calculate scores and sort
        for path in paths:
            path.confidence_score = calculate_ranking_score(path)
        
        return sorted(paths, key=lambda p: p.confidence_score or 0, reverse=True)
    
    async def generate_customized_learning_paths(self, user_id: str, target_skills: List[str],
                                               customization: Dict[str, Any],
                                               db = None) -> List[Dict[str, Any]]:
        """
        Generate customized learning paths based on detailed preferences.
        
        Args:
            user_id: User identifier
            target_skills: Skills to learn
            customization: Customization parameters (time, budget, style, etc.)
            db: Database session
            
        Returns:
            List of customized learning paths
        """
        try:
            logger.info(f"Generating customized learning paths for user {user_id}")
            
            # Extract customization parameters
            difficulty_preference = customization.get('difficulty_preference', 'intermediate')
            time_commitment = customization.get('time_commitment_hours_per_week', 10)
            budget_max = customization.get('budget_max')
            preferred_providers = customization.get('preferred_providers', [])
            learning_style = customization.get('learning_style', 'mixed')
            include_projects = customization.get('include_projects', True)
            
            learning_paths = []
            
            for skill in target_skills:
                # Generate learning path for each skill
                path = await self._generate_skill_learning_path(
                    skill=skill,
                    difficulty_preference=difficulty_preference,
                    time_commitment=time_commitment,
                    budget_max=budget_max,
                    preferred_providers=preferred_providers,
                    learning_style=learning_style,
                    include_projects=include_projects
                )
                
                learning_paths.append(path)
            
            return learning_paths
            
        except Exception as e:
            logger.error(f"Error generating customized learning paths: {e}")
            raise ServiceException(f"Failed to generate customized learning paths: {str(e)}")
    
    async def _generate_skill_learning_path(self, skill: str, difficulty_preference: str,
                                          time_commitment: int, budget_max: Optional[float],
                                          preferred_providers: List[str], learning_style: str,
                                          include_projects: bool) -> Dict[str, Any]:
        """Generate learning path for a specific skill with customization."""
        try:
            # Get skill information from taxonomy
            skill_info = self._get_skill_info(skill)
            
            # Generate learning resources based on preferences
            resources = await self._get_customized_resources(
                skill=skill,
                difficulty_preference=difficulty_preference,
                budget_max=budget_max,
                preferred_providers=preferred_providers,
                learning_style=learning_style
            )
            
            # Generate milestones
            milestones = self._generate_skill_milestones(skill, difficulty_preference)
            
            # Calculate timeline based on time commitment
            total_hours = sum(resource.get('duration_hours', 0) for resource in resources)
            estimated_weeks = max(1, total_hours // time_commitment)
            
            # Add projects if requested
            projects = []
            if include_projects:
                projects = await self._get_skill_projects(skill, difficulty_preference)
            
            # Calculate estimated cost
            estimated_cost = sum(
                resource.get('cost', 0) for resource in resources 
                if resource.get('cost', 0) > 0
            )
            
            return {
                'skill': skill,
                'title': f"Master {skill.title()}",
                'difficulty_level': difficulty_preference,
                'estimated_duration_weeks': estimated_weeks,
                'estimated_cost': estimated_cost,
                'resources': resources,
                'milestones': milestones,
                'projects': projects,
                'customization_applied': {
                    'difficulty_preference': difficulty_preference,
                    'time_commitment_hours_per_week': time_commitment,
                    'budget_max': budget_max,
                    'preferred_providers': preferred_providers,
                    'learning_style': learning_style,
                    'include_projects': include_projects
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating skill learning path for {skill}: {e}")
            raise ServiceException(f"Failed to generate learning path for {skill}: {str(e)}")
    
    async def _get_customized_resources(self, skill: str, difficulty_preference: str,
                                      budget_max: Optional[float], preferred_providers: List[str],
                                      learning_style: str) -> List[Dict[str, Any]]:
        """Get customized learning resources based on preferences."""
        try:
            # Base resources for the skill
            base_resources = self._get_base_skill_resources(skill, difficulty_preference)
            
            # Filter by budget
            if budget_max is not None:
                base_resources = [
                    resource for resource in base_resources
                    if resource.get('cost', 0) <= budget_max
                ]
            
            # Filter by preferred providers
            if preferred_providers:
                filtered_resources = []
                for resource in base_resources:
                    if resource.get('provider', '').lower() in [p.lower() for p in preferred_providers]:
                        filtered_resources.append(resource)
                # If no resources match preferred providers, include all
                if filtered_resources:
                    base_resources = filtered_resources
            
            # Customize based on learning style
            if learning_style == 'visual':
                # Prioritize video courses and interactive content
                base_resources.sort(key=lambda x: (
                    x.get('type') in ['course', 'tutorial'] and 'video' in x.get('title', '').lower(),
                    x.get('rating', 0)
                ), reverse=True)
            elif learning_style == 'hands_on':
                # Prioritize projects and practical exercises
                base_resources.sort(key=lambda x: (
                    x.get('type') in ['project', 'tutorial'],
                    x.get('rating', 0)
                ), reverse=True)
            elif learning_style == 'reading':
                # Prioritize books and documentation
                base_resources.sort(key=lambda x: (
                    x.get('type') in ['book', 'documentation'],
                    x.get('rating', 0)
                ), reverse=True)
            else:  # mixed
                # Balance different types
                base_resources.sort(key=lambda x: x.get('rating', 0), reverse=True)
            
            return base_resources[:10]  # Limit to top 10 resources
            
        except Exception as e:
            logger.error(f"Error getting customized resources: {e}")
            return []
    
    def _get_base_skill_resources(self, skill: str, difficulty: str) -> List[Dict[str, Any]]:
        """Get base learning resources for a skill."""
        # This would typically query a database of learning resources
        # For now, return mock data based on skill and difficulty
        
        skill_lower = skill.lower()
        
        if 'python' in skill_lower:
            return [
                {
                    'title': f'Complete Python {difficulty.title()} Course',
                    'type': 'course',
                    'provider': 'Coursera',
                    'url': 'https://coursera.org/python',
                    'rating': 4.7,
                    'duration_hours': 40,
                    'cost': 49.99 if difficulty != 'beginner' else 0,
                    'prerequisites': [] if difficulty == 'beginner' else ['programming_basics']
                },
                {
                    'title': 'Python Official Documentation',
                    'type': 'documentation',
                    'provider': 'Python.org',
                    'url': 'https://docs.python.org',
                    'rating': 4.8,
                    'duration_hours': 20,
                    'cost': 0,
                    'prerequisites': []
                },
                {
                    'title': f'Python {difficulty.title()} Projects',
                    'type': 'project',
                    'provider': 'GitHub',
                    'url': 'https://github.com/python-projects',
                    'rating': 4.5,
                    'duration_hours': 30,
                    'cost': 0,
                    'prerequisites': ['python_basics']
                }
            ]
        elif 'javascript' in skill_lower:
            return [
                {
                    'title': f'JavaScript {difficulty.title()} Bootcamp',
                    'type': 'course',
                    'provider': 'Udemy',
                    'url': 'https://udemy.com/javascript',
                    'rating': 4.6,
                    'duration_hours': 35,
                    'cost': 89.99,
                    'prerequisites': [] if difficulty == 'beginner' else ['html', 'css']
                },
                {
                    'title': 'MDN JavaScript Guide',
                    'type': 'documentation',
                    'provider': 'Mozilla',
                    'url': 'https://developer.mozilla.org/en-US/docs/Web/JavaScript',
                    'rating': 4.9,
                    'duration_hours': 15,
                    'cost': 0,
                    'prerequisites': []
                }
            ]
        elif 'machine learning' in skill_lower or 'ml' in skill_lower:
            return [
                {
                    'title': f'Machine Learning {difficulty.title()} Specialization',
                    'type': 'course',
                    'provider': 'Coursera',
                    'url': 'https://coursera.org/ml',
                    'rating': 4.8,
                    'duration_hours': 60,
                    'cost': 79.99,
                    'prerequisites': ['python', 'statistics', 'linear_algebra']
                },
                {
                    'title': 'Scikit-learn Documentation',
                    'type': 'documentation',
                    'provider': 'Scikit-learn',
                    'url': 'https://scikit-learn.org/stable/',
                    'rating': 4.7,
                    'duration_hours': 25,
                    'cost': 0,
                    'prerequisites': ['python']
                }
            ]
        else:
            # Generic resources for unknown skills
            return [
                {
                    'title': f'Complete {skill.title()} Course',
                    'type': 'course',
                    'provider': 'Online Learning Platform',
                    'url': f'https://example.com/{skill.lower()}',
                    'rating': 4.5,
                    'duration_hours': 30,
                    'cost': 49.99,
                    'prerequisites': []
                },
                {
                    'title': f'{skill.title()} Documentation',
                    'type': 'documentation',
                    'provider': 'Official',
                    'url': f'https://docs.{skill.lower()}.org',
                    'rating': 4.6,
                    'duration_hours': 15,
                    'cost': 0,
                    'prerequisites': []
                }
            ]
    
    def _generate_skill_milestones(self, skill: str, difficulty: str) -> List[Dict[str, Any]]:
        """Generate learning milestones for a skill."""
        skill_lower = skill.lower()
        
        if difficulty == 'beginner':
            return [
                {
                    'title': f'Learn {skill.title()} Basics',
                    'description': f'Understand fundamental concepts of {skill}',
                    'estimated_weeks': 2,
                    'completion_criteria': [
                        'Complete introductory course',
                        'Understand basic syntax and concepts',
                        'Complete practice exercises'
                    ]
                },
                {
                    'title': f'Build First {skill.title()} Project',
                    'description': f'Create a simple project using {skill}',
                    'estimated_weeks': 2,
                    'completion_criteria': [
                        'Design and implement a basic project',
                        'Document the code',
                        'Test the functionality'
                    ]
                }
            ]
        elif difficulty == 'intermediate':
            return [
                {
                    'title': f'Master {skill.title()} Intermediate Concepts',
                    'description': f'Learn advanced features and best practices',
                    'estimated_weeks': 3,
                    'completion_criteria': [
                        'Complete intermediate course modules',
                        'Understand design patterns',
                        'Practice code optimization'
                    ]
                },
                {
                    'title': f'Build Complex {skill.title()} Application',
                    'description': f'Create a full-featured application',
                    'estimated_weeks': 4,
                    'completion_criteria': [
                        'Implement complex features',
                        'Add error handling and testing',
                        'Deploy to production'
                    ]
                }
            ]
        else:  # advanced
            return [
                {
                    'title': f'Master {skill.title()} Architecture',
                    'description': f'Learn system design and architecture patterns',
                    'estimated_weeks': 4,
                    'completion_criteria': [
                        'Study architectural patterns',
                        'Design scalable systems',
                        'Implement performance optimizations'
                    ]
                },
                {
                    'title': f'Contribute to {skill.title()} Open Source',
                    'description': f'Contribute to open source projects',
                    'estimated_weeks': 6,
                    'completion_criteria': [
                        'Find suitable open source projects',
                        'Submit meaningful contributions',
                        'Collaborate with maintainers'
                    ]
                }
            ]
    
    async def _get_skill_projects(self, skill: str, difficulty: str) -> List[Dict[str, Any]]:
        """Get project recommendations for a skill."""
        # This would typically query GitHub API or project database
        # For now, return mock project data
        
        skill_lower = skill.lower()
        
        if 'python' in skill_lower:
            if difficulty == 'beginner':
                return [
                    {
                        'title': 'Calculator App',
                        'description': 'Build a simple calculator with GUI',
                        'difficulty': 'beginner',
                        'estimated_hours': 8,
                        'skills_practiced': ['python', 'tkinter', 'basic_programming'],
                        'github_url': 'https://github.com/example/python-calculator'
                    },
                    {
                        'title': 'To-Do List Manager',
                        'description': 'Create a command-line to-do list application',
                        'difficulty': 'beginner',
                        'estimated_hours': 12,
                        'skills_practiced': ['python', 'file_handling', 'data_structures'],
                        'github_url': 'https://github.com/example/todo-manager'
                    }
                ]
            elif difficulty == 'intermediate':
                return [
                    {
                        'title': 'Web Scraper with API',
                        'description': 'Build a web scraper that exposes data via REST API',
                        'difficulty': 'intermediate',
                        'estimated_hours': 25,
                        'skills_practiced': ['python', 'web_scraping', 'flask', 'api_design'],
                        'github_url': 'https://github.com/example/web-scraper-api'
                    },
                    {
                        'title': 'Data Analysis Dashboard',
                        'description': 'Create an interactive dashboard for data visualization',
                        'difficulty': 'intermediate',
                        'estimated_hours': 30,
                        'skills_practiced': ['python', 'pandas', 'plotly', 'streamlit'],
                        'github_url': 'https://github.com/example/data-dashboard'
                    }
                ]
            else:  # advanced
                return [
                    {
                        'title': 'Microservices Architecture',
                        'description': 'Build a distributed system with multiple microservices',
                        'difficulty': 'advanced',
                        'estimated_hours': 60,
                        'skills_practiced': ['python', 'microservices', 'docker', 'kubernetes', 'api_gateway'],
                        'github_url': 'https://github.com/example/microservices-system'
                    }
                ]
        
        # Default projects for other skills
        return [
            {
                'title': f'{skill.title()} Practice Project',
                'description': f'Hands-on project to practice {skill} skills',
                'difficulty': difficulty,
                'estimated_hours': 20,
                'skills_practiced': [skill.lower(), 'problem_solving'],
                'github_url': f'https://github.com/example/{skill.lower()}-project'
            }
        ]
    
    def _get_skill_info(self, skill: str) -> Dict[str, Any]:
        """Get skill information from taxonomy."""
        skill_lower = skill.lower()
        
        # Check if skill exists in taxonomy
        for category, skills in self.skill_taxonomy.items():
            if skill_lower in skills:
                return skills[skill_lower]
        
        # Return default info for unknown skills
        return {
            'difficulty': 'intermediate',
            'category': 'general',
            'prerequisites': []
        }
    
    async def generate_learning_paths_for_role(self, user_id: str, target_role: str,
                                             db = None) -> List[Dict[str, Any]]:
        """Generate learning paths specifically for a target role."""
        try:
            # Get required skills for the target role
            role_skills = self._get_role_required_skills(target_role)
            
            # Generate learning paths for each required skill
            learning_paths = []
            for skill in role_skills:
                path = await self._generate_skill_learning_path(
                    skill=skill,
                    difficulty_preference='intermediate',
                    time_commitment=10,
                    budget_max=None,
                    preferred_providers=[],
                    learning_style='mixed',
                    include_projects=True
                )
                learning_paths.append(path)
            
            return learning_paths
            
        except Exception as e:
            logger.error(f"Error generating learning paths for role {target_role}: {e}")
            raise ServiceException(f"Failed to generate learning paths for role: {str(e)}")
    
    def _get_role_required_skills(self, target_role: str) -> List[str]:
        """Get required skills for a target role."""
        role_lower = target_role.lower()
        
        role_skills_map = {
            'software engineer': ['python', 'javascript', 'sql', 'git', 'algorithms'],
            'data scientist': ['python', 'machine learning', 'statistics', 'sql', 'pandas'],
            'frontend developer': ['javascript', 'react', 'html', 'css', 'typescript'],
            'backend developer': ['python', 'sql', 'api design', 'docker', 'microservices'],
            'devops engineer': ['docker', 'kubernetes', 'aws', 'linux', 'terraform'],
            'machine learning engineer': ['python', 'machine learning', 'tensorflow', 'docker', 'mlops']
        }
        
        for role, skills in role_skills_map.items():
            if role in role_lower:
                return skills
        
        # Default skills for unknown roles
        return ['communication', 'problem solving', 'teamwork']
    
    async def get_user_learning_paths(self, user_id: str, db = None) -> List[Dict[str, Any]]:
        """Get all learning paths for a user."""
        try:
            # This would typically fetch from database
            # For now, return empty list as placeholder
            return []
            
        except Exception as e:
            logger.error(f"Error getting user learning paths: {e}")
            raise ServiceException(f"Failed to get user learning paths: {str(e)}")