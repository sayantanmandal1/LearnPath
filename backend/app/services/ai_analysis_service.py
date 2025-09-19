"""
AI Analysis Service with Gemini Integration

This service provides comprehensive AI-powered career analysis using Google's Gemini API.
It aggregates profile data from multiple sources and generates intelligent insights.
"""
import os
import json
import asyncio
import logging
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass

import httpx
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.models.profile import UserProfile
from app.models.resume import ResumeData
from app.models.platform_account import PlatformAccount
from app.models.analysis_result import AnalysisResult, AnalysisType
from app.schemas.profile import ProfileResponse
from app.schemas.resume import ParsedResumeData
from app.core.exceptions import ProcessingError, ValidationError

logger = logging.getLogger(__name__)


@dataclass
class CompleteProfileData:
    """Aggregated profile data for AI analysis"""
    user_id: str
    basic_profile: Optional[ProfileResponse]
    resume_data: Optional[ParsedResumeData]
    platform_data: Dict[str, Any]
    career_preferences: Dict[str, Any]
    
    def to_analysis_context(self) -> str:
        """Convert profile data to text context for AI analysis"""
        context_parts = []
        
        # Basic profile information
        if self.basic_profile:
            context_parts.append(f"Current Role: {self.basic_profile.current_role or 'Not specified'}")
            context_parts.append(f"Experience: {self.basic_profile.experience_years or 0} years")
            context_parts.append(f"Location: {self.basic_profile.location or 'Not specified'}")
            context_parts.append(f"Dream Job: {self.basic_profile.dream_job or 'Not specified'}")
            context_parts.append(f"Industry: {self.basic_profile.industry or 'Not specified'}")
            context_parts.append(f"Desired Role: {self.basic_profile.desired_role or 'Not specified'}")
            context_parts.append(f"Career Goals: {self.basic_profile.career_goals or 'Not specified'}")
            
        # Resume data
        if self.resume_data:
            if self.resume_data.summary:
                context_parts.append(f"Professional Summary: {self.resume_data.summary}")
            
            if self.resume_data.work_experience:
                context_parts.append("Work Experience:")
                for exp in self.resume_data.work_experience:
                    exp_text = f"- {exp.position} at {exp.company}"
                    if exp.technologies:
                        exp_text += f" (Technologies: {', '.join(exp.technologies)})"
                    context_parts.append(exp_text)
            
            if self.resume_data.skills:
                context_parts.append("Skills:")
                for skill_cat in self.resume_data.skills:
                    context_parts.append(f"- {skill_cat.category}: {', '.join(skill_cat.skills)}")
        
        # Platform data
        if self.platform_data:
            for platform, data in self.platform_data.items():
                if platform == "github" and data:
                    context_parts.append(f"GitHub: {data.get('public_repos', 0)} repositories, {data.get('followers', 0)} followers")
                elif platform == "leetcode" and data:
                    context_parts.append(f"LeetCode: {data.get('problems_solved', 0)} problems solved")
                elif platform == "linkedin" and data:
                    context_parts.append(f"LinkedIn: {data.get('connections', 0)} connections")
        
        return "\n".join(context_parts)


@dataclass
class SkillAssessment:
    """AI-generated skill assessment"""
    technical_skills: Dict[str, float]  # skill -> proficiency (0-1)
    soft_skills: Dict[str, float]
    skill_strengths: List[str]
    skill_gaps: List[str]
    improvement_areas: List[str]
    market_relevance_score: float
    confidence_score: float


@dataclass
class CareerRecommendation:
    """AI-generated career recommendation"""
    recommended_role: str
    match_score: float
    reasoning: str
    required_skills: List[str]
    skill_gaps: List[str]
    preparation_timeline: str
    salary_range: Optional[str]
    market_demand: str


@dataclass
class LearningPath:
    """AI-generated learning path"""
    title: str
    description: str
    target_skills: List[str]
    learning_modules: List[Dict[str, Any]]
    estimated_duration: str
    difficulty_level: str
    resources: List[Dict[str, str]]


@dataclass
class ProjectSuggestion:
    """AI-generated project suggestion"""
    title: str
    description: str
    technologies: List[str]
    difficulty_level: str
    estimated_duration: str
    learning_outcomes: List[str]
    portfolio_value: str


@dataclass
class CareerAnalysis:
    """Complete AI analysis result"""
    user_id: str
    skill_assessment: SkillAssessment
    career_recommendations: List[CareerRecommendation]
    learning_paths: List[LearningPath]
    project_suggestions: List[ProjectSuggestion]
    market_insights: Dict[str, Any]
    analysis_timestamp: datetime
    gemini_request_id: Optional[str]


class GeminiAPIClient:
    """Client for Google Gemini API integration"""
    
    def __init__(self):
        # Try to get API key from settings first, then fallback to os.getenv
        try:
            from app.core.config import settings
            self.api_key = settings.GEMINI_API_KEY or os.getenv("GEMINI_API_KEY")
        except ImportError:
            self.api_key = os.getenv("GEMINI_API_KEY")
        
        self.base_url = "https://generativelanguage.googleapis.com/v1beta"
        self.model = "gemini-1.5-flash"
        self.timeout = 60.0
        
        if not self.api_key:
            logger.warning("Gemini API key not configured")
    
    async def generate_content(self, prompt: str, temperature: float = 0.3) -> Tuple[str, Optional[str]]:
        """
        Generate content using Gemini API with enhanced error handling
        
        Args:
            prompt: Input prompt for generation
            temperature: Generation temperature (0.0-1.0)
            
        Returns:
            Tuple[str, Optional[str]]: Generated content and request ID
        """
        from app.core.exceptions import MLModelError
        from app.core.error_handling_decorators import with_gemini_error_handling
        
        if not self.api_key:
            raise MLModelError(
                model_name="Gemini API",
                detail="API key not configured",
                fallback_available=True
            )
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/models/{self.model}:generateContent?key={self.api_key}",
                    json={
                        "contents": [{
                            "parts": [{"text": prompt}]
                        }],
                        "generationConfig": {
                            "temperature": temperature,
                            "maxOutputTokens": 4096,
                            "topP": 0.8,
                            "topK": 40
                        }
                    }
                )
                
                # Enhanced error handling for different status codes
                if response.status_code == 401:
                    raise MLModelError(
                        model_name="Gemini API",
                        detail="Authentication failed - invalid API key",
                        fallback_available=True
                    )
                elif response.status_code == 403:
                    raise MLModelError(
                        model_name="Gemini API", 
                        detail="Access forbidden - check API permissions",
                        fallback_available=True
                    )
                elif response.status_code == 429:
                    raise MLModelError(
                        model_name="Gemini API",
                        detail="Rate limit exceeded - too many requests",
                        fallback_available=True
                    )
                elif response.status_code == 503:
                    raise MLModelError(
                        model_name="Gemini API",
                        detail="Service temporarily unavailable",
                        fallback_available=True
                    )
                elif response.status_code != 200:
                    logger.error(f"Gemini API error: {response.status_code} - {response.text}")
                    raise MLModelError(
                        model_name="Gemini API",
                        detail=f"API request failed with status {response.status_code}",
                        fallback_available=True
                    )
                
                result = response.json()
                
                # Check for API response errors
                if "error" in result:
                    error_detail = result["error"].get("message", "Unknown API error")
                    raise MLModelError(
                        model_name="Gemini API",
                        detail=f"API returned error: {error_detail}",
                        fallback_available=True
                    )
                
                if "candidates" not in result or not result["candidates"]:
                    raise MLModelError(
                        model_name="Gemini API",
                        detail="No content generated by API",
                        fallback_available=True
                    )
                
                # Check for content filtering
                candidate = result["candidates"][0]
                if candidate.get("finishReason") == "SAFETY":
                    raise MLModelError(
                        model_name="Gemini API",
                        detail="Content was filtered for safety reasons",
                        fallback_available=True
                    )
                
                content = candidate["content"]["parts"][0]["text"]
                request_id = response.headers.get("x-request-id")
                
                return content, request_id
                
        except httpx.TimeoutException:
            raise MLModelError(
                model_name="Gemini API",
                detail="Request timed out - service may be overloaded",
                fallback_available=True
            )
        except httpx.ConnectError:
            raise MLModelError(
                model_name="Gemini API", 
                detail="Connection failed - network or service issue",
                fallback_available=True
            )
        except MLModelError:
            # Re-raise MLModelError as-is
            raise
        except Exception as e:
            logger.error(f"Gemini API request failed: {str(e)}")
            raise MLModelError(
                model_name="Gemini API",
                detail=f"Unexpected error: {str(e)}",
                fallback_available=True,
                original_error=e
            )
    
    async def analyze_with_structured_output(self, prompt: str, expected_schema: str) -> Tuple[Dict[str, Any], Optional[str]]:
        """
        Generate structured JSON output using Gemini API
        
        Args:
            prompt: Analysis prompt
            expected_schema: Expected JSON schema description
            
        Returns:
            Tuple[Dict[str, Any], Optional[str]]: Parsed JSON result and request ID
        """
        structured_prompt = f"""
        {prompt}
        
        Please respond with valid JSON only, following this schema:
        {expected_schema}
        
        Ensure the response is valid JSON that can be parsed directly.
        """
        
        content, request_id = await self.generate_content(structured_prompt, temperature=0.1)
        
        try:
            # Clean the response
            clean_content = content.strip()
            if clean_content.startswith("```json"):
                clean_content = clean_content[7:]
            if clean_content.endswith("```"):
                clean_content = clean_content[:-3]
            
            parsed_json = json.loads(clean_content)
            return parsed_json, request_id
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Gemini JSON response: {str(e)}")
            logger.error(f"Raw response: {content}")
            raise ProcessingError(f"Invalid JSON response from Gemini API: {str(e)}")


class AIAnalysisService:
    """Main AI analysis service with Gemini integration"""
    
    def __init__(self):
        self.gemini_client = GeminiAPIClient()
        self.fallback_enabled = True
    
    async def analyze_complete_profile(self, user_id: str, db: AsyncSession) -> CareerAnalysis:
        """
        Perform comprehensive AI analysis of user profile
        
        Args:
            user_id: User ID to analyze
            db: Database session
            
        Returns:
            CareerAnalysis: Complete analysis results
        """
        try:
            # Aggregate profile data
            profile_data = await self._aggregate_profile_data(user_id, db)
            
            # Generate AI analysis
            analysis = await self._generate_comprehensive_analysis(profile_data)
            
            # Store analysis results
            await self._store_analysis_results(analysis, db)
            
            logger.info(f"AI analysis completed for user {user_id}")
            return analysis
            
        except Exception as e:
            logger.error(f"AI analysis failed for user {user_id}: {str(e)}")
            if self.fallback_enabled:
                return await self._fallback_analysis(user_id, db)
            raise ProcessingError(f"AI analysis failed: {str(e)}")
    
    async def _aggregate_profile_data(self, user_id: str, db: AsyncSession) -> CompleteProfileData:
        """Aggregate all available profile data for analysis"""
        # Get basic profile
        profile_result = await db.execute(select(UserProfile).where(UserProfile.user_id == user_id))
        profile = profile_result.scalar_one_or_none()
        
        # Get resume data
        resume_result = await db.execute(
            select(ResumeData)
            .where(ResumeData.user_id == user_id)
            .order_by(ResumeData.created_at.desc())
        )
        latest_resume = resume_result.first()
        
        # Get platform data
        platform_result = await db.execute(
            select(PlatformAccount).where(PlatformAccount.user_id == user_id)
        )
        platform_accounts = platform_result.scalars().all()
        
        # Aggregate platform data
        platform_data = {}
        for account in platform_accounts:
            platform_data[account.platform] = account.data
        
        # Convert to response format
        profile_response = None
        if profile:
            profile_response = ProfileResponse.from_orm(profile)
        
        # Parse resume data
        resume_data = None
        if latest_resume and latest_resume.parsed_sections:
            try:
                resume_data = ParsedResumeData(**latest_resume.parsed_sections)
            except Exception as e:
                logger.warning(f"Failed to parse resume data: {str(e)}")
        
        # Extract career preferences
        career_preferences = {}
        if profile:
            career_preferences = {
                "desired_role": profile.desired_role,
                "career_goals": profile.career_goals,
                "timeframe": profile.timeframe,
                "salary_expectation": profile.salary_expectation,
                "work_type": profile.work_type,
                "company_size": profile.company_size,
                "work_culture": profile.work_culture
            }
        
        return CompleteProfileData(
            user_id=user_id,
            basic_profile=profile_response,
            resume_data=resume_data,
            platform_data=platform_data,
            career_preferences=career_preferences
        )
    
    async def _generate_comprehensive_analysis(self, profile_data: CompleteProfileData) -> CareerAnalysis:
        """Generate comprehensive AI analysis using Gemini"""
        context = profile_data.to_analysis_context()
        
        # Generate skill assessment
        skill_assessment = await self._generate_skill_assessment(context)
        
        # Generate career recommendations
        career_recommendations = await self._generate_career_recommendations(context, skill_assessment)
        
        # Generate learning paths
        learning_paths = await self._generate_learning_paths(context, skill_assessment)
        
        # Generate project suggestions
        project_suggestions = await self._generate_project_suggestions(context, skill_assessment)
        
        # Generate market insights
        market_insights = await self._generate_market_insights(context)
        
        return CareerAnalysis(
            user_id=profile_data.user_id,
            skill_assessment=skill_assessment,
            career_recommendations=career_recommendations,
            learning_paths=learning_paths,
            project_suggestions=project_suggestions,
            market_insights=market_insights,
            analysis_timestamp=datetime.utcnow(),
            gemini_request_id=None  # Will be set by individual API calls
        )
    
    async def _generate_skill_assessment(self, context: str) -> SkillAssessment:
        """Generate AI-powered skill assessment"""
        prompt = f"""
        Analyze the following professional profile and provide a comprehensive skill assessment:
        
        {context}
        
        Evaluate:
        1. Technical skills with proficiency levels (0.0-1.0)
        2. Soft skills with proficiency levels (0.0-1.0)
        3. Top 5 skill strengths
        4. Top 5 skill gaps for career advancement
        5. Areas needing improvement
        6. Market relevance score (0.0-1.0)
        7. Overall confidence in assessment (0.0-1.0)
        
        Focus on skills relevant to the Indian tech market and current industry trends.
        """
        
        schema = """
        {
            "technical_skills": {"skill_name": proficiency_score},
            "soft_skills": {"skill_name": proficiency_score},
            "skill_strengths": ["strength1", "strength2", ...],
            "skill_gaps": ["gap1", "gap2", ...],
            "improvement_areas": ["area1", "area2", ...],
            "market_relevance_score": 0.0-1.0,
            "confidence_score": 0.0-1.0
        }
        """
        
        try:
            result, request_id = await self.gemini_client.analyze_with_structured_output(prompt, schema)
            
            return SkillAssessment(
                technical_skills=result.get("technical_skills", {}),
                soft_skills=result.get("soft_skills", {}),
                skill_strengths=result.get("skill_strengths", []),
                skill_gaps=result.get("skill_gaps", []),
                improvement_areas=result.get("improvement_areas", []),
                market_relevance_score=result.get("market_relevance_score", 0.0),
                confidence_score=result.get("confidence_score", 0.0)
            )
            
        except Exception as e:
            logger.error(f"Skill assessment generation failed: {str(e)}")
            return self._fallback_skill_assessment()
    
    async def _generate_career_recommendations(self, context: str, skill_assessment: SkillAssessment) -> List[CareerRecommendation]:
        """Generate AI-powered career recommendations"""
        prompt = f"""
        Based on the following profile and skill assessment, provide career recommendations:
        
        Profile Context:
        {context}
        
        Skill Strengths: {', '.join(skill_assessment.skill_strengths)}
        Skill Gaps: {', '.join(skill_assessment.skill_gaps)}
        
        Provide 3-5 career recommendations with:
        1. Recommended role title
        2. Match score (0.0-1.0) based on current skills
        3. Detailed reasoning for the recommendation
        4. Required skills for the role
        5. Skill gaps to address
        6. Preparation timeline (e.g., "6-12 months")
        7. Expected salary range in Indian market
        8. Market demand assessment
        
        Focus on roles in the Indian tech industry with good growth prospects.
        """
        
        schema = """
        [
            {
                "recommended_role": "Role Title",
                "match_score": 0.0-1.0,
                "reasoning": "Detailed explanation",
                "required_skills": ["skill1", "skill2", ...],
                "skill_gaps": ["gap1", "gap2", ...],
                "preparation_timeline": "X months",
                "salary_range": "X-Y LPA",
                "market_demand": "High/Medium/Low with explanation"
            }
        ]
        """
        
        try:
            result, request_id = await self.gemini_client.analyze_with_structured_output(prompt, schema)
            
            recommendations = []
            for rec in result:
                recommendations.append(CareerRecommendation(
                    recommended_role=rec.get("recommended_role", ""),
                    match_score=rec.get("match_score", 0.0),
                    reasoning=rec.get("reasoning", ""),
                    required_skills=rec.get("required_skills", []),
                    skill_gaps=rec.get("skill_gaps", []),
                    preparation_timeline=rec.get("preparation_timeline", ""),
                    salary_range=rec.get("salary_range"),
                    market_demand=rec.get("market_demand", "")
                ))
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Career recommendations generation failed: {str(e)}")
            return self._fallback_career_recommendations()
    
    async def _generate_learning_paths(self, context: str, skill_assessment: SkillAssessment) -> List[LearningPath]:
        """Generate AI-powered learning paths"""
        prompt = f"""
        Based on the profile and skill gaps, create personalized learning paths:
        
        Profile Context:
        {context}
        
        Skill Gaps to Address: {', '.join(skill_assessment.skill_gaps)}
        Improvement Areas: {', '.join(skill_assessment.improvement_areas)}
        
        Create 2-3 learning paths with:
        1. Path title and description
        2. Target skills to develop
        3. Learning modules with topics and subtopics
        4. Estimated duration
        5. Difficulty level (Beginner/Intermediate/Advanced)
        6. Recommended resources (courses, books, tutorials)
        
        Focus on practical, industry-relevant learning that can be completed while working.
        """
        
        schema = """
        [
            {
                "title": "Learning Path Title",
                "description": "Path description",
                "target_skills": ["skill1", "skill2", ...],
                "learning_modules": [
                    {
                        "module_name": "Module Title",
                        "topics": ["topic1", "topic2", ...],
                        "duration": "X weeks"
                    }
                ],
                "estimated_duration": "X months",
                "difficulty_level": "Beginner/Intermediate/Advanced",
                "resources": [
                    {
                        "type": "course/book/tutorial",
                        "title": "Resource Title",
                        "provider": "Provider Name",
                        "url": "optional_url"
                    }
                ]
            }
        ]
        """
        
        try:
            result, request_id = await self.gemini_client.analyze_with_structured_output(prompt, schema)
            
            learning_paths = []
            for path in result:
                learning_paths.append(LearningPath(
                    title=path.get("title", ""),
                    description=path.get("description", ""),
                    target_skills=path.get("target_skills", []),
                    learning_modules=path.get("learning_modules", []),
                    estimated_duration=path.get("estimated_duration", ""),
                    difficulty_level=path.get("difficulty_level", ""),
                    resources=path.get("resources", [])
                ))
            
            return learning_paths
            
        except Exception as e:
            logger.error(f"Learning paths generation failed: {str(e)}")
            return self._fallback_learning_paths()
    
    async def _generate_project_suggestions(self, context: str, skill_assessment: SkillAssessment) -> List[ProjectSuggestion]:
        """Generate AI-powered project suggestions"""
        prompt = f"""
        Based on the profile and skill assessment, suggest portfolio projects:
        
        Profile Context:
        {context}
        
        Current Technical Skills: {', '.join(skill_assessment.technical_skills.keys())}
        Skills to Develop: {', '.join(skill_assessment.skill_gaps)}
        
        Suggest 3-5 projects with:
        1. Project title and description
        2. Technologies to use
        3. Difficulty level
        4. Estimated duration
        5. Learning outcomes
        6. Portfolio value explanation
        
        Focus on projects that demonstrate skills relevant to target roles and can be showcased to employers.
        """
        
        schema = """
        [
            {
                "title": "Project Title",
                "description": "Detailed project description",
                "technologies": ["tech1", "tech2", ...],
                "difficulty_level": "Beginner/Intermediate/Advanced",
                "estimated_duration": "X weeks",
                "learning_outcomes": ["outcome1", "outcome2", ...],
                "portfolio_value": "Explanation of portfolio value"
            }
        ]
        """
        
        try:
            result, request_id = await self.gemini_client.analyze_with_structured_output(prompt, schema)
            
            project_suggestions = []
            for project in result:
                project_suggestions.append(ProjectSuggestion(
                    title=project.get("title", ""),
                    description=project.get("description", ""),
                    technologies=project.get("technologies", []),
                    difficulty_level=project.get("difficulty_level", ""),
                    estimated_duration=project.get("estimated_duration", ""),
                    learning_outcomes=project.get("learning_outcomes", []),
                    portfolio_value=project.get("portfolio_value", "")
                ))
            
            return project_suggestions
            
        except Exception as e:
            logger.error(f"Project suggestions generation failed: {str(e)}")
            return self._fallback_project_suggestions()
    
    async def _generate_market_insights(self, context: str) -> Dict[str, Any]:
        """Generate AI-powered market insights"""
        prompt = f"""
        Analyze the current Indian tech job market and provide insights relevant to this profile:
        
        Profile Context:
        {context}
        
        Provide market insights including:
        1. Industry trends relevant to the profile
        2. In-demand skills in the Indian market
        3. Salary trends for relevant roles
        4. Growth opportunities in different tech cities
        5. Emerging technologies to watch
        6. Market demand forecast
        
        Focus on actionable insights for career planning in the Indian tech ecosystem.
        """
        
        schema = """
        {
            "industry_trends": ["trend1", "trend2", ...],
            "in_demand_skills": ["skill1", "skill2", ...],
            "salary_trends": {
                "role_name": "salary_range_and_trend"
            },
            "city_opportunities": {
                "city_name": "opportunity_description"
            },
            "emerging_technologies": ["tech1", "tech2", ...],
            "market_forecast": "Overall market outlook",
            "actionable_insights": ["insight1", "insight2", ...]
        }
        """
        
        try:
            result, request_id = await self.gemini_client.analyze_with_structured_output(prompt, schema)
            return result
            
        except Exception as e:
            logger.error(f"Market insights generation failed: {str(e)}")
            return self._fallback_market_insights()
    
    async def _store_analysis_results(self, analysis: CareerAnalysis, db: AsyncSession) -> None:
        """Store analysis results in database"""
        try:
            # Store skill assessment
            skill_result = AnalysisResult(
                user_id=analysis.user_id,
                analysis_type=AnalysisType.SKILL_ASSESSMENT,
                result_data={
                    "technical_skills": analysis.skill_assessment.technical_skills,
                    "soft_skills": analysis.skill_assessment.soft_skills,
                    "skill_strengths": analysis.skill_assessment.skill_strengths,
                    "skill_gaps": analysis.skill_assessment.skill_gaps,
                    "improvement_areas": analysis.skill_assessment.improvement_areas,
                    "market_relevance_score": analysis.skill_assessment.market_relevance_score,
                    "confidence_score": analysis.skill_assessment.confidence_score
                },
                confidence_score=analysis.skill_assessment.confidence_score,
                gemini_request_id=analysis.gemini_request_id
            )
            db.add(skill_result)
            
            # Store career recommendations
            career_result = AnalysisResult(
                user_id=analysis.user_id,
                analysis_type=AnalysisType.CAREER_RECOMMENDATION,
                result_data={
                    "recommendations": [
                        {
                            "recommended_role": rec.recommended_role,
                            "match_score": rec.match_score,
                            "reasoning": rec.reasoning,
                            "required_skills": rec.required_skills,
                            "skill_gaps": rec.skill_gaps,
                            "preparation_timeline": rec.preparation_timeline,
                            "salary_range": rec.salary_range,
                            "market_demand": rec.market_demand
                        }
                        for rec in analysis.career_recommendations
                    ]
                },
                confidence_score=sum(rec.match_score for rec in analysis.career_recommendations) / len(analysis.career_recommendations) if analysis.career_recommendations else 0.0,
                gemini_request_id=analysis.gemini_request_id
            )
            db.add(career_result)
            
            # Store learning paths
            learning_result = AnalysisResult(
                user_id=analysis.user_id,
                analysis_type=AnalysisType.LEARNING_PATH,
                result_data={
                    "learning_paths": [
                        {
                            "title": path.title,
                            "description": path.description,
                            "target_skills": path.target_skills,
                            "learning_modules": path.learning_modules,
                            "estimated_duration": path.estimated_duration,
                            "difficulty_level": path.difficulty_level,
                            "resources": path.resources
                        }
                        for path in analysis.learning_paths
                    ]
                },
                confidence_score=0.8,  # Default confidence for learning paths
                gemini_request_id=analysis.gemini_request_id
            )
            db.add(learning_result)
            
            # Store project suggestions
            project_result = AnalysisResult(
                user_id=analysis.user_id,
                analysis_type=AnalysisType.PROJECT_SUGGESTION,
                result_data={
                    "project_suggestions": [
                        {
                            "title": proj.title,
                            "description": proj.description,
                            "technologies": proj.technologies,
                            "difficulty_level": proj.difficulty_level,
                            "estimated_duration": proj.estimated_duration,
                            "learning_outcomes": proj.learning_outcomes,
                            "portfolio_value": proj.portfolio_value
                        }
                        for proj in analysis.project_suggestions
                    ]
                },
                confidence_score=0.8,  # Default confidence for project suggestions
                gemini_request_id=analysis.gemini_request_id
            )
            db.add(project_result)
            
            # Store market insights
            market_result = AnalysisResult(
                user_id=analysis.user_id,
                analysis_type=AnalysisType.MARKET_ANALYSIS,
                result_data=analysis.market_insights,
                confidence_score=0.7,  # Default confidence for market insights
                gemini_request_id=analysis.gemini_request_id
            )
            db.add(market_result)
            
            await db.commit()
            logger.info(f"Analysis results stored for user {analysis.user_id}")
            
        except Exception as e:
            logger.error(f"Failed to store analysis results: {str(e)}")
            await db.rollback()
            raise ProcessingError(f"Failed to store analysis results: {str(e)}")
    
    # Fallback methods for when Gemini API is unavailable
    
    def _fallback_skill_assessment(self) -> SkillAssessment:
        """Fallback skill assessment when AI is unavailable"""
        return SkillAssessment(
            technical_skills={"Python": 0.7, "JavaScript": 0.6, "SQL": 0.5},
            soft_skills={"Communication": 0.6, "Problem Solving": 0.7, "Teamwork": 0.6},
            skill_strengths=["Programming", "Analytical Thinking"],
            skill_gaps=["System Design", "Leadership", "Cloud Technologies"],
            improvement_areas=["Advanced Programming", "Architecture Design"],
            market_relevance_score=0.6,
            confidence_score=0.3  # Low confidence for fallback
        )
    
    def _fallback_career_recommendations(self) -> List[CareerRecommendation]:
        """Fallback career recommendations when AI is unavailable"""
        return [
            CareerRecommendation(
                recommended_role="Software Developer",
                match_score=0.7,
                reasoning="Based on programming skills and experience",
                required_skills=["Programming", "Problem Solving", "Version Control"],
                skill_gaps=["System Design", "Testing"],
                preparation_timeline="3-6 months",
                salary_range="8-15 LPA",
                market_demand="High demand in Indian tech market"
            )
        ]
    
    def _fallback_learning_paths(self) -> List[LearningPath]:
        """Fallback learning paths when AI is unavailable"""
        return [
            LearningPath(
                title="Full Stack Development Path",
                description="Comprehensive path to become a full stack developer",
                target_skills=["React", "Node.js", "Database Design"],
                learning_modules=[
                    {
                        "module_name": "Frontend Development",
                        "topics": ["HTML/CSS", "JavaScript", "React"],
                        "duration": "8 weeks"
                    }
                ],
                estimated_duration="6 months",
                difficulty_level="Intermediate",
                resources=[
                    {
                        "type": "course",
                        "title": "Full Stack Web Development",
                        "provider": "Online Platform",
                        "url": ""
                    }
                ]
            )
        ]
    
    def _fallback_project_suggestions(self) -> List[ProjectSuggestion]:
        """Fallback project suggestions when AI is unavailable"""
        return [
            ProjectSuggestion(
                title="Personal Portfolio Website",
                description="Build a responsive portfolio website to showcase your skills",
                technologies=["HTML", "CSS", "JavaScript", "React"],
                difficulty_level="Beginner",
                estimated_duration="2-3 weeks",
                learning_outcomes=["Frontend Development", "Responsive Design", "Deployment"],
                portfolio_value="Essential for job applications and personal branding"
            )
        ]
    
    def _fallback_market_insights(self) -> Dict[str, Any]:
        """Fallback market insights when AI is unavailable"""
        return {
            "industry_trends": ["Remote Work", "Cloud Computing", "AI/ML Integration"],
            "in_demand_skills": ["Python", "React", "AWS", "Docker", "Kubernetes"],
            "salary_trends": {
                "Software Developer": "8-15 LPA with 15% annual growth",
                "Full Stack Developer": "10-18 LPA with strong demand"
            },
            "city_opportunities": {
                "Bangalore": "Highest number of tech opportunities",
                "Hyderabad": "Growing startup ecosystem",
                "Pune": "Strong IT services presence"
            },
            "emerging_technologies": ["AI/ML", "Blockchain", "IoT", "Edge Computing"],
            "market_forecast": "Strong growth expected in Indian tech sector",
            "actionable_insights": [
                "Focus on cloud technologies for better opportunities",
                "Build strong portfolio with real projects",
                "Consider remote-first companies for better work-life balance"
            ]
        }
    
    async def _fallback_analysis(self, user_id: str, db: AsyncSession) -> CareerAnalysis:
        """Complete fallback analysis when AI is unavailable"""
        logger.warning(f"Using fallback analysis for user {user_id}")
        
        return CareerAnalysis(
            user_id=user_id,
            skill_assessment=self._fallback_skill_assessment(),
            career_recommendations=self._fallback_career_recommendations(),
            learning_paths=self._fallback_learning_paths(),
            project_suggestions=self._fallback_project_suggestions(),
            market_insights=self._fallback_market_insights(),
            analysis_timestamp=datetime.utcnow(),
            gemini_request_id=None
        )
    
    # Public methods for individual analysis components
    
    async def generate_skill_assessment(self, user_id: str, db: AsyncSession) -> SkillAssessment:
        """Generate only skill assessment for a user"""
        profile_data = await self._aggregate_profile_data(user_id, db)
        context = profile_data.to_analysis_context()
        return await self._generate_skill_assessment(context)
    
    async def generate_career_recommendations(self, user_id: str, db: AsyncSession) -> List[CareerRecommendation]:
        """Generate only career recommendations for a user"""
        profile_data = await self._aggregate_profile_data(user_id, db)
        context = profile_data.to_analysis_context()
        skill_assessment = await self._generate_skill_assessment(context)
        return await self._generate_career_recommendations(context, skill_assessment)
    
    async def generate_learning_paths(self, user_id: str, db: AsyncSession) -> List[LearningPath]:
        """Generate only learning paths for a user"""
        profile_data = await self._aggregate_profile_data(user_id, db)
        context = profile_data.to_analysis_context()
        skill_assessment = await self._generate_skill_assessment(context)
        return await self._generate_learning_paths(context, skill_assessment)
    
    async def generate_project_suggestions(self, user_id: str, db: AsyncSession) -> List[ProjectSuggestion]:
        """Generate only project suggestions for a user"""
        profile_data = await self._aggregate_profile_data(user_id, db)
        context = profile_data.to_analysis_context()
        skill_assessment = await self._generate_skill_assessment(context)
        return await self._generate_project_suggestions(context, skill_assessment)
    
    async def get_cached_analysis(self, user_id: str, analysis_type: AnalysisType, db: AsyncSession) -> Optional[Dict[str, Any]]:
        """Get cached analysis results from database"""
        try:
            result = await db.execute(
                select(AnalysisResult)
                .where(
                    AnalysisResult.user_id == user_id,
                    AnalysisResult.analysis_type == analysis_type
                )
                .order_by(AnalysisResult.created_at.desc())
            )
            
            latest_result = result.scalar_one_or_none()
            if latest_result:
                return latest_result.result_data
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get cached analysis: {str(e)}")
            return None
    
    async def is_analysis_stale(self, user_id: str, analysis_type: AnalysisType, db: AsyncSession, max_age_hours: int = 24) -> bool:
        """Check if cached analysis is stale and needs refresh"""
        try:
            result = await db.execute(
                select(AnalysisResult)
                .where(
                    AnalysisResult.user_id == user_id,
                    AnalysisResult.analysis_type == analysis_type
                )
                .order_by(AnalysisResult.created_at.desc())
            )
            
            latest_result = result.scalar_one_or_none()
            if not latest_result:
                return True
            
            age_hours = (datetime.utcnow() - latest_result.created_at).total_seconds() / 3600
            return age_hours > max_age_hours
            
        except Exception as e:
            logger.error(f"Failed to check analysis staleness: {str(e)}")
            return True  # Assume stale on error