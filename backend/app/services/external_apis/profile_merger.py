"""Profile data merging and consolidation service."""

import logging
from typing import Dict, List, Optional, Any, Set
from datetime import datetime
from pydantic import BaseModel

from .github_client import GitHubProfile
from .leetcode_scraper import LeetCodeProfile
from .linkedin_scraper import LinkedInProfile
from .data_validator import DataValidator, ValidationResult, DataQuality


logger = logging.getLogger(__name__)


class MergedProfile(BaseModel):
    """Consolidated profile from multiple sources."""
    # Basic Information
    name: Optional[str] = None
    username: Optional[str] = None
    email: Optional[str] = None
    location: Optional[str] = None
    bio: Optional[str] = None
    
    # Professional Information
    current_company: Optional[str] = None
    current_position: Optional[str] = None
    industry: Optional[str] = None
    experience_years: Optional[int] = None
    
    # Skills and Technologies
    programming_languages: Dict[str, int] = {}  # language -> proficiency score
    technical_skills: Dict[str, int] = {}  # skill -> proficiency score
    frameworks_tools: List[str] = []
    
    # Coding Statistics
    github_stats: Optional[Dict[str, Any]] = None
    leetcode_stats: Optional[Dict[str, Any]] = None
    total_repositories: int = 0
    total_stars: int = 0
    total_problems_solved: int = 0
    
    # Professional Network
    github_followers: int = 0
    linkedin_connections: Optional[int] = None
    
    # Data Quality and Sources
    data_sources: List[str] = []
    data_quality_score: float = 0.0
    confidence_level: str = "low"  # low, medium, high
    last_updated: datetime
    
    # Validation Results
    validation_summary: Dict[str, Any] = {}


class ProfileMerger:
    """Service for merging and consolidating profile data from multiple sources."""
    
    def __init__(self):
        self.data_validator = DataValidator()
        self.skill_weights = {
            "github": 0.4,
            "leetcode": 0.3,
            "linkedin": 0.3
        }
    
    def merge_profiles(
        self,
        github_profile: Optional[Dict[str, Any]] = None,
        leetcode_profile: Optional[Dict[str, Any]] = None,
        linkedin_profile: Optional[Dict[str, Any]] = None,
        validation_results: Optional[Dict[str, ValidationResult]] = None
    ) -> MergedProfile:
        """Merge profiles from multiple sources into a consolidated profile."""
        
        logger.info("Starting profile merge process")
        
        # Initialize merged profile
        merged = MergedProfile(last_updated=datetime.utcnow())
        
        # Track data sources
        if github_profile:
            merged.data_sources.append("github")
        if leetcode_profile:
            merged.data_sources.append("leetcode")
        if linkedin_profile:
            merged.data_sources.append("linkedin")
        
        # Merge basic information
        merged = self._merge_basic_info(merged, github_profile, leetcode_profile, linkedin_profile)
        
        # Merge professional information
        merged = self._merge_professional_info(merged, github_profile, leetcode_profile, linkedin_profile)
        
        # Merge skills and technologies
        merged = self._merge_skills(merged, github_profile, leetcode_profile, linkedin_profile)
        
        # Merge coding statistics
        merged = self._merge_coding_stats(merged, github_profile, leetcode_profile)
        
        # Merge network information
        merged = self._merge_network_info(merged, github_profile, linkedin_profile)
        
        # Calculate data quality and confidence
        merged = self._calculate_quality_metrics(merged, validation_results)
        
        # Add validation summary
        if validation_results:
            merged.validation_summary = self._create_validation_summary(validation_results)
        
        logger.info(f"Profile merge completed. Sources: {merged.data_sources}, Quality: {merged.confidence_level}")
        
        return merged
    
    def _merge_basic_info(
        self,
        merged: MergedProfile,
        github_profile: Optional[Dict[str, Any]],
        leetcode_profile: Optional[Dict[str, Any]],
        linkedin_profile: Optional[Dict[str, Any]]
    ) -> MergedProfile:
        """Merge basic personal information."""
        
        # Name - prioritize LinkedIn > GitHub > LeetCode
        if linkedin_profile and linkedin_profile.get("name"):
            merged.name = linkedin_profile["name"]
        elif github_profile and github_profile.get("name"):
            merged.name = github_profile["name"]
        elif leetcode_profile and leetcode_profile.get("real_name"):
            merged.name = leetcode_profile["real_name"]
        
        # Username - prioritize GitHub
        if github_profile and github_profile.get("username"):
            merged.username = github_profile["username"]
        elif leetcode_profile and leetcode_profile.get("username"):
            merged.username = leetcode_profile["username"]
        
        # Email - from GitHub
        if github_profile and github_profile.get("email"):
            merged.email = github_profile["email"]
        
        # Location - prioritize LinkedIn > GitHub > LeetCode
        if linkedin_profile and linkedin_profile.get("location"):
            merged.location = linkedin_profile["location"]
        elif github_profile and github_profile.get("location"):
            merged.location = github_profile["location"]
        elif leetcode_profile and leetcode_profile.get("country"):
            merged.location = leetcode_profile["country"]
        
        # Bio - prioritize LinkedIn summary > GitHub bio
        if linkedin_profile and linkedin_profile.get("summary"):
            merged.bio = linkedin_profile["summary"]
        elif github_profile and github_profile.get("bio"):
            merged.bio = github_profile["bio"]
        
        return merged
    
    def _merge_professional_info(
        self,
        merged: MergedProfile,
        github_profile: Optional[Dict[str, Any]],
        leetcode_profile: Optional[Dict[str, Any]],
        linkedin_profile: Optional[Dict[str, Any]]
    ) -> MergedProfile:
        """Merge professional information."""
        
        # Current company - prioritize LinkedIn > GitHub > LeetCode
        if linkedin_profile and linkedin_profile.get("current_company"):
            merged.current_company = linkedin_profile["current_company"]
        elif github_profile and github_profile.get("company"):
            merged.current_company = github_profile["company"]
        elif leetcode_profile and leetcode_profile.get("company"):
            merged.current_company = leetcode_profile["company"]
        
        # Current position - from LinkedIn
        if linkedin_profile and linkedin_profile.get("current_position"):
            merged.current_position = linkedin_profile["current_position"]
        
        # Industry - from LinkedIn
        if linkedin_profile and linkedin_profile.get("industry"):
            merged.industry = linkedin_profile["industry"]
        
        # Experience years - estimate from LinkedIn experience or GitHub account age
        if linkedin_profile and linkedin_profile.get("experience"):
            merged.experience_years = self._estimate_experience_years(linkedin_profile["experience"])
        elif github_profile and github_profile.get("created_at"):
            # Rough estimate based on GitHub account age
            try:
                created_date = datetime.fromisoformat(github_profile["created_at"].replace('Z', '+00:00'))
                years_active = (datetime.utcnow() - created_date).days / 365.25
                merged.experience_years = max(1, int(years_active))
            except:
                pass
        
        return merged
    
    def _merge_skills(
        self,
        merged: MergedProfile,
        github_profile: Optional[Dict[str, Any]],
        leetcode_profile: Optional[Dict[str, Any]],
        linkedin_profile: Optional[Dict[str, Any]]
    ) -> MergedProfile:
        """Merge skills and technologies with weighted scoring."""
        
        programming_languages = {}
        technical_skills = {}
        frameworks_tools = set()
        
        # GitHub languages
        if github_profile and github_profile.get("languages"):
            for lang, bytes_count in github_profile["languages"].items():
                # Normalize and weight GitHub language data
                normalized_lang = self._normalize_language(lang)
                if normalized_lang:
                    score = min(100, int(bytes_count / 1000))  # Convert bytes to score
                    programming_languages[normalized_lang] = programming_languages.get(normalized_lang, 0) + \
                                                           int(score * self.skill_weights["github"])
        
        # LeetCode languages and skills
        if leetcode_profile:
            if leetcode_profile.get("languages_used"):
                for lang, problem_count in leetcode_profile["languages_used"].items():
                    normalized_lang = self._normalize_language(lang)
                    if normalized_lang:
                        score = min(100, problem_count * 2)  # 2 points per problem
                        programming_languages[normalized_lang] = programming_languages.get(normalized_lang, 0) + \
                                                               int(score * self.skill_weights["leetcode"])
            
            if leetcode_profile.get("skill_tags"):
                for skill, problem_count in leetcode_profile["skill_tags"].items():
                    normalized_skill = self._normalize_skill(skill)
                    if normalized_skill:
                        score = min(100, problem_count * 3)  # 3 points per problem with this skill
                        technical_skills[normalized_skill] = technical_skills.get(normalized_skill, 0) + \
                                                           int(score * self.skill_weights["leetcode"])
        
        # LinkedIn skills
        if linkedin_profile and linkedin_profile.get("skills"):
            for skill_data in linkedin_profile["skills"]:
                if isinstance(skill_data, dict):
                    skill_name = skill_data.get("name")
                    endorsements = skill_data.get("endorsements", 0)
                    is_top_skill = skill_data.get("is_top_skill", False)
                    
                    if skill_name:
                        # Check if it's a programming language
                        normalized_lang = self._normalize_language(skill_name)
                        if normalized_lang:
                            score = min(100, endorsements * 5 + (20 if is_top_skill else 0))
                            programming_languages[normalized_lang] = programming_languages.get(normalized_lang, 0) + \
                                                                   int(score * self.skill_weights["linkedin"])
                        else:
                            # Treat as technical skill
                            normalized_skill = self._normalize_skill(skill_name)
                            if normalized_skill:
                                score = min(100, endorsements * 5 + (20 if is_top_skill else 0))
                                technical_skills[normalized_skill] = technical_skills.get(normalized_skill, 0) + \
                                                                    int(score * self.skill_weights["linkedin"])
        
        # Extract frameworks and tools from GitHub repositories
        if github_profile and github_profile.get("repositories"):
            for repo in github_profile["repositories"]:
                if repo.get("topics"):
                    for topic in repo["topics"]:
                        frameworks_tools.add(topic.title())
        
        merged.programming_languages = programming_languages
        merged.technical_skills = technical_skills
        merged.frameworks_tools = list(frameworks_tools)
        
        return merged
    
    def _merge_coding_stats(
        self,
        merged: MergedProfile,
        github_profile: Optional[Dict[str, Any]],
        leetcode_profile: Optional[Dict[str, Any]]
    ) -> MergedProfile:
        """Merge coding statistics."""
        
        # GitHub stats
        if github_profile:
            merged.github_stats = {
                "public_repos": github_profile.get("public_repos", 0),
                "total_stars": github_profile.get("total_stars", 0),
                "total_commits": github_profile.get("total_commits", 0),
                "followers": github_profile.get("followers", 0),
                "contribution_years": github_profile.get("contribution_years", [])
            }
            merged.total_repositories = github_profile.get("public_repos", 0)
            merged.total_stars = github_profile.get("total_stars", 0)
        
        # LeetCode stats
        if leetcode_profile and leetcode_profile.get("stats"):
            stats = leetcode_profile["stats"]
            merged.leetcode_stats = {
                "total_solved": stats.get("total_solved", 0),
                "easy_solved": stats.get("easy_solved", 0),
                "medium_solved": stats.get("medium_solved", 0),
                "hard_solved": stats.get("hard_solved", 0),
                "acceptance_rate": stats.get("acceptance_rate", 0),
                "ranking": stats.get("ranking"),
                "reputation": stats.get("reputation")
            }
            merged.total_problems_solved = stats.get("total_solved", 0)
        
        return merged
    
    def _merge_network_info(
        self,
        merged: MergedProfile,
        github_profile: Optional[Dict[str, Any]],
        linkedin_profile: Optional[Dict[str, Any]]
    ) -> MergedProfile:
        """Merge professional network information."""
        
        if github_profile:
            merged.github_followers = github_profile.get("followers", 0)
        
        if linkedin_profile:
            merged.linkedin_connections = linkedin_profile.get("connections_count")
        
        return merged
    
    def _calculate_quality_metrics(
        self,
        merged: MergedProfile,
        validation_results: Optional[Dict[str, ValidationResult]]
    ) -> MergedProfile:
        """Calculate overall data quality and confidence metrics."""
        
        quality_scores = []
        
        if validation_results:
            for source, result in validation_results.items():
                quality_scores.append(result.confidence_score)
        
        # Calculate completeness score
        completeness_score = 0.0
        total_fields = 10  # Number of key fields to check
        
        if merged.name:
            completeness_score += 1
        if merged.email:
            completeness_score += 1
        if merged.location:
            completeness_score += 1
        if merged.current_company:
            completeness_score += 1
        if merged.current_position:
            completeness_score += 1
        if merged.programming_languages:
            completeness_score += 1
        if merged.technical_skills:
            completeness_score += 1
        if merged.github_stats:
            completeness_score += 1
        if merged.leetcode_stats:
            completeness_score += 1
        if len(merged.data_sources) > 1:
            completeness_score += 1
        
        completeness_score = completeness_score / total_fields
        
        # Combine validation scores with completeness
        if quality_scores:
            avg_validation_score = sum(quality_scores) / len(quality_scores)
            merged.data_quality_score = (avg_validation_score * 0.7) + (completeness_score * 0.3)
        else:
            merged.data_quality_score = completeness_score
        
        # Determine confidence level
        if merged.data_quality_score >= 0.8:
            merged.confidence_level = "high"
        elif merged.data_quality_score >= 0.6:
            merged.confidence_level = "medium"
        else:
            merged.confidence_level = "low"
        
        return merged
    
    def _create_validation_summary(self, validation_results: Dict[str, ValidationResult]) -> Dict[str, Any]:
        """Create a summary of validation results."""
        summary = {
            "total_sources": len(validation_results),
            "valid_sources": 0,
            "quality_breakdown": {},
            "common_issues": [],
            "recommendations": []
        }
        
        all_errors = []
        all_warnings = []
        
        for source, result in validation_results.items():
            if result.is_valid:
                summary["valid_sources"] += 1
            
            summary["quality_breakdown"][source] = {
                "quality": result.quality.value,
                "confidence": result.confidence_score,
                "errors": len(result.errors),
                "warnings": len(result.warnings)
            }
            
            all_errors.extend(result.errors)
            all_warnings.extend(result.warnings)
        
        # Identify common issues
        error_counts = {}
        for error in all_errors:
            error_type = error.split(":")[0] if ":" in error else error
            error_counts[error_type] = error_counts.get(error_type, 0) + 1
        
        summary["common_issues"] = [
            {"issue": issue, "count": count}
            for issue, count in sorted(error_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        ]
        
        # Generate recommendations
        if summary["valid_sources"] == 0:
            summary["recommendations"].append("All profile sources have validation issues. Consider updating profile information.")
        elif summary["valid_sources"] < len(validation_results):
            summary["recommendations"].append("Some profile sources have issues. Review and update incomplete profiles.")
        
        if len(validation_results) == 1:
            summary["recommendations"].append("Consider adding more profile sources for better data completeness.")
        
        return summary
    
    def _estimate_experience_years(self, experience_list: List[Dict[str, Any]]) -> int:
        """Estimate years of experience from LinkedIn experience data."""
        if not experience_list:
            return 0
        
        total_months = 0
        for exp in experience_list:
            if exp.get("duration"):
                # Parse duration strings like "2 years 3 months", "1 year", "6 months"
                duration = exp["duration"].lower()
                years = 0
                months = 0
                
                if "year" in duration:
                    year_match = duration.split("year")[0].strip().split()[-1]
                    try:
                        years = int(year_match)
                    except:
                        years = 1
                
                if "month" in duration:
                    month_match = duration.split("month")[0].strip().split()[-1]
                    try:
                        months = int(month_match)
                    except:
                        months = 6
                
                total_months += (years * 12) + months
        
        return max(1, int(total_months / 12))
    
    def _normalize_language(self, language: str) -> Optional[str]:
        """Normalize programming language names."""
        if not language:
            return None
        
        lang_map = {
            "javascript": "JavaScript",
            "typescript": "TypeScript",
            "python": "Python",
            "java": "Java",
            "c++": "C++",
            "c#": "C#",
            "go": "Go",
            "rust": "Rust",
            "kotlin": "Kotlin",
            "swift": "Swift",
            "php": "PHP",
            "ruby": "Ruby",
            "scala": "Scala",
            "r": "R",
            "matlab": "MATLAB",
            "sql": "SQL",
            "html": "HTML",
            "css": "CSS"
        }
        
        return lang_map.get(language.lower(), language.title())
    
    def _normalize_skill(self, skill: str) -> Optional[str]:
        """Normalize technical skill names."""
        if not skill:
            return None
        
        skill_map = {
            "machine learning": "Machine Learning",
            "artificial intelligence": "Artificial Intelligence",
            "data science": "Data Science",
            "web development": "Web Development",
            "mobile development": "Mobile Development",
            "cloud computing": "Cloud Computing",
            "devops": "DevOps",
            "database": "Database Management",
            "api": "API Development",
            "microservices": "Microservices",
            "docker": "Docker",
            "kubernetes": "Kubernetes",
            "aws": "AWS",
            "azure": "Azure",
            "gcp": "Google Cloud Platform"
        }
        
        return skill_map.get(skill.lower(), skill.title())