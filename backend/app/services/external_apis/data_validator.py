"""Data validation and cleaning pipelines for external API data."""

import logging
import re
from typing import Any, Dict, List, Optional, Union, Callable
from datetime import datetime, timedelta
from pydantic import BaseModel, ValidationError, validator
from enum import Enum

from .github_client import GitHubProfile, GitHubRepository
from .leetcode_scraper import LeetCodeProfile, LeetCodeStats
from .linkedin_scraper import LinkedInProfile, LinkedInExperience


logger = logging.getLogger(__name__)


class DataQuality(Enum):
    """Data quality levels."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INVALID = "invalid"


class ValidationResult(BaseModel):
    """Result of data validation."""
    is_valid: bool
    quality: DataQuality
    errors: List[str] = []
    warnings: List[str] = []
    cleaned_data: Optional[Dict[str, Any]] = None
    confidence_score: float = 0.0


class DataValidator:
    """Comprehensive data validator and cleaner for external API data."""
    
    def __init__(self):
        self.skill_patterns = self._load_skill_patterns()
        self.company_aliases = self._load_company_aliases()
        self.language_mappings = self._load_language_mappings()
    
    def _load_skill_patterns(self) -> Dict[str, List[str]]:
        """Load skill normalization patterns."""
        return {
            "python": ["python", "python3", "py", "python programming"],
            "javascript": ["javascript", "js", "node.js", "nodejs", "ecmascript"],
            "java": ["java", "java programming", "java development"],
            "react": ["react", "reactjs", "react.js", "react native"],
            "angular": ["angular", "angularjs", "angular.js"],
            "vue": ["vue", "vuejs", "vue.js"],
            "docker": ["docker", "containerization", "containers"],
            "kubernetes": ["kubernetes", "k8s", "container orchestration"],
            "aws": ["aws", "amazon web services", "amazon aws"],
            "machine_learning": ["machine learning", "ml", "artificial intelligence", "ai"],
            "data_science": ["data science", "data analysis", "data analytics"],
            "sql": ["sql", "mysql", "postgresql", "sqlite", "database"],
            "git": ["git", "version control", "source control", "github", "gitlab"]
        }
    
    def _load_company_aliases(self) -> Dict[str, str]:
        """Load company name normalization mappings."""
        return {
            "google inc": "Google",
            "alphabet inc": "Google",
            "microsoft corporation": "Microsoft",
            "amazon.com inc": "Amazon",
            "meta platforms": "Meta",
            "facebook inc": "Meta",
            "apple inc": "Apple",
            "netflix inc": "Netflix",
            "tesla inc": "Tesla",
            "uber technologies": "Uber"
        }
    
    def _load_language_mappings(self) -> Dict[str, str]:
        """Load programming language normalization mappings."""
        return {
            "c++": "C++",
            "c#": "C#",
            "javascript": "JavaScript",
            "typescript": "TypeScript",
            "python": "Python",
            "java": "Java",
            "go": "Go",
            "rust": "Rust",
            "kotlin": "Kotlin",
            "swift": "Swift",
            "php": "PHP",
            "ruby": "Ruby",
            "scala": "Scala",
            "r": "R",
            "matlab": "MATLAB"
        }
    
    def validate_github_profile(self, profile: GitHubProfile) -> ValidationResult:
        """Validate and clean GitHub profile data."""
        errors = []
        warnings = []
        quality_score = 1.0
        
        try:
            # Validate basic fields
            if not profile.username or len(profile.username.strip()) == 0:
                errors.append("Username is required")
                quality_score -= 0.3
            
            if profile.public_repos < 0:
                errors.append("Public repos count cannot be negative")
                quality_score -= 0.2
            
            # Validate repositories
            valid_repos = []
            for repo in profile.repositories:
                repo_validation = self._validate_repository(repo)
                if repo_validation.is_valid:
                    valid_repos.append(repo_validation.cleaned_data)
                else:
                    warnings.extend([f"Repository {repo.name}: {error}" for error in repo_validation.errors])
                    quality_score -= 0.1
            
            # Clean and normalize languages
            cleaned_languages = self._normalize_languages(profile.languages)
            
            # Calculate quality based on data completeness
            if profile.name:
                quality_score += 0.1
            if profile.bio:
                quality_score += 0.1
            if profile.company:
                quality_score += 0.1
            if len(valid_repos) > 0:
                quality_score += 0.2
            if len(cleaned_languages) > 0:
                quality_score += 0.2
            
            # Determine quality level
            if quality_score >= 0.8:
                quality = DataQuality.HIGH
            elif quality_score >= 0.6:
                quality = DataQuality.MEDIUM
            else:
                quality = DataQuality.LOW
            
            cleaned_data = {
                "username": profile.username.strip(),
                "name": profile.name.strip() if profile.name else None,
                "bio": self._clean_text(profile.bio) if profile.bio else None,
                "company": self._normalize_company(profile.company) if profile.company else None,
                "location": profile.location.strip() if profile.location else None,
                "public_repos": profile.public_repos,
                "followers": profile.followers,
                "following": profile.following,
                "repositories": valid_repos,
                "languages": cleaned_languages,
                "total_stars": max(0, profile.total_stars),
                "created_at": profile.created_at,
                "contribution_years": sorted(list(set(profile.contribution_years)))
            }
            
            return ValidationResult(
                is_valid=len(errors) == 0,
                quality=quality,
                errors=errors,
                warnings=warnings,
                cleaned_data=cleaned_data,
                confidence_score=min(1.0, max(0.0, quality_score))
            )
            
        except Exception as e:
            logger.error(f"Error validating GitHub profile: {str(e)}")
            return ValidationResult(
                is_valid=False,
                quality=DataQuality.INVALID,
                errors=[f"Validation error: {str(e)}"],
                confidence_score=0.0
            )
    
    def _validate_repository(self, repo: GitHubRepository) -> ValidationResult:
        """Validate individual repository data."""
        errors = []
        warnings = []
        
        if not repo.name or len(repo.name.strip()) == 0:
            errors.append("Repository name is required")
        
        if repo.stars < 0:
            errors.append("Stars count cannot be negative")
        
        if repo.forks < 0:
            errors.append("Forks count cannot be negative")
        
        # Clean languages
        cleaned_languages = self._normalize_languages(repo.languages)
        
        cleaned_data = {
            "name": repo.name.strip(),
            "full_name": repo.full_name.strip(),
            "description": self._clean_text(repo.description) if repo.description else None,
            "language": self._normalize_language(repo.language) if repo.language else None,
            "languages": cleaned_languages,
            "stars": max(0, repo.stars),
            "forks": max(0, repo.forks),
            "size": max(0, repo.size),
            "topics": [topic.strip().lower() for topic in repo.topics if topic.strip()],
            "is_fork": repo.is_fork,
            "created_at": repo.created_at,
            "updated_at": repo.updated_at
        }
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            quality=DataQuality.HIGH if len(errors) == 0 and len(warnings) == 0 else DataQuality.MEDIUM,
            errors=errors,
            warnings=warnings,
            cleaned_data=cleaned_data,
            confidence_score=1.0 if len(errors) == 0 else 0.5
        )
    
    def validate_leetcode_profile(self, profile: LeetCodeProfile) -> ValidationResult:
        """Validate and clean LeetCode profile data."""
        errors = []
        warnings = []
        quality_score = 1.0
        
        try:
            # Validate username
            if not profile.username or len(profile.username.strip()) == 0:
                errors.append("Username is required")
                quality_score -= 0.3
            
            # Validate stats
            stats_validation = self._validate_leetcode_stats(profile.stats)
            if not stats_validation.is_valid:
                errors.extend(stats_validation.errors)
                quality_score -= 0.4
            
            # Clean skill tags
            cleaned_skill_tags = self._normalize_skill_tags(profile.skill_tags)
            
            # Clean languages
            cleaned_languages = self._normalize_languages(profile.languages_used)
            
            # Calculate quality score
            if profile.real_name:
                quality_score += 0.1
            if profile.company:
                quality_score += 0.1
            if len(profile.solved_problems) > 0:
                quality_score += 0.2
            if len(cleaned_skill_tags) > 0:
                quality_score += 0.2
            
            # Determine quality level
            if quality_score >= 0.8:
                quality = DataQuality.HIGH
            elif quality_score >= 0.6:
                quality = DataQuality.MEDIUM
            else:
                quality = DataQuality.LOW
            
            cleaned_data = {
                "username": profile.username.strip(),
                "real_name": profile.real_name.strip() if profile.real_name else None,
                "country": profile.country.strip() if profile.country else None,
                "company": self._normalize_company(profile.company) if profile.company else None,
                "school": profile.school.strip() if profile.school else None,
                "stats": stats_validation.cleaned_data,
                "solved_problems": len(profile.solved_problems),
                "skill_tags": cleaned_skill_tags,
                "languages_used": cleaned_languages,
                "recent_contests": len(profile.recent_contests)
            }
            
            return ValidationResult(
                is_valid=len(errors) == 0,
                quality=quality,
                errors=errors,
                warnings=warnings,
                cleaned_data=cleaned_data,
                confidence_score=min(1.0, max(0.0, quality_score))
            )
            
        except Exception as e:
            logger.error(f"Error validating LeetCode profile: {str(e)}")
            return ValidationResult(
                is_valid=False,
                quality=DataQuality.INVALID,
                errors=[f"Validation error: {str(e)}"],
                confidence_score=0.0
            )
    
    def _validate_leetcode_stats(self, stats: LeetCodeStats) -> ValidationResult:
        """Validate LeetCode statistics."""
        errors = []
        
        if stats.total_solved < 0:
            errors.append("Total solved cannot be negative")
        
        if stats.easy_solved < 0:
            errors.append("Easy solved cannot be negative")
        
        if stats.medium_solved < 0:
            errors.append("Medium solved cannot be negative")
        
        if stats.hard_solved < 0:
            errors.append("Hard solved cannot be negative")
        
        if stats.acceptance_rate < 0 or stats.acceptance_rate > 100:
            errors.append("Acceptance rate must be between 0 and 100")
        
        # Check if totals add up
        calculated_total = stats.easy_solved + stats.medium_solved + stats.hard_solved
        if abs(calculated_total - stats.total_solved) > 1:  # Allow small discrepancy
            errors.append("Difficulty breakdown doesn't match total solved")
        
        cleaned_data = {
            "total_solved": max(0, stats.total_solved),
            "easy_solved": max(0, stats.easy_solved),
            "medium_solved": max(0, stats.medium_solved),
            "hard_solved": max(0, stats.hard_solved),
            "acceptance_rate": max(0, min(100, stats.acceptance_rate)),
            "ranking": stats.ranking if stats.ranking and stats.ranking > 0 else None,
            "reputation": stats.reputation if stats.reputation and stats.reputation > 0 else None
        }
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            quality=DataQuality.HIGH if len(errors) == 0 else DataQuality.LOW,
            errors=errors,
            cleaned_data=cleaned_data,
            confidence_score=1.0 if len(errors) == 0 else 0.3
        )
    
    def validate_linkedin_profile(self, profile: LinkedInProfile) -> ValidationResult:
        """Validate and clean LinkedIn profile data."""
        errors = []
        warnings = []
        quality_score = 1.0
        
        try:
            # Validate basic fields
            if not profile.name or len(profile.name.strip()) == 0:
                errors.append("Name is required")
                quality_score -= 0.3
            
            # Validate experience entries
            cleaned_experience = []
            for exp in profile.experience:
                exp_validation = self._validate_experience(exp)
                if exp_validation.is_valid:
                    cleaned_experience.append(exp_validation.cleaned_data)
                else:
                    warnings.extend(exp_validation.errors)
                    quality_score -= 0.1
            
            # Clean skills
            cleaned_skills = self._normalize_linkedin_skills(profile.skills)
            
            # Calculate quality score
            if profile.headline:
                quality_score += 0.1
            if profile.summary:
                quality_score += 0.2
            if len(cleaned_experience) > 0:
                quality_score += 0.3
            if len(cleaned_skills) > 0:
                quality_score += 0.2
            
            # Determine quality level
            if quality_score >= 0.8:
                quality = DataQuality.HIGH
            elif quality_score >= 0.6:
                quality = DataQuality.MEDIUM
            else:
                quality = DataQuality.LOW
            
            cleaned_data = {
                "name": profile.name.strip(),
                "headline": self._clean_text(profile.headline) if profile.headline else None,
                "location": profile.location.strip() if profile.location else None,
                "industry": profile.industry.strip() if profile.industry else None,
                "summary": self._clean_text(profile.summary) if profile.summary else None,
                "current_company": self._normalize_company(profile.current_company) if profile.current_company else None,
                "current_position": profile.current_position.strip() if profile.current_position else None,
                "experience": cleaned_experience,
                "skills": cleaned_skills,
                "profile_url": profile.profile_url
            }
            
            return ValidationResult(
                is_valid=len(errors) == 0,
                quality=quality,
                errors=errors,
                warnings=warnings,
                cleaned_data=cleaned_data,
                confidence_score=min(1.0, max(0.0, quality_score))
            )
            
        except Exception as e:
            logger.error(f"Error validating LinkedIn profile: {str(e)}")
            return ValidationResult(
                is_valid=False,
                quality=DataQuality.INVALID,
                errors=[f"Validation error: {str(e)}"],
                confidence_score=0.0
            )
    
    def _validate_experience(self, experience: LinkedInExperience) -> ValidationResult:
        """Validate individual experience entry."""
        errors = []
        
        if not experience.company or len(experience.company.strip()) == 0:
            errors.append("Company name is required")
        
        if not experience.position or len(experience.position.strip()) == 0:
            errors.append("Position is required")
        
        cleaned_data = {
            "company": self._normalize_company(experience.company) if experience.company else None,
            "position": experience.position.strip() if experience.position else None,
            "duration": experience.duration.strip() if experience.duration else None,
            "location": experience.location.strip() if experience.location else None,
            "description": self._clean_text(experience.description) if experience.description else None,
            "is_current": experience.is_current
        }
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            quality=DataQuality.HIGH if len(errors) == 0 else DataQuality.LOW,
            errors=errors,
            cleaned_data=cleaned_data,
            confidence_score=1.0 if len(errors) == 0 else 0.5
        )
    
    def _normalize_languages(self, languages: Dict[str, int]) -> Dict[str, int]:
        """Normalize programming language names."""
        normalized = {}
        for lang, count in languages.items():
            if lang and count > 0:
                normalized_name = self._normalize_language(lang)
                if normalized_name:
                    normalized[normalized_name] = normalized.get(normalized_name, 0) + count
        return normalized
    
    def _normalize_language(self, language: str) -> Optional[str]:
        """Normalize a single programming language name."""
        if not language:
            return None
        
        lang_lower = language.lower().strip()
        return self.language_mappings.get(lang_lower, language.title())
    
    def _normalize_skill_tags(self, skill_tags: Dict[str, int]) -> Dict[str, int]:
        """Normalize skill tags from LeetCode."""
        normalized = {}
        for tag, count in skill_tags.items():
            if tag and count > 0:
                normalized_tag = self._normalize_skill(tag)
                if normalized_tag:
                    normalized[normalized_tag] = normalized.get(normalized_tag, 0) + count
        return normalized
    
    def _normalize_skill(self, skill: str) -> Optional[str]:
        """Normalize a skill name."""
        if not skill:
            return None
        
        skill_lower = skill.lower().strip()
        
        # Check for exact matches in patterns
        for normalized_skill, patterns in self.skill_patterns.items():
            if skill_lower in patterns:
                return normalized_skill.replace('_', ' ').title()
        
        # Return cleaned version if no match found
        return skill.strip().title()
    
    def _normalize_linkedin_skills(self, skills) -> List[Dict[str, Any]]:
        """Normalize LinkedIn skills."""
        normalized = []
        for skill in skills:
            if hasattr(skill, 'name') and skill.name:
                normalized_name = self._normalize_skill(skill.name)
                if normalized_name:
                    normalized.append({
                        "name": normalized_name,
                        "endorsements": getattr(skill, 'endorsements', 0),
                        "is_top_skill": getattr(skill, 'is_top_skill', False)
                    })
        return normalized
    
    def _normalize_company(self, company: str) -> Optional[str]:
        """Normalize company name."""
        if not company:
            return None
        
        company_clean = company.strip()
        company_lower = company_clean.lower()
        
        # Check for known aliases
        normalized = self.company_aliases.get(company_lower)
        if normalized:
            return normalized
        
        # Clean common suffixes
        suffixes = [' inc', ' inc.', ' corporation', ' corp', ' corp.', ' llc', ' ltd', ' ltd.']
        for suffix in suffixes:
            if company_lower.endswith(suffix):
                company_clean = company_clean[:-len(suffix)].strip()
                break
        
        return company_clean
    
    def _clean_text(self, text: str) -> Optional[str]:
        """Clean and normalize text content."""
        if not text:
            return None
        
        # Remove extra whitespace
        cleaned = re.sub(r'\s+', ' ', text.strip())
        
        # Remove common unwanted characters
        cleaned = re.sub(r'[^\w\s\-.,!?():]', '', cleaned)
        
        return cleaned if len(cleaned) > 0 else None
    
    def validate_combined_profile(self, profiles: Dict[str, Any]) -> ValidationResult:
        """Validate combined profile data from multiple sources."""
        errors = []
        warnings = []
        quality_scores = []
        
        # Validate each source
        for source, profile_data in profiles.items():
            if source == "github" and profile_data:
                validation = self.validate_github_profile(profile_data)
                quality_scores.append(validation.confidence_score)
                if not validation.is_valid:
                    errors.extend([f"GitHub: {error}" for error in validation.errors])
                warnings.extend([f"GitHub: {warning}" for warning in validation.warnings])
            
            elif source == "leetcode" and profile_data:
                validation = self.validate_leetcode_profile(profile_data)
                quality_scores.append(validation.confidence_score)
                if not validation.is_valid:
                    errors.extend([f"LeetCode: {error}" for error in validation.errors])
                warnings.extend([f"LeetCode: {warning}" for warning in validation.warnings])
            
            elif source == "linkedin" and profile_data:
                validation = self.validate_linkedin_profile(profile_data)
                quality_scores.append(validation.confidence_score)
                if not validation.is_valid:
                    errors.extend([f"LinkedIn: {error}" for error in validation.errors])
                warnings.extend([f"LinkedIn: {warning}" for warning in validation.warnings])
        
        # Calculate overall quality
        overall_score = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
        
        if overall_score >= 0.8:
            quality = DataQuality.HIGH
        elif overall_score >= 0.6:
            quality = DataQuality.MEDIUM
        elif overall_score >= 0.3:
            quality = DataQuality.LOW
        else:
            quality = DataQuality.INVALID
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            quality=quality,
            errors=errors,
            warnings=warnings,
            confidence_score=overall_score
        )