"""HackerRank scraper for skill certifications and challenge completions."""

import logging
import re
from typing import Dict, List, Optional, Any
from datetime import datetime
from pydantic import BaseModel

from .base_client import BaseAPIClient, APIError, RetryConfig


logger = logging.getLogger(__name__)


class HackerRankCertification(BaseModel):
    """HackerRank skill certification."""
    skill: str
    level: str  # Basic, Intermediate, Advanced
    certificate_url: Optional[str] = None
    earned_at: datetime
    score: Optional[int] = None
    max_score: Optional[int] = None


class HackerRankChallenge(BaseModel):
    """HackerRank challenge completion."""
    challenge_name: str
    domain: str  # Algorithms, Data Structures, etc.
    subdomain: str
    difficulty: str  # Easy, Medium, Hard
    score: int
    max_score: int
    language: str
    solved_at: datetime
    submission_id: Optional[str] = None


class HackerRankContest(BaseModel):
    """HackerRank contest participation."""
    contest_name: str
    rank: int
    score: int
    max_score: int
    problems_solved: int
    total_problems: int
    start_time: datetime
    end_time: datetime


class HackerRankStats(BaseModel):
    """HackerRank user statistics."""
    total_score: int
    challenges_solved: int
    certifications_earned: int
    contests_participated: int
    rank: Optional[int] = None
    badges_earned: int
    submissions_count: int
    languages_used: int
    domains_active: int


class HackerRankProfile(BaseModel):
    """HackerRank user profile data model."""
    username: str
    name: Optional[str] = None
    country: Optional[str] = None
    company: Optional[str] = None
    school: Optional[str] = None
    avatar: Optional[str] = None
    stats: HackerRankStats
    certifications: List[HackerRankCertification] = []
    solved_challenges: List[HackerRankChallenge] = []
    contest_history: List[HackerRankContest] = []
    domain_scores: Dict[str, int] = {}  # domain -> score
    skill_levels: Dict[str, str] = {}  # skill -> level
    languages_used: Dict[str, int] = {}  # language -> count
    badges: List[str] = []


class HackerRankScraper(BaseAPIClient):
    """HackerRank scraper for skill assessments and challenge data."""
    
    def __init__(self, timeout: float = 30.0):
        super().__init__(
            base_url="https://www.hackerrank.com",
            timeout=timeout,
            retry_config=RetryConfig(max_retries=3, base_delay=2.0)
        )
    
    def _get_auth_headers(self) -> Dict[str, str]:
        """HackerRank doesn't require authentication for public profiles."""
        return {}
    
    def _get_default_headers(self) -> Dict[str, str]:
        """Get headers for HackerRank requests."""
        return {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Referer": "https://www.hackerrank.com/"
        }
    
    async def get_user_profile(self, username: str) -> HackerRankProfile:
        """Get comprehensive HackerRank user profile."""
        try:
            # Get basic user info
            user_data = await self._get_user_info(username)
            
            # Get user statistics
            stats = await self._get_user_stats(username)
            
            # Get certifications
            certifications = await self._get_certifications(username)
            
            # Get solved challenges
            solved_challenges = await self._get_solved_challenges(username)
            
            # Get contest history
            contest_history = await self._get_contest_history(username)
            
            # Analyze domain scores and skills
            domain_scores = self._analyze_domain_scores(solved_challenges)
            skill_levels = self._analyze_skill_levels(certifications)
            languages_used = self._analyze_languages(solved_challenges)
            
            # Get badges
            badges = await self._get_badges(username)
            
            return HackerRankProfile(
                username=username,
                name=user_data.get("name"),
                country=user_data.get("country"),
                company=user_data.get("company"),
                school=user_data.get("school"),
                avatar=user_data.get("avatar"),
                stats=stats,
                certifications=certifications,
                solved_challenges=solved_challenges,
                contest_history=contest_history,
                domain_scores=domain_scores,
                skill_levels=skill_levels,
                languages_used=languages_used,
                badges=badges
            )
            
        except APIError as e:
            if e.status_code == 404:
                raise APIError(f"HackerRank user '{username}' not found", status_code=404)
            raise APIError(f"Failed to fetch HackerRank profile for '{username}': {e.message}")
    
    async def _get_user_info(self, username: str) -> Dict[str, Any]:
        """Get basic user information."""
        try:
            response = await self.get(f"/profile/{username}")
            html_content = response.get("content", "")
            
            # Parse user info from HTML
            user_info = self._parse_user_info_html(html_content)
            return user_info
            
        except APIError as e:
            if e.status_code == 404:
                raise APIError(f"User {username} not found")
            raise
    
    def _parse_user_info_html(self, html_content: str) -> Dict[str, Any]:
        """Parse user information from HTML content."""
        # Simplified implementation - in production use proper HTML parsing
        user_info = {}
        
        # Extract name
        name_match = re.search(r'<h1[^>]*class="[^"]*profile-heading[^"]*"[^>]*>([^<]+)</h1>', html_content)
        if name_match:
            user_info["name"] = name_match.group(1).strip()
        
        # Extract country
        country_match = re.search(r'Country.*?<div[^>]*>([^<]+)</div>', html_content, re.IGNORECASE | re.DOTALL)
        if country_match:
            user_info["country"] = country_match.group(1).strip()
        
        # Extract company
        company_match = re.search(r'Company.*?<div[^>]*>([^<]+)</div>', html_content, re.IGNORECASE | re.DOTALL)
        if company_match:
            user_info["company"] = company_match.group(1).strip()
        
        # Extract school
        school_match = re.search(r'School.*?<div[^>]*>([^<]+)</div>', html_content, re.IGNORECASE | re.DOTALL)
        if school_match:
            user_info["school"] = school_match.group(1).strip()
        
        # Extract avatar
        avatar_match = re.search(r'<img[^>]*class="[^"]*avatar[^"]*"[^>]*src="([^"]+)"', html_content)
        if avatar_match:
            user_info["avatar"] = avatar_match.group(1)
        
        return user_info
    
    async def _get_user_stats(self, username: str) -> HackerRankStats:
        """Get user statistics."""
        try:
            response = await self.get(f"/profile/{username}")
            html_content = response.get("content", "")
            
            # Parse statistics from HTML
            stats_data = self._parse_stats_html(html_content)
            
            return HackerRankStats(
                total_score=stats_data.get("total_score", 0),
                challenges_solved=stats_data.get("challenges_solved", 0),
                certifications_earned=stats_data.get("certifications_earned", 0),
                contests_participated=stats_data.get("contests_participated", 0),
                rank=stats_data.get("rank"),
                badges_earned=stats_data.get("badges_earned", 0),
                submissions_count=stats_data.get("submissions_count", 0),
                languages_used=stats_data.get("languages_used", 0),
                domains_active=stats_data.get("domains_active", 0)
            )
            
        except APIError as e:
            logger.warning(f"Failed to get user stats: {e.message}")
            return HackerRankStats(
                total_score=0,
                challenges_solved=0,
                certifications_earned=0,
                contests_participated=0,
                badges_earned=0,
                submissions_count=0,
                languages_used=0,
                domains_active=0
            )
    
    def _parse_stats_html(self, html_content: str) -> Dict[str, Any]:
        """Parse statistics from HTML content."""
        stats = {}
        
        # Extract total score
        score_match = re.search(r'Total Score.*?(\d+)', html_content, re.IGNORECASE | re.DOTALL)
        if score_match:
            stats["total_score"] = int(score_match.group(1))
        
        # Extract challenges solved
        challenges_match = re.search(r'Challenges Solved.*?(\d+)', html_content, re.IGNORECASE | re.DOTALL)
        if challenges_match:
            stats["challenges_solved"] = int(challenges_match.group(1))
        
        # Extract certifications
        cert_match = re.search(r'Certifications.*?(\d+)', html_content, re.IGNORECASE | re.DOTALL)
        if cert_match:
            stats["certifications_earned"] = int(cert_match.group(1))
        
        # Extract rank
        rank_match = re.search(r'Rank.*?(\d+)', html_content, re.IGNORECASE | re.DOTALL)
        if rank_match:
            stats["rank"] = int(rank_match.group(1))
        
        # Extract badges
        badges_match = re.search(r'Badges.*?(\d+)', html_content, re.IGNORECASE | re.DOTALL)
        if badges_match:
            stats["badges_earned"] = int(badges_match.group(1))
        
        return stats
    
    async def _get_certifications(self, username: str) -> List[HackerRankCertification]:
        """Get user's skill certifications."""
        try:
            response = await self.get(f"/profile/{username}")
            html_content = response.get("content", "")
            
            # Parse certifications from HTML
            certifications = self._parse_certifications_html(html_content)
            return certifications
            
        except APIError as e:
            logger.warning(f"Failed to get certifications: {e.message}")
            return []
    
    def _parse_certifications_html(self, html_content: str) -> List[HackerRankCertification]:
        """Parse certifications from HTML content."""
        # Simplified implementation
        certifications = []
        
        # This would require parsing the certifications section
        # For now, return empty list as placeholder
        logger.warning("Certifications parsing not fully implemented - using mock data")
        
        return certifications
    
    async def _get_solved_challenges(self, username: str) -> List[HackerRankChallenge]:
        """Get user's solved challenges."""
        try:
            # HackerRank might have pagination for challenges
            response = await self.get(f"/profile/{username}")
            html_content = response.get("content", "")
            
            # Parse solved challenges from HTML
            challenges = self._parse_challenges_html(html_content)
            return challenges
            
        except APIError as e:
            logger.warning(f"Failed to get solved challenges: {e.message}")
            return []
    
    def _parse_challenges_html(self, html_content: str) -> List[HackerRankChallenge]:
        """Parse solved challenges from HTML content."""
        # Simplified implementation
        challenges = []
        
        # This would require parsing the challenges section
        # For now, return empty list as placeholder
        logger.warning("Challenges parsing not fully implemented - using mock data")
        
        return challenges
    
    async def _get_contest_history(self, username: str) -> List[HackerRankContest]:
        """Get user's contest participation history."""
        try:
            response = await self.get(f"/profile/{username}")
            html_content = response.get("content", "")
            
            # Parse contest history from HTML
            contests = self._parse_contests_html(html_content)
            return contests
            
        except APIError as e:
            logger.warning(f"Failed to get contest history: {e.message}")
            return []
    
    def _parse_contests_html(self, html_content: str) -> List[HackerRankContest]:
        """Parse contest history from HTML content."""
        # Simplified implementation
        contests = []
        
        # This would require parsing the contests section
        # For now, return empty list as placeholder
        logger.warning("Contest history parsing not fully implemented - using mock data")
        
        return contests
    
    async def _get_badges(self, username: str) -> List[str]:
        """Get user's earned badges."""
        try:
            response = await self.get(f"/profile/{username}")
            html_content = response.get("content", "")
            
            # Parse badges from HTML
            badges = self._parse_badges_html(html_content)
            return badges
            
        except APIError as e:
            logger.warning(f"Failed to get badges: {e.message}")
            return []
    
    def _parse_badges_html(self, html_content: str) -> List[str]:
        """Parse badges from HTML content."""
        badges = []
        
        # Extract badge names from HTML
        badge_matches = re.findall(r'badge[^>]*title="([^"]+)"', html_content, re.IGNORECASE)
        badges.extend(badge_matches)
        
        return list(set(badges))  # Remove duplicates
    
    def _analyze_domain_scores(self, challenges: List[HackerRankChallenge]) -> Dict[str, int]:
        """Analyze scores by domain."""
        domain_scores = {}
        
        for challenge in challenges:
            domain = challenge.domain
            domain_scores[domain] = domain_scores.get(domain, 0) + challenge.score
        
        return domain_scores
    
    def _analyze_skill_levels(self, certifications: List[HackerRankCertification]) -> Dict[str, str]:
        """Analyze skill levels from certifications."""
        skill_levels = {}
        
        for cert in certifications:
            skill_levels[cert.skill] = cert.level
        
        return skill_levels
    
    def _analyze_languages(self, challenges: List[HackerRankChallenge]) -> Dict[str, int]:
        """Analyze programming languages used."""
        language_counts = {}
        
        for challenge in challenges:
            lang = challenge.language
            language_counts[lang] = language_counts.get(lang, 0) + 1
        
        return language_counts
    
    async def validate_username(self, username: str) -> bool:
        """Validate if a HackerRank username exists."""
        try:
            await self._get_user_info(username)
            return True
        except APIError:
            return False
    
    async def get_skill_assessment_details(self, skill: str) -> Optional[Dict[str, Any]]:
        """Get details about a specific skill assessment."""
        try:
            # This would require accessing HackerRank's skill assessment pages
            # For now, return basic structure
            return {
                "skill": skill,
                "levels": ["Basic", "Intermediate", "Advanced"],
                "topics": [],
                "duration": "90 minutes",
                "questions": "Multiple choice and coding"
            }
            
        except APIError as e:
            logger.warning(f"Failed to get skill assessment details: {e.message}")
            return None