"""AtCoder scraper for competitive programming statistics and achievements."""

import logging
import re
from typing import Dict, List, Optional, Any
from datetime import datetime
from pydantic import BaseModel

from .base_client import BaseAPIClient, APIError, RetryConfig


logger = logging.getLogger(__name__)


class AtCoderContest(BaseModel):
    """AtCoder contest participation."""
    contest_name: str
    contest_type: str
    rank: int
    performance: int
    rating_change: int
    new_rating: int
    date: datetime


class AtCoderProblem(BaseModel):
    """AtCoder problem solution."""
    contest_name: str
    problem_id: str
    problem_title: str
    difficulty: Optional[str] = None
    score: Optional[int] = None
    solved_at: datetime
    language: str
    execution_time: Optional[int] = None  # in ms
    memory_usage: Optional[int] = None  # in KB


class AtCoderStats(BaseModel):
    """AtCoder user statistics."""
    current_rating: int
    max_rating: int
    rank: str
    contests_participated: int
    problems_solved: int
    submissions_count: int
    accepted_count: int
    acceptance_rate: float
    registration_date: Optional[datetime] = None


class AtCoderProfile(BaseModel):
    """AtCoder user profile data model."""
    username: str
    country: Optional[str] = None
    birth_year: Optional[int] = None
    twitter_id: Optional[str] = None
    stats: AtCoderStats
    contest_history: List[AtCoderContest] = []
    solved_problems: List[AtCoderProblem] = []
    difficulty_distribution: Dict[str, int] = {}  # difficulty -> count
    languages_used: Dict[str, int] = {}  # language -> count
    rating_graph: List[Dict[str, Any]] = []


class AtCoderScraper(BaseAPIClient):
    """AtCoder scraper for competitive programming data."""
    
    def __init__(self, timeout: float = 30.0):
        super().__init__(
            base_url="https://atcoder.jp",
            timeout=timeout,
            retry_config=RetryConfig(max_retries=3, base_delay=2.0)
        )
    
    def _get_auth_headers(self) -> Dict[str, str]:
        """AtCoder doesn't require authentication for public profiles."""
        return {}
    
    def _get_default_headers(self) -> Dict[str, str]:
        """Get headers for AtCoder requests."""
        return {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
        }
    
    async def get_user_profile(self, username: str) -> AtCoderProfile:
        """Get comprehensive AtCoder user profile."""
        try:
            # Get basic user info and stats
            user_data = await self._get_user_info(username)
            stats = await self._get_user_stats(username)
            
            # Get contest history
            contest_history = await self._get_contest_history(username)
            
            # Get solved problems
            solved_problems = await self._get_solved_problems(username)
            
            # Analyze difficulty distribution and languages
            difficulty_dist = self._analyze_difficulty_distribution(solved_problems)
            languages_used = self._analyze_languages(solved_problems)
            
            # Create rating graph
            rating_graph = self._create_rating_graph(contest_history)
            
            return AtCoderProfile(
                username=username,
                country=user_data.get("country"),
                birth_year=user_data.get("birth_year"),
                twitter_id=user_data.get("twitter_id"),
                stats=stats,
                contest_history=contest_history,
                solved_problems=solved_problems,
                difficulty_distribution=difficulty_dist,
                languages_used=languages_used,
                rating_graph=rating_graph
            )
            
        except APIError as e:
            if e.status_code == 404:
                raise APIError(f"AtCoder user '{username}' not found", status_code=404)
            raise APIError(f"Failed to fetch AtCoder profile for '{username}': {e.message}")
    
    async def _get_user_info(self, username: str) -> Dict[str, Any]:
        """Get basic user information by scraping profile page."""
        try:
            response = await self.get(f"/users/{username}")
            html_content = response.get("content", "")
            
            # Parse basic info from HTML (simplified implementation)
            user_info = self._parse_user_info_html(html_content)
            return user_info
            
        except APIError as e:
            if e.status_code == 404:
                raise APIError(f"User {username} not found")
            raise
    
    def _parse_user_info_html(self, html_content: str) -> Dict[str, Any]:
        """Parse user information from HTML content."""
        # This is a simplified implementation
        # In production, use proper HTML parsing with BeautifulSoup
        
        user_info = {}
        
        # Extract country (example pattern)
        country_match = re.search(r'Country.*?<td[^>]*>([^<]+)</td>', html_content, re.IGNORECASE | re.DOTALL)
        if country_match:
            user_info["country"] = country_match.group(1).strip()
        
        # Extract birth year (example pattern)
        birth_match = re.search(r'Birth Year.*?<td[^>]*>(\d{4})</td>', html_content, re.IGNORECASE | re.DOTALL)
        if birth_match:
            user_info["birth_year"] = int(birth_match.group(1))
        
        # Extract Twitter ID (example pattern)
        twitter_match = re.search(r'Twitter.*?<td[^>]*>@([^<]+)</td>', html_content, re.IGNORECASE | re.DOTALL)
        if twitter_match:
            user_info["twitter_id"] = twitter_match.group(1).strip()
        
        return user_info
    
    async def _get_user_stats(self, username: str) -> AtCoderStats:
        """Get user statistics by scraping profile page."""
        try:
            response = await self.get(f"/users/{username}")
            html_content = response.get("content", "")
            
            # Parse statistics from HTML
            stats_data = self._parse_stats_html(html_content)
            
            return AtCoderStats(
                current_rating=stats_data.get("current_rating", 0),
                max_rating=stats_data.get("max_rating", 0),
                rank=stats_data.get("rank", "Unrated"),
                contests_participated=stats_data.get("contests_participated", 0),
                problems_solved=stats_data.get("problems_solved", 0),
                submissions_count=stats_data.get("submissions_count", 0),
                accepted_count=stats_data.get("accepted_count", 0),
                acceptance_rate=stats_data.get("acceptance_rate", 0.0),
                registration_date=stats_data.get("registration_date")
            )
            
        except APIError as e:
            logger.warning(f"Failed to get user stats: {e.message}")
            # Return default stats
            return AtCoderStats(
                current_rating=0,
                max_rating=0,
                rank="Unrated",
                contests_participated=0,
                problems_solved=0,
                submissions_count=0,
                accepted_count=0,
                acceptance_rate=0.0
            )
    
    def _parse_stats_html(self, html_content: str) -> Dict[str, Any]:
        """Parse statistics from HTML content."""
        # Simplified implementation - in production use proper HTML parsing
        stats = {}
        
        # Extract rating (example patterns)
        rating_match = re.search(r'Rating.*?<span[^>]*>(\d+)</span>', html_content, re.IGNORECASE | re.DOTALL)
        if rating_match:
            stats["current_rating"] = int(rating_match.group(1))
            stats["max_rating"] = int(rating_match.group(1))  # Simplified
        
        # Extract rank
        rank_match = re.search(r'<span[^>]*class="[^"]*user-[^"]*"[^>]*>([^<]+)</span>', html_content)
        if rank_match:
            stats["rank"] = rank_match.group(1).strip()
        
        # Extract contest count
        contest_match = re.search(r'Rated Matches.*?(\d+)', html_content, re.IGNORECASE | re.DOTALL)
        if contest_match:
            stats["contests_participated"] = int(contest_match.group(1))
        
        # Extract problems solved (simplified)
        stats["problems_solved"] = 0
        stats["submissions_count"] = 0
        stats["accepted_count"] = 0
        stats["acceptance_rate"] = 0.0
        
        return stats
    
    async def _get_contest_history(self, username: str) -> List[AtCoderContest]:
        """Get contest participation history."""
        try:
            response = await self.get(f"/users/{username}/history")
            html_content = response.get("content", "")
            
            # Parse contest history from HTML
            contests = self._parse_contest_history_html(html_content)
            return contests
            
        except APIError as e:
            logger.warning(f"Failed to get contest history: {e.message}")
            return []
    
    def _parse_contest_history_html(self, html_content: str) -> List[AtCoderContest]:
        """Parse contest history from HTML content."""
        # Simplified implementation
        # In production, use proper HTML parsing to extract contest data
        contests = []
        
        # This would require parsing the contest history table
        # For now, return empty list as placeholder
        logger.warning("Contest history parsing not fully implemented - using mock data")
        
        return contests
    
    async def _get_solved_problems(self, username: str) -> List[AtCoderProblem]:
        """Get solved problems list."""
        try:
            # AtCoder doesn't have a direct API for this
            # Would need to scrape submissions page or use unofficial APIs
            response = await self.get(f"/users/{username}/history/share")
            html_content = response.get("content", "")
            
            # Parse solved problems from HTML
            problems = self._parse_solved_problems_html(html_content)
            return problems
            
        except APIError as e:
            logger.warning(f"Failed to get solved problems: {e.message}")
            return []
    
    def _parse_solved_problems_html(self, html_content: str) -> List[AtCoderProblem]:
        """Parse solved problems from HTML content."""
        # Simplified implementation
        problems = []
        
        # This would require parsing the submissions table
        # For now, return empty list as placeholder
        logger.warning("Solved problems parsing not fully implemented - using mock data")
        
        return problems
    
    def _analyze_difficulty_distribution(self, problems: List[AtCoderProblem]) -> Dict[str, int]:
        """Analyze difficulty distribution of solved problems."""
        difficulty_counts = {}
        
        for problem in problems:
            if problem.difficulty:
                difficulty_counts[problem.difficulty] = difficulty_counts.get(problem.difficulty, 0) + 1
        
        return difficulty_counts
    
    def _analyze_languages(self, problems: List[AtCoderProblem]) -> Dict[str, int]:
        """Analyze programming languages used."""
        language_counts = {}
        
        for problem in problems:
            lang = problem.language
            language_counts[lang] = language_counts.get(lang, 0) + 1
        
        return language_counts
    
    def _create_rating_graph(self, contests: List[AtCoderContest]) -> List[Dict[str, Any]]:
        """Create rating progression graph data."""
        rating_points = []
        
        for contest in contests:
            rating_points.append({
                "date": contest.date.isoformat(),
                "rating": contest.new_rating,
                "contest_name": contest.contest_name,
                "rank": contest.rank,
                "rating_change": contest.rating_change,
                "performance": contest.performance
            })
        
        return rating_points
    
    async def validate_username(self, username: str) -> bool:
        """Validate if an AtCoder username exists."""
        try:
            await self._get_user_info(username)
            return True
        except APIError:
            return False
    
    async def get_contest_details(self, contest_id: str) -> Optional[Dict[str, Any]]:
        """Get details about a specific contest."""
        try:
            response = await self.get(f"/contests/{contest_id}")
            html_content = response.get("content", "")
            
            # Parse contest details from HTML
            contest_info = self._parse_contest_details_html(html_content)
            return contest_info
            
        except APIError as e:
            logger.warning(f"Failed to get contest details: {e.message}")
            return None
    
    def _parse_contest_details_html(self, html_content: str) -> Dict[str, Any]:
        """Parse contest details from HTML content."""
        # Simplified implementation
        contest_info = {
            "name": "Unknown Contest",
            "start_time": None,
            "duration": None,
            "type": "Unknown",
            "problems_count": 0
        }
        
        # Extract contest name
        name_match = re.search(r'<title>([^<]+)</title>', html_content)
        if name_match:
            contest_info["name"] = name_match.group(1).strip()
        
        return contest_info