"""LeetCode scraper for coding statistics and problem-solving patterns."""

import logging
import re
from typing import Dict, List, Optional, Any
from datetime import datetime
from pydantic import BaseModel

from .base_client import BaseAPIClient, APIError, RetryConfig


logger = logging.getLogger(__name__)


class LeetCodeProblem(BaseModel):
    """LeetCode problem data model."""
    title: str
    difficulty: str
    category: str
    tags: List[str] = []
    acceptance_rate: Optional[float] = None
    solved_at: Optional[datetime] = None


class LeetCodeStats(BaseModel):
    """LeetCode user statistics."""
    total_solved: int
    easy_solved: int
    medium_solved: int
    hard_solved: int
    acceptance_rate: float
    ranking: Optional[int] = None
    reputation: Optional[int] = None


class LeetCodeContest(BaseModel):
    """LeetCode contest performance."""
    contest_name: str
    rank: int
    score: int
    finish_time: Optional[int] = None  # in seconds
    problems_solved: int
    date: datetime


class LeetCodeProfile(BaseModel):
    """LeetCode user profile data model."""
    username: str
    real_name: Optional[str] = None
    avatar: Optional[str] = None
    country: Optional[str] = None
    company: Optional[str] = None
    school: Optional[str] = None
    stats: LeetCodeStats
    solved_problems: List[LeetCodeProblem] = []
    recent_contests: List[LeetCodeContest] = []
    skill_tags: Dict[str, int] = {}  # tag -> problem count
    languages_used: Dict[str, int] = {}  # language -> submission count
    streak_info: Dict[str, Any] = {}


class LeetCodeScraper(BaseAPIClient):
    """LeetCode scraper for user statistics and problem-solving patterns."""
    
    def __init__(self, timeout: float = 30.0):
        # LeetCode doesn't have an official API, so we use GraphQL endpoint
        super().__init__(
            base_url="https://leetcode.com",
            timeout=timeout,
            retry_config=RetryConfig(max_retries=3, base_delay=2.0)
        )
    
    def _get_auth_headers(self) -> Dict[str, str]:
        """LeetCode doesn't require authentication for public profiles."""
        return {}
    
    def _get_default_headers(self) -> Dict[str, str]:
        """Get headers for LeetCode requests."""
        return {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Referer": "https://leetcode.com/"
        }
    
    async def get_user_profile(self, username: str) -> LeetCodeProfile:
        """Get comprehensive LeetCode user profile."""
        try:
            # Get basic user info and stats
            user_data = await self._get_user_basic_info(username)
            stats = await self._get_user_stats(username)
            
            # Get solved problems
            solved_problems = await self._get_solved_problems(username)
            
            # Get contest history
            contests = await self._get_contest_history(username)
            
            # Analyze skill tags and languages
            skill_tags = self._analyze_skill_tags(solved_problems)
            languages_used = await self._get_language_stats(username)
            
            # Get streak information
            streak_info = await self._get_streak_info(username)
            
            return LeetCodeProfile(
                username=username,
                real_name=user_data.get("realName"),
                avatar=user_data.get("avatar"),
                country=user_data.get("profile", {}).get("countryName"),
                company=user_data.get("profile", {}).get("company"),
                school=user_data.get("profile", {}).get("school"),
                stats=stats,
                solved_problems=solved_problems,
                recent_contests=contests,
                skill_tags=skill_tags,
                languages_used=languages_used,
                streak_info=streak_info
            )
            
        except APIError as e:
            if "does not exist" in str(e).lower() or e.status_code == 404:
                raise APIError(f"LeetCode user '{username}' not found", status_code=404)
            raise APIError(f"Failed to fetch LeetCode profile for '{username}': {e.message}")
    
    async def _get_user_basic_info(self, username: str) -> Dict[str, Any]:
        """Get basic user information using GraphQL."""
        query = """
        query getUserProfile($username: String!) {
            matchedUser(username: $username) {
                username
                profile {
                    realName
                    aboutMe
                    avatar
                    location
                    countryName
                    company
                    school
                    websites
                    skillTags
                }
            }
        }
        """
        
        response = await self.post("/graphql", data={
            "query": query,
            "variables": {"username": username}
        })
        
        if not response.get("data", {}).get("matchedUser"):
            raise APIError(f"User {username} does not exist")
        
        return response["data"]["matchedUser"]
    
    async def _get_user_stats(self, username: str) -> LeetCodeStats:
        """Get user problem-solving statistics."""
        query = """
        query getUserStats($username: String!) {
            matchedUser(username: $username) {
                submitStats: submitStatsGlobal {
                    acSubmissionNum {
                        difficulty
                        count
                        submissions
                    }
                }
                profile {
                    ranking
                    reputation
                }
            }
        }
        """
        
        response = await self.post("/graphql", data={
            "query": query,
            "variables": {"username": username}
        })
        
        user_data = response.get("data", {}).get("matchedUser", {})
        submit_stats = user_data.get("submitStats", {}).get("acSubmissionNum", [])
        profile = user_data.get("profile", {})
        
        # Parse difficulty-based stats
        easy_solved = 0
        medium_solved = 0
        hard_solved = 0
        total_submissions = 0
        total_accepted = 0
        
        for stat in submit_stats:
            difficulty = stat.get("difficulty", "").lower()
            count = stat.get("count", 0)
            submissions = stat.get("submissions", 0)
            
            total_accepted += count
            total_submissions += submissions
            
            if difficulty == "easy":
                easy_solved = count
            elif difficulty == "medium":
                medium_solved = count
            elif difficulty == "hard":
                hard_solved = count
        
        acceptance_rate = (total_accepted / total_submissions * 100) if total_submissions > 0 else 0
        
        return LeetCodeStats(
            total_solved=total_accepted,
            easy_solved=easy_solved,
            medium_solved=medium_solved,
            hard_solved=hard_solved,
            acceptance_rate=round(acceptance_rate, 2),
            ranking=profile.get("ranking"),
            reputation=profile.get("reputation")
        )
    
    async def _get_solved_problems(self, username: str, limit: int = 100) -> List[LeetCodeProblem]:
        """Get list of solved problems with details."""
        query = """
        query getRecentSubmissions($username: String!, $limit: Int!) {
            recentSubmissionList(username: $username, limit: $limit) {
                title
                titleSlug
                timestamp
                statusDisplay
                lang
            }
        }
        """
        
        try:
            response = await self.post("/graphql", data={
                "query": query,
                "variables": {"username": username, "limit": limit}
            })
            
            submissions = response.get("data", {}).get("recentSubmissionList", [])
            problems = []
            seen_problems = set()
            
            for submission in submissions:
                if submission.get("statusDisplay") == "Accepted":
                    title = submission.get("title")
                    if title and title not in seen_problems:
                        seen_problems.add(title)
                        
                        # Get problem details
                        problem_details = await self._get_problem_details(submission.get("titleSlug"))
                        
                        problem = LeetCodeProblem(
                            title=title,
                            difficulty=problem_details.get("difficulty", "Unknown"),
                            category=problem_details.get("categoryTitle", "Unknown"),
                            tags=problem_details.get("topicTags", []),
                            acceptance_rate=problem_details.get("acRate"),
                            solved_at=datetime.fromtimestamp(int(submission.get("timestamp", 0)))
                        )
                        
                        problems.append(problem)
            
            return problems
            
        except APIError as e:
            logger.warning(f"Failed to fetch solved problems: {e.message}")
            return []
    
    async def _get_problem_details(self, title_slug: str) -> Dict[str, Any]:
        """Get detailed information about a specific problem."""
        query = """
        query getProblemDetails($titleSlug: String!) {
            question(titleSlug: $titleSlug) {
                difficulty
                categoryTitle
                topicTags {
                    name
                }
                acRate
                stats
            }
        }
        """
        
        try:
            response = await self.post("/graphql", data={
                "query": query,
                "variables": {"titleSlug": title_slug}
            })
            
            question_data = response.get("data", {}).get("question", {})
            
            return {
                "difficulty": question_data.get("difficulty"),
                "categoryTitle": question_data.get("categoryTitle"),
                "topicTags": [tag.get("name") for tag in question_data.get("topicTags", [])],
                "acRate": question_data.get("acRate")
            }
            
        except APIError as e:
            logger.warning(f"Failed to fetch problem details for {title_slug}: {e.message}")
            return {}
    
    async def _get_contest_history(self, username: str, limit: int = 10) -> List[LeetCodeContest]:
        """Get recent contest participation history."""
        query = """
        query getContestHistory($username: String!) {
            userContestRanking(username: $username) {
                attendedContestsCount
                rating
                globalRanking
                totalParticipants
                topPercentage
                badge {
                    name
                }
            }
            userContestRankingHistory(username: $username) {
                attended
                trendDirection
                problemsSolved
                totalProblems
                finishTimeInSeconds
                rating
                ranking
                contest {
                    title
                    startTime
                }
            }
        }
        """
        
        try:
            response = await self.post("/graphql", data={
                "query": query,
                "variables": {"username": username}
            })
            
            contest_history = response.get("data", {}).get("userContestRankingHistory", [])
            contests = []
            
            for contest_data in contest_history[-limit:]:  # Get recent contests
                if contest_data.get("attended"):
                    contest_info = contest_data.get("contest", {})
                    
                    contest = LeetCodeContest(
                        contest_name=contest_info.get("title", "Unknown Contest"),
                        rank=contest_data.get("ranking", 0),
                        score=contest_data.get("rating", 0),
                        finish_time=contest_data.get("finishTimeInSeconds"),
                        problems_solved=contest_data.get("problemsSolved", 0),
                        date=datetime.fromtimestamp(contest_info.get("startTime", 0))
                    )
                    
                    contests.append(contest)
            
            return contests
            
        except APIError as e:
            logger.warning(f"Failed to fetch contest history: {e.message}")
            return []
    
    def _analyze_skill_tags(self, problems: List[LeetCodeProblem]) -> Dict[str, int]:
        """Analyze skill tags from solved problems."""
        skill_tags = {}
        
        for problem in problems:
            for tag in problem.tags:
                skill_tags[tag] = skill_tags.get(tag, 0) + 1
        
        return skill_tags
    
    async def _get_language_stats(self, username: str) -> Dict[str, int]:
        """Get programming language usage statistics."""
        query = """
        query getLanguageStats($username: String!) {
            matchedUser(username: $username) {
                languageProblemCount {
                    languageName
                    problemsSolved
                }
            }
        }
        """
        
        try:
            response = await self.post("/graphql", data={
                "query": query,
                "variables": {"username": username}
            })
            
            language_data = response.get("data", {}).get("matchedUser", {}).get("languageProblemCount", [])
            languages = {}
            
            for lang_stat in language_data:
                lang_name = lang_stat.get("languageName")
                problems_solved = lang_stat.get("problemsSolved", 0)
                if lang_name and problems_solved > 0:
                    languages[lang_name] = problems_solved
            
            return languages
            
        except APIError as e:
            logger.warning(f"Failed to fetch language stats: {e.message}")
            return {}
    
    async def _get_streak_info(self, username: str) -> Dict[str, Any]:
        """Get user's solving streak information."""
        # This would require more complex scraping or calendar data
        # For now, return basic structure
        return {
            "current_streak": 0,
            "longest_streak": 0,
            "total_active_days": 0
        }
    
    async def validate_username(self, username: str) -> bool:
        """Validate if a LeetCode username exists."""
        try:
            await self._get_user_basic_info(username)
            return True
        except APIError:
            return False