"""Codeforces scraper for competitive programming statistics and achievements."""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from pydantic import BaseModel

from .base_client import BaseAPIClient, APIError, RetryConfig


logger = logging.getLogger(__name__)


class CodeforcesContest(BaseModel):
    """Codeforces contest participation."""
    contest_id: int
    contest_name: str
    rank: int
    old_rating: int
    new_rating: int
    rating_change: int
    date: datetime


class CodeforcesProblem(BaseModel):
    """Codeforces problem solution."""
    contest_id: int
    problem_index: str
    problem_name: str
    problem_tags: List[str] = []
    rating: Optional[int] = None
    solved_at: datetime
    language: str
    verdict: str


class CodeforcesStats(BaseModel):
    """Codeforces user statistics."""
    current_rating: int
    max_rating: int
    rank: str
    max_rank: str
    contests_participated: int
    problems_solved: int
    contribution: int
    friend_count: int
    registration_date: datetime


class CodeforcesProfile(BaseModel):
    """Codeforces user profile data model."""
    handle: str
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    country: Optional[str] = None
    city: Optional[str] = None
    organization: Optional[str] = None
    avatar: Optional[str] = None
    stats: CodeforcesStats
    contest_history: List[CodeforcesContest] = []
    solved_problems: List[CodeforcesProblem] = []
    problem_tags: Dict[str, int] = {}  # tag -> count
    languages_used: Dict[str, int] = {}  # language -> count
    rating_graph: List[Dict[str, Any]] = []


class CodeforcesScraper(BaseAPIClient):
    """Codeforces API client for competitive programming data."""
    
    def __init__(self, timeout: float = 30.0):
        super().__init__(
            base_url="https://codeforces.com/api",
            timeout=timeout,
            retry_config=RetryConfig(max_retries=3, base_delay=1.0)
        )
    
    def _get_auth_headers(self) -> Dict[str, str]:
        """Codeforces API doesn't require authentication for public data."""
        return {}
    
    async def get_user_profile(self, handle: str) -> CodeforcesProfile:
        """Get comprehensive Codeforces user profile."""
        try:
            # Get basic user info
            user_data = await self._get_user_info(handle)
            
            # Get contest history
            contest_history = await self._get_contest_history(handle)
            
            # Get solved problems
            solved_problems = await self._get_solved_problems(handle)
            
            # Analyze problem tags and languages
            problem_tags = self._analyze_problem_tags(solved_problems)
            languages_used = self._analyze_languages(solved_problems)
            
            # Create rating graph from contest history
            rating_graph = self._create_rating_graph(contest_history)
            
            # Calculate statistics
            stats = self._calculate_stats(user_data, contest_history, solved_problems)
            
            return CodeforcesProfile(
                handle=user_data["handle"],
                first_name=user_data.get("firstName"),
                last_name=user_data.get("lastName"),
                country=user_data.get("country"),
                city=user_data.get("city"),
                organization=user_data.get("organization"),
                avatar=user_data.get("avatar"),
                stats=stats,
                contest_history=contest_history,
                solved_problems=solved_problems,
                problem_tags=problem_tags,
                languages_used=languages_used,
                rating_graph=rating_graph
            )
            
        except APIError as e:
            if "not found" in str(e).lower():
                raise APIError(f"Codeforces user '{handle}' not found", status_code=404)
            raise APIError(f"Failed to fetch Codeforces profile for '{handle}': {e.message}")
    
    async def _get_user_info(self, handle: str) -> Dict[str, Any]:
        """Get basic user information."""
        response = await self.get("/user.info", params={"handles": handle})
        
        if response.get("status") != "OK":
            raise APIError(f"API error: {response.get('comment', 'Unknown error')}")
        
        result = response.get("result", [])
        if not result:
            raise APIError(f"User {handle} not found")
        
        return result[0]
    
    async def _get_contest_history(self, handle: str) -> List[CodeforcesContest]:
        """Get user's contest participation history."""
        try:
            response = await self.get("/user.rating", params={"handle": handle})
            
            if response.get("status") != "OK":
                logger.warning(f"Failed to get contest history: {response.get('comment')}")
                return []
            
            contests = []
            for contest_data in response.get("result", []):
                contest = CodeforcesContest(
                    contest_id=contest_data["contestId"],
                    contest_name=contest_data["contestName"],
                    rank=contest_data["rank"],
                    old_rating=contest_data["oldRating"],
                    new_rating=contest_data["newRating"],
                    rating_change=contest_data["newRating"] - contest_data["oldRating"],
                    date=datetime.fromtimestamp(contest_data["ratingUpdateTimeSeconds"])
                )
                contests.append(contest)
            
            return contests
            
        except APIError as e:
            logger.warning(f"Failed to get contest history: {e.message}")
            return []
    
    async def _get_solved_problems(self, handle: str, limit: int = 1000) -> List[CodeforcesProblem]:
        """Get user's solved problems."""
        try:
            response = await self.get("/user.status", params={
                "handle": handle,
                "from": 1,
                "count": limit
            })
            
            if response.get("status") != "OK":
                logger.warning(f"Failed to get submissions: {response.get('comment')}")
                return []
            
            # Filter for accepted solutions
            submissions = response.get("result", [])
            solved_problems = []
            seen_problems = set()
            
            for submission in submissions:
                if submission.get("verdict") == "OK":  # Accepted
                    problem = submission.get("problem", {})
                    problem_key = f"{problem.get('contestId', 0)}{problem.get('index', '')}"
                    
                    if problem_key not in seen_problems:
                        seen_problems.add(problem_key)
                        
                        solved_problem = CodeforcesProblem(
                            contest_id=problem.get("contestId", 0),
                            problem_index=problem.get("index", ""),
                            problem_name=problem.get("name", "Unknown"),
                            problem_tags=problem.get("tags", []),
                            rating=problem.get("rating"),
                            solved_at=datetime.fromtimestamp(submission.get("creationTimeSeconds", 0)),
                            language=submission.get("programmingLanguage", "Unknown"),
                            verdict=submission.get("verdict", "OK")
                        )
                        
                        solved_problems.append(solved_problem)
            
            return solved_problems
            
        except APIError as e:
            logger.warning(f"Failed to get solved problems: {e.message}")
            return []
    
    def _analyze_problem_tags(self, problems: List[CodeforcesProblem]) -> Dict[str, int]:
        """Analyze problem tags to identify skill areas."""
        tag_counts = {}
        
        for problem in problems:
            for tag in problem.problem_tags:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
        
        return tag_counts
    
    def _analyze_languages(self, problems: List[CodeforcesProblem]) -> Dict[str, int]:
        """Analyze programming languages used."""
        language_counts = {}
        
        for problem in problems:
            lang = problem.language
            language_counts[lang] = language_counts.get(lang, 0) + 1
        
        return language_counts
    
    def _create_rating_graph(self, contests: List[CodeforcesContest]) -> List[Dict[str, Any]]:
        """Create rating progression graph data."""
        rating_points = []
        
        for contest in contests:
            rating_points.append({
                "date": contest.date.isoformat(),
                "rating": contest.new_rating,
                "contest_name": contest.contest_name,
                "rank": contest.rank,
                "rating_change": contest.rating_change
            })
        
        return rating_points
    
    def _calculate_stats(
        self,
        user_data: Dict[str, Any],
        contests: List[CodeforcesContest],
        problems: List[CodeforcesProblem]
    ) -> CodeforcesStats:
        """Calculate comprehensive user statistics."""
        current_rating = user_data.get("rating", 0)
        max_rating = user_data.get("maxRating", current_rating)
        rank = user_data.get("rank", "unrated")
        max_rank = user_data.get("maxRank", rank)
        
        return CodeforcesStats(
            current_rating=current_rating,
            max_rating=max_rating,
            rank=rank,
            max_rank=max_rank,
            contests_participated=len(contests),
            problems_solved=len(problems),
            contribution=user_data.get("contribution", 0),
            friend_count=user_data.get("friendOfCount", 0),
            registration_date=datetime.fromtimestamp(user_data.get("registrationTimeSeconds", 0))
        )
    
    async def get_contest_standings(self, contest_id: int, handle: str) -> Optional[Dict[str, Any]]:
        """Get user's standing in a specific contest."""
        try:
            response = await self.get("/contest.standings", params={
                "contestId": contest_id,
                "handles": handle
            })
            
            if response.get("status") != "OK":
                return None
            
            rows = response.get("result", {}).get("rows", [])
            if rows:
                return {
                    "rank": rows[0]["rank"],
                    "points": rows[0]["points"],
                    "penalty": rows[0]["penalty"],
                    "successful_hacks": rows[0]["successfulHackCount"],
                    "unsuccessful_hacks": rows[0]["unsuccessfulHackCount"]
                }
            
            return None
            
        except APIError as e:
            logger.warning(f"Failed to get contest standings: {e.message}")
            return None
    
    async def validate_handle(self, handle: str) -> bool:
        """Validate if a Codeforces handle exists."""
        try:
            await self._get_user_info(handle)
            return True
        except APIError:
            return False
    
    async def get_problem_details(self, contest_id: int, problem_index: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific problem."""
        try:
            response = await self.get("/contest.problems", params={"contestId": contest_id})
            
            if response.get("status") != "OK":
                return None
            
            problems = response.get("result", {}).get("problems", [])
            for problem in problems:
                if problem.get("index") == problem_index:
                    return {
                        "name": problem.get("name"),
                        "type": problem.get("type"),
                        "points": problem.get("points"),
                        "rating": problem.get("rating"),
                        "tags": problem.get("tags", [])
                    }
            
            return None
            
        except APIError as e:
            logger.warning(f"Failed to get problem details: {e.message}")
            return None