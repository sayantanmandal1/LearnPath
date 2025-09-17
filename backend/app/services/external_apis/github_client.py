"""GitHub API client for repository analysis and language detection."""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from pydantic import BaseModel

from .base_client import BaseAPIClient, APIError, RetryConfig


logger = logging.getLogger(__name__)


class GitHubRepository(BaseModel):
    """GitHub repository data model."""
    name: str
    full_name: str
    description: Optional[str]
    language: Optional[str]
    languages: Dict[str, int] = {}
    stars: int
    forks: int
    size: int
    created_at: datetime
    updated_at: datetime
    topics: List[str] = []
    is_fork: bool
    is_private: bool


class GitHubProfile(BaseModel):
    """GitHub user profile data model."""
    username: str
    name: Optional[str] = None
    bio: Optional[str] = None
    company: Optional[str] = None
    location: Optional[str] = None
    email: Optional[str] = None
    public_repos: int
    followers: int
    following: int
    created_at: datetime
    repositories: List[GitHubRepository] = []
    languages: Dict[str, int] = {}
    total_stars: int = 0
    total_commits: int = 0
    contribution_years: List[int] = []


class GitHubClient(BaseAPIClient):
    """GitHub API client with repository analysis capabilities."""
    
    def __init__(self, api_token: Optional[str] = None, timeout: float = 30.0):
        super().__init__(
            base_url="https://api.github.com",
            api_key=api_token,
            timeout=timeout,
            retry_config=RetryConfig(max_retries=3, base_delay=2.0)
        )
    
    def _get_auth_headers(self) -> Dict[str, str]:
        """Get GitHub authentication headers."""
        if self.api_key:
            return {"Authorization": f"token {self.api_key}"}
        return {}
    
    def _update_rate_limit_info(self, response):
        """Update GitHub-specific rate limit information."""
        remaining = response.headers.get('X-RateLimit-Remaining')
        reset_time = response.headers.get('X-RateLimit-Reset')
        
        if remaining and int(remaining) == 0 and reset_time:
            from datetime import datetime
            self._rate_limit_reset = datetime.fromtimestamp(int(reset_time))
    
    async def get_user_profile(self, username: str) -> GitHubProfile:
        """Get comprehensive GitHub user profile with repository analysis."""
        try:
            # Get basic user info
            user_data = await self.get(f"/users/{username}")
            
            # Get repositories
            repositories = await self._get_user_repositories(username)
            
            # Analyze repositories for languages and metrics
            languages, total_stars, contribution_years = await self._analyze_repositories(repositories)
            
            # Get commit count (approximate from recent activity)
            total_commits = await self._estimate_commit_count(username)
            
            return GitHubProfile(
                username=user_data["login"],
                name=user_data.get("name"),
                bio=user_data.get("bio"),
                company=user_data.get("company"),
                location=user_data.get("location"),
                email=user_data.get("email"),
                public_repos=user_data["public_repos"],
                followers=user_data["followers"],
                following=user_data["following"],
                created_at=datetime.fromisoformat(user_data["created_at"].replace('Z', '+00:00')),
                repositories=repositories,
                languages=languages,
                total_stars=total_stars,
                total_commits=total_commits,
                contribution_years=contribution_years
            )
            
        except APIError as e:
            if e.status_code == 404:
                raise APIError(f"GitHub user '{username}' not found", status_code=404)
            raise APIError(f"Failed to fetch GitHub profile for '{username}': {e.message}")
    
    async def _get_user_repositories(self, username: str, max_repos: int = 100) -> List[GitHubRepository]:
        """Get user repositories with detailed information."""
        repositories = []
        page = 1
        per_page = min(max_repos, 100)
        
        while len(repositories) < max_repos:
            try:
                repos_data = await self.get(
                    f"/users/{username}/repos",
                    params={
                        "page": page,
                        "per_page": per_page,
                        "sort": "updated",
                        "direction": "desc"
                    }
                )
                
                if not repos_data:
                    break
                
                for repo_data in repos_data:
                    if len(repositories) >= max_repos:
                        break
                    
                    # Get detailed language information
                    languages = await self._get_repository_languages(repo_data["full_name"])
                    
                    repository = GitHubRepository(
                        name=repo_data["name"],
                        full_name=repo_data["full_name"],
                        description=repo_data.get("description"),
                        language=repo_data.get("language"),
                        languages=languages,
                        stars=repo_data["stargazers_count"],
                        forks=repo_data["forks_count"],
                        size=repo_data["size"],
                        created_at=datetime.fromisoformat(repo_data["created_at"].replace('Z', '+00:00')),
                        updated_at=datetime.fromisoformat(repo_data["updated_at"].replace('Z', '+00:00')),
                        topics=repo_data.get("topics", []),
                        is_fork=repo_data["fork"],
                        is_private=repo_data["private"]
                    )
                    
                    repositories.append(repository)
                
                if len(repos_data) < per_page:
                    break
                    
                page += 1
                
            except APIError as e:
                logger.warning(f"Failed to fetch repositories page {page}: {e.message}")
                break
        
        return repositories
    
    async def _get_repository_languages(self, full_name: str) -> Dict[str, int]:
        """Get programming languages used in a repository."""
        try:
            return await self.get(f"/repos/{full_name}/languages")
        except APIError as e:
            logger.warning(f"Failed to fetch languages for {full_name}: {e.message}")
            return {}
    
    async def _analyze_repositories(self, repositories: List[GitHubRepository]) -> tuple[Dict[str, int], int, List[int]]:
        """Analyze repositories to extract language usage, stars, and contribution years."""
        languages = {}
        total_stars = 0
        years = set()
        
        for repo in repositories:
            # Skip forks for language analysis (optional)
            if not repo.is_fork:
                # Aggregate languages
                for lang, bytes_count in repo.languages.items():
                    languages[lang] = languages.get(lang, 0) + bytes_count
                
                # Count stars
                total_stars += repo.stars
            
            # Track contribution years
            years.add(repo.created_at.year)
            years.add(repo.updated_at.year)
        
        return languages, total_stars, sorted(list(years))
    
    async def _estimate_commit_count(self, username: str) -> int:
        """Estimate total commit count from recent activity."""
        try:
            # Get recent events to estimate activity
            events = await self.get(f"/users/{username}/events", params={"per_page": 100})
            
            # Count push events as a proxy for commits
            push_events = [e for e in events if e.get("type") == "PushEvent"]
            
            # Rough estimation: multiply recent push events by a factor
            # This is a simplified approach - in production, you might want to use
            # the GraphQL API for more accurate commit counts
            estimated_commits = len(push_events) * 10  # Rough multiplier
            
            return max(estimated_commits, len(push_events))
            
        except APIError as e:
            logger.warning(f"Failed to estimate commit count: {e.message}")
            return 0
    
    async def get_repository_details(self, full_name: str) -> GitHubRepository:
        """Get detailed information about a specific repository."""
        try:
            repo_data = await self.get(f"/repos/{full_name}")
            languages = await self._get_repository_languages(full_name)
            
            return GitHubRepository(
                name=repo_data["name"],
                full_name=repo_data["full_name"],
                description=repo_data.get("description"),
                language=repo_data.get("language"),
                languages=languages,
                stars=repo_data["stargazers_count"],
                forks=repo_data["forks_count"],
                size=repo_data["size"],
                created_at=datetime.fromisoformat(repo_data["created_at"].replace('Z', '+00:00')),
                updated_at=datetime.fromisoformat(repo_data["updated_at"].replace('Z', '+00:00')),
                topics=repo_data.get("topics", []),
                is_fork=repo_data["fork"],
                is_private=repo_data["private"]
            )
            
        except APIError as e:
            if e.status_code == 404:
                raise APIError(f"Repository '{full_name}' not found", status_code=404)
            raise APIError(f"Failed to fetch repository details: {e.message}")
    
    async def search_repositories(
        self,
        query: str,
        language: Optional[str] = None,
        sort: str = "stars",
        limit: int = 10
    ) -> List[GitHubRepository]:
        """Search for repositories matching criteria."""
        search_query = query
        if language:
            search_query += f" language:{language}"
        
        try:
            results = await self.get(
                "/search/repositories",
                params={
                    "q": search_query,
                    "sort": sort,
                    "order": "desc",
                    "per_page": min(limit, 100)
                }
            )
            
            repositories = []
            for repo_data in results.get("items", []):
                languages = await self._get_repository_languages(repo_data["full_name"])
                
                repository = GitHubRepository(
                    name=repo_data["name"],
                    full_name=repo_data["full_name"],
                    description=repo_data.get("description"),
                    language=repo_data.get("language"),
                    languages=languages,
                    stars=repo_data["stargazers_count"],
                    forks=repo_data["forks_count"],
                    size=repo_data["size"],
                    created_at=datetime.fromisoformat(repo_data["created_at"].replace('Z', '+00:00')),
                    updated_at=datetime.fromisoformat(repo_data["updated_at"].replace('Z', '+00:00')),
                    topics=repo_data.get("topics", []),
                    is_fork=repo_data["fork"],
                    is_private=repo_data["private"]
                )
                
                repositories.append(repository)
            
            return repositories
            
        except APIError as e:
            raise APIError(f"Failed to search repositories: {e.message}")