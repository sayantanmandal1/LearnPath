"""Kaggle scraper for competition rankings and dataset contributions."""

import logging
import re
from typing import Dict, List, Optional, Any
from datetime import datetime
from pydantic import BaseModel

from .base_client import BaseAPIClient, APIError, RetryConfig


logger = logging.getLogger(__name__)


class KaggleCompetition(BaseModel):
    """Kaggle competition participation."""
    competition_name: str
    competition_type: str  # Featured, Research, Getting Started, etc.
    rank: Optional[int] = None
    percentile: Optional[float] = None
    score: Optional[float] = None
    entries: int = 0
    team_size: int = 1
    medal: Optional[str] = None  # Gold, Silver, Bronze
    ended_at: datetime
    prize_pool: Optional[str] = None


class KaggleDataset(BaseModel):
    """Kaggle dataset contribution."""
    title: str
    subtitle: Optional[str] = None
    description: Optional[str] = None
    size: Optional[str] = None
    file_count: int = 0
    download_count: int = 0
    vote_count: int = 0
    view_count: int = 0
    created_at: datetime
    updated_at: datetime
    tags: List[str] = []
    license_name: Optional[str] = None


class KaggleNotebook(BaseModel):
    """Kaggle notebook/kernel."""
    title: str
    language: str  # Python, R, etc.
    notebook_type: str  # Script, Notebook
    vote_count: int = 0
    comment_count: int = 0
    view_count: int = 0
    fork_count: int = 0
    created_at: datetime
    updated_at: datetime
    tags: List[str] = []
    medal: Optional[str] = None


class KaggleStats(BaseModel):
    """Kaggle user statistics."""
    competitions_entered: int
    competitions_won: int
    highest_rank: Optional[int] = None
    current_tier: str  # Novice, Contributor, Expert, Master, Grandmaster
    datasets_created: int
    notebooks_created: int
    total_votes: int
    total_medals: int
    gold_medals: int
    silver_medals: int
    bronze_medals: int
    followers: int
    following: int


class KaggleProfile(BaseModel):
    """Kaggle user profile data model."""
    username: str
    display_name: Optional[str] = None
    bio: Optional[str] = None
    location: Optional[str] = None
    website: Optional[str] = None
    twitter: Optional[str] = None
    github: Optional[str] = None
    linkedin: Optional[str] = None
    avatar: Optional[str] = None
    stats: KaggleStats
    competitions: List[KaggleCompetition] = []
    datasets: List[KaggleDataset] = []
    notebooks: List[KaggleNotebook] = []
    skills: List[str] = []
    achievements: List[str] = []
    tier_progression: Dict[str, Any] = {}


class KaggleScraper(BaseAPIClient):
    """Kaggle scraper for competition and dataset data."""
    
    def __init__(self, timeout: float = 30.0):
        super().__init__(
            base_url="https://www.kaggle.com",
            timeout=timeout,
            retry_config=RetryConfig(max_retries=3, base_delay=2.0)
        )
    
    def _get_auth_headers(self) -> Dict[str, str]:
        """Kaggle doesn't require authentication for public profiles."""
        return {}
    
    def _get_default_headers(self) -> Dict[str, str]:
        """Get headers for Kaggle requests."""
        return {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Referer": "https://www.kaggle.com/"
        }
    
    async def get_user_profile(self, username: str) -> KaggleProfile:
        """Get comprehensive Kaggle user profile."""
        try:
            # Get basic user info
            user_data = await self._get_user_info(username)
            
            # Get user statistics
            stats = await self._get_user_stats(username)
            
            # Get competitions
            competitions = await self._get_competitions(username)
            
            # Get datasets
            datasets = await self._get_datasets(username)
            
            # Get notebooks
            notebooks = await self._get_notebooks(username)
            
            # Get skills and achievements
            skills = await self._get_skills(username)
            achievements = await self._get_achievements(username)
            
            # Get tier progression
            tier_progression = await self._get_tier_progression(username)
            
            return KaggleProfile(
                username=username,
                display_name=user_data.get("display_name"),
                bio=user_data.get("bio"),
                location=user_data.get("location"),
                website=user_data.get("website"),
                twitter=user_data.get("twitter"),
                github=user_data.get("github"),
                linkedin=user_data.get("linkedin"),
                avatar=user_data.get("avatar"),
                stats=stats,
                competitions=competitions,
                datasets=datasets,
                notebooks=notebooks,
                skills=skills,
                achievements=achievements,
                tier_progression=tier_progression
            )
            
        except APIError as e:
            if e.status_code == 404:
                raise APIError(f"Kaggle user '{username}' not found", status_code=404)
            raise APIError(f"Failed to fetch Kaggle profile for '{username}': {e.message}")
    
    async def _get_user_info(self, username: str) -> Dict[str, Any]:
        """Get basic user information."""
        try:
            response = await self.get(f"/{username}")
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
        user_info = {}
        
        # Extract display name
        name_match = re.search(r'<h1[^>]*>([^<]+)</h1>', html_content)
        if name_match:
            user_info["display_name"] = name_match.group(1).strip()
        
        # Extract bio
        bio_match = re.search(r'<div[^>]*class="[^"]*bio[^"]*"[^>]*>([^<]+)</div>', html_content)
        if bio_match:
            user_info["bio"] = bio_match.group(1).strip()
        
        # Extract location
        location_match = re.search(r'Location.*?<span[^>]*>([^<]+)</span>', html_content, re.IGNORECASE | re.DOTALL)
        if location_match:
            user_info["location"] = location_match.group(1).strip()
        
        # Extract social links
        twitter_match = re.search(r'twitter\.com/([^"\'>\s]+)', html_content)
        if twitter_match:
            user_info["twitter"] = twitter_match.group(1)
        
        github_match = re.search(r'github\.com/([^"\'>\s]+)', html_content)
        if github_match:
            user_info["github"] = github_match.group(1)
        
        linkedin_match = re.search(r'linkedin\.com/in/([^"\'>\s]+)', html_content)
        if linkedin_match:
            user_info["linkedin"] = linkedin_match.group(1)
        
        # Extract avatar
        avatar_match = re.search(r'<img[^>]*class="[^"]*avatar[^"]*"[^>]*src="([^"]+)"', html_content)
        if avatar_match:
            user_info["avatar"] = avatar_match.group(1)
        
        return user_info
    
    async def _get_user_stats(self, username: str) -> KaggleStats:
        """Get user statistics."""
        try:
            response = await self.get(f"/{username}")
            html_content = response.get("content", "")
            
            # Parse statistics from HTML
            stats_data = self._parse_stats_html(html_content)
            
            return KaggleStats(
                competitions_entered=stats_data.get("competitions_entered", 0),
                competitions_won=stats_data.get("competitions_won", 0),
                highest_rank=stats_data.get("highest_rank"),
                current_tier=stats_data.get("current_tier", "Novice"),
                datasets_created=stats_data.get("datasets_created", 0),
                notebooks_created=stats_data.get("notebooks_created", 0),
                total_votes=stats_data.get("total_votes", 0),
                total_medals=stats_data.get("total_medals", 0),
                gold_medals=stats_data.get("gold_medals", 0),
                silver_medals=stats_data.get("silver_medals", 0),
                bronze_medals=stats_data.get("bronze_medals", 0),
                followers=stats_data.get("followers", 0),
                following=stats_data.get("following", 0)
            )
            
        except APIError as e:
            logger.warning(f"Failed to get user stats: {e.message}")
            return KaggleStats(
                competitions_entered=0,
                competitions_won=0,
                current_tier="Novice",
                datasets_created=0,
                notebooks_created=0,
                total_votes=0,
                total_medals=0,
                gold_medals=0,
                silver_medals=0,
                bronze_medals=0,
                followers=0,
                following=0
            )
    
    def _parse_stats_html(self, html_content: str) -> Dict[str, Any]:
        """Parse statistics from HTML content."""
        stats = {}
        
        # Extract tier
        tier_match = re.search(r'<div[^>]*class="[^"]*tier[^"]*"[^>]*>([^<]+)</div>', html_content)
        if tier_match:
            stats["current_tier"] = tier_match.group(1).strip()
        
        # Extract competitions count
        comp_match = re.search(r'Competitions.*?(\d+)', html_content, re.IGNORECASE | re.DOTALL)
        if comp_match:
            stats["competitions_entered"] = int(comp_match.group(1))
        
        # Extract datasets count
        dataset_match = re.search(r'Datasets.*?(\d+)', html_content, re.IGNORECASE | re.DOTALL)
        if dataset_match:
            stats["datasets_created"] = int(dataset_match.group(1))
        
        # Extract notebooks count
        notebook_match = re.search(r'Notebooks.*?(\d+)', html_content, re.IGNORECASE | re.DOTALL)
        if notebook_match:
            stats["notebooks_created"] = int(notebook_match.group(1))
        
        # Extract medals
        gold_match = re.search(r'Gold.*?(\d+)', html_content, re.IGNORECASE | re.DOTALL)
        if gold_match:
            stats["gold_medals"] = int(gold_match.group(1))
        
        silver_match = re.search(r'Silver.*?(\d+)', html_content, re.IGNORECASE | re.DOTALL)
        if silver_match:
            stats["silver_medals"] = int(silver_match.group(1))
        
        bronze_match = re.search(r'Bronze.*?(\d+)', html_content, re.IGNORECASE | re.DOTALL)
        if bronze_match:
            stats["bronze_medals"] = int(bronze_match.group(1))
        
        stats["total_medals"] = stats.get("gold_medals", 0) + stats.get("silver_medals", 0) + stats.get("bronze_medals", 0)
        
        # Extract followers/following
        followers_match = re.search(r'Followers.*?(\d+)', html_content, re.IGNORECASE | re.DOTALL)
        if followers_match:
            stats["followers"] = int(followers_match.group(1))
        
        following_match = re.search(r'Following.*?(\d+)', html_content, re.IGNORECASE | re.DOTALL)
        if following_match:
            stats["following"] = int(following_match.group(1))
        
        return stats
    
    async def _get_competitions(self, username: str) -> List[KaggleCompetition]:
        """Get user's competition history."""
        try:
            response = await self.get(f"/{username}/competitions")
            html_content = response.get("content", "")
            
            # Parse competitions from HTML
            competitions = self._parse_competitions_html(html_content)
            return competitions
            
        except APIError as e:
            logger.warning(f"Failed to get competitions: {e.message}")
            return []
    
    def _parse_competitions_html(self, html_content: str) -> List[KaggleCompetition]:
        """Parse competitions from HTML content."""
        # Simplified implementation
        competitions = []
        
        # This would require parsing the competitions table
        # For now, return empty list as placeholder
        logger.warning("Competition parsing not fully implemented - using mock data")
        
        return competitions
    
    async def _get_datasets(self, username: str) -> List[KaggleDataset]:
        """Get user's datasets."""
        try:
            response = await self.get(f"/{username}/datasets")
            html_content = response.get("content", "")
            
            # Parse datasets from HTML
            datasets = self._parse_datasets_html(html_content)
            return datasets
            
        except APIError as e:
            logger.warning(f"Failed to get datasets: {e.message}")
            return []
    
    def _parse_datasets_html(self, html_content: str) -> List[KaggleDataset]:
        """Parse datasets from HTML content."""
        # Simplified implementation
        datasets = []
        
        # This would require parsing the datasets section
        # For now, return empty list as placeholder
        logger.warning("Dataset parsing not fully implemented - using mock data")
        
        return datasets
    
    async def _get_notebooks(self, username: str) -> List[KaggleNotebook]:
        """Get user's notebooks."""
        try:
            response = await self.get(f"/{username}/notebooks")
            html_content = response.get("content", "")
            
            # Parse notebooks from HTML
            notebooks = self._parse_notebooks_html(html_content)
            return notebooks
            
        except APIError as e:
            logger.warning(f"Failed to get notebooks: {e.message}")
            return []
    
    def _parse_notebooks_html(self, html_content: str) -> List[KaggleNotebook]:
        """Parse notebooks from HTML content."""
        # Simplified implementation
        notebooks = []
        
        # This would require parsing the notebooks section
        # For now, return empty list as placeholder
        logger.warning("Notebook parsing not fully implemented - using mock data")
        
        return notebooks
    
    async def _get_skills(self, username: str) -> List[str]:
        """Get user's skills."""
        try:
            response = await self.get(f"/{username}")
            html_content = response.get("content", "")
            
            # Parse skills from HTML
            skills = self._parse_skills_html(html_content)
            return skills
            
        except APIError as e:
            logger.warning(f"Failed to get skills: {e.message}")
            return []
    
    def _parse_skills_html(self, html_content: str) -> List[str]:
        """Parse skills from HTML content."""
        skills = []
        
        # Extract skills from profile
        skill_matches = re.findall(r'skill[^>]*>([^<]+)</[^>]*>', html_content, re.IGNORECASE)
        skills.extend([skill.strip() for skill in skill_matches])
        
        return list(set(skills))  # Remove duplicates
    
    async def _get_achievements(self, username: str) -> List[str]:
        """Get user's achievements."""
        try:
            response = await self.get(f"/{username}")
            html_content = response.get("content", "")
            
            # Parse achievements from HTML
            achievements = self._parse_achievements_html(html_content)
            return achievements
            
        except APIError as e:
            logger.warning(f"Failed to get achievements: {e.message}")
            return []
    
    def _parse_achievements_html(self, html_content: str) -> List[str]:
        """Parse achievements from HTML content."""
        achievements = []
        
        # Extract achievement names
        achievement_matches = re.findall(r'achievement[^>]*title="([^"]+)"', html_content, re.IGNORECASE)
        achievements.extend(achievement_matches)
        
        return list(set(achievements))  # Remove duplicates
    
    async def _get_tier_progression(self, username: str) -> Dict[str, Any]:
        """Get user's tier progression information."""
        try:
            response = await self.get(f"/{username}")
            html_content = response.get("content", "")
            
            # Parse tier progression from HTML
            tier_info = self._parse_tier_progression_html(html_content)
            return tier_info
            
        except APIError as e:
            logger.warning(f"Failed to get tier progression: {e.message}")
            return {}
    
    def _parse_tier_progression_html(self, html_content: str) -> Dict[str, Any]:
        """Parse tier progression from HTML content."""
        tier_info = {
            "current_tier": "Novice",
            "next_tier": None,
            "progress_percentage": 0,
            "requirements_met": 0,
            "total_requirements": 0
        }
        
        # Extract current tier
        tier_match = re.search(r'Current.*?tier.*?([A-Za-z]+)', html_content, re.IGNORECASE | re.DOTALL)
        if tier_match:
            tier_info["current_tier"] = tier_match.group(1).strip()
        
        return tier_info
    
    async def validate_username(self, username: str) -> bool:
        """Validate if a Kaggle username exists."""
        try:
            await self._get_user_info(username)
            return True
        except APIError:
            return False
    
    async def get_competition_details(self, competition_name: str) -> Optional[Dict[str, Any]]:
        """Get details about a specific competition."""
        try:
            response = await self.get(f"/c/{competition_name}")
            html_content = response.get("content", "")
            
            # Parse competition details from HTML
            comp_info = self._parse_competition_details_html(html_content)
            return comp_info
            
        except APIError as e:
            logger.warning(f"Failed to get competition details: {e.message}")
            return None
    
    def _parse_competition_details_html(self, html_content: str) -> Dict[str, Any]:
        """Parse competition details from HTML content."""
        comp_info = {
            "title": "Unknown Competition",
            "description": "",
            "prize": None,
            "participants": 0,
            "deadline": None,
            "evaluation": "Unknown",
            "type": "Unknown"
        }
        
        # Extract competition title
        title_match = re.search(r'<h1[^>]*>([^<]+)</h1>', html_content)
        if title_match:
            comp_info["title"] = title_match.group(1).strip()
        
        # Extract prize information
        prize_match = re.search(r'Prize.*?\$([0-9,]+)', html_content, re.IGNORECASE | re.DOTALL)
        if prize_match:
            comp_info["prize"] = f"${prize_match.group(1)}"
        
        # Extract participant count
        participants_match = re.search(r'(\d+)\s+teams?', html_content, re.IGNORECASE)
        if participants_match:
            comp_info["participants"] = int(participants_match.group(1).replace(',', ''))
        
        return comp_info