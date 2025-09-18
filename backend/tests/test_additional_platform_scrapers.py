"""Tests for additional platform scrapers (Codeforces, AtCoder, HackerRank, Kaggle)."""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

from app.services.external_apis.codeforces_scraper import (
    CodeforcesScraper,
    CodeforcesProfile,
    CodeforcesStats,
    CodeforcesContest,
    CodeforcesProblem
)
from app.services.external_apis.atcoder_scraper import (
    AtCoderScraper,
    AtCoderProfile,
    AtCoderStats,
    AtCoderContest,
    AtCoderProblem
)
from app.services.external_apis.hackerrank_scraper import (
    HackerRankScraper,
    HackerRankProfile,
    HackerRankStats,
    HackerRankCertification,
    HackerRankChallenge
)
from app.services.external_apis.kaggle_scraper import (
    KaggleScraper,
    KaggleProfile,
    KaggleStats,
    KaggleCompetition,
    KaggleDataset
)
from app.services.external_apis.base_client import APIError


class TestCodeforcesScraper:
    """Test cases for Codeforces scraper."""
    
    @pytest.fixture
    def scraper(self):
        """Create a CodeforcesScraper instance for testing."""
        return CodeforcesScraper()
    
    @pytest.fixture
    def mock_user_data(self):
        """Mock Codeforces user data."""
        return {
            "handle": "testuser",
            "firstName": "Test",
            "lastName": "User",
            "country": "TestCountry",
            "city": "TestCity",
            "organization": "TestOrg",
            "rating": 1500,
            "maxRating": 1600,
            "rank": "expert",
            "maxRank": "candidate master",
            "contribution": 10,
            "friendOfCount": 5,
            "registrationTimeSeconds": 1234567890
        }
    
    @pytest.fixture
    def mock_contest_data(self):
        """Mock Codeforces contest data."""
        return [
            {
                "contestId": 1234,
                "contestName": "Test Contest",
                "rank": 100,
                "oldRating": 1400,
                "newRating": 1500,
                "ratingUpdateTimeSeconds": 1234567890
            }
        ]
    
    @pytest.fixture
    def mock_submission_data(self):
        """Mock Codeforces submission data."""
        return [
            {
                "problem": {
                    "contestId": 1234,
                    "index": "A",
                    "name": "Test Problem",
                    "tags": ["implementation", "math"],
                    "rating": 1200
                },
                "verdict": "OK",
                "programmingLanguage": "C++17",
                "creationTimeSeconds": 1234567890
            }
        ]
    
    def test_scraper_initialization(self, scraper):
        """Test scraper initialization."""
        assert scraper.base_url == "https://codeforces.com/api"
        assert scraper.timeout == 30.0
    
    @pytest.mark.asyncio
    async def test_get_user_info_success(self, scraper, mock_user_data):
        """Test successful user info retrieval."""
        mock_response = {
            "status": "OK",
            "result": [mock_user_data]
        }
        
        with patch.object(scraper, 'get', return_value=mock_response):
            result = await scraper._get_user_info("testuser")
            
        assert result == mock_user_data
    
    @pytest.mark.asyncio
    async def test_get_user_info_not_found(self, scraper):
        """Test user not found error."""
        mock_response = {
            "status": "OK",
            "result": []
        }
        
        with patch.object(scraper, 'get', return_value=mock_response):
            with pytest.raises(APIError, match="User testuser not found"):
                await scraper._get_user_info("testuser")
    
    @pytest.mark.asyncio
    async def test_get_contest_history(self, scraper, mock_contest_data):
        """Test contest history retrieval."""
        mock_response = {
            "status": "OK",
            "result": mock_contest_data
        }
        
        with patch.object(scraper, 'get', return_value=mock_response):
            contests = await scraper._get_contest_history("testuser")
            
        assert len(contests) == 1
        assert contests[0].contest_id == 1234
        assert contests[0].contest_name == "Test Contest"
        assert contests[0].rating_change == 100
    
    @pytest.mark.asyncio
    async def test_get_solved_problems(self, scraper, mock_submission_data):
        """Test solved problems retrieval."""
        mock_response = {
            "status": "OK",
            "result": mock_submission_data
        }
        
        with patch.object(scraper, 'get', return_value=mock_response):
            problems = await scraper._get_solved_problems("testuser")
            
        assert len(problems) == 1
        assert problems[0].contest_id == 1234
        assert problems[0].problem_index == "A"
        assert problems[0].problem_name == "Test Problem"
        assert "implementation" in problems[0].problem_tags
    
    @pytest.mark.asyncio
    async def test_validate_handle_success(self, scraper, mock_user_data):
        """Test successful handle validation."""
        mock_response = {
            "status": "OK",
            "result": [mock_user_data]
        }
        
        with patch.object(scraper, 'get', return_value=mock_response):
            is_valid = await scraper.validate_handle("testuser")
            
        assert is_valid is True
    
    @pytest.mark.asyncio
    async def test_validate_handle_failure(self, scraper):
        """Test handle validation failure."""
        with patch.object(scraper, 'get', side_effect=APIError("User not found")):
            is_valid = await scraper.validate_handle("nonexistent")
            
        assert is_valid is False
    
    def test_analyze_problem_tags(self, scraper):
        """Test problem tags analysis."""
        problems = [
            CodeforcesProblem(
                contest_id=1,
                problem_index="A",
                problem_name="Test",
                problem_tags=["math", "implementation"],
                solved_at=datetime.now(),
                language="C++",
                verdict="OK"
            ),
            CodeforcesProblem(
                contest_id=2,
                problem_index="B",
                problem_name="Test2",
                problem_tags=["math", "greedy"],
                solved_at=datetime.now(),
                language="Python",
                verdict="OK"
            )
        ]
        
        tag_counts = scraper._analyze_problem_tags(problems)
        
        assert tag_counts["math"] == 2
        assert tag_counts["implementation"] == 1
        assert tag_counts["greedy"] == 1
    
    def test_analyze_languages(self, scraper):
        """Test programming languages analysis."""
        problems = [
            CodeforcesProblem(
                contest_id=1,
                problem_index="A",
                problem_name="Test",
                problem_tags=[],
                solved_at=datetime.now(),
                language="C++17",
                verdict="OK"
            ),
            CodeforcesProblem(
                contest_id=2,
                problem_index="B",
                problem_name="Test2",
                problem_tags=[],
                solved_at=datetime.now(),
                language="C++17",
                verdict="OK"
            ),
            CodeforcesProblem(
                contest_id=3,
                problem_index="C",
                problem_name="Test3",
                problem_tags=[],
                solved_at=datetime.now(),
                language="Python3",
                verdict="OK"
            )
        ]
        
        lang_counts = scraper._analyze_languages(problems)
        
        assert lang_counts["C++17"] == 2
        assert lang_counts["Python3"] == 1


class TestAtCoderScraper:
    """Test cases for AtCoder scraper."""
    
    @pytest.fixture
    def scraper(self):
        """Create an AtCoderScraper instance for testing."""
        return AtCoderScraper()
    
    def test_scraper_initialization(self, scraper):
        """Test scraper initialization."""
        assert scraper.base_url == "https://atcoder.jp"
        assert scraper.timeout == 30.0
    
    @pytest.mark.asyncio
    async def test_validate_username_success(self, scraper):
        """Test successful username validation."""
        mock_response = {"content": "<html>Valid user page</html>"}
        
        with patch.object(scraper, 'get', return_value=mock_response):
            is_valid = await scraper.validate_username("testuser")
            
        assert is_valid is True
    
    @pytest.mark.asyncio
    async def test_validate_username_failure(self, scraper):
        """Test username validation failure."""
        with patch.object(scraper, 'get', side_effect=APIError("User not found", status_code=404)):
            is_valid = await scraper.validate_username("nonexistent")
            
        assert is_valid is False
    
    def test_parse_user_info_html(self, scraper):
        """Test HTML parsing for user info."""
        html_content = """
        <html>
            <td>Country</td><td>Japan</td>
            <td>Birth Year</td><td>1990</td>
            <td>Twitter</td><td>@testuser</td>
        </html>
        """
        
        user_info = scraper._parse_user_info_html(html_content)
        
        assert user_info.get("country") == "Japan"
        assert user_info.get("birth_year") == 1990
        assert user_info.get("twitter_id") == "testuser"
    
    def test_parse_stats_html(self, scraper):
        """Test HTML parsing for statistics."""
        html_content = """
        <html>
            <span>1500</span>
            <span class="user-blue">testuser</span>
            Rated Matches 50
        </html>
        """
        
        stats_data = scraper._parse_stats_html(html_content)
        
        # Basic structure test - actual parsing would be more complex
        assert isinstance(stats_data, dict)
    
    def test_analyze_difficulty_distribution(self, scraper):
        """Test difficulty distribution analysis."""
        problems = [
            AtCoderProblem(
                contest_name="ABC123",
                problem_id="A",
                problem_title="Test",
                difficulty="Easy",
                solved_at=datetime.now(),
                language="C++"
            ),
            AtCoderProblem(
                contest_name="ABC124",
                problem_id="B",
                problem_title="Test2",
                difficulty="Easy",
                solved_at=datetime.now(),
                language="Python"
            ),
            AtCoderProblem(
                contest_name="ABC125",
                problem_id="C",
                problem_title="Test3",
                difficulty="Medium",
                solved_at=datetime.now(),
                language="C++"
            )
        ]
        
        difficulty_dist = scraper._analyze_difficulty_distribution(problems)
        
        assert difficulty_dist["Easy"] == 2
        assert difficulty_dist["Medium"] == 1


class TestHackerRankScraper:
    """Test cases for HackerRank scraper."""
    
    @pytest.fixture
    def scraper(self):
        """Create a HackerRankScraper instance for testing."""
        return HackerRankScraper()
    
    def test_scraper_initialization(self, scraper):
        """Test scraper initialization."""
        assert scraper.base_url == "https://www.hackerrank.com"
        assert scraper.timeout == 30.0
    
    @pytest.mark.asyncio
    async def test_validate_username_success(self, scraper):
        """Test successful username validation."""
        mock_response = {"content": "<html>Valid user page</html>"}
        
        with patch.object(scraper, 'get', return_value=mock_response):
            is_valid = await scraper.validate_username("testuser")
            
        assert is_valid is True
    
    @pytest.mark.asyncio
    async def test_validate_username_failure(self, scraper):
        """Test username validation failure."""
        with patch.object(scraper, 'get', side_effect=APIError("User not found", status_code=404)):
            is_valid = await scraper.validate_username("nonexistent")
            
        assert is_valid is False
    
    def test_parse_user_info_html(self, scraper):
        """Test HTML parsing for user info."""
        html_content = """
        <html>
            <h1 class="profile-heading">Test User</h1>
            <div>Country</div><div>USA</div>
            <div>Company</div><div>TestCorp</div>
            <div>School</div><div>TestUniv</div>
            <img class="avatar" src="avatar.jpg">
        </html>
        """
        
        user_info = scraper._parse_user_info_html(html_content)
        
        assert user_info.get("name") == "Test User"
        assert user_info.get("country") == "USA"
        assert user_info.get("company") == "TestCorp"
        assert user_info.get("school") == "TestUniv"
        assert user_info.get("avatar") == "avatar.jpg"
    
    def test_parse_stats_html(self, scraper):
        """Test HTML parsing for statistics."""
        html_content = """
        <html>
            Total Score 1500
            Challenges Solved 100
            Certifications 5
            Rank 1000
            Badges 10
        </html>
        """
        
        stats_data = scraper._parse_stats_html(html_content)
        
        assert stats_data.get("total_score") == 1500
        assert stats_data.get("challenges_solved") == 100
        assert stats_data.get("certifications_earned") == 5
        assert stats_data.get("rank") == 1000
        assert stats_data.get("badges_earned") == 10
    
    def test_analyze_domain_scores(self, scraper):
        """Test domain scores analysis."""
        challenges = [
            HackerRankChallenge(
                challenge_name="Test1",
                domain="Algorithms",
                subdomain="Sorting",
                difficulty="Easy",
                score=10,
                max_score=10,
                language="Python",
                solved_at=datetime.now()
            ),
            HackerRankChallenge(
                challenge_name="Test2",
                domain="Algorithms",
                subdomain="Graph Theory",
                difficulty="Medium",
                score=20,
                max_score=20,
                language="C++",
                solved_at=datetime.now()
            ),
            HackerRankChallenge(
                challenge_name="Test3",
                domain="Data Structures",
                subdomain="Arrays",
                difficulty="Easy",
                score=15,
                max_score=15,
                language="Java",
                solved_at=datetime.now()
            )
        ]
        
        domain_scores = scraper._analyze_domain_scores(challenges)
        
        assert domain_scores["Algorithms"] == 30
        assert domain_scores["Data Structures"] == 15
    
    def test_analyze_skill_levels(self, scraper):
        """Test skill levels analysis."""
        certifications = [
            HackerRankCertification(
                skill="Python",
                level="Intermediate",
                earned_at=datetime.now(),
                score=80,
                max_score=100
            ),
            HackerRankCertification(
                skill="Java",
                level="Basic",
                earned_at=datetime.now(),
                score=70,
                max_score=100
            )
        ]
        
        skill_levels = scraper._analyze_skill_levels(certifications)
        
        assert skill_levels["Python"] == "Intermediate"
        assert skill_levels["Java"] == "Basic"


class TestKaggleScraper:
    """Test cases for Kaggle scraper."""
    
    @pytest.fixture
    def scraper(self):
        """Create a KaggleScraper instance for testing."""
        return KaggleScraper()
    
    def test_scraper_initialization(self, scraper):
        """Test scraper initialization."""
        assert scraper.base_url == "https://www.kaggle.com"
        assert scraper.timeout == 30.0
    
    @pytest.mark.asyncio
    async def test_validate_username_success(self, scraper):
        """Test successful username validation."""
        mock_response = {"content": "<html>Valid user page</html>"}
        
        with patch.object(scraper, 'get', return_value=mock_response):
            is_valid = await scraper.validate_username("testuser")
            
        assert is_valid is True
    
    @pytest.mark.asyncio
    async def test_validate_username_failure(self, scraper):
        """Test username validation failure."""
        with patch.object(scraper, 'get', side_effect=APIError("User not found", status_code=404)):
            is_valid = await scraper.validate_username("nonexistent")
            
        assert is_valid is False
    
    def test_parse_user_info_html(self, scraper):
        """Test HTML parsing for user info."""
        html_content = """
        <html>
            <h1>Test User</h1>
            <div class="bio">Data Scientist</div>
            <span>Location</span><span>San Francisco</span>
            <a href="https://twitter.com/testuser">Twitter</a>
            <a href="https://github.com/testuser">GitHub</a>
            <a href="https://linkedin.com/in/testuser">LinkedIn</a>
            <img class="avatar" src="avatar.jpg">
        </html>
        """
        
        user_info = scraper._parse_user_info_html(html_content)
        
        assert user_info.get("display_name") == "Test User"
        assert user_info.get("bio") == "Data Scientist"
        assert user_info.get("twitter") == "testuser"
        assert user_info.get("github") == "testuser"
        assert user_info.get("linkedin") == "testuser"
        assert user_info.get("avatar") == "avatar.jpg"
    
    def test_parse_stats_html(self, scraper):
        """Test HTML parsing for statistics."""
        html_content = """
        <html>
            <div class="tier">Expert</div>
            Competitions 25
            Datasets 10
            Notebooks 50
            Gold 2
            Silver 5
            Bronze 8
            Followers 1000
            Following 500
        </html>
        """
        
        stats_data = scraper._parse_stats_html(html_content)
        
        assert stats_data.get("current_tier") == "Expert"
        assert stats_data.get("competitions_entered") == 25
        assert stats_data.get("datasets_created") == 10
        assert stats_data.get("notebooks_created") == 50
        assert stats_data.get("gold_medals") == 2
        assert stats_data.get("silver_medals") == 5
        assert stats_data.get("bronze_medals") == 8
        assert stats_data.get("total_medals") == 15
        assert stats_data.get("followers") == 1000
        assert stats_data.get("following") == 500
    
    def test_parse_skills_html(self, scraper):
        """Test HTML parsing for skills."""
        html_content = """
        <html>
            <div class="skill">Machine Learning</div>
            <div class="skill">Deep Learning</div>
            <div class="skill">Python</div>
        </html>
        """
        
        skills = scraper._parse_skills_html(html_content)
        
        assert "Machine Learning" in skills
        assert "Deep Learning" in skills
        assert "Python" in skills
    
    def test_parse_achievements_html(self, scraper):
        """Test HTML parsing for achievements."""
        html_content = """
        <html>
            <div class="achievement" title="Competition Expert">Expert</div>
            <div class="achievement" title="Notebook Master">Master</div>
        </html>
        """
        
        achievements = scraper._parse_achievements_html(html_content)
        
        assert "Competition Expert" in achievements
        assert "Notebook Master" in achievements


class TestPlatformModels:
    """Test platform-specific data models."""
    
    def test_codeforces_profile_model(self):
        """Test CodeforcesProfile model."""
        stats = CodeforcesStats(
            current_rating=1500,
            max_rating=1600,
            rank="expert",
            max_rank="candidate master",
            contests_participated=50,
            problems_solved=200,
            contribution=10,
            friend_count=5,
            registration_date=datetime.now()
        )
        
        profile = CodeforcesProfile(
            handle="testuser",
            first_name="Test",
            last_name="User",
            country="TestCountry",
            stats=stats
        )
        
        assert profile.handle == "testuser"
        assert profile.stats.current_rating == 1500
        assert profile.stats.rank == "expert"
    
    def test_atcoder_profile_model(self):
        """Test AtCoderProfile model."""
        stats = AtCoderStats(
            current_rating=1200,
            max_rating=1300,
            rank="brown",
            contests_participated=20,
            problems_solved=100,
            submissions_count=150,
            accepted_count=100,
            acceptance_rate=66.7
        )
        
        profile = AtCoderProfile(
            username="testuser",
            country="Japan",
            stats=stats
        )
        
        assert profile.username == "testuser"
        assert profile.stats.current_rating == 1200
        assert profile.stats.acceptance_rate == 66.7
    
    def test_hackerrank_profile_model(self):
        """Test HackerRankProfile model."""
        stats = HackerRankStats(
            total_score=1000,
            challenges_solved=50,
            certifications_earned=3,
            contests_participated=10,
            badges_earned=5,
            submissions_count=100,
            languages_used=3,
            domains_active=5
        )
        
        profile = HackerRankProfile(
            username="testuser",
            name="Test User",
            country="USA",
            stats=stats
        )
        
        assert profile.username == "testuser"
        assert profile.stats.total_score == 1000
        assert profile.stats.certifications_earned == 3
    
    def test_kaggle_profile_model(self):
        """Test KaggleProfile model."""
        stats = KaggleStats(
            competitions_entered=15,
            competitions_won=2,
            current_tier="Expert",
            datasets_created=5,
            notebooks_created=20,
            total_votes=100,
            total_medals=8,
            gold_medals=2,
            silver_medals=3,
            bronze_medals=3,
            followers=500,
            following=200
        )
        
        profile = KaggleProfile(
            username="testuser",
            display_name="Test User",
            location="San Francisco",
            stats=stats
        )
        
        assert profile.username == "testuser"
        assert profile.stats.current_tier == "Expert"
        assert profile.stats.total_medals == 8


if __name__ == "__main__":
    pytest.main([__file__])