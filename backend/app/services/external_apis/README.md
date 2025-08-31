# External API Integration Services

This module provides comprehensive integration with external platforms for extracting user profile data, including GitHub, LeetCode, and LinkedIn. The services include intelligent retry mechanisms, data validation, caching, and graceful error handling.

## Features

- **Multi-platform Integration**: GitHub, LeetCode, and LinkedIn profile extraction
- **Intelligent Retry Logic**: Exponential backoff with jitter for handling rate limits
- **Data Validation & Cleaning**: Comprehensive validation and normalization of extracted data
- **Caching**: Configurable caching to reduce API calls and improve performance
- **Graceful Degradation**: Continue operation even when some services fail
- **Comprehensive Error Handling**: Detailed error reporting and recovery strategies
- **Rate Limit Compliance**: Respect platform rate limits and ToS requirements

## Architecture

```
external_apis/
├── __init__.py                 # Module exports
├── base_client.py             # Base HTTP client with retry logic
├── github_client.py           # GitHub API integration
├── leetcode_scraper.py        # LeetCode data scraping
├── linkedin_scraper.py        # LinkedIn profile extraction
├── data_validator.py          # Data validation and cleaning
├── integration_service.py     # Orchestration service
└── README.md                  # This file
```

## Quick Start

### Basic Usage

```python
from app.services.external_apis import ExternalAPIIntegrationService, ProfileExtractionRequest

# Initialize service
service = ExternalAPIIntegrationService(github_token="your_token_here")

# Create extraction request
request = ProfileExtractionRequest(
    github_username="octocat",
    leetcode_username="example_user",
    linkedin_url="https://linkedin.com/in/example",
    timeout_seconds=60,
    enable_validation=True
)

# Extract profiles
result = await service.extract_comprehensive_profile(request)

if result.success:
    print(f"Extracted data from {len(result.sources_successful)} sources")
    if result.github_profile:
        print(f"GitHub: {result.github_profile['username']}")
    if result.leetcode_profile:
        print(f"LeetCode: {result.leetcode_profile['stats']['total_solved']} problems solved")
```

### Individual Client Usage

```python
from app.services.external_apis import GitHubClient, LeetCodeScraper

# GitHub API
async with GitHubClient(api_token="token") as client:
    profile = await client.get_user_profile("username")
    print(f"User has {profile.public_repos} repositories")

# LeetCode Scraper
async with LeetCodeScraper() as scraper:
    profile = await scraper.get_user_profile("username")
    print(f"Solved {profile.stats.total_solved} problems")
```

## Configuration

Configure the services using environment variables or the configuration class:

```python
from app.core.external_api_config import external_api_config

# Available configuration options:
# - GITHUB_TOKEN: GitHub API token for higher rate limits
# - GITHUB_TIMEOUT: Request timeout in seconds (default: 30)
# - LEETCODE_TIMEOUT: Request timeout in seconds (default: 30)
# - LINKEDIN_TIMEOUT: Request timeout in seconds (default: 30)
# - ENABLE_API_CACHING: Enable response caching (default: True)
# - API_CACHE_TTL_SECONDS: Cache TTL in seconds (default: 3600)
# - RATE_LIMIT_MAX_RETRIES: Maximum retry attempts (default: 3)
# - RATE_LIMIT_BASE_DELAY: Base delay between retries (default: 1.0)
```

## API Endpoints

The module provides REST API endpoints for external profile extraction:

### Extract Profiles
```http
POST /api/v1/external-profiles/extract
Content-Type: application/json

{
    "github_username": "octocat",
    "leetcode_username": "example_user",
    "linkedin_url": "https://linkedin.com/in/example",
    "timeout_seconds": 60,
    "enable_validation": true,
    "enable_graceful_degradation": true
}
```

### Validate Profile Sources
```http
POST /api/v1/external-profiles/validate
Content-Type: application/json

{
    "github_username": "octocat",
    "leetcode_username": "example_user",
    "linkedin_url": "https://linkedin.com/in/example"
}
```

### Cache Management
```http
GET /api/v1/external-profiles/cache/stats
DELETE /api/v1/external-profiles/cache/clear
```

## Data Models

### GitHub Profile
```python
{
    "username": "octocat",
    "name": "The Octocat",
    "bio": "GitHub's mascot",
    "company": "GitHub",
    "public_repos": 8,
    "followers": 4000,
    "languages": {"JavaScript": 1000, "Python": 800},
    "repositories": [...],
    "total_stars": 150,
    "contribution_years": [2020, 2021, 2022, 2023]
}
```

### LeetCode Profile
```python
{
    "username": "example_user",
    "real_name": "John Doe",
    "stats": {
        "total_solved": 150,
        "easy_solved": 80,
        "medium_solved": 60,
        "hard_solved": 10,
        "acceptance_rate": 85.5
    },
    "skill_tags": {"Array": 25, "Dynamic Programming": 15},
    "languages_used": {"Python": 100, "Java": 50}
}
```

### LinkedIn Profile
```python
{
    "name": "John Doe",
    "headline": "Software Engineer at Tech Company",
    "current_company": "Tech Company",
    "experience": [...],
    "skills": [...],
    "profile_url": "https://linkedin.com/in/johndoe"
}
```

## Error Handling

The services implement comprehensive error handling:

### Error Types
- **APIError**: General API errors with status codes
- **RateLimitError**: Rate limit exceeded errors with retry information
- **ValidationError**: Data validation failures

### Graceful Degradation
When `enable_graceful_degradation=True`, the service will:
- Continue processing even if some sources fail
- Return partial data with error information
- Provide detailed error messages for debugging

### Example Error Response
```python
{
    "success": True,  # Partial success
    "github_profile": {...},  # Successfully extracted
    "leetcode_profile": None,  # Failed to extract
    "errors": {
        "leetcode": "User not found"
    },
    "warnings": ["LeetCode profile data quality is low"],
    "sources_successful": ["github"],
    "sources_attempted": ["github", "leetcode"]
}
```

## Data Validation

The `DataValidator` class provides comprehensive data cleaning and validation:

### Features
- **Skill Normalization**: Standardize skill names (e.g., "javascript" → "JavaScript")
- **Company Name Cleaning**: Remove common suffixes and normalize names
- **Language Mapping**: Consistent programming language names
- **Data Quality Scoring**: Assign quality scores (HIGH/MEDIUM/LOW/INVALID)
- **Error Detection**: Identify invalid or suspicious data

### Example Usage
```python
from app.services.external_apis import DataValidator

validator = DataValidator()
result = validator.validate_github_profile(profile)

if result.is_valid:
    cleaned_data = result.cleaned_data
    quality = result.quality  # DataQuality.HIGH/MEDIUM/LOW
    confidence = result.confidence_score  # 0.0 to 1.0
```

## Rate Limiting & Compliance

### GitHub API
- Requires authentication token for higher rate limits
- Implements intelligent retry with exponential backoff
- Respects GitHub's rate limit headers

### LeetCode Scraping
- Uses GraphQL endpoints where possible
- Implements conservative rate limiting
- May be blocked by anti-bot measures

### LinkedIn Scraping
- **WARNING**: LinkedIn heavily restricts scraping
- Current implementation returns mock data
- Consider using LinkedIn's official APIs for production
- Respects robots.txt and ToS requirements

## Caching

The service includes intelligent caching to improve performance:

### Features
- **Configurable TTL**: Default 1 hour, configurable per environment
- **Source-specific Caching**: Separate cache keys for each platform
- **Automatic Expiration**: Expired entries are automatically refreshed
- **Cache Statistics**: Monitor cache hit rates and performance

### Cache Management
```python
# Get cache statistics
stats = service.get_cache_stats()
print(f"Cache hit rate: {stats['valid_entries']}/{stats['total_entries']}")

# Clear cache
service.clear_cache()
```

## Testing

The module includes comprehensive tests:

```bash
# Run all tests
pytest tests/test_external_apis.py -v

# Run specific test categories
pytest tests/test_external_apis.py::TestGitHubClient -v
pytest tests/test_external_apis.py::TestDataValidator -v

# Run integration tests (requires network access)
pytest tests/test_external_apis.py -m integration -v
```

## Security Considerations

### API Keys
- Store GitHub tokens securely in environment variables
- Never commit API keys to version control
- Use different tokens for development and production

### Rate Limiting
- Implement proper backoff strategies
- Monitor API usage to avoid hitting limits
- Consider using multiple tokens for higher throughput

### Data Privacy
- Cache only necessary data
- Implement data retention policies
- Respect user privacy and platform ToS

## Troubleshooting

### Common Issues

#### GitHub Rate Limits
```python
# Solution: Use authentication token
client = GitHubClient(api_token="your_token_here")
```

#### LeetCode Scraping Blocked
```python
# LeetCode may block scraping attempts
# Consider using official APIs when available
# Implement longer delays between requests
```

#### LinkedIn Access Denied
```python
# LinkedIn heavily restricts scraping
# Current implementation uses mock data
# Consider LinkedIn's official APIs for production use
```

### Debug Mode
Enable debug logging to troubleshoot issues:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# This will show detailed request/response information
```

## Performance Optimization

### Best Practices
1. **Use Caching**: Enable caching for frequently accessed profiles
2. **Batch Requests**: Process multiple profiles concurrently
3. **Optimize Timeouts**: Set appropriate timeouts for your use case
4. **Monitor Performance**: Track extraction times and success rates

### Example Concurrent Processing
```python
import asyncio

async def extract_multiple_profiles(usernames):
    tasks = []
    for username in usernames:
        request = ProfileExtractionRequest(github_username=username)
        task = service.extract_comprehensive_profile(request)
        tasks.append(task)
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    return results
```

## Contributing

When contributing to this module:

1. **Follow the existing patterns** for error handling and validation
2. **Add comprehensive tests** for new functionality
3. **Update documentation** for any API changes
4. **Respect platform ToS** and rate limiting requirements
5. **Consider privacy implications** of data extraction

## License & Legal

This module is designed to respect the Terms of Service of all integrated platforms:

- **GitHub**: Uses official API with proper authentication
- **LeetCode**: Uses public GraphQL endpoints with rate limiting
- **LinkedIn**: Current implementation uses mock data due to ToS restrictions

Always review and comply with platform-specific terms of service before using in production.