# Vector Database Integration

This module provides comprehensive vector database integration for the AI Career Recommender system, enabling semantic search and similarity matching across profiles, jobs, skills, and learning resources.

## Overview

The vector database integration consists of several key components:

- **Base Interface**: Abstract base class defining common operations
- **Provider Implementations**: Pinecone and Weaviate clients
- **Manager**: Factory and connection management
- **Embedding Service**: High-level embedding generation and storage
- **Semantic Search**: Intelligent search and matching capabilities

## Supported Providers

### Pinecone
- **Type**: Managed vector database service
- **Pros**: Easy setup, serverless, good performance
- **Cons**: Requires API key, limited free tier
- **Best for**: Production deployments, scalable applications

### Weaviate
- **Type**: Open-source vector database
- **Pros**: Self-hosted, feature-rich, GraphQL API
- **Cons**: Requires infrastructure setup
- **Best for**: On-premise deployments, custom requirements

## Configuration

Set the following environment variables:

```bash
# Provider selection
VECTOR_DB_PROVIDER="pinecone"  # or "weaviate"

# Pinecone configuration
PINECONE_API_KEY="your-api-key"
PINECONE_ENVIRONMENT="us-west1-gcp-free"
PINECONE_INDEX_NAME="ai-career-recommender"

# Weaviate configuration
WEAVIATE_URL="http://localhost:8080"
WEAVIATE_API_KEY="optional-api-key"

# Embedding configuration
EMBEDDING_DIMENSION=384
EMBEDDING_MODEL="sentence-transformers/all-MiniLM-L6-v2"
VECTOR_DB_BATCH_SIZE=100
VECTOR_DB_MAX_RETRIES=3
VECTOR_DB_TIMEOUT=30
```

## Usage Examples

### Basic Embedding Operations

```python
from app.services.embedding_service import embedding_service

# Initialize service
await embedding_service.initialize()

# Generate embedding
embedding = await embedding_service.generate_embedding("Python programming")

# Store profile embedding
profile_data = {
    "skills": ["Python", "Machine Learning"],
    "experience_level": "senior",
    "dream_job": "ML Engineer"
}
await embedding_service.store_profile_embedding("user123", profile_data)
```

### Semantic Search

```python
from app.services.semantic_search_service import semantic_search_service, SearchRequest, SearchType

# Search for similar profiles
request = SearchRequest(
    search_type=SearchType.PROFILE_SIMILARITY,
    query_data={"user_id": "user123"},
    top_k=10,
    min_score=0.7
)

matches = await semantic_search_service.search(request)
for match in matches:
    print(f"Similar user: {match.id} (score: {match.score})")
```

### Job Matching

```python
# Find matching jobs for a user
request = SearchRequest(
    search_type=SearchType.JOB_MATCHING,
    query_data={"user_id": "user123"},
    top_k=20
)

job_matches = await semantic_search_service.search(request)
for match in job_matches:
    job_title = match.metadata.get("title")
    company = match.metadata.get("company")
    print(f"Job match: {job_title} at {company} (score: {match.score})")
```

### Learning Resource Recommendations

```python
# Find learning resources for skill gaps
request = SearchRequest(
    search_type=SearchType.LEARNING_RESOURCES,
    query_data={"skill_gaps": ["Machine Learning", "Deep Learning"]},
    top_k=15
)

resources = await semantic_search_service.search(request)
for resource in resources:
    title = resource.metadata.get("title")
    provider = resource.metadata.get("provider")
    print(f"Resource: {title} by {provider}")
```

## API Endpoints

The vector search functionality is exposed through REST API endpoints:

### Embedding Management
- `POST /api/v1/vector-search/embeddings/generate` - Generate embedding for text
- `POST /api/v1/vector-search/embeddings/profiles` - Store profile embedding
- `POST /api/v1/vector-search/embeddings/jobs` - Store job embedding
- `POST /api/v1/vector-search/embeddings/skills` - Store skill embedding
- `POST /api/v1/vector-search/embeddings/resources` - Store resource embedding

### Semantic Search
- `POST /api/v1/vector-search/search` - General semantic search
- `GET /api/v1/vector-search/search/similar-profiles/{user_id}` - Find similar profiles
- `GET /api/v1/vector-search/search/matching-jobs/{user_id}` - Find matching jobs
- `GET /api/v1/vector-search/search/similar-skills/{skill_name}` - Find similar skills
- `POST /api/v1/vector-search/search/learning-resources` - Find learning resources

### Management
- `DELETE /api/v1/vector-search/embeddings/{index_type}/{item_id}` - Delete embedding
- `GET /api/v1/vector-search/health` - Health check
- `POST /api/v1/vector-search/search/batch` - Batch search

## Data Models

### Vector Record
```python
@dataclass
class VectorRecord:
    id: str
    vector: Union[List[float], np.ndarray]
    metadata: Dict[str, Any]
    namespace: Optional[str] = None
```

### Search Result
```python
@dataclass
class SearchResult:
    id: str
    score: float
    metadata: Dict[str, Any]
    namespace: Optional[str] = None
```

### Search Match (Enhanced)
```python
@dataclass
class SearchMatch:
    id: str
    score: float
    metadata: Dict[str, Any]
    match_reasons: List[str]
    confidence_level: str  # 'high', 'medium', 'low'
    search_type: SearchType
```

## Index Structure

The system creates separate indexes for different data types:

### Profiles Index
- **Purpose**: Store user profile embeddings
- **Metadata**: user_id, experience_level, primary_skills, created_at
- **Use Cases**: Find similar users, career path recommendations

### Jobs Index
- **Purpose**: Store job posting embeddings
- **Metadata**: job_id, title, company, location, experience_level, salary_range
- **Use Cases**: Job matching, market analysis

### Skills Index
- **Purpose**: Store skill embeddings
- **Metadata**: skill_name, category, popularity_score
- **Use Cases**: Skill similarity, gap analysis

### Learning Resources Index
- **Purpose**: Store learning resource embeddings
- **Metadata**: resource_id, type, provider, difficulty_level, rating
- **Use Cases**: Course recommendations, learning path generation

## Performance Considerations

### Embedding Generation
- Uses sentence-transformers for high-quality embeddings
- Implements caching to avoid regenerating embeddings
- Supports batch processing for efficiency

### Vector Storage
- Batch upserts for better performance
- Configurable batch sizes
- Retry mechanisms for reliability

### Search Optimization
- Efficient similarity computation
- Metadata filtering to reduce search space
- Configurable result limits

### Caching Strategy
- In-memory caching for frequently accessed embeddings
- Redis caching for API responses
- TTL-based cache invalidation

## Error Handling

The system implements comprehensive error handling:

- **Connection Errors**: Automatic retry with exponential backoff
- **API Limits**: Rate limiting and queuing mechanisms
- **Data Validation**: Input validation and sanitization
- **Graceful Degradation**: Fallback to cached results when possible

## Monitoring and Health Checks

### Health Check Endpoint
```python
GET /api/v1/vector-search/health
```

Returns:
```json
{
  "status": "healthy",
  "provider": "pinecone",
  "clients": {
    "profiles": {"status": "healthy", "vector_count": 1000},
    "jobs": {"status": "healthy", "vector_count": 5000},
    "skills": {"status": "healthy", "vector_count": 500},
    "learning_resources": {"status": "healthy", "vector_count": 2000}
  }
}
```

### Metrics
- Embedding generation latency
- Search query performance
- Index size and growth
- Error rates and types

## Testing

Run the comprehensive test suite:

```bash
# Unit tests
pytest backend/tests/test_vector_db_integration.py -v

# Integration tests (requires vector database)
pytest backend/tests/test_vector_db_integration.py -v -m integration

# Demo script
python backend/examples/vector_db_demo.py
```

## Troubleshooting

### Common Issues

1. **Connection Failures**
   - Check API keys and credentials
   - Verify network connectivity
   - Review firewall settings

2. **Slow Performance**
   - Increase batch sizes
   - Check index size and fragmentation
   - Monitor resource usage

3. **Search Quality Issues**
   - Verify embedding model compatibility
   - Check data quality and preprocessing
   - Adjust similarity thresholds

4. **Memory Issues**
   - Reduce batch sizes
   - Implement streaming for large datasets
   - Monitor memory usage patterns

### Debug Mode

Enable debug logging:
```bash
LOG_LEVEL=DEBUG python your_app.py
```

### Performance Profiling

Use the built-in profiling tools:
```python
from app.services.vector_db.vector_db_manager import vector_db_manager

# Get detailed health information
health = await vector_db_manager.health_check()
print(health)
```

## Future Enhancements

- Support for additional vector database providers (Qdrant, Milvus)
- Advanced filtering and faceted search
- Real-time embedding updates
- Distributed search across multiple indexes
- Machine learning model versioning and A/B testing
- Advanced analytics and search insights