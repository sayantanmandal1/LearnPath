"""
Vector Database Integration Tests

Comprehensive tests for vector database functionality including
embedding generation, storage, retrieval, and semantic search.
"""

import pytest
import asyncio
import numpy as np
from unittest.mock import Mock, patch, AsyncMock
from typing import List, Dict, Any

from app.services.vector_db.base_vector_db import VectorRecord, SearchResult, QueryFilter
from app.services.vector_db.pinecone_client import PineconeClient
from app.services.vector_db.weaviate_client import WeaviateClient
from app.services.vector_db.vector_db_manager import VectorDBManager
from app.services.embedding_service import EmbeddingService
from app.services.semantic_search_service import SemanticSearchService, SearchRequest, SearchType
from app.core.vector_db_config import VectorDBConfig


class TestVectorDBBase:
    """Base test class with common fixtures and utilities."""
    
    @pytest.fixture
    def mock_config(self):
        """Mock vector database configuration."""
        return {
            "api_key": "test-api-key",
            "environment": "test-env",
            "timeout": 30,
            "max_retries": 3,
            "batch_size": 100
        }
    
    @pytest.fixture
    def sample_vector_record(self):
        """Sample vector record for testing."""
        return VectorRecord(
            id="test-id-1",
            vector=np.random.rand(384).tolist(),
            metadata={
                "user_id": "user123",
                "skills": ["Python", "Machine Learning"],
                "experience_level": "senior"
            }
        )
    
    @pytest.fixture
    def sample_search_results(self):
        """Sample search results for testing."""
        return [
            SearchResult(
                id="result-1",
                score=0.95,
                metadata={"title": "Software Engineer", "company": "TechCorp"},
                namespace=None
            ),
            SearchResult(
                id="result-2", 
                score=0.87,
                metadata={"title": "Data Scientist", "company": "DataCorp"},
                namespace=None
            )
        ]


class TestPineconeClient(TestVectorDBBase):
    """Test Pinecone vector database client."""
    
    @pytest.fixture
    def pinecone_client(self, mock_config):
        """Create Pinecone client with mocked dependencies."""
        with patch('app.services.vector_db.pinecone_client.pinecone') as mock_pinecone:
            mock_pinecone.Pinecone = Mock()
            mock_pinecone.ServerlessSpec = Mock()
            client = PineconeClient(mock_config)
            return client
    
    @pytest.mark.asyncio
    async def test_connect(self, pinecone_client):
        """Test Pinecone connection."""
        with patch('app.services.vector_db.pinecone_client.Pinecone') as mock_pinecone_class:
            mock_client = Mock()
            mock_pinecone_class.return_value = mock_client
            
            await pinecone_client.connect()
            
            assert pinecone_client._client == mock_client
            mock_pinecone_class.assert_called_once_with(api_key="test-api-key")
    
    @pytest.mark.asyncio
    async def test_create_index(self, pinecone_client):
        """Test index creation."""
        mock_client = Mock()
        mock_client.list_indexes.return_value = []
        mock_client.describe_index.return_value.status = {'ready': True}
        pinecone_client._client = mock_client
        
        result = await pinecone_client.create_index("test-index", 384, "cosine")
        
        assert result is True
        mock_client.create_index.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_upsert_vectors(self, pinecone_client, sample_vector_record):
        """Test vector upsert."""
        mock_client = Mock()
        mock_index = Mock()
        mock_client.Index.return_value = mock_index
        pinecone_client._client = mock_client
        
        result = await pinecone_client.upsert_vectors("test-index", [sample_vector_record])
        
        assert result is True
        mock_index.upsert.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_search_vectors(self, pinecone_client):
        """Test vector search."""
        mock_client = Mock()
        mock_index = Mock()
        mock_response = Mock()
        mock_response.matches = [
            Mock(id="result-1", score=0.95, metadata={"title": "Engineer"})
        ]
        mock_index.query.return_value = mock_response
        mock_client.Index.return_value = mock_index
        pinecone_client._client = mock_client
        
        query_vector = np.random.rand(384)
        results = await pinecone_client.search_vectors("test-index", query_vector, top_k=5)
        
        assert len(results) == 1
        assert results[0].id == "result-1"
        assert results[0].score == 0.95
        mock_index.query.assert_called_once()


class TestWeaviateClient(TestVectorDBBase):
    """Test Weaviate vector database client."""
    
    @pytest.fixture
    def weaviate_client(self, mock_config):
        """Create Weaviate client with mocked dependencies."""
        mock_config["url"] = "http://localhost:8080"
        with patch('app.services.vector_db.weaviate_client.weaviate') as mock_weaviate:
            mock_weaviate.Client = Mock()
            mock_weaviate.AuthApiKey = Mock()
            client = WeaviateClient(mock_config)
            return client
    
    @pytest.mark.asyncio
    async def test_connect(self, weaviate_client):
        """Test Weaviate connection."""
        with patch('app.services.vector_db.weaviate_client.Client') as mock_client_class:
            mock_client = Mock()
            mock_client.schema.get.return_value = {}
            mock_client_class.return_value = mock_client
            
            await weaviate_client.connect()
            
            assert weaviate_client._client == mock_client
            mock_client.schema.get.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_create_index(self, weaviate_client):
        """Test class creation."""
        mock_client = Mock()
        mock_client.schema.get.return_value = {"classes": []}
        weaviate_client._client = mock_client
        
        result = await weaviate_client.create_index("TestClass", 384, "cosine")
        
        assert result is True
        mock_client.schema.create_class.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_search_vectors(self, weaviate_client):
        """Test vector search."""
        mock_client = Mock()
        mock_query = Mock()
        mock_query.with_near_vector.return_value = mock_query
        mock_query.with_limit.return_value = mock_query
        mock_query.with_additional.return_value = mock_query
        mock_query.do.return_value = {
            "data": {
                "Get": {
                    "TestClass": [
                        {
                            "content": "test content",
                            "metadata": {"title": "Test"},
                            "_additional": {"id": "test-1", "certainty": 0.9}
                        }
                    ]
                }
            }
        }
        mock_client.query.get.return_value = mock_query
        weaviate_client._client = mock_client
        
        query_vector = np.random.rand(384)
        results = await weaviate_client.search_vectors("TestClass", query_vector, top_k=5)
        
        assert len(results) == 1
        assert results[0].id == "test-1"
        assert results[0].score == 0.9


class TestVectorDBManager:
    """Test vector database manager."""
    
    @pytest.fixture
    def mock_vector_config(self):
        """Mock vector database configuration."""
        config = Mock(spec=VectorDBConfig)
        config.VECTOR_DB_PROVIDER = "pinecone"
        config.PINECONE_API_KEY = "test-key"
        config.PINECONE_ENVIRONMENT = "test-env"
        config.TIMEOUT_SECONDS = 30
        config.MAX_RETRIES = 3
        config.BATCH_SIZE = 100
        return config
    
    @pytest.fixture
    def vector_manager(self, mock_vector_config):
        """Create vector database manager."""
        return VectorDBManager(mock_vector_config)
    
    @pytest.mark.asyncio
    async def test_initialize(self, vector_manager):
        """Test manager initialization."""
        with patch.object(vector_manager, '_create_client') as mock_create:
            mock_client = AsyncMock()
            mock_client.connect = AsyncMock()
            mock_client.index_exists = AsyncMock(return_value=False)
            mock_client.create_index = AsyncMock(return_value=True)
            mock_create.return_value = mock_client
            
            await vector_manager.initialize()
            
            assert vector_manager._initialized is True
            assert len(vector_manager._clients) > 0
    
    def test_create_client_pinecone(self, vector_manager):
        """Test Pinecone client creation."""
        with patch('app.services.vector_db.vector_db_manager.PineconeClient') as mock_pinecone:
            client = vector_manager._create_client()
            mock_pinecone.assert_called_once()
    
    def test_get_client(self, vector_manager):
        """Test getting client for index type."""
        mock_client = Mock()
        vector_manager._clients["profiles"] = mock_client
        vector_manager._initialized = True
        
        client = vector_manager.get_client("profiles")
        assert client == mock_client
    
    def test_get_client_not_initialized(self, vector_manager):
        """Test getting client when not initialized."""
        with pytest.raises(RuntimeError):
            vector_manager.get_client("profiles")
    
    @pytest.mark.asyncio
    async def test_health_check(self, vector_manager):
        """Test health check."""
        mock_client = AsyncMock()
        mock_client.health_check = AsyncMock(return_value={"status": "healthy"})
        vector_manager._clients["profiles"] = mock_client
        vector_manager._initialized = True
        
        health = await vector_manager.health_check()
        
        assert health["overall_status"] == "healthy"
        assert "profiles" in health["clients"]


class TestEmbeddingService:
    """Test embedding service."""
    
    @pytest.fixture
    def embedding_service(self):
        """Create embedding service."""
        service = EmbeddingService()
        # Mock NLP engine
        service.nlp_engine = AsyncMock()
        service.nlp_engine.generate_embeddings = AsyncMock(
            return_value=np.random.rand(1, 384)
        )
        return service
    
    @pytest.mark.asyncio
    async def test_generate_embedding(self, embedding_service):
        """Test embedding generation."""
        text = "Python programming language"
        embedding = await embedding_service.generate_embedding(text)
        
        assert isinstance(embedding, np.ndarray)
        assert len(embedding) == 384
    
    @pytest.mark.asyncio
    async def test_store_profile_embedding(self, embedding_service):
        """Test storing profile embedding."""
        with patch('app.services.embedding_service.vector_db_manager') as mock_manager:
            mock_client = AsyncMock()
            mock_client.upsert_vectors = AsyncMock(return_value=True)
            mock_manager.get_client.return_value = mock_client
            mock_manager.get_index_name.return_value = "test-profiles"
            
            profile_data = {
                "skills": ["Python", "Machine Learning"],
                "experience_level": "senior",
                "dream_job": "Data Scientist"
            }
            
            result = await embedding_service.store_profile_embedding("user123", profile_data)
            
            assert result is True
            mock_client.upsert_vectors.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_search_similar_profiles(self, embedding_service):
        """Test searching similar profiles."""
        with patch('app.services.embedding_service.vector_db_manager') as mock_manager:
            mock_client = AsyncMock()
            mock_vector = Mock()
            mock_vector.vector = np.random.rand(384)
            mock_client.get_vector = AsyncMock(return_value=mock_vector)
            mock_client.search_vectors = AsyncMock(return_value=[
                SearchResult("user456", 0.9, {"skills": ["Python"]}, None)
            ])
            mock_manager.get_client.return_value = mock_client
            mock_manager.get_index_name.return_value = "test-profiles"
            
            results = await embedding_service.search_similar_profiles("user123", top_k=5)
            
            assert len(results) == 1
            assert results[0].id == "user456"
            assert results[0].score == 0.9


class TestSemanticSearchService:
    """Test semantic search service."""
    
    @pytest.fixture
    def search_service(self):
        """Create semantic search service."""
        return SemanticSearchService()
    
    @pytest.mark.asyncio
    async def test_profile_similarity_search(self, search_service):
        """Test profile similarity search."""
        with patch('app.services.semantic_search_service.embedding_service') as mock_embedding:
            mock_embedding.search_similar_profiles = AsyncMock(return_value=[
                SearchResult("user456", 0.9, {"skills": ["Python"]}, None)
            ])
            
            request = SearchRequest(
                search_type=SearchType.PROFILE_SIMILARITY,
                query_data={"user_id": "user123"},
                top_k=5
            )
            
            results = await search_service.search(request)
            
            assert len(results) == 1
            assert results[0].id == "user456"
            assert results[0].confidence_level == "high"
    
    @pytest.mark.asyncio
    async def test_job_matching_search(self, search_service):
        """Test job matching search."""
        with patch('app.services.semantic_search_service.embedding_service') as mock_embedding:
            mock_embedding.search_matching_jobs = AsyncMock(return_value=[
                SearchResult("job123", 0.85, {"title": "Software Engineer"}, None)
            ])
            
            request = SearchRequest(
                search_type=SearchType.JOB_MATCHING,
                query_data={"user_id": "user123"},
                top_k=10
            )
            
            results = await search_service.search(request)
            
            assert len(results) == 1
            assert results[0].id == "job123"
            assert "Software Engineer" in results[0].match_reasons[0]
    
    @pytest.mark.asyncio
    async def test_skill_similarity_search(self, search_service):
        """Test skill similarity search."""
        with patch('app.services.semantic_search_service.embedding_service') as mock_embedding:
            mock_embedding.find_similar_skills = AsyncMock(return_value=[
                SearchResult("skill456", 0.92, {"skill_name": "JavaScript"}, None)
            ])
            
            request = SearchRequest(
                search_type=SearchType.SKILL_SIMILARITY,
                query_data={"skill_name": "Python"},
                top_k=10
            )
            
            results = await search_service.search(request)
            
            assert len(results) == 1
            assert results[0].id == "skill456"
    
    @pytest.mark.asyncio
    async def test_learning_resources_search(self, search_service):
        """Test learning resources search."""
        with patch('app.services.semantic_search_service.embedding_service') as mock_embedding:
            mock_embedding.search_learning_resources = AsyncMock(return_value=[
                SearchResult("course123", 0.88, {"title": "Python Basics", "rating": 4.5}, None)
            ])
            
            request = SearchRequest(
                search_type=SearchType.LEARNING_RESOURCES,
                query_data={"skill_gaps": ["Python", "Django"]},
                top_k=15
            )
            
            results = await search_service.search(request)
            
            assert len(results) == 1
            assert results[0].id == "course123"
            assert "Python Basics" in results[0].match_reasons[0]
    
    def test_calculate_confidence_level(self, search_service):
        """Test confidence level calculation."""
        assert search_service._calculate_confidence_level(0.9) == "high"
        assert search_service._calculate_confidence_level(0.7) == "medium"
        assert search_service._calculate_confidence_level(0.5) == "low"
    
    @pytest.mark.asyncio
    async def test_explain_search_results(self, search_service):
        """Test search results explanation."""
        from app.services.semantic_search_service import SearchMatch
        
        matches = [
            SearchMatch("id1", 0.9, {}, ["reason1"], "high", SearchType.JOB_MATCHING),
            SearchMatch("id2", 0.7, {}, ["reason2"], "medium", SearchType.JOB_MATCHING)
        ]
        
        explanation = await search_service.explain_search_results(matches)
        
        assert explanation["confidence_distribution"]["high"] == 1
        assert explanation["confidence_distribution"]["medium"] == 1
        assert explanation["average_score"] == 0.8
        assert len(explanation["insights"]) > 0
    
    @pytest.mark.asyncio
    async def test_batch_search(self, search_service):
        """Test batch search functionality."""
        with patch.object(search_service, 'search') as mock_search:
            mock_search.return_value = [
                Mock(id="result1", score=0.9)
            ]
            
            requests = [
                SearchRequest(SearchType.PROFILE_SIMILARITY, {"user_id": "user1"}),
                SearchRequest(SearchType.JOB_MATCHING, {"user_id": "user2"})
            ]
            
            results = await search_service.batch_search(requests)
            
            assert len(results) == 2
            assert "0" in results
            assert "1" in results
            assert mock_search.call_count == 2


class TestVectorSearchAPI:
    """Test vector search API endpoints."""
    
    @pytest.fixture
    def mock_current_user(self):
        """Mock current user for authentication."""
        return {"user_id": "test-user", "email": "test@example.com"}
    
    @pytest.mark.asyncio
    async def test_generate_embedding_endpoint(self, mock_current_user):
        """Test embedding generation endpoint."""
        from app.api.v1.endpoints.vector_search import generate_embedding
        from app.api.v1.endpoints.vector_search import EmbeddingRequest
        
        with patch('app.api.v1.endpoints.vector_search.embedding_service') as mock_service:
            mock_service.generate_embedding = AsyncMock(
                return_value=np.random.rand(384)
            )
            
            request = EmbeddingRequest(text="Python programming")
            response = await generate_embedding(request, mock_current_user)
            
            assert response.dimension == 384
            assert len(response.embedding) == 384
    
    @pytest.mark.asyncio
    async def test_semantic_search_endpoint(self, mock_current_user):
        """Test semantic search endpoint."""
        from app.api.v1.endpoints.vector_search import semantic_search
        from app.api.v1.endpoints.vector_search import SemanticSearchRequest
        
        with patch('app.api.v1.endpoints.vector_search.semantic_search_service') as mock_service:
            mock_match = Mock()
            mock_match.id = "result1"
            mock_match.score = 0.9
            mock_match.metadata = {"title": "Engineer"}
            mock_match.match_reasons = ["skill match"]
            mock_match.confidence_level = "high"
            
            mock_service.search = AsyncMock(return_value=[mock_match])
            mock_service.explain_search_results = AsyncMock(return_value={
                "explanation": "Found 1 match",
                "insights": ["Good match"]
            })
            
            request = SemanticSearchRequest(
                search_type="job_matching",
                query_data={"user_id": "user123"},
                top_k=10
            )
            
            response = await semantic_search(request, mock_current_user)
            
            assert response.total_results == 1
            assert response.matches[0].id == "result1"
            assert response.search_type == "job_matching"


# Integration Tests

class TestVectorDBIntegration:
    """Integration tests for vector database functionality."""
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_end_to_end_profile_search(self):
        """Test end-to-end profile similarity search."""
        # This would require actual vector database setup
        # For now, we'll mock the entire flow
        
        with patch('app.services.vector_db.vector_db_manager.vector_db_manager') as mock_manager:
            # Mock the entire flow
            mock_client = AsyncMock()
            mock_manager.get_client.return_value = mock_client
            mock_manager.get_index_name.return_value = "test-profiles"
            
            # Mock profile storage
            mock_client.upsert_vectors = AsyncMock(return_value=True)
            
            # Mock profile search
            mock_vector = Mock()
            mock_vector.vector = np.random.rand(384)
            mock_client.get_vector = AsyncMock(return_value=mock_vector)
            mock_client.search_vectors = AsyncMock(return_value=[
                SearchResult("similar_user", 0.9, {"skills": ["Python"]}, None)
            ])
            
            # Initialize services
            embedding_service = EmbeddingService()
            embedding_service.nlp_engine = AsyncMock()
            embedding_service.nlp_engine.generate_embeddings = AsyncMock(
                return_value=np.random.rand(1, 384)
            )
            
            # Store profile
            profile_data = {
                "skills": ["Python", "Machine Learning"],
                "experience_level": "senior"
            }
            
            store_result = await embedding_service.store_profile_embedding("user123", profile_data)
            assert store_result is True
            
            # Search similar profiles
            search_results = await embedding_service.search_similar_profiles("user123", top_k=5)
            assert len(search_results) == 1
            assert search_results[0].id == "similar_user"
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_end_to_end_job_matching(self):
        """Test end-to-end job matching flow."""
        with patch('app.services.vector_db.vector_db_manager.vector_db_manager') as mock_manager:
            # Mock clients for both profiles and jobs
            mock_profiles_client = AsyncMock()
            mock_jobs_client = AsyncMock()
            
            def get_client_side_effect(index_type):
                if index_type == "profiles":
                    return mock_profiles_client
                elif index_type == "jobs":
                    return mock_jobs_client
                
            mock_manager.get_client.side_effect = get_client_side_effect
            mock_manager.get_index_name.side_effect = lambda x: f"test-{x}"
            
            # Mock user profile retrieval
            mock_vector = Mock()
            mock_vector.vector = np.random.rand(384)
            mock_profiles_client.get_vector = AsyncMock(return_value=mock_vector)
            
            # Mock job search
            mock_jobs_client.search_vectors = AsyncMock(return_value=[
                SearchResult("job123", 0.85, {"title": "Software Engineer"}, None)
            ])
            
            # Initialize service
            embedding_service = EmbeddingService()
            
            # Search matching jobs
            job_results = await embedding_service.search_matching_jobs("user123", top_k=10)
            assert len(job_results) == 1
            assert job_results[0].id == "job123"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])