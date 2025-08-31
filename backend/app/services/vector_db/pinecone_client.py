"""
Pinecone Vector Database Client

Implementation of vector database interface using Pinecone as the backend.
Provides efficient vector storage, retrieval, and similarity search capabilities.
"""

import logging
import asyncio
from typing import List, Dict, Any, Optional, Union
import numpy as np
from concurrent.futures import ThreadPoolExecutor

try:
    import pinecone
    from pinecone import Pinecone, ServerlessSpec
except ImportError:
    pinecone = None
    Pinecone = None
    ServerlessSpec = None

from .base_vector_db import BaseVectorDB, VectorRecord, SearchResult, QueryFilter

logger = logging.getLogger(__name__)


class PineconeClient(BaseVectorDB):
    """Pinecone implementation of vector database interface."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Pinecone client.
        
        Args:
            config: Configuration containing API key, environment, etc.
        """
        super().__init__(config)
        
        if not pinecone:
            raise ImportError("pinecone-client package is required for PineconeClient")
        
        self.api_key = config.get("api_key")
        self.environment = config.get("environment", "us-west1-gcp-free")
        self.timeout = config.get("timeout", 30)
        self.max_retries = config.get("max_retries", 3)
        self.batch_size = config.get("batch_size", 100)
        
        if not self.api_key:
            raise ValueError("Pinecone API key is required")
        
        # Thread pool for blocking operations
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        logger.info("Pinecone client initialized")
    
    async def connect(self) -> None:
        """Establish connection to Pinecone."""
        try:
            def _connect():
                self._client = Pinecone(api_key=self.api_key)
                return self._client
            
            loop = asyncio.get_event_loop()
            self._client = await loop.run_in_executor(self.executor, _connect)
            
            logger.info("Connected to Pinecone successfully")
            
        except Exception as e:
            logger.error(f"Failed to connect to Pinecone: {str(e)}")
            raise
    
    async def disconnect(self) -> None:
        """Close connection to Pinecone."""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)
        self._client = None
        logger.info("Disconnected from Pinecone")
    
    async def create_index(self, index_name: str, dimension: int, 
                          metric: str = "cosine", **kwargs) -> bool:
        """
        Create a new Pinecone index.
        
        Args:
            index_name: Name of the index
            dimension: Vector dimension
            metric: Similarity metric
            **kwargs: Additional configuration
            
        Returns:
            True if index created successfully
        """
        try:
            def _create_index():
                # Check if index already exists
                existing_indexes = self._client.list_indexes()
                if any(idx.name == index_name for idx in existing_indexes):
                    logger.info(f"Index {index_name} already exists")
                    return True
                
                # Create serverless index (free tier)
                spec = ServerlessSpec(
                    cloud="aws",
                    region="us-west-2"
                )
                
                self._client.create_index(
                    name=index_name,
                    dimension=dimension,
                    metric=metric,
                    spec=spec
                )
                
                # Wait for index to be ready
                import time
                while not self._client.describe_index(index_name).status['ready']:
                    time.sleep(1)
                
                return True
            
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(self.executor, _create_index)
            
            logger.info(f"Created Pinecone index: {index_name}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to create index {index_name}: {str(e)}")
            return False
    
    async def delete_index(self, index_name: str) -> bool:
        """Delete a Pinecone index."""
        try:
            def _delete_index():
                self._client.delete_index(index_name)
                return True
            
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(self.executor, _delete_index)
            
            logger.info(f"Deleted Pinecone index: {index_name}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to delete index {index_name}: {str(e)}")
            return False
    
    async def list_indexes(self) -> List[str]:
        """List all Pinecone indexes."""
        try:
            def _list_indexes():
                indexes = self._client.list_indexes()
                return [idx.name for idx in indexes]
            
            loop = asyncio.get_event_loop()
            indexes = await loop.run_in_executor(self.executor, _list_indexes)
            
            return indexes
            
        except Exception as e:
            logger.error(f"Failed to list indexes: {str(e)}")
            return []
    
    async def index_exists(self, index_name: str) -> bool:
        """Check if Pinecone index exists."""
        try:
            indexes = await self.list_indexes()
            return index_name in indexes
        except Exception as e:
            logger.error(f"Failed to check if index exists: {str(e)}")
            return False
    
    async def upsert_vectors(self, index_name: str, vectors: List[VectorRecord],
                           namespace: Optional[str] = None) -> bool:
        """
        Upsert vectors to Pinecone index.
        
        Args:
            index_name: Name of the index
            vectors: List of vector records
            namespace: Optional namespace
            
        Returns:
            True if upsert successful
        """
        try:
            def _upsert_vectors():
                index = self._client.Index(index_name)
                
                # Convert vectors to Pinecone format
                pinecone_vectors = []
                for vector in vectors:
                    pinecone_vector = {
                        "id": vector.id,
                        "values": self._convert_numpy_to_list(vector.vector),
                        "metadata": vector.metadata
                    }
                    pinecone_vectors.append(pinecone_vector)
                
                # Batch upsert
                for i in range(0, len(pinecone_vectors), self.batch_size):
                    batch = pinecone_vectors[i:i + self.batch_size]
                    index.upsert(vectors=batch, namespace=namespace)
                
                return True
            
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(self.executor, _upsert_vectors)
            
            logger.info(f"Upserted {len(vectors)} vectors to index {index_name}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to upsert vectors to {index_name}: {str(e)}")
            return False
    
    async def delete_vectors(self, index_name: str, vector_ids: List[str],
                           namespace: Optional[str] = None) -> bool:
        """Delete vectors from Pinecone index."""
        try:
            def _delete_vectors():
                index = self._client.Index(index_name)
                
                # Batch delete
                for i in range(0, len(vector_ids), self.batch_size):
                    batch = vector_ids[i:i + self.batch_size]
                    index.delete(ids=batch, namespace=namespace)
                
                return True
            
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(self.executor, _delete_vectors)
            
            logger.info(f"Deleted {len(vector_ids)} vectors from index {index_name}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to delete vectors from {index_name}: {str(e)}")
            return False
    
    async def search_vectors(self, index_name: str, query_vector: Union[List[float], np.ndarray],
                           top_k: int = 10, filters: Optional[List[QueryFilter]] = None,
                           namespace: Optional[str] = None, 
                           include_metadata: bool = True) -> List[SearchResult]:
        """
        Search for similar vectors in Pinecone.
        
        Args:
            index_name: Name of the index
            query_vector: Query vector
            top_k: Number of results to return
            filters: Optional filters
            namespace: Optional namespace
            include_metadata: Whether to include metadata
            
        Returns:
            List of search results
        """
        try:
            def _search_vectors():
                index = self._client.Index(index_name)
                
                # Convert filters to Pinecone format
                pinecone_filter = None
                if filters:
                    pinecone_filter = self._convert_filters_to_pinecone(filters)
                
                # Perform search
                response = index.query(
                    vector=self._convert_numpy_to_list(query_vector),
                    top_k=top_k,
                    filter=pinecone_filter,
                    namespace=namespace,
                    include_metadata=include_metadata
                )
                
                # Convert results
                results = []
                for match in response.matches:
                    result = SearchResult(
                        id=match.id,
                        score=match.score,
                        metadata=match.metadata if include_metadata else {},
                        namespace=namespace
                    )
                    results.append(result)
                
                return results
            
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(self.executor, _search_vectors)
            
            logger.info(f"Found {len(results)} similar vectors in index {index_name}")
            return results
            
        except Exception as e:
            logger.error(f"Failed to search vectors in {index_name}: {str(e)}")
            return []
    
    async def get_vector(self, index_name: str, vector_id: str,
                        namespace: Optional[str] = None) -> Optional[VectorRecord]:
        """Retrieve a specific vector from Pinecone."""
        try:
            def _get_vector():
                index = self._client.Index(index_name)
                
                response = index.fetch(ids=[vector_id], namespace=namespace)
                
                if vector_id in response.vectors:
                    vector_data = response.vectors[vector_id]
                    return VectorRecord(
                        id=vector_id,
                        vector=vector_data.values,
                        metadata=vector_data.metadata,
                        namespace=namespace
                    )
                return None
            
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(self.executor, _get_vector)
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to get vector {vector_id} from {index_name}: {str(e)}")
            return None
    
    async def get_index_stats(self, index_name: str) -> Dict[str, Any]:
        """Get statistics about a Pinecone index."""
        try:
            def _get_stats():
                index = self._client.Index(index_name)
                stats = index.describe_index_stats()
                
                return {
                    "total_vector_count": stats.total_vector_count,
                    "dimension": stats.dimension,
                    "index_fullness": stats.index_fullness,
                    "namespaces": dict(stats.namespaces) if stats.namespaces else {}
                }
            
            loop = asyncio.get_event_loop()
            stats = await loop.run_in_executor(self.executor, _get_stats)
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get stats for index {index_name}: {str(e)}")
            return {}
    
    async def update_metadata(self, index_name: str, vector_id: str,
                            metadata: Dict[str, Any], 
                            namespace: Optional[str] = None) -> bool:
        """Update metadata for a specific vector in Pinecone."""
        try:
            def _update_metadata():
                index = self._client.Index(index_name)
                
                # Pinecone doesn't have direct metadata update, so we need to upsert
                # First fetch the vector
                response = index.fetch(ids=[vector_id], namespace=namespace)
                
                if vector_id not in response.vectors:
                    return False
                
                vector_data = response.vectors[vector_id]
                
                # Upsert with new metadata
                index.upsert(
                    vectors=[{
                        "id": vector_id,
                        "values": vector_data.values,
                        "metadata": metadata
                    }],
                    namespace=namespace
                )
                
                return True
            
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(self.executor, _update_metadata)
            
            logger.info(f"Updated metadata for vector {vector_id} in index {index_name}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to update metadata for {vector_id} in {index_name}: {str(e)}")
            return False
    
    def _convert_filters_to_pinecone(self, filters: List[QueryFilter]) -> Dict[str, Any]:
        """Convert QueryFilter objects to Pinecone filter format."""
        if not filters:
            return {}
        
        pinecone_filter = {}
        
        for filter_obj in filters:
            field = filter_obj.field
            operator = filter_obj.operator
            value = filter_obj.value
            
            if operator == "eq":
                pinecone_filter[field] = {"$eq": value}
            elif operator == "ne":
                pinecone_filter[field] = {"$ne": value}
            elif operator == "in":
                pinecone_filter[field] = {"$in": value}
            elif operator == "nin":
                pinecone_filter[field] = {"$nin": value}
            elif operator == "gt":
                pinecone_filter[field] = {"$gt": value}
            elif operator == "gte":
                pinecone_filter[field] = {"$gte": value}
            elif operator == "lt":
                pinecone_filter[field] = {"$lt": value}
            elif operator == "lte":
                pinecone_filter[field] = {"$lte": value}
        
        return pinecone_filter