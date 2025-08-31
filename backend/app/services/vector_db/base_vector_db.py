"""
Base Vector Database Interface

Abstract base class for vector database implementations providing
a common interface for different vector database providers.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np
from dataclasses import dataclass
from enum import Enum


class VectorDBProvider(Enum):
    """Supported vector database providers."""
    PINECONE = "pinecone"
    WEAVIATE = "weaviate"


@dataclass
class VectorRecord:
    """Represents a vector record with metadata."""
    id: str
    vector: Union[List[float], np.ndarray]
    metadata: Dict[str, Any]
    namespace: Optional[str] = None


@dataclass
class SearchResult:
    """Represents a search result from vector database."""
    id: str
    score: float
    metadata: Dict[str, Any]
    namespace: Optional[str] = None


@dataclass
class QueryFilter:
    """Filter criteria for vector searches."""
    field: str
    operator: str  # 'eq', 'ne', 'in', 'nin', 'gt', 'gte', 'lt', 'lte'
    value: Any


class BaseVectorDB(ABC):
    """Abstract base class for vector database implementations."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize vector database connection.
        
        Args:
            config: Database configuration parameters
        """
        self.config = config
        self._client = None
    
    @abstractmethod
    async def connect(self) -> None:
        """Establish connection to vector database."""
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """Close connection to vector database."""
        pass
    
    @abstractmethod
    async def create_index(self, index_name: str, dimension: int, 
                          metric: str = "cosine", **kwargs) -> bool:
        """
        Create a new vector index.
        
        Args:
            index_name: Name of the index
            dimension: Vector dimension
            metric: Similarity metric ('cosine', 'euclidean', 'dotproduct')
            **kwargs: Additional index configuration
            
        Returns:
            True if index created successfully
        """
        pass
    
    @abstractmethod
    async def delete_index(self, index_name: str) -> bool:
        """
        Delete a vector index.
        
        Args:
            index_name: Name of the index to delete
            
        Returns:
            True if index deleted successfully
        """
        pass
    
    @abstractmethod
    async def list_indexes(self) -> List[str]:
        """
        List all available indexes.
        
        Returns:
            List of index names
        """
        pass
    
    @abstractmethod
    async def index_exists(self, index_name: str) -> bool:
        """
        Check if index exists.
        
        Args:
            index_name: Name of the index
            
        Returns:
            True if index exists
        """
        pass
    
    @abstractmethod
    async def upsert_vectors(self, index_name: str, vectors: List[VectorRecord],
                           namespace: Optional[str] = None) -> bool:
        """
        Insert or update vectors in the index.
        
        Args:
            index_name: Name of the index
            vectors: List of vector records to upsert
            namespace: Optional namespace for the vectors
            
        Returns:
            True if upsert successful
        """
        pass
    
    @abstractmethod
    async def delete_vectors(self, index_name: str, vector_ids: List[str],
                           namespace: Optional[str] = None) -> bool:
        """
        Delete vectors from the index.
        
        Args:
            index_name: Name of the index
            vector_ids: List of vector IDs to delete
            namespace: Optional namespace
            
        Returns:
            True if deletion successful
        """
        pass
    
    @abstractmethod
    async def search_vectors(self, index_name: str, query_vector: Union[List[float], np.ndarray],
                           top_k: int = 10, filters: Optional[List[QueryFilter]] = None,
                           namespace: Optional[str] = None, 
                           include_metadata: bool = True) -> List[SearchResult]:
        """
        Search for similar vectors.
        
        Args:
            index_name: Name of the index
            query_vector: Query vector
            top_k: Number of results to return
            filters: Optional filters to apply
            namespace: Optional namespace to search in
            include_metadata: Whether to include metadata in results
            
        Returns:
            List of search results
        """
        pass
    
    @abstractmethod
    async def get_vector(self, index_name: str, vector_id: str,
                        namespace: Optional[str] = None) -> Optional[VectorRecord]:
        """
        Retrieve a specific vector by ID.
        
        Args:
            index_name: Name of the index
            vector_id: ID of the vector to retrieve
            namespace: Optional namespace
            
        Returns:
            Vector record if found, None otherwise
        """
        pass
    
    @abstractmethod
    async def get_index_stats(self, index_name: str) -> Dict[str, Any]:
        """
        Get statistics about an index.
        
        Args:
            index_name: Name of the index
            
        Returns:
            Dictionary containing index statistics
        """
        pass
    
    @abstractmethod
    async def update_metadata(self, index_name: str, vector_id: str,
                            metadata: Dict[str, Any], 
                            namespace: Optional[str] = None) -> bool:
        """
        Update metadata for a specific vector.
        
        Args:
            index_name: Name of the index
            vector_id: ID of the vector
            metadata: New metadata to set
            namespace: Optional namespace
            
        Returns:
            True if update successful
        """
        pass
    
    def _convert_numpy_to_list(self, vector: Union[List[float], np.ndarray]) -> List[float]:
        """Convert numpy array to list if needed."""
        if isinstance(vector, np.ndarray):
            return vector.tolist()
        return vector
    
    def _validate_vector_dimension(self, vector: Union[List[float], np.ndarray], 
                                 expected_dim: int) -> bool:
        """Validate vector dimension."""
        if isinstance(vector, np.ndarray):
            return vector.shape[0] == expected_dim
        return len(vector) == expected_dim
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on the vector database.
        
        Returns:
            Health status information
        """
        try:
            indexes = await self.list_indexes()
            return {
                "status": "healthy",
                "provider": self.__class__.__name__,
                "indexes_count": len(indexes),
                "indexes": indexes
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "provider": self.__class__.__name__,
                "error": str(e)
            }