"""
Vector Database Package

Package for vector database implementations and management.
Provides unified interface for different vector database providers.
"""

from .base_vector_db import BaseVectorDB, VectorRecord, SearchResult, QueryFilter, VectorDBProvider
from .pinecone_client import PineconeClient
from .weaviate_client import WeaviateClient
from .vector_db_manager import VectorDBManager, vector_db_manager

__all__ = [
    "BaseVectorDB",
    "VectorRecord", 
    "SearchResult",
    "QueryFilter",
    "VectorDBProvider",
    "PineconeClient",
    "WeaviateClient", 
    "VectorDBManager",
    "vector_db_manager"
]