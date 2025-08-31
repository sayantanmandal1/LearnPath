"""
Vector Database Manager

Factory and management class for vector database operations.
Provides a unified interface for different vector database providers.
"""

import logging
from typing import Dict, Any, Optional, List
from enum import Enum

from ...core.vector_db_config import vector_db_config, INDEX_CONFIGS, VectorDBConfig
from .base_vector_db import BaseVectorDB, VectorDBProvider
from .pinecone_client import PineconeClient
from .weaviate_client import WeaviateClient

logger = logging.getLogger(__name__)


class VectorDBManager:
    """
    Manager class for vector database operations.
    Handles initialization, connection management, and provides unified interface.
    """
    
    def __init__(self, config: Optional[VectorDBConfig] = None):
        """
        Initialize vector database manager.
        
        Args:
            config: Vector database configuration
        """
        self.config = config or vector_db_config
        self._clients: Dict[str, BaseVectorDB] = {}
        self._initialized = False
        
        logger.info(f"Vector DB Manager initialized with provider: {self.config.VECTOR_DB_PROVIDER}")
    
    async def initialize(self) -> None:
        """Initialize vector database connections and indexes."""
        try:
            # Create clients for each index type
            for index_type, index_config in INDEX_CONFIGS.items():
                client = self._create_client()
                await client.connect()
                
                # Create index if it doesn't exist
                index_name = index_config["name"]
                if not await client.index_exists(index_name):
                    await client.create_index(
                        index_name=index_name,
                        dimension=index_config["dimension"],
                        metric=index_config["metric"]
                    )
                    logger.info(f"Created index: {index_name}")
                else:
                    logger.info(f"Index already exists: {index_name}")
                
                self._clients[index_type] = client
            
            self._initialized = True
            logger.info("Vector database manager initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize vector database manager: {str(e)}")
            raise
    
    def _create_client(self) -> BaseVectorDB:
        """Create vector database client based on configuration."""
        provider = self.config.VECTOR_DB_PROVIDER.lower()
        
        if provider == VectorDBProvider.PINECONE.value:
            return PineconeClient({
                "api_key": self.config.PINECONE_API_KEY,
                "environment": self.config.PINECONE_ENVIRONMENT,
                "timeout": self.config.TIMEOUT_SECONDS,
                "max_retries": self.config.MAX_RETRIES,
                "batch_size": self.config.BATCH_SIZE
            })
        elif provider == VectorDBProvider.WEAVIATE.value:
            return WeaviateClient({
                "url": self.config.WEAVIATE_URL,
                "api_key": self.config.WEAVIATE_API_KEY,
                "timeout": self.config.TIMEOUT_SECONDS,
                "max_retries": self.config.MAX_RETRIES,
                "batch_size": self.config.BATCH_SIZE
            })
        else:
            raise ValueError(f"Unsupported vector database provider: {provider}")
    
    def get_client(self, index_type: str) -> BaseVectorDB:
        """
        Get vector database client for specific index type.
        
        Args:
            index_type: Type of index ('profiles', 'jobs', 'skills', 'learning_resources')
            
        Returns:
            Vector database client
        """
        if not self._initialized:
            raise RuntimeError("Vector database manager not initialized. Call initialize() first.")
        
        if index_type not in self._clients:
            raise ValueError(f"Unknown index type: {index_type}")
        
        return self._clients[index_type]
    
    def get_index_name(self, index_type: str) -> str:
        """
        Get index name for specific index type.
        
        Args:
            index_type: Type of index
            
        Returns:
            Index name
        """
        if index_type not in INDEX_CONFIGS:
            raise ValueError(f"Unknown index type: {index_type}")
        
        return INDEX_CONFIGS[index_type]["name"]
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on all vector database connections.
        
        Returns:
            Health status for all clients
        """
        health_status = {
            "overall_status": "healthy",
            "provider": self.config.VECTOR_DB_PROVIDER,
            "clients": {}
        }
        
        if not self._initialized:
            health_status["overall_status"] = "not_initialized"
            return health_status
        
        unhealthy_count = 0
        
        for index_type, client in self._clients.items():
            try:
                client_health = await client.health_check()
                health_status["clients"][index_type] = client_health
                
                if client_health["status"] != "healthy":
                    unhealthy_count += 1
                    
            except Exception as e:
                health_status["clients"][index_type] = {
                    "status": "error",
                    "error": str(e)
                }
                unhealthy_count += 1
        
        if unhealthy_count > 0:
            health_status["overall_status"] = "degraded" if unhealthy_count < len(self._clients) else "unhealthy"
        
        return health_status
    
    async def cleanup(self) -> None:
        """Clean up all vector database connections."""
        for client in self._clients.values():
            try:
                await client.disconnect()
            except Exception as e:
                logger.error(f"Error disconnecting client: {str(e)}")
        
        self._clients.clear()
        self._initialized = False
        logger.info("Vector database manager cleaned up")


# Global vector database manager instance
vector_db_manager = VectorDBManager()


# Convenience functions for common operations
async def get_profiles_client() -> BaseVectorDB:
    """Get client for user profiles index."""
    return vector_db_manager.get_client("profiles")


async def get_jobs_client() -> BaseVectorDB:
    """Get client for jobs index."""
    return vector_db_manager.get_client("jobs")


async def get_skills_client() -> BaseVectorDB:
    """Get client for skills index."""
    return vector_db_manager.get_client("skills")


async def get_learning_resources_client() -> BaseVectorDB:
    """Get client for learning resources index."""
    return vector_db_manager.get_client("learning_resources")