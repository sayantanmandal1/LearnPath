"""
Vector Database Configuration

Configuration settings for vector database integration including
Pinecone and Weaviate setup parameters.
"""

import os
from typing import Optional, Dict, Any
from pydantic_settings import BaseSettings
from pydantic import Field


class VectorDBConfig(BaseSettings):
    """Vector database configuration settings."""
    
    # Vector database provider ('pinecone' or 'weaviate')
    VECTOR_DB_PROVIDER: str = Field(default="pinecone", env="VECTOR_DB_PROVIDER")
    
    # Pinecone configuration
    PINECONE_API_KEY: Optional[str] = Field(default=None, env="PINECONE_API_KEY")
    PINECONE_ENVIRONMENT: str = Field(default="us-west1-gcp-free", env="PINECONE_ENVIRONMENT")
    PINECONE_INDEX_NAME: str = Field(default="ai-career-recommender", env="PINECONE_INDEX_NAME")
    
    # Weaviate configuration
    WEAVIATE_URL: str = Field(default="http://localhost:8080", env="WEAVIATE_URL")
    WEAVIATE_API_KEY: Optional[str] = Field(default=None, env="WEAVIATE_API_KEY")
    
    # Embedding configuration
    EMBEDDING_DIMENSION: int = Field(default=384, env="EMBEDDING_DIMENSION")  # all-MiniLM-L6-v2 dimension
    EMBEDDING_MODEL: str = Field(default="sentence-transformers/all-MiniLM-L6-v2", env="EMBEDDING_MODEL")
    
    # Performance settings
    BATCH_SIZE: int = Field(default=100, env="VECTOR_DB_BATCH_SIZE")
    MAX_RETRIES: int = Field(default=3, env="VECTOR_DB_MAX_RETRIES")
    TIMEOUT_SECONDS: int = Field(default=30, env="VECTOR_DB_TIMEOUT")
    
    # Index configuration
    SIMILARITY_METRIC: str = Field(default="cosine", env="VECTOR_DB_SIMILARITY_METRIC")
    
    model_config = {
        "env_file": ".env",
        "case_sensitive": True,
        "extra": "ignore"
    }


# Global configuration instance
vector_db_config = VectorDBConfig()


# Index configurations for different data types
INDEX_CONFIGS = {
    "profiles": {
        "name": f"{vector_db_config.PINECONE_INDEX_NAME}-profiles",
        "dimension": vector_db_config.EMBEDDING_DIMENSION,
        "metric": vector_db_config.SIMILARITY_METRIC,
        "metadata_config": {
            "indexed": ["user_id", "experience_level", "primary_skills", "created_at"]
        }
    },
    "jobs": {
        "name": f"{vector_db_config.PINECONE_INDEX_NAME}-jobs",
        "dimension": vector_db_config.EMBEDDING_DIMENSION,
        "metric": vector_db_config.SIMILARITY_METRIC,
        "metadata_config": {
            "indexed": ["job_id", "title", "company", "location", "experience_level", "salary_range"]
        }
    },
    "skills": {
        "name": f"{vector_db_config.PINECONE_INDEX_NAME}-skills",
        "dimension": vector_db_config.EMBEDDING_DIMENSION,
        "metric": vector_db_config.SIMILARITY_METRIC,
        "metadata_config": {
            "indexed": ["skill_name", "category", "popularity_score"]
        }
    },
    "learning_resources": {
        "name": f"{vector_db_config.PINECONE_INDEX_NAME}-resources",
        "dimension": vector_db_config.EMBEDDING_DIMENSION,
        "metric": vector_db_config.SIMILARITY_METRIC,
        "metadata_config": {
            "indexed": ["resource_id", "type", "provider", "difficulty_level", "rating"]
        }
    }
}