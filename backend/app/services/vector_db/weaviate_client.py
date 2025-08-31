"""
Weaviate Vector Database Client

Implementation of vector database interface using Weaviate as the backend.
Provides efficient vector storage, retrieval, and similarity search capabilities.
"""

import logging
import asyncio
from typing import List, Dict, Any, Optional, Union
import numpy as np
from concurrent.futures import ThreadPoolExecutor

try:
    import weaviate
    from weaviate.client import Client
    from weaviate.auth import AuthApiKey
except ImportError:
    weaviate = None
    Client = None
    AuthApiKey = None

from .base_vector_db import BaseVectorDB, VectorRecord, SearchResult, QueryFilter

logger = logging.getLogger(__name__)


class WeaviateClient(BaseVectorDB):
    """Weaviate implementation of vector database interface."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Weaviate client.
        
        Args:
            config: Configuration containing URL, API key, etc.
        """
        super().__init__(config)
        
        if not weaviate:
            raise ImportError("weaviate-client package is required for WeaviateClient")
        
        self.url = config.get("url", "http://localhost:8080")
        self.api_key = config.get("api_key")
        self.timeout = config.get("timeout", 30)
        self.max_retries = config.get("max_retries", 3)
        self.batch_size = config.get("batch_size", 100)
        
        # Thread pool for blocking operations
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        logger.info("Weaviate client initialized")
    
    async def connect(self) -> None:
        """Establish connection to Weaviate."""
        try:
            def _connect():
                auth_config = None
                if self.api_key:
                    auth_config = AuthApiKey(api_key=self.api_key)
                
                self._client = Client(
                    url=self.url,
                    auth_client_secret=auth_config,
                    timeout_config=(self.timeout, self.timeout)
                )
                
                # Test connection
                self._client.schema.get()
                return self._client
            
            loop = asyncio.get_event_loop()
            self._client = await loop.run_in_executor(self.executor, _connect)
            
            logger.info("Connected to Weaviate successfully")
            
        except Exception as e:
            logger.error(f"Failed to connect to Weaviate: {str(e)}")
            raise
    
    async def disconnect(self) -> None:
        """Close connection to Weaviate."""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)
        self._client = None
        logger.info("Disconnected from Weaviate")
    
    async def create_index(self, index_name: str, dimension: int, 
                          metric: str = "cosine", **kwargs) -> bool:
        """
        Create a new Weaviate class (index).
        
        Args:
            index_name: Name of the class
            dimension: Vector dimension
            metric: Similarity metric
            **kwargs: Additional configuration
            
        Returns:
            True if class created successfully
        """
        try:
            def _create_class():
                # Check if class already exists
                schema = self._client.schema.get()
                existing_classes = [cls["class"] for cls in schema.get("classes", [])]
                
                if index_name in existing_classes:
                    logger.info(f"Class {index_name} already exists")
                    return True
                
                # Create class schema
                class_schema = {
                    "class": index_name,
                    "description": f"Vector index for {index_name}",
                    "vectorizer": "none",  # We'll provide vectors manually
                    "properties": [
                        {
                            "name": "content",
                            "dataType": ["text"],
                            "description": "Content text"
                        },
                        {
                            "name": "metadata",
                            "dataType": ["object"],
                            "description": "Associated metadata"
                        }
                    ]
                }
                
                # Add custom properties from kwargs
                if "properties" in kwargs:
                    class_schema["properties"].extend(kwargs["properties"])
                
                self._client.schema.create_class(class_schema)
                return True
            
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(self.executor, _create_class)
            
            logger.info(f"Created Weaviate class: {index_name}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to create class {index_name}: {str(e)}")
            return False
    
    async def delete_index(self, index_name: str) -> bool:
        """Delete a Weaviate class."""
        try:
            def _delete_class():
                self._client.schema.delete_class(index_name)
                return True
            
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(self.executor, _delete_class)
            
            logger.info(f"Deleted Weaviate class: {index_name}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to delete class {index_name}: {str(e)}")
            return False
    
    async def list_indexes(self) -> List[str]:
        """List all Weaviate classes."""
        try:
            def _list_classes():
                schema = self._client.schema.get()
                return [cls["class"] for cls in schema.get("classes", [])]
            
            loop = asyncio.get_event_loop()
            classes = await loop.run_in_executor(self.executor, _list_classes)
            
            return classes
            
        except Exception as e:
            logger.error(f"Failed to list classes: {str(e)}")
            return []
    
    async def index_exists(self, index_name: str) -> bool:
        """Check if Weaviate class exists."""
        try:
            classes = await self.list_indexes()
            return index_name in classes
        except Exception as e:
            logger.error(f"Failed to check if class exists: {str(e)}")
            return False
    
    async def upsert_vectors(self, index_name: str, vectors: List[VectorRecord],
                           namespace: Optional[str] = None) -> bool:
        """
        Upsert vectors to Weaviate class.
        
        Args:
            index_name: Name of the class
            vectors: List of vector records
            namespace: Optional namespace (not used in Weaviate)
            
        Returns:
            True if upsert successful
        """
        try:
            def _upsert_vectors():
                # Use batch import for efficiency
                with self._client.batch as batch:
                    batch.batch_size = self.batch_size
                    
                    for vector in vectors:
                        properties = {
                            "content": vector.metadata.get("content", ""),
                            "metadata": vector.metadata
                        }
                        
                        batch.add_data_object(
                            data_object=properties,
                            class_name=index_name,
                            uuid=vector.id,
                            vector=self._convert_numpy_to_list(vector.vector)
                        )
                
                return True
            
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(self.executor, _upsert_vectors)
            
            logger.info(f"Upserted {len(vectors)} vectors to class {index_name}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to upsert vectors to {index_name}: {str(e)}")
            return False
    
    async def delete_vectors(self, index_name: str, vector_ids: List[str],
                           namespace: Optional[str] = None) -> bool:
        """Delete vectors from Weaviate class."""
        try:
            def _delete_vectors():
                for vector_id in vector_ids:
                    self._client.data_object.delete(
                        uuid=vector_id,
                        class_name=index_name
                    )
                return True
            
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(self.executor, _delete_vectors)
            
            logger.info(f"Deleted {len(vector_ids)} vectors from class {index_name}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to delete vectors from {index_name}: {str(e)}")
            return False
    
    async def search_vectors(self, index_name: str, query_vector: Union[List[float], np.ndarray],
                           top_k: int = 10, filters: Optional[List[QueryFilter]] = None,
                           namespace: Optional[str] = None, 
                           include_metadata: bool = True) -> List[SearchResult]:
        """
        Search for similar vectors in Weaviate.
        
        Args:
            index_name: Name of the class
            query_vector: Query vector
            top_k: Number of results to return
            filters: Optional filters
            namespace: Optional namespace (not used)
            include_metadata: Whether to include metadata
            
        Returns:
            List of search results
        """
        try:
            def _search_vectors():
                query = self._client.query.get(index_name, ["content", "metadata"])
                
                # Add vector search
                query = query.with_near_vector({
                    "vector": self._convert_numpy_to_list(query_vector)
                })
                
                # Add filters if provided
                if filters:
                    where_filter = self._convert_filters_to_weaviate(filters)
                    query = query.with_where(where_filter)
                
                # Set limit
                query = query.with_limit(top_k)
                
                # Add additional fields
                query = query.with_additional(["certainty", "id"])
                
                response = query.do()
                
                # Convert results
                results = []
                if "data" in response and "Get" in response["data"]:
                    objects = response["data"]["Get"].get(index_name, [])
                    
                    for obj in objects:
                        additional = obj.get("_additional", {})
                        result = SearchResult(
                            id=additional.get("id", ""),
                            score=additional.get("certainty", 0.0),
                            metadata=obj.get("metadata", {}) if include_metadata else {},
                            namespace=namespace
                        )
                        results.append(result)
                
                return results
            
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(self.executor, _search_vectors)
            
            logger.info(f"Found {len(results)} similar vectors in class {index_name}")
            return results
            
        except Exception as e:
            logger.error(f"Failed to search vectors in {index_name}: {str(e)}")
            return []
    
    async def get_vector(self, index_name: str, vector_id: str,
                        namespace: Optional[str] = None) -> Optional[VectorRecord]:
        """Retrieve a specific vector from Weaviate."""
        try:
            def _get_vector():
                response = self._client.data_object.get_by_id(
                    uuid=vector_id,
                    class_name=index_name,
                    with_vector=True
                )
                
                if response:
                    return VectorRecord(
                        id=vector_id,
                        vector=response.get("vector", []),
                        metadata=response.get("properties", {}),
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
        """Get statistics about a Weaviate class."""
        try:
            def _get_stats():
                # Get class schema
                schema = self._client.schema.get(index_name)
                
                # Get object count
                response = self._client.query.aggregate(index_name).with_meta_count().do()
                
                count = 0
                if "data" in response and "Aggregate" in response["data"]:
                    aggregate_data = response["data"]["Aggregate"].get(index_name, [])
                    if aggregate_data:
                        count = aggregate_data[0].get("meta", {}).get("count", 0)
                
                return {
                    "total_vector_count": count,
                    "class_name": index_name,
                    "properties": schema.get("properties", []),
                    "vectorizer": schema.get("vectorizer", "none")
                }
            
            loop = asyncio.get_event_loop()
            stats = await loop.run_in_executor(self.executor, _get_stats)
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get stats for class {index_name}: {str(e)}")
            return {}
    
    async def update_metadata(self, index_name: str, vector_id: str,
                            metadata: Dict[str, Any], 
                            namespace: Optional[str] = None) -> bool:
        """Update metadata for a specific vector in Weaviate."""
        try:
            def _update_metadata():
                # Get current object
                current_obj = self._client.data_object.get_by_id(
                    uuid=vector_id,
                    class_name=index_name
                )
                
                if not current_obj:
                    return False
                
                # Update properties
                properties = current_obj.get("properties", {})
                properties["metadata"] = metadata
                
                # Replace the object
                self._client.data_object.replace(
                    uuid=vector_id,
                    class_name=index_name,
                    data_object=properties
                )
                
                return True
            
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(self.executor, _update_metadata)
            
            logger.info(f"Updated metadata for vector {vector_id} in class {index_name}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to update metadata for {vector_id} in {index_name}: {str(e)}")
            return False
    
    def _convert_filters_to_weaviate(self, filters: List[QueryFilter]) -> Dict[str, Any]:
        """Convert QueryFilter objects to Weaviate where filter format."""
        if not filters:
            return {}
        
        conditions = []
        
        for filter_obj in filters:
            field = f"metadata.{filter_obj.field}"
            operator = filter_obj.operator
            value = filter_obj.value
            
            condition = {"path": [field]}
            
            if operator == "eq":
                condition["operator"] = "Equal"
                condition["valueText"] = str(value)
            elif operator == "ne":
                condition["operator"] = "NotEqual"
                condition["valueText"] = str(value)
            elif operator == "gt":
                condition["operator"] = "GreaterThan"
                condition["valueNumber"] = float(value)
            elif operator == "gte":
                condition["operator"] = "GreaterThanEqual"
                condition["valueNumber"] = float(value)
            elif operator == "lt":
                condition["operator"] = "LessThan"
                condition["valueNumber"] = float(value)
            elif operator == "lte":
                condition["operator"] = "LessThanEqual"
                condition["valueNumber"] = float(value)
            
            conditions.append(condition)
        
        if len(conditions) == 1:
            return conditions[0]
        else:
            return {
                "operator": "And",
                "operands": conditions
            }