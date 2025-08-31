"""
Embedding Service

Service for generating, storing, and managing embeddings for profiles, jobs,
skills, and learning resources using vector databases.
"""

import logging
import asyncio
from typing import List, Dict, Any, Optional, Union, Tuple
import numpy as np
from datetime import datetime
import hashlib
import json

from ..core.vector_db_config import vector_db_config
from .vector_db.vector_db_manager import vector_db_manager
from .vector_db.base_vector_db import VectorRecord, SearchResult, QueryFilter

# Import NLP engine for embedding generation
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..', 'machinelearningmodel'))

try:
    from nlp_engine import NLPEngine
except ImportError:
    # Fallback for testing
    NLPEngine = None

logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    Service for managing embeddings across different data types.
    Handles generation, storage, retrieval, and similarity search.
    """
    
    def __init__(self):
        """Initialize embedding service."""
        self.nlp_engine = None
        self._embedding_cache = {}  # Simple in-memory cache
        
        logger.info("Embedding service initialized")
    
    async def initialize(self) -> None:
        """Initialize the embedding service and dependencies."""
        try:
            # Initialize NLP engine for embedding generation
            if NLPEngine:
                self.nlp_engine = NLPEngine()
            else:
                logger.warning("NLP Engine not available, using mock embeddings")
            
            # Initialize vector database manager
            await vector_db_manager.initialize()
            
            logger.info("Embedding service initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize embedding service: {str(e)}")
            raise
    
    async def generate_embedding(self, text: str, cache_key: Optional[str] = None) -> np.ndarray:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
            cache_key: Optional cache key for the embedding
            
        Returns:
            Embedding vector as numpy array
        """
        # Check cache first
        if cache_key and cache_key in self._embedding_cache:
            return self._embedding_cache[cache_key]
        
        try:
            if self.nlp_engine:
                embeddings = await self.nlp_engine.generate_embeddings([text])
                embedding = embeddings[0]
            else:
                # Mock embedding for testing
                embedding = np.random.rand(vector_db_config.EMBEDDING_DIMENSION)
            
            # Cache the embedding
            if cache_key:
                self._embedding_cache[cache_key] = embedding
            
            return embedding
            
        except Exception as e:
            logger.error(f"Failed to generate embedding: {str(e)}")
            raise
    
    async def generate_embeddings_batch(self, texts: List[str]) -> List[np.ndarray]:
        """
        Generate embeddings for multiple texts efficiently.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        try:
            if self.nlp_engine:
                embeddings = await self.nlp_engine.generate_embeddings(texts)
                return [embeddings[i] for i in range(len(texts))]
            else:
                # Mock embeddings for testing
                return [np.random.rand(vector_db_config.EMBEDDING_DIMENSION) for _ in texts]
                
        except Exception as e:
            logger.error(f"Failed to generate batch embeddings: {str(e)}")
            raise
    
    # Profile Embedding Methods
    
    async def store_profile_embedding(self, user_id: str, profile_data: Dict[str, Any]) -> bool:
        """
        Generate and store embedding for user profile.
        
        Args:
            user_id: Unique user identifier
            profile_data: Profile data including skills, experience, etc.
            
        Returns:
            True if successful
        """
        try:
            # Create profile text for embedding
            profile_text = self._create_profile_text(profile_data)
            
            # Generate embedding
            embedding = await self.generate_embedding(profile_text, f"profile_{user_id}")
            
            # Create vector record
            vector_record = VectorRecord(
                id=user_id,
                vector=embedding,
                metadata={
                    "user_id": user_id,
                    "experience_level": profile_data.get("experience_level", ""),
                    "primary_skills": profile_data.get("primary_skills", []),
                    "dream_job": profile_data.get("dream_job", ""),
                    "created_at": datetime.utcnow().isoformat(),
                    "profile_hash": self._hash_profile_data(profile_data)
                }
            )
            
            # Store in vector database
            client = vector_db_manager.get_client("profiles")
            index_name = vector_db_manager.get_index_name("profiles")
            
            success = await client.upsert_vectors(index_name, [vector_record])
            
            if success:
                logger.info(f"Stored profile embedding for user {user_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to store profile embedding for user {user_id}: {str(e)}")
            return False
    
    async def search_similar_profiles(self, user_id: str, top_k: int = 10,
                                    filters: Optional[List[QueryFilter]] = None) -> List[SearchResult]:
        """
        Find profiles similar to the given user.
        
        Args:
            user_id: User ID to find similar profiles for
            top_k: Number of similar profiles to return
            filters: Optional filters to apply
            
        Returns:
            List of similar profiles
        """
        try:
            # Get user's profile embedding
            client = vector_db_manager.get_client("profiles")
            index_name = vector_db_manager.get_index_name("profiles")
            
            user_vector = await client.get_vector(index_name, user_id)
            if not user_vector:
                logger.warning(f"Profile embedding not found for user {user_id}")
                return []
            
            # Search for similar profiles
            results = await client.search_vectors(
                index_name=index_name,
                query_vector=user_vector.vector,
                top_k=top_k + 1,  # +1 to exclude self
                filters=filters
            )
            
            # Filter out the user's own profile
            filtered_results = [r for r in results if r.id != user_id][:top_k]
            
            logger.info(f"Found {len(filtered_results)} similar profiles for user {user_id}")
            return filtered_results
            
        except Exception as e:
            logger.error(f"Failed to search similar profiles for user {user_id}: {str(e)}")
            return []
    
    # Job Embedding Methods
    
    async def store_job_embedding(self, job_id: str, job_data: Dict[str, Any]) -> bool:
        """
        Generate and store embedding for job posting.
        
        Args:
            job_id: Unique job identifier
            job_data: Job data including title, description, requirements
            
        Returns:
            True if successful
        """
        try:
            # Create job text for embedding
            job_text = self._create_job_text(job_data)
            
            # Generate embedding
            embedding = await self.generate_embedding(job_text, f"job_{job_id}")
            
            # Create vector record
            vector_record = VectorRecord(
                id=job_id,
                vector=embedding,
                metadata={
                    "job_id": job_id,
                    "title": job_data.get("title", ""),
                    "company": job_data.get("company", ""),
                    "location": job_data.get("location", ""),
                    "experience_level": job_data.get("experience_level", ""),
                    "salary_range": job_data.get("salary_range", []),
                    "required_skills": job_data.get("required_skills", []),
                    "created_at": datetime.utcnow().isoformat()
                }
            )
            
            # Store in vector database
            client = vector_db_manager.get_client("jobs")
            index_name = vector_db_manager.get_index_name("jobs")
            
            success = await client.upsert_vectors(index_name, [vector_record])
            
            if success:
                logger.info(f"Stored job embedding for job {job_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to store job embedding for job {job_id}: {str(e)}")
            return False
    
    async def search_matching_jobs(self, user_id: str, top_k: int = 20,
                                 filters: Optional[List[QueryFilter]] = None) -> List[SearchResult]:
        """
        Find jobs matching user's profile.
        
        Args:
            user_id: User ID to find matching jobs for
            top_k: Number of matching jobs to return
            filters: Optional filters to apply
            
        Returns:
            List of matching jobs
        """
        try:
            # Get user's profile embedding
            profiles_client = vector_db_manager.get_client("profiles")
            profiles_index = vector_db_manager.get_index_name("profiles")
            
            user_vector = await profiles_client.get_vector(profiles_index, user_id)
            if not user_vector:
                logger.warning(f"Profile embedding not found for user {user_id}")
                return []
            
            # Search for matching jobs
            jobs_client = vector_db_manager.get_client("jobs")
            jobs_index = vector_db_manager.get_index_name("jobs")
            
            results = await jobs_client.search_vectors(
                index_name=jobs_index,
                query_vector=user_vector.vector,
                top_k=top_k,
                filters=filters
            )
            
            logger.info(f"Found {len(results)} matching jobs for user {user_id}")
            return results
            
        except Exception as e:
            logger.error(f"Failed to search matching jobs for user {user_id}: {str(e)}")
            return []
    
    # Skill Embedding Methods
    
    async def store_skill_embedding(self, skill_name: str, skill_data: Dict[str, Any]) -> bool:
        """
        Generate and store embedding for skill.
        
        Args:
            skill_name: Name of the skill
            skill_data: Skill data including description, category
            
        Returns:
            True if successful
        """
        try:
            # Create skill text for embedding
            skill_text = self._create_skill_text(skill_name, skill_data)
            
            # Generate embedding
            skill_id = self._generate_skill_id(skill_name)
            embedding = await self.generate_embedding(skill_text, f"skill_{skill_id}")
            
            # Create vector record
            vector_record = VectorRecord(
                id=skill_id,
                vector=embedding,
                metadata={
                    "skill_name": skill_name,
                    "category": skill_data.get("category", ""),
                    "description": skill_data.get("description", ""),
                    "popularity_score": skill_data.get("popularity_score", 0.0),
                    "created_at": datetime.utcnow().isoformat()
                }
            )
            
            # Store in vector database
            client = vector_db_manager.get_client("skills")
            index_name = vector_db_manager.get_index_name("skills")
            
            success = await client.upsert_vectors(index_name, [vector_record])
            
            if success:
                logger.info(f"Stored skill embedding for skill {skill_name}")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to store skill embedding for skill {skill_name}: {str(e)}")
            return False
    
    async def find_similar_skills(self, skill_name: str, top_k: int = 10) -> List[SearchResult]:
        """
        Find skills similar to the given skill.
        
        Args:
            skill_name: Skill name to find similar skills for
            top_k: Number of similar skills to return
            
        Returns:
            List of similar skills
        """
        try:
            # Generate embedding for the query skill
            skill_text = f"Skill: {skill_name}"
            query_embedding = await self.generate_embedding(skill_text)
            
            # Search for similar skills
            client = vector_db_manager.get_client("skills")
            index_name = vector_db_manager.get_index_name("skills")
            
            results = await client.search_vectors(
                index_name=index_name,
                query_vector=query_embedding,
                top_k=top_k
            )
            
            logger.info(f"Found {len(results)} similar skills for {skill_name}")
            return results
            
        except Exception as e:
            logger.error(f"Failed to find similar skills for {skill_name}: {str(e)}")
            return []
    
    # Learning Resource Embedding Methods
    
    async def store_learning_resource_embedding(self, resource_id: str, 
                                              resource_data: Dict[str, Any]) -> bool:
        """
        Generate and store embedding for learning resource.
        
        Args:
            resource_id: Unique resource identifier
            resource_data: Resource data including title, description, skills
            
        Returns:
            True if successful
        """
        try:
            # Create resource text for embedding
            resource_text = self._create_resource_text(resource_data)
            
            # Generate embedding
            embedding = await self.generate_embedding(resource_text, f"resource_{resource_id}")
            
            # Create vector record
            vector_record = VectorRecord(
                id=resource_id,
                vector=embedding,
                metadata={
                    "resource_id": resource_id,
                    "title": resource_data.get("title", ""),
                    "type": resource_data.get("type", ""),
                    "provider": resource_data.get("provider", ""),
                    "difficulty_level": resource_data.get("difficulty_level", ""),
                    "rating": resource_data.get("rating", 0.0),
                    "target_skills": resource_data.get("target_skills", []),
                    "created_at": datetime.utcnow().isoformat()
                }
            )
            
            # Store in vector database
            client = vector_db_manager.get_client("learning_resources")
            index_name = vector_db_manager.get_index_name("learning_resources")
            
            success = await client.upsert_vectors(index_name, [vector_record])
            
            if success:
                logger.info(f"Stored learning resource embedding for resource {resource_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to store learning resource embedding for resource {resource_id}: {str(e)}")
            return False
    
    async def search_learning_resources(self, skill_gaps: List[str], top_k: int = 15,
                                      filters: Optional[List[QueryFilter]] = None) -> List[SearchResult]:
        """
        Find learning resources for skill gaps.
        
        Args:
            skill_gaps: List of skills to learn
            top_k: Number of resources to return
            filters: Optional filters to apply
            
        Returns:
            List of relevant learning resources
        """
        try:
            # Create query text from skill gaps
            query_text = f"Learning resources for skills: {', '.join(skill_gaps)}"
            query_embedding = await self.generate_embedding(query_text)
            
            # Search for relevant resources
            client = vector_db_manager.get_client("learning_resources")
            index_name = vector_db_manager.get_index_name("learning_resources")
            
            results = await client.search_vectors(
                index_name=index_name,
                query_vector=query_embedding,
                top_k=top_k,
                filters=filters
            )
            
            logger.info(f"Found {len(results)} learning resources for skill gaps")
            return results
            
        except Exception as e:
            logger.error(f"Failed to search learning resources: {str(e)}")
            return []
    
    # Utility Methods
    
    def _create_profile_text(self, profile_data: Dict[str, Any]) -> str:
        """Create text representation of user profile for embedding."""
        parts = []
        
        if "skills" in profile_data:
            parts.append(f"Skills: {', '.join(profile_data['skills'])}")
        
        if "experience_level" in profile_data:
            parts.append(f"Experience Level: {profile_data['experience_level']}")
        
        if "dream_job" in profile_data:
            parts.append(f"Target Role: {profile_data['dream_job']}")
        
        if "experience" in profile_data:
            exp_texts = []
            for exp in profile_data["experience"]:
                if isinstance(exp, dict):
                    exp_texts.append(f"{exp.get('position', '')} at {exp.get('company', '')}")
            if exp_texts:
                parts.append(f"Experience: {'; '.join(exp_texts)}")
        
        if "education" in profile_data:
            edu_texts = []
            for edu in profile_data["education"]:
                if isinstance(edu, dict):
                    edu_texts.append(edu.get("degree_institution", ""))
            if edu_texts:
                parts.append(f"Education: {'; '.join(edu_texts)}")
        
        return ". ".join(parts)
    
    def _create_job_text(self, job_data: Dict[str, Any]) -> str:
        """Create text representation of job posting for embedding."""
        parts = []
        
        if "title" in job_data:
            parts.append(f"Job Title: {job_data['title']}")
        
        if "company" in job_data:
            parts.append(f"Company: {job_data['company']}")
        
        if "description" in job_data:
            parts.append(f"Description: {job_data['description']}")
        
        if "required_skills" in job_data:
            parts.append(f"Required Skills: {', '.join(job_data['required_skills'])}")
        
        if "experience_level" in job_data:
            parts.append(f"Experience Level: {job_data['experience_level']}")
        
        return ". ".join(parts)
    
    def _create_skill_text(self, skill_name: str, skill_data: Dict[str, Any]) -> str:
        """Create text representation of skill for embedding."""
        parts = [f"Skill: {skill_name}"]
        
        if "description" in skill_data:
            parts.append(f"Description: {skill_data['description']}")
        
        if "category" in skill_data:
            parts.append(f"Category: {skill_data['category']}")
        
        return ". ".join(parts)
    
    def _create_resource_text(self, resource_data: Dict[str, Any]) -> str:
        """Create text representation of learning resource for embedding."""
        parts = []
        
        if "title" in resource_data:
            parts.append(f"Course: {resource_data['title']}")
        
        if "description" in resource_data:
            parts.append(f"Description: {resource_data['description']}")
        
        if "target_skills" in resource_data:
            parts.append(f"Skills Taught: {', '.join(resource_data['target_skills'])}")
        
        if "type" in resource_data:
            parts.append(f"Type: {resource_data['type']}")
        
        if "difficulty_level" in resource_data:
            parts.append(f"Level: {resource_data['difficulty_level']}")
        
        return ". ".join(parts)
    
    def _hash_profile_data(self, profile_data: Dict[str, Any]) -> str:
        """Generate hash of profile data for change detection."""
        profile_str = json.dumps(profile_data, sort_keys=True)
        return hashlib.md5(profile_str.encode()).hexdigest()
    
    def _generate_skill_id(self, skill_name: str) -> str:
        """Generate consistent ID for skill name."""
        return hashlib.md5(skill_name.lower().encode()).hexdigest()
    
    async def update_embedding(self, index_type: str, item_id: str, 
                             item_data: Dict[str, Any]) -> bool:
        """
        Update embedding for an existing item.
        
        Args:
            index_type: Type of index ('profiles', 'jobs', 'skills', 'learning_resources')
            item_id: ID of the item to update
            item_data: Updated item data
            
        Returns:
            True if successful
        """
        try:
            if index_type == "profiles":
                return await self.store_profile_embedding(item_id, item_data)
            elif index_type == "jobs":
                return await self.store_job_embedding(item_id, item_data)
            elif index_type == "skills":
                skill_name = item_data.get("skill_name", item_id)
                return await self.store_skill_embedding(skill_name, item_data)
            elif index_type == "learning_resources":
                return await self.store_learning_resource_embedding(item_id, item_data)
            else:
                raise ValueError(f"Unknown index type: {index_type}")
                
        except Exception as e:
            logger.error(f"Failed to update embedding for {index_type}/{item_id}: {str(e)}")
            return False
    
    async def delete_embedding(self, index_type: str, item_id: str) -> bool:
        """
        Delete embedding for an item.
        
        Args:
            index_type: Type of index
            item_id: ID of the item to delete
            
        Returns:
            True if successful
        """
        try:
            client = vector_db_manager.get_client(index_type)
            index_name = vector_db_manager.get_index_name(index_type)
            
            success = await client.delete_vectors(index_name, [item_id])
            
            if success:
                logger.info(f"Deleted embedding for {index_type}/{item_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to delete embedding for {index_type}/{item_id}: {str(e)}")
            return False
    
    async def cleanup(self) -> None:
        """Clean up embedding service resources."""
        self._embedding_cache.clear()
        
        if self.nlp_engine and hasattr(self.nlp_engine, 'cleanup'):
            self.nlp_engine.cleanup()
        
        await vector_db_manager.cleanup()
        
        logger.info("Embedding service cleaned up")


# Global embedding service instance
embedding_service = EmbeddingService()