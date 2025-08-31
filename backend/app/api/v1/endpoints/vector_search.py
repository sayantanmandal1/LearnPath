"""
Vector Search API Endpoints

REST API endpoints for semantic search and vector database operations.
Provides access to similarity search, embedding management, and search analytics.
"""

import logging
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Depends, Query, BackgroundTasks
from pydantic import BaseModel, Field

from ....services.embedding_service import embedding_service
from ....services.semantic_search_service import semantic_search_service, SearchRequest, SearchType
from ....services.vector_db.base_vector_db import QueryFilter
from ....core.exceptions import APIException
from ...dependencies import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/vector-search", tags=["Vector Search"])


# Request/Response Models

class EmbeddingRequest(BaseModel):
    """Request model for generating embeddings."""
    text: str = Field(..., description="Text to generate embedding for")
    cache_key: Optional[str] = Field(None, description="Optional cache key")


class EmbeddingResponse(BaseModel):
    """Response model for embedding generation."""
    embedding: List[float] = Field(..., description="Generated embedding vector")
    dimension: int = Field(..., description="Embedding dimension")


class ProfileEmbeddingRequest(BaseModel):
    """Request model for storing profile embedding."""
    user_id: str = Field(..., description="User ID")
    profile_data: Dict[str, Any] = Field(..., description="Profile data")


class JobEmbeddingRequest(BaseModel):
    """Request model for storing job embedding."""
    job_id: str = Field(..., description="Job ID")
    job_data: Dict[str, Any] = Field(..., description="Job data")


class SkillEmbeddingRequest(BaseModel):
    """Request model for storing skill embedding."""
    skill_name: str = Field(..., description="Skill name")
    skill_data: Dict[str, Any] = Field(..., description="Skill data")


class ResourceEmbeddingRequest(BaseModel):
    """Request model for storing learning resource embedding."""
    resource_id: str = Field(..., description="Resource ID")
    resource_data: Dict[str, Any] = Field(..., description="Resource data")


class SearchFilter(BaseModel):
    """Search filter model."""
    field: str = Field(..., description="Field to filter on")
    operator: str = Field(..., description="Filter operator")
    value: Any = Field(..., description="Filter value")


class SemanticSearchRequest(BaseModel):
    """Request model for semantic search."""
    search_type: str = Field(..., description="Type of search")
    query_data: Dict[str, Any] = Field(..., description="Query parameters")
    top_k: int = Field(default=10, ge=1, le=100, description="Number of results")
    filters: Optional[List[SearchFilter]] = Field(None, description="Optional filters")
    min_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Minimum similarity score")


class SearchMatch(BaseModel):
    """Search match response model."""
    id: str = Field(..., description="Match ID")
    score: float = Field(..., description="Similarity score")
    metadata: Dict[str, Any] = Field(..., description="Match metadata")
    match_reasons: List[str] = Field(..., description="Reasons for match")
    confidence_level: str = Field(..., description="Confidence level")


class SemanticSearchResponse(BaseModel):
    """Response model for semantic search."""
    matches: List[SearchMatch] = Field(..., description="Search matches")
    total_results: int = Field(..., description="Total number of results")
    search_type: str = Field(..., description="Type of search performed")
    explanation: Optional[Dict[str, Any]] = Field(None, description="Search explanation")


class HealthCheckResponse(BaseModel):
    """Health check response model."""
    status: str = Field(..., description="Overall health status")
    provider: str = Field(..., description="Vector database provider")
    clients: Dict[str, Any] = Field(..., description="Client health status")


# Embedding Management Endpoints

@router.post("/embeddings/generate", response_model=EmbeddingResponse)
async def generate_embedding(
    request: EmbeddingRequest,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Generate embedding for text."""
    try:
        embedding = await embedding_service.generate_embedding(
            text=request.text,
            cache_key=request.cache_key
        )
        
        return EmbeddingResponse(
            embedding=embedding.tolist(),
            dimension=len(embedding)
        )
        
    except Exception as e:
        logger.error(f"Failed to generate embedding: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to generate embedding")


@router.post("/embeddings/profiles")
async def store_profile_embedding(
    request: ProfileEmbeddingRequest,
    background_tasks: BackgroundTasks,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Store embedding for user profile."""
    try:
        # Store embedding in background
        background_tasks.add_task(
            embedding_service.store_profile_embedding,
            request.user_id,
            request.profile_data
        )
        
        return {"message": "Profile embedding storage initiated", "user_id": request.user_id}
        
    except Exception as e:
        logger.error(f"Failed to store profile embedding: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to store profile embedding")


@router.post("/embeddings/jobs")
async def store_job_embedding(
    request: JobEmbeddingRequest,
    background_tasks: BackgroundTasks,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Store embedding for job posting."""
    try:
        # Store embedding in background
        background_tasks.add_task(
            embedding_service.store_job_embedding,
            request.job_id,
            request.job_data
        )
        
        return {"message": "Job embedding storage initiated", "job_id": request.job_id}
        
    except Exception as e:
        logger.error(f"Failed to store job embedding: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to store job embedding")


@router.post("/embeddings/skills")
async def store_skill_embedding(
    request: SkillEmbeddingRequest,
    background_tasks: BackgroundTasks,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Store embedding for skill."""
    try:
        # Store embedding in background
        background_tasks.add_task(
            embedding_service.store_skill_embedding,
            request.skill_name,
            request.skill_data
        )
        
        return {"message": "Skill embedding storage initiated", "skill_name": request.skill_name}
        
    except Exception as e:
        logger.error(f"Failed to store skill embedding: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to store skill embedding")


@router.post("/embeddings/resources")
async def store_resource_embedding(
    request: ResourceEmbeddingRequest,
    background_tasks: BackgroundTasks,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Store embedding for learning resource."""
    try:
        # Store embedding in background
        background_tasks.add_task(
            embedding_service.store_learning_resource_embedding,
            request.resource_id,
            request.resource_data
        )
        
        return {"message": "Resource embedding storage initiated", "resource_id": request.resource_id}
        
    except Exception as e:
        logger.error(f"Failed to store resource embedding: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to store resource embedding")


# Semantic Search Endpoints

@router.post("/search", response_model=SemanticSearchResponse)
async def semantic_search(
    request: SemanticSearchRequest,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Perform semantic search."""
    try:
        # Convert search type string to enum
        try:
            search_type = SearchType(request.search_type)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid search type: {request.search_type}")
        
        # Convert filters
        filters = None
        if request.filters:
            filters = [
                QueryFilter(field=f.field, operator=f.operator, value=f.value)
                for f in request.filters
            ]
        
        # Create search request
        search_req = SearchRequest(
            search_type=search_type,
            query_data=request.query_data,
            top_k=request.top_k,
            filters=filters,
            min_score=request.min_score
        )
        
        # Perform search
        matches = await semantic_search_service.search(search_req)
        
        # Get explanation
        explanation = await semantic_search_service.explain_search_results(matches)
        
        # Convert matches to response format
        response_matches = [
            SearchMatch(
                id=match.id,
                score=match.score,
                metadata=match.metadata,
                match_reasons=match.match_reasons,
                confidence_level=match.confidence_level
            )
            for match in matches
        ]
        
        return SemanticSearchResponse(
            matches=response_matches,
            total_results=len(response_matches),
            search_type=request.search_type,
            explanation=explanation
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Semantic search failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Semantic search failed")


@router.get("/search/similar-profiles/{user_id}")
async def find_similar_profiles(
    user_id: str,
    top_k: int = Query(default=10, ge=1, le=50),
    min_score: float = Query(default=0.0, ge=0.0, le=1.0),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Find profiles similar to a specific user."""
    try:
        request = SemanticSearchRequest(
            search_type="profile_similarity",
            query_data={"user_id": user_id},
            top_k=top_k,
            min_score=min_score
        )
        
        return await semantic_search(request, current_user)
        
    except Exception as e:
        logger.error(f"Failed to find similar profiles: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to find similar profiles")


@router.get("/search/matching-jobs/{user_id}")
async def find_matching_jobs(
    user_id: str,
    top_k: int = Query(default=20, ge=1, le=100),
    min_score: float = Query(default=0.0, ge=0.0, le=1.0),
    experience_level: Optional[str] = Query(None),
    location: Optional[str] = Query(None),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Find jobs matching a user's profile."""
    try:
        # Build filters
        filters = []
        if experience_level:
            filters.append(SearchFilter(field="experience_level", operator="eq", value=experience_level))
        if location:
            filters.append(SearchFilter(field="location", operator="eq", value=location))
        
        request = SemanticSearchRequest(
            search_type="job_matching",
            query_data={"user_id": user_id},
            top_k=top_k,
            filters=filters if filters else None,
            min_score=min_score
        )
        
        return await semantic_search(request, current_user)
        
    except Exception as e:
        logger.error(f"Failed to find matching jobs: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to find matching jobs")


@router.get("/search/similar-skills/{skill_name}")
async def find_similar_skills(
    skill_name: str,
    top_k: int = Query(default=10, ge=1, le=50),
    min_score: float = Query(default=0.0, ge=0.0, le=1.0),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Find skills similar to a specific skill."""
    try:
        request = SemanticSearchRequest(
            search_type="skill_similarity",
            query_data={"skill_name": skill_name},
            top_k=top_k,
            min_score=min_score
        )
        
        return await semantic_search(request, current_user)
        
    except Exception as e:
        logger.error(f"Failed to find similar skills: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to find similar skills")


@router.post("/search/learning-resources")
async def find_learning_resources(
    skill_gaps: List[str],
    top_k: int = Query(default=15, ge=1, le=50),
    min_score: float = Query(default=0.0, ge=0.0, le=1.0),
    difficulty_level: Optional[str] = Query(None),
    resource_type: Optional[str] = Query(None),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Find learning resources for skill gaps."""
    try:
        # Build filters
        filters = []
        if difficulty_level:
            filters.append(SearchFilter(field="difficulty_level", operator="eq", value=difficulty_level))
        if resource_type:
            filters.append(SearchFilter(field="type", operator="eq", value=resource_type))
        
        request = SemanticSearchRequest(
            search_type="learning_resources",
            query_data={"skill_gaps": skill_gaps},
            top_k=top_k,
            filters=filters if filters else None,
            min_score=min_score
        )
        
        return await semantic_search(request, current_user)
        
    except Exception as e:
        logger.error(f"Failed to find learning resources: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to find learning resources")


# Management Endpoints

@router.delete("/embeddings/{index_type}/{item_id}")
async def delete_embedding(
    index_type: str,
    item_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Delete embedding for an item."""
    try:
        valid_types = ["profiles", "jobs", "skills", "learning_resources"]
        if index_type not in valid_types:
            raise HTTPException(status_code=400, detail=f"Invalid index type. Must be one of: {valid_types}")
        
        success = await embedding_service.delete_embedding(index_type, item_id)
        
        if success:
            return {"message": f"Embedding deleted successfully", "index_type": index_type, "item_id": item_id}
        else:
            raise HTTPException(status_code=404, detail="Embedding not found or deletion failed")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete embedding: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to delete embedding")


@router.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """Check vector database health."""
    try:
        from ....services.vector_db.vector_db_manager import vector_db_manager
        
        health_status = await vector_db_manager.health_check()
        
        return HealthCheckResponse(
            status=health_status["overall_status"],
            provider=health_status["provider"],
            clients=health_status["clients"]
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Health check failed")


@router.get("/search/suggestions/{search_type}")
async def get_search_suggestions(
    search_type: str,
    query: str = Query(..., min_length=1),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Get search suggestions for partial queries."""
    try:
        # Convert search type string to enum
        try:
            search_type_enum = SearchType(search_type)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid search type: {search_type}")
        
        suggestions = await semantic_search_service.get_search_suggestions(
            search_type_enum, query
        )
        
        return {"suggestions": suggestions}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get search suggestions: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get search suggestions")


# Batch Operations

@router.post("/search/batch")
async def batch_search(
    requests: List[SemanticSearchRequest],
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Perform multiple searches in batch."""
    try:
        if len(requests) > 10:
            raise HTTPException(status_code=400, detail="Maximum 10 requests allowed in batch")
        
        # Convert requests
        search_requests = []
        for req in requests:
            try:
                search_type = SearchType(req.search_type)
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid search type: {req.search_type}")
            
            filters = None
            if req.filters:
                filters = [
                    QueryFilter(field=f.field, operator=f.operator, value=f.value)
                    for f in req.filters
                ]
            
            search_requests.append(SearchRequest(
                search_type=search_type,
                query_data=req.query_data,
                top_k=req.top_k,
                filters=filters,
                min_score=req.min_score
            ))
        
        # Perform batch search
        results = await semantic_search_service.batch_search(search_requests)
        
        # Convert results
        response_results = {}
        for key, matches in results.items():
            response_results[key] = [
                SearchMatch(
                    id=match.id,
                    score=match.score,
                    metadata=match.metadata,
                    match_reasons=match.match_reasons,
                    confidence_level=match.confidence_level
                )
                for match in matches
            ]
        
        return {"results": response_results}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch search failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Batch search failed")