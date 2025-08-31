"""
Semantic Search Service

High-level service for semantic search operations across different data types.
Provides intelligent matching, ranking, and filtering capabilities.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np

from .embedding_service import embedding_service
from .vector_db.base_vector_db import QueryFilter, SearchResult

logger = logging.getLogger(__name__)


class SearchType(Enum):
    """Types of semantic searches supported."""
    PROFILE_SIMILARITY = "profile_similarity"
    JOB_MATCHING = "job_matching"
    SKILL_SIMILARITY = "skill_similarity"
    LEARNING_RESOURCES = "learning_resources"
    CAREER_PATHS = "career_paths"


@dataclass
class SearchRequest:
    """Request object for semantic search operations."""
    search_type: SearchType
    query_data: Dict[str, Any]
    top_k: int = 10
    filters: Optional[List[QueryFilter]] = None
    include_scores: bool = True
    min_score: float = 0.0


@dataclass
class SearchMatch:
    """Enhanced search result with additional context."""
    id: str
    score: float
    metadata: Dict[str, Any]
    match_reasons: List[str]
    confidence_level: str  # 'high', 'medium', 'low'
    search_type: SearchType


class SemanticSearchService:
    """
    High-level semantic search service providing intelligent matching
    and ranking across different data types.
    """
    
    def __init__(self):
        """Initialize semantic search service."""
        self.confidence_thresholds = {
            "high": 0.8,
            "medium": 0.6,
            "low": 0.4
        }
        
        logger.info("Semantic search service initialized")
    
    async def search(self, request: SearchRequest) -> List[SearchMatch]:
        """
        Perform semantic search based on request type.
        
        Args:
            request: Search request with parameters
            
        Returns:
            List of search matches with enhanced metadata
        """
        try:
            if request.search_type == SearchType.PROFILE_SIMILARITY:
                return await self._search_similar_profiles(request)
            elif request.search_type == SearchType.JOB_MATCHING:
                return await self._search_matching_jobs(request)
            elif request.search_type == SearchType.SKILL_SIMILARITY:
                return await self._search_similar_skills(request)
            elif request.search_type == SearchType.LEARNING_RESOURCES:
                return await self._search_learning_resources(request)
            elif request.search_type == SearchType.CAREER_PATHS:
                return await self._search_career_paths(request)
            else:
                raise ValueError(f"Unsupported search type: {request.search_type}")
                
        except Exception as e:
            logger.error(f"Search failed for type {request.search_type}: {str(e)}")
            return []
    
    async def _search_similar_profiles(self, request: SearchRequest) -> List[SearchMatch]:
        """Search for profiles similar to a given user."""
        user_id = request.query_data.get("user_id")
        if not user_id:
            raise ValueError("user_id is required for profile similarity search")
        
        # Get similar profiles
        results = await embedding_service.search_similar_profiles(
            user_id=user_id,
            top_k=request.top_k,
            filters=request.filters
        )
        
        # Convert to enhanced search matches
        matches = []
        for result in results:
            if result.score >= request.min_score:
                match = await self._create_profile_match(result, request.search_type)
                matches.append(match)
        
        return matches
    
    async def _search_matching_jobs(self, request: SearchRequest) -> List[SearchMatch]:
        """Search for jobs matching a user's profile."""
        user_id = request.query_data.get("user_id")
        if not user_id:
            raise ValueError("user_id is required for job matching search")
        
        # Get matching jobs
        results = await embedding_service.search_matching_jobs(
            user_id=user_id,
            top_k=request.top_k,
            filters=request.filters
        )
        
        # Convert to enhanced search matches
        matches = []
        for result in results:
            if result.score >= request.min_score:
                match = await self._create_job_match(result, request.search_type, user_id)
                matches.append(match)
        
        return matches
    
    async def _search_similar_skills(self, request: SearchRequest) -> List[SearchMatch]:
        """Search for skills similar to a given skill."""
        skill_name = request.query_data.get("skill_name")
        if not skill_name:
            raise ValueError("skill_name is required for skill similarity search")
        
        # Get similar skills
        results = await embedding_service.find_similar_skills(
            skill_name=skill_name,
            top_k=request.top_k
        )
        
        # Convert to enhanced search matches
        matches = []
        for result in results:
            if result.score >= request.min_score:
                match = await self._create_skill_match(result, request.search_type)
                matches.append(match)
        
        return matches
    
    async def _search_learning_resources(self, request: SearchRequest) -> List[SearchMatch]:
        """Search for learning resources based on skill gaps."""
        skill_gaps = request.query_data.get("skill_gaps", [])
        if not skill_gaps:
            raise ValueError("skill_gaps is required for learning resources search")
        
        # Get relevant learning resources
        results = await embedding_service.search_learning_resources(
            skill_gaps=skill_gaps,
            top_k=request.top_k,
            filters=request.filters
        )
        
        # Convert to enhanced search matches
        matches = []
        for result in results:
            if result.score >= request.min_score:
                match = await self._create_resource_match(result, request.search_type, skill_gaps)
                matches.append(match)
        
        return matches
    
    async def _search_career_paths(self, request: SearchRequest) -> List[SearchMatch]:
        """Search for career paths based on current profile and target role."""
        user_id = request.query_data.get("user_id")
        target_role = request.query_data.get("target_role")
        
        if not user_id:
            raise ValueError("user_id is required for career path search")
        
        # This is a more complex search that combines job matching with career progression
        # First, find jobs similar to the target role
        job_filters = []
        if target_role:
            job_filters.append(QueryFilter(field="title", operator="eq", value=target_role))
        
        job_results = await embedding_service.search_matching_jobs(
            user_id=user_id,
            top_k=request.top_k * 2,  # Get more results for filtering
            filters=job_filters
        )
        
        # Convert to career path matches
        matches = []
        for result in job_results:
            if result.score >= request.min_score:
                match = await self._create_career_path_match(result, request.search_type, user_id)
                matches.append(match)
        
        return matches[:request.top_k]
    
    async def _create_profile_match(self, result: SearchResult, search_type: SearchType) -> SearchMatch:
        """Create enhanced match for profile similarity."""
        match_reasons = []
        
        # Analyze similarity reasons
        metadata = result.metadata
        if "primary_skills" in metadata:
            match_reasons.append(f"Similar skills: {', '.join(metadata['primary_skills'][:3])}")
        
        if "experience_level" in metadata:
            match_reasons.append(f"Similar experience level: {metadata['experience_level']}")
        
        confidence_level = self._calculate_confidence_level(result.score)
        
        return SearchMatch(
            id=result.id,
            score=result.score,
            metadata=metadata,
            match_reasons=match_reasons,
            confidence_level=confidence_level,
            search_type=search_type
        )
    
    async def _create_job_match(self, result: SearchResult, search_type: SearchType, user_id: str) -> SearchMatch:
        """Create enhanced match for job matching."""
        match_reasons = []
        
        # Analyze job match reasons
        metadata = result.metadata
        if "title" in metadata:
            match_reasons.append(f"Role match: {metadata['title']}")
        
        if "required_skills" in metadata:
            skills = metadata["required_skills"][:3]
            match_reasons.append(f"Skill alignment: {', '.join(skills)}")
        
        if "experience_level" in metadata:
            match_reasons.append(f"Experience level: {metadata['experience_level']}")
        
        confidence_level = self._calculate_confidence_level(result.score)
        
        return SearchMatch(
            id=result.id,
            score=result.score,
            metadata=metadata,
            match_reasons=match_reasons,
            confidence_level=confidence_level,
            search_type=search_type
        )
    
    async def _create_skill_match(self, result: SearchResult, search_type: SearchType) -> SearchMatch:
        """Create enhanced match for skill similarity."""
        match_reasons = []
        
        # Analyze skill similarity reasons
        metadata = result.metadata
        if "category" in metadata:
            match_reasons.append(f"Same category: {metadata['category']}")
        
        if "skill_name" in metadata:
            match_reasons.append(f"Similar skill: {metadata['skill_name']}")
        
        confidence_level = self._calculate_confidence_level(result.score)
        
        return SearchMatch(
            id=result.id,
            score=result.score,
            metadata=metadata,
            match_reasons=match_reasons,
            confidence_level=confidence_level,
            search_type=search_type
        )
    
    async def _create_resource_match(self, result: SearchResult, search_type: SearchType, 
                                   skill_gaps: List[str]) -> SearchMatch:
        """Create enhanced match for learning resources."""
        match_reasons = []
        
        # Analyze resource match reasons
        metadata = result.metadata
        if "title" in metadata:
            match_reasons.append(f"Course: {metadata['title']}")
        
        if "target_skills" in metadata:
            target_skills = metadata["target_skills"]
            matching_skills = [skill for skill in target_skills if skill in skill_gaps]
            if matching_skills:
                match_reasons.append(f"Teaches: {', '.join(matching_skills[:3])}")
        
        if "difficulty_level" in metadata:
            match_reasons.append(f"Level: {metadata['difficulty_level']}")
        
        if "rating" in metadata and metadata["rating"] > 0:
            match_reasons.append(f"Rating: {metadata['rating']}/5.0")
        
        confidence_level = self._calculate_confidence_level(result.score)
        
        return SearchMatch(
            id=result.id,
            score=result.score,
            metadata=metadata,
            match_reasons=match_reasons,
            confidence_level=confidence_level,
            search_type=search_type
        )
    
    async def _create_career_path_match(self, result: SearchResult, search_type: SearchType, 
                                      user_id: str) -> SearchMatch:
        """Create enhanced match for career paths."""
        match_reasons = []
        
        # Analyze career path reasons
        metadata = result.metadata
        if "title" in metadata:
            match_reasons.append(f"Target role: {metadata['title']}")
        
        if "company" in metadata:
            match_reasons.append(f"Company: {metadata['company']}")
        
        if "required_skills" in metadata:
            skills = metadata["required_skills"][:3]
            match_reasons.append(f"Skills needed: {', '.join(skills)}")
        
        confidence_level = self._calculate_confidence_level(result.score)
        
        return SearchMatch(
            id=result.id,
            score=result.score,
            metadata=metadata,
            match_reasons=match_reasons,
            confidence_level=confidence_level,
            search_type=search_type
        )
    
    def _calculate_confidence_level(self, score: float) -> str:
        """Calculate confidence level based on similarity score."""
        if score >= self.confidence_thresholds["high"]:
            return "high"
        elif score >= self.confidence_thresholds["medium"]:
            return "medium"
        else:
            return "low"
    
    async def batch_search(self, requests: List[SearchRequest]) -> Dict[str, List[SearchMatch]]:
        """
        Perform multiple searches in batch.
        
        Args:
            requests: List of search requests
            
        Returns:
            Dictionary mapping request index to results
        """
        results = {}
        
        for i, request in enumerate(requests):
            try:
                matches = await self.search(request)
                results[str(i)] = matches
            except Exception as e:
                logger.error(f"Batch search failed for request {i}: {str(e)}")
                results[str(i)] = []
        
        return results
    
    async def get_search_suggestions(self, search_type: SearchType, 
                                   partial_query: str) -> List[Dict[str, Any]]:
        """
        Get search suggestions based on partial query.
        
        Args:
            search_type: Type of search
            partial_query: Partial search query
            
        Returns:
            List of search suggestions
        """
        suggestions = []
        
        try:
            if search_type == SearchType.SKILL_SIMILARITY:
                # Find skills that match the partial query
                # This would typically query a skills database
                suggestions = [
                    {"text": "Python", "type": "skill"},
                    {"text": "JavaScript", "type": "skill"},
                    {"text": "Machine Learning", "type": "skill"}
                ]
            elif search_type == SearchType.JOB_MATCHING:
                # Find job titles that match the partial query
                suggestions = [
                    {"text": "Software Engineer", "type": "job_title"},
                    {"text": "Data Scientist", "type": "job_title"},
                    {"text": "Product Manager", "type": "job_title"}
                ]
            
            # Filter suggestions based on partial query
            if partial_query:
                suggestions = [
                    s for s in suggestions 
                    if partial_query.lower() in s["text"].lower()
                ]
            
        except Exception as e:
            logger.error(f"Failed to get search suggestions: {str(e)}")
        
        return suggestions[:10]  # Limit to 10 suggestions
    
    async def explain_search_results(self, matches: List[SearchMatch]) -> Dict[str, Any]:
        """
        Provide explanation for search results.
        
        Args:
            matches: List of search matches
            
        Returns:
            Explanation of search results
        """
        if not matches:
            return {"explanation": "No matches found", "insights": []}
        
        # Analyze the results
        high_confidence_count = sum(1 for m in matches if m.confidence_level == "high")
        medium_confidence_count = sum(1 for m in matches if m.confidence_level == "medium")
        low_confidence_count = sum(1 for m in matches if m.confidence_level == "low")
        
        avg_score = sum(m.score for m in matches) / len(matches)
        
        insights = []
        
        if high_confidence_count > 0:
            insights.append(f"{high_confidence_count} high-confidence matches found")
        
        if avg_score > 0.8:
            insights.append("Very strong matches found")
        elif avg_score > 0.6:
            insights.append("Good matches found")
        else:
            insights.append("Moderate matches found - consider refining search criteria")
        
        # Analyze common themes in match reasons
        all_reasons = []
        for match in matches:
            all_reasons.extend(match.match_reasons)
        
        if all_reasons:
            insights.append(f"Common match factors: {', '.join(set(all_reasons[:5]))}")
        
        return {
            "explanation": f"Found {len(matches)} matches with average similarity of {avg_score:.2f}",
            "confidence_distribution": {
                "high": high_confidence_count,
                "medium": medium_confidence_count,
                "low": low_confidence_count
            },
            "insights": insights,
            "average_score": avg_score
        }


# Global semantic search service instance
semantic_search_service = SemanticSearchService()