"""
Vector Database Demo

Demonstration script showing vector database integration capabilities
including embedding generation, storage, and semantic search.
"""

import asyncio
import logging
import sys
import os
from typing import List, Dict, Any

# Add parent directories to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../..', 'machinelearningmodel'))

from app.services.embedding_service import embedding_service
from app.services.semantic_search_service import semantic_search_service, SearchRequest, SearchType
from app.services.vector_db.vector_db_manager import vector_db_manager
from app.core.vector_db_config import vector_db_config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VectorDBDemo:
    """Demo class for vector database functionality."""
    
    def __init__(self):
        """Initialize demo."""
        self.sample_profiles = [
            {
                "user_id": "user_001",
                "skills": ["Python", "Machine Learning", "TensorFlow", "Data Science"],
                "experience_level": "senior",
                "dream_job": "ML Engineer",
                "experience": [
                    {"position": "Data Scientist", "company": "TechCorp"}
                ]
            },
            {
                "user_id": "user_002", 
                "skills": ["JavaScript", "React", "Node.js", "Frontend Development"],
                "experience_level": "mid",
                "dream_job": "Frontend Developer",
                "experience": [
                    {"position": "Frontend Developer", "company": "WebCorp"}
                ]
            },
            {
                "user_id": "user_003",
                "skills": ["Python", "Django", "PostgreSQL", "Backend Development"],
                "experience_level": "senior",
                "dream_job": "Backend Engineer",
                "experience": [
                    {"position": "Backend Engineer", "company": "ServerCorp"}
                ]
            }
        ]
        
        self.sample_jobs = [
            {
                "job_id": "job_001",
                "title": "Senior ML Engineer",
                "company": "AI Innovations",
                "description": "Build and deploy machine learning models at scale",
                "required_skills": ["Python", "TensorFlow", "Machine Learning", "MLOps"],
                "experience_level": "senior",
                "location": "San Francisco"
            },
            {
                "job_id": "job_002",
                "title": "Frontend Developer",
                "company": "Modern Web Co",
                "description": "Create beautiful and responsive web applications",
                "required_skills": ["JavaScript", "React", "CSS", "HTML"],
                "experience_level": "mid",
                "location": "New York"
            },
            {
                "job_id": "job_003",
                "title": "Full Stack Developer",
                "company": "Startup Inc",
                "description": "Work on both frontend and backend systems",
                "required_skills": ["Python", "JavaScript", "React", "Django"],
                "experience_level": "mid",
                "location": "Austin"
            }
        ]
        
        self.sample_skills = [
            {
                "skill_name": "Python",
                "category": "Programming Language",
                "description": "High-level programming language for general-purpose programming",
                "popularity_score": 0.95
            },
            {
                "skill_name": "Machine Learning",
                "category": "Technology",
                "description": "Field of AI that uses algorithms to learn from data",
                "popularity_score": 0.90
            },
            {
                "skill_name": "React",
                "category": "Framework",
                "description": "JavaScript library for building user interfaces",
                "popularity_score": 0.85
            }
        ]
        
        self.sample_resources = [
            {
                "resource_id": "course_001",
                "title": "Complete Python Bootcamp",
                "type": "course",
                "provider": "Udemy",
                "description": "Learn Python from beginner to advanced",
                "target_skills": ["Python", "Programming Fundamentals"],
                "difficulty_level": "beginner",
                "rating": 4.6
            },
            {
                "resource_id": "course_002",
                "title": "Machine Learning A-Z",
                "type": "course", 
                "provider": "Coursera",
                "description": "Comprehensive machine learning course",
                "target_skills": ["Machine Learning", "Python", "Data Science"],
                "difficulty_level": "intermediate",
                "rating": 4.8
            }
        ]
    
    async def initialize_services(self):
        """Initialize all required services."""
        logger.info("Initializing vector database services...")
        
        try:
            # Initialize embedding service
            await embedding_service.initialize()
            logger.info("✓ Embedding service initialized")
            
            # Check vector database health
            health = await vector_db_manager.health_check()
            logger.info(f"✓ Vector database health: {health['overall_status']}")
            
        except Exception as e:
            logger.error(f"Failed to initialize services: {str(e)}")
            logger.info("Using mock mode for demonstration...")
            # Continue with mock mode
    
    async def demo_embedding_generation(self):
        """Demonstrate embedding generation."""
        logger.info("\n=== Embedding Generation Demo ===")
        
        try:
            # Generate embeddings for sample texts
            texts = [
                "Python machine learning engineer with 5 years experience",
                "Frontend developer specializing in React and JavaScript",
                "Data scientist with expertise in deep learning"
            ]
            
            for text in texts:
                embedding = await embedding_service.generate_embedding(text)
                logger.info(f"Generated embedding for: '{text[:50]}...'")
                logger.info(f"  Embedding dimension: {len(embedding)}")
                logger.info(f"  Sample values: {embedding[:5].tolist()}")
                
        except Exception as e:
            logger.error(f"Embedding generation failed: {str(e)}")
    
    async def demo_profile_storage(self):
        """Demonstrate profile embedding storage."""
        logger.info("\n=== Profile Storage Demo ===")
        
        try:
            for profile in self.sample_profiles:
                success = await embedding_service.store_profile_embedding(
                    profile["user_id"], profile
                )
                
                if success:
                    logger.info(f"✓ Stored profile for {profile['user_id']}")
                else:
                    logger.warning(f"✗ Failed to store profile for {profile['user_id']}")
                    
        except Exception as e:
            logger.error(f"Profile storage failed: {str(e)}")
    
    async def demo_job_storage(self):
        """Demonstrate job embedding storage."""
        logger.info("\n=== Job Storage Demo ===")
        
        try:
            for job in self.sample_jobs:
                success = await embedding_service.store_job_embedding(
                    job["job_id"], job
                )
                
                if success:
                    logger.info(f"✓ Stored job {job['job_id']}: {job['title']}")
                else:
                    logger.warning(f"✗ Failed to store job {job['job_id']}")
                    
        except Exception as e:
            logger.error(f"Job storage failed: {str(e)}")
    
    async def demo_skill_storage(self):
        """Demonstrate skill embedding storage."""
        logger.info("\n=== Skill Storage Demo ===")
        
        try:
            for skill in self.sample_skills:
                success = await embedding_service.store_skill_embedding(
                    skill["skill_name"], skill
                )
                
                if success:
                    logger.info(f"✓ Stored skill: {skill['skill_name']}")
                else:
                    logger.warning(f"✗ Failed to store skill: {skill['skill_name']}")
                    
        except Exception as e:
            logger.error(f"Skill storage failed: {str(e)}")
    
    async def demo_resource_storage(self):
        """Demonstrate learning resource embedding storage."""
        logger.info("\n=== Learning Resource Storage Demo ===")
        
        try:
            for resource in self.sample_resources:
                success = await embedding_service.store_learning_resource_embedding(
                    resource["resource_id"], resource
                )
                
                if success:
                    logger.info(f"✓ Stored resource: {resource['title']}")
                else:
                    logger.warning(f"✗ Failed to store resource: {resource['title']}")
                    
        except Exception as e:
            logger.error(f"Resource storage failed: {str(e)}")
    
    async def demo_profile_similarity_search(self):
        """Demonstrate profile similarity search."""
        logger.info("\n=== Profile Similarity Search Demo ===")
        
        try:
            # Search for profiles similar to user_001
            request = SearchRequest(
                search_type=SearchType.PROFILE_SIMILARITY,
                query_data={"user_id": "user_001"},
                top_k=5,
                min_score=0.0
            )
            
            matches = await semantic_search_service.search(request)
            
            logger.info(f"Found {len(matches)} similar profiles for user_001:")
            for match in matches:
                logger.info(f"  • {match.id} (score: {match.score:.3f}, confidence: {match.confidence_level})")
                logger.info(f"    Reasons: {', '.join(match.match_reasons)}")
                
        except Exception as e:
            logger.error(f"Profile similarity search failed: {str(e)}")
    
    async def demo_job_matching_search(self):
        """Demonstrate job matching search."""
        logger.info("\n=== Job Matching Search Demo ===")
        
        try:
            # Find jobs matching user_001's profile
            request = SearchRequest(
                search_type=SearchType.JOB_MATCHING,
                query_data={"user_id": "user_001"},
                top_k=10,
                min_score=0.0
            )
            
            matches = await semantic_search_service.search(request)
            
            logger.info(f"Found {len(matches)} matching jobs for user_001:")
            for match in matches:
                job_title = match.metadata.get("title", "Unknown")
                company = match.metadata.get("company", "Unknown")
                logger.info(f"  • {job_title} at {company} (score: {match.score:.3f})")
                logger.info(f"    Reasons: {', '.join(match.match_reasons)}")
                
        except Exception as e:
            logger.error(f"Job matching search failed: {str(e)}")
    
    async def demo_skill_similarity_search(self):
        """Demonstrate skill similarity search."""
        logger.info("\n=== Skill Similarity Search Demo ===")
        
        try:
            # Find skills similar to Python
            request = SearchRequest(
                search_type=SearchType.SKILL_SIMILARITY,
                query_data={"skill_name": "Python"},
                top_k=5,
                min_score=0.0
            )
            
            matches = await semantic_search_service.search(request)
            
            logger.info(f"Found {len(matches)} skills similar to Python:")
            for match in matches:
                skill_name = match.metadata.get("skill_name", "Unknown")
                category = match.metadata.get("category", "Unknown")
                logger.info(f"  • {skill_name} ({category}) - score: {match.score:.3f}")
                
        except Exception as e:
            logger.error(f"Skill similarity search failed: {str(e)}")
    
    async def demo_learning_resources_search(self):
        """Demonstrate learning resources search."""
        logger.info("\n=== Learning Resources Search Demo ===")
        
        try:
            # Find resources for skill gaps
            skill_gaps = ["Machine Learning", "Deep Learning", "PyTorch"]
            
            request = SearchRequest(
                search_type=SearchType.LEARNING_RESOURCES,
                query_data={"skill_gaps": skill_gaps},
                top_k=10,
                min_score=0.0
            )
            
            matches = await semantic_search_service.search(request)
            
            logger.info(f"Found {len(matches)} learning resources for skill gaps:")
            for match in matches:
                title = match.metadata.get("title", "Unknown")
                provider = match.metadata.get("provider", "Unknown")
                rating = match.metadata.get("rating", 0)
                logger.info(f"  • {title} by {provider} (rating: {rating}, score: {match.score:.3f})")
                logger.info(f"    Reasons: {', '.join(match.match_reasons)}")
                
        except Exception as e:
            logger.error(f"Learning resources search failed: {str(e)}")
    
    async def demo_batch_search(self):
        """Demonstrate batch search functionality."""
        logger.info("\n=== Batch Search Demo ===")
        
        try:
            # Create multiple search requests
            requests = [
                SearchRequest(
                    search_type=SearchType.PROFILE_SIMILARITY,
                    query_data={"user_id": "user_001"},
                    top_k=3
                ),
                SearchRequest(
                    search_type=SearchType.JOB_MATCHING,
                    query_data={"user_id": "user_002"},
                    top_k=3
                ),
                SearchRequest(
                    search_type=SearchType.SKILL_SIMILARITY,
                    query_data={"skill_name": "React"},
                    top_k=3
                )
            ]
            
            results = await semantic_search_service.batch_search(requests)
            
            logger.info(f"Batch search completed with {len(results)} result sets:")
            for i, (key, matches) in enumerate(results.items()):
                search_type = requests[i].search_type.value
                logger.info(f"  Request {key} ({search_type}): {len(matches)} matches")
                
        except Exception as e:
            logger.error(f"Batch search failed: {str(e)}")
    
    async def demo_search_explanation(self):
        """Demonstrate search result explanation."""
        logger.info("\n=== Search Explanation Demo ===")
        
        try:
            # Perform a search and get explanation
            request = SearchRequest(
                search_type=SearchType.JOB_MATCHING,
                query_data={"user_id": "user_001"},
                top_k=5
            )
            
            matches = await semantic_search_service.search(request)
            explanation = await semantic_search_service.explain_search_results(matches)
            
            logger.info("Search Results Explanation:")
            logger.info(f"  {explanation['explanation']}")
            logger.info(f"  Average score: {explanation['average_score']:.3f}")
            logger.info(f"  Confidence distribution: {explanation['confidence_distribution']}")
            logger.info("  Insights:")
            for insight in explanation['insights']:
                logger.info(f"    - {insight}")
                
        except Exception as e:
            logger.error(f"Search explanation failed: {str(e)}")
    
    async def run_full_demo(self):
        """Run the complete vector database demo."""
        logger.info("🚀 Starting Vector Database Integration Demo")
        logger.info(f"Provider: {vector_db_config.VECTOR_DB_PROVIDER}")
        logger.info(f"Embedding Model: {vector_db_config.EMBEDDING_MODEL}")
        logger.info(f"Embedding Dimension: {vector_db_config.EMBEDDING_DIMENSION}")
        
        try:
            # Initialize services
            await self.initialize_services()
            
            # Demo embedding generation
            await self.demo_embedding_generation()
            
            # Demo data storage
            await self.demo_profile_storage()
            await self.demo_job_storage()
            await self.demo_skill_storage()
            await self.demo_resource_storage()
            
            # Demo search functionality
            await self.demo_profile_similarity_search()
            await self.demo_job_matching_search()
            await self.demo_skill_similarity_search()
            await self.demo_learning_resources_search()
            
            # Demo advanced features
            await self.demo_batch_search()
            await self.demo_search_explanation()
            
            logger.info("\n✅ Vector Database Demo completed successfully!")
            
        except Exception as e:
            logger.error(f"Demo failed: {str(e)}")
        
        finally:
            # Cleanup
            try:
                await embedding_service.cleanup()
                logger.info("✓ Services cleaned up")
            except Exception as e:
                logger.error(f"Cleanup failed: {str(e)}")


async def main():
    """Main demo function."""
    demo = VectorDBDemo()
    await demo.run_full_demo()


if __name__ == "__main__":
    asyncio.run(main())