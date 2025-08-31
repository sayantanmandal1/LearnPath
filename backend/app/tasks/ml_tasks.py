"""
Machine Learning background tasks
"""
import asyncio
from typing import Dict, List, Any
import structlog
from celery import current_task

from app.core.celery_app import celery_app, TASK_PRIORITIES
from app.services.cache_service import get_cache_service, CacheKeyBuilder
from app.core.database import AsyncSessionLocal

logger = structlog.get_logger()


@celery_app.task(
    bind=True,
    name="app.tasks.ml_tasks.generate_career_recommendations",
    priority=TASK_PRIORITIES["HIGH"],
    max_retries=3,
    default_retry_delay=60
)
def generate_career_recommendations(self, user_id: str, profile_data: Dict[str, Any]) -> Dict[str, Any]:
    """Generate career recommendations for a user (background task)"""
    try:
        # Update task progress
        current_task.update_state(
            state="PROGRESS",
            meta={"current": 10, "total": 100, "status": "Initializing ML models"}
        )
        
        # Import ML services (lazy import to avoid circular dependencies)
        from app.services.recommendation_service import RecommendationService
        from machinelearningmodel.recommendation_engine import RecommendationEngine
        
        # Initialize services
        ml_engine = RecommendationEngine()
        recommendation_service = RecommendationService()
        
        current_task.update_state(
            state="PROGRESS",
            meta={"current": 30, "total": 100, "status": "Processing user profile"}
        )
        
        # Run async operations in sync context
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # Generate recommendations
            recommendations = loop.run_until_complete(
                recommendation_service._generate_recommendations_async(user_id, profile_data)
            )
            
            current_task.update_state(
                state="PROGRESS",
                meta={"current": 80, "total": 100, "status": "Caching results"}
            )
            
            # Cache the results
            cache_service = loop.run_until_complete(get_cache_service())
            cache_key = CacheKeyBuilder.career_recommendations(user_id, str(hash(str(profile_data))))
            
            loop.run_until_complete(
                cache_service.set(cache_key, recommendations, ttl=3600)  # 1 hour cache
            )
            
            current_task.update_state(
                state="SUCCESS",
                meta={"current": 100, "total": 100, "status": "Completed"}
            )
            
            return {
                "status": "success",
                "user_id": user_id,
                "recommendations_count": len(recommendations),
                "cache_key": cache_key
            }
            
        finally:
            loop.close()
            
    except Exception as e:
        logger.error("Career recommendations task failed", user_id=user_id, error=str(e))
        current_task.update_state(
            state="FAILURE",
            meta={"error": str(e), "status": "Failed to generate recommendations"}
        )
        raise self.retry(exc=e, countdown=60, max_retries=3)


@celery_app.task(
    bind=True,
    name="app.tasks.ml_tasks.generate_learning_path",
    priority=TASK_PRIORITIES["HIGH"],
    max_retries=3
)
def generate_learning_path(self, user_id: str, skill_gaps: Dict[str, float], target_role: str) -> Dict[str, Any]:
    """Generate personalized learning path (background task)"""
    try:
        current_task.update_state(
            state="PROGRESS",
            meta={"current": 20, "total": 100, "status": "Analyzing skill gaps"}
        )
        
        from app.services.learning_path_service import LearningPathService
        from machinelearningmodel.learning_path_optimizer import LearningPathOptimizer
        
        # Initialize services
        optimizer = LearningPathOptimizer()
        learning_service = LearningPathService()
        
        current_task.update_state(
            state="PROGRESS",
            meta={"current": 50, "total": 100, "status": "Generating learning path"}
        )
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # Generate learning path
            learning_path = loop.run_until_complete(
                learning_service._generate_learning_path_async(user_id, skill_gaps, target_role)
            )
            
            current_task.update_state(
                state="PROGRESS",
                meta={"current": 90, "total": 100, "status": "Caching results"}
            )
            
            # Cache the results
            cache_service = loop.run_until_complete(get_cache_service())
            cache_key = CacheKeyBuilder.learning_path(user_id, str(hash(str(skill_gaps))))
            
            loop.run_until_complete(
                cache_service.set(cache_key, learning_path, ttl=7200)  # 2 hours cache
            )
            
            return {
                "status": "success",
                "user_id": user_id,
                "learning_path_id": learning_path.get("id"),
                "estimated_duration_weeks": learning_path.get("estimated_duration_weeks"),
                "cache_key": cache_key
            }
            
        finally:
            loop.close()
            
    except Exception as e:
        logger.error("Learning path generation task failed", user_id=user_id, error=str(e))
        raise self.retry(exc=e, countdown=60, max_retries=3)


@celery_app.task(
    bind=True,
    name="app.tasks.ml_tasks.process_resume_analysis",
    priority=TASK_PRIORITIES["MEDIUM"],
    max_retries=2
)
def process_resume_analysis(self, user_id: str, resume_content: str) -> Dict[str, Any]:
    """Process resume analysis with NLP (background task)"""
    try:
        current_task.update_state(
            state="PROGRESS",
            meta={"current": 25, "total": 100, "status": "Initializing NLP models"}
        )
        
        from machinelearningmodel.nlp_engine import NLPEngine
        from machinelearningmodel.skill_classifier import SkillClassifier
        
        # Initialize NLP services
        nlp_engine = NLPEngine()
        skill_classifier = SkillClassifier()
        
        current_task.update_state(
            state="PROGRESS",
            meta={"current": 50, "total": 100, "status": "Extracting skills and experience"}
        )
        
        # Process resume
        extracted_skills = nlp_engine.extract_skills(resume_content)
        classified_skills = skill_classifier.classify_skills(extracted_skills)
        experience_data = nlp_engine.extract_experience(resume_content)
        
        current_task.update_state(
            state="PROGRESS",
            meta={"current": 80, "total": 100, "status": "Generating embeddings"}
        )
        
        # Generate profile embeddings
        profile_embedding = nlp_engine.generate_profile_embedding({
            "skills": classified_skills,
            "experience": experience_data,
            "resume_text": resume_content
        })
        
        result = {
            "user_id": user_id,
            "extracted_skills": classified_skills,
            "experience_data": experience_data,
            "profile_embedding": profile_embedding.tolist() if hasattr(profile_embedding, 'tolist') else profile_embedding,
            "processing_timestamp": current_task.request.id
        }
        
        # Cache the results
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            cache_service = loop.run_until_complete(get_cache_service())
            cache_key = f"resume_analysis:{user_id}"
            loop.run_until_complete(
                cache_service.set(cache_key, result, ttl=86400)  # 24 hours cache
            )
        finally:
            loop.close()
        
        return result
        
    except Exception as e:
        logger.error("Resume analysis task failed", user_id=user_id, error=str(e))
        raise self.retry(exc=e, countdown=120, max_retries=2)


@celery_app.task(
    name="app.tasks.ml_tasks.update_ml_models",
    priority=TASK_PRIORITIES["LOW"]
)
def update_ml_models() -> Dict[str, Any]:
    """Periodic task to update ML models with new data"""
    try:
        from machinelearningmodel.recommendation_engine import RecommendationEngine
        from machinelearningmodel.skill_classifier import SkillClassifier
        
        logger.info("Starting ML model update process")
        
        # Update recommendation models
        rec_engine = RecommendationEngine()
        rec_engine.retrain_models()
        
        # Update skill classification models
        skill_classifier = SkillClassifier()
        skill_classifier.update_model()
        
        # Clear ML model caches
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            cache_service = loop.run_until_complete(get_cache_service())
            loop.run_until_complete(
                cache_service.invalidation.invalidate_ml_model_cache("recommendation_engine")
            )
            loop.run_until_complete(
                cache_service.invalidation.invalidate_ml_model_cache("skill_classifier")
            )
        finally:
            loop.close()
        
        logger.info("ML model update completed successfully")
        
        return {
            "status": "success",
            "updated_models": ["recommendation_engine", "skill_classifier"],
            "timestamp": current_task.request.id
        }
        
    except Exception as e:
        logger.error("ML model update failed", error=str(e))
        raise


@celery_app.task(
    bind=True,
    name="app.tasks.ml_tasks.batch_profile_analysis",
    priority=TASK_PRIORITIES["LOW"],
    max_retries=1
)
def batch_profile_analysis(self, user_ids: List[str]) -> Dict[str, Any]:
    """Batch process multiple user profiles for analysis"""
    try:
        results = {}
        total_users = len(user_ids)
        
        for i, user_id in enumerate(user_ids):
            current_task.update_state(
                state="PROGRESS",
                meta={
                    "current": i + 1,
                    "total": total_users,
                    "status": f"Processing user {user_id}"
                }
            )
            
            # Process individual user (this would trigger other tasks)
            # For now, just simulate processing
            results[user_id] = {
                "status": "processed",
                "timestamp": current_task.request.id
            }
        
        return {
            "status": "success",
            "processed_users": len(results),
            "results": results
        }
        
    except Exception as e:
        logger.error("Batch profile analysis failed", error=str(e))
        raise self.retry(exc=e, countdown=300, max_retries=1)