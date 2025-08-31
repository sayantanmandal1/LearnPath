"""
FastAPI main application entry point
"""
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import make_asgi_app

from app.core.config import settings
from app.core.database import init_db
from app.core.logging import setup_logging
from app.core.redis import redis_manager
from app.middleware.error_handler import ErrorHandlerMiddleware
from app.middleware.logging import LoggingMiddleware
from app.middleware.metrics import MetricsMiddleware
from app.core.rate_limiting import RateLimitMiddleware
from app.api.v1.router import api_router
from app.services.performance_monitoring import performance_monitor
from app.core.monitoring import system_monitor
from app.core.graceful_degradation import degradation_manager
from app.core.alerting import alerting_system
from app.services.data_pipeline.pipeline_initializer import initialize_pipeline_automation, shutdown_pipeline_automation

logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager"""
    # Startup
    setup_logging()
    logger.info("Starting AI Career Recommender API", version=settings.VERSION)
    
    # Initialize database
    await init_db()
    logger.info("Database initialized")
    
    # Initialize Redis
    await redis_manager.connect()
    logger.info("Redis connected")
    
    # Start performance monitoring
    await performance_monitor.collect_metrics()
    logger.info("Performance monitoring started")
    
    # Start system monitoring
    await system_monitor.start_monitoring()
    logger.info("System monitoring started")
    
    # Start health monitoring for graceful degradation
    await degradation_manager.start_health_monitoring()
    logger.info("Health monitoring started")
    
    # Register external services for monitoring
    degradation_manager.register_service("github", fallback_handler=None)
    degradation_manager.register_service("leetcode", fallback_handler=None)
    degradation_manager.register_service("linkedin", fallback_handler=None)
    degradation_manager.register_service("job_scrapers", fallback_handler=None)
    
    # Initialize pipeline automation system
    try:
        await initialize_pipeline_automation()
        logger.info("Pipeline automation system initialized")
    except Exception as e:
        logger.error("Failed to initialize pipeline automation", error=str(e))
        # Continue startup even if pipeline automation fails
    
    yield
    
    # Shutdown
    # Shutdown pipeline automation system
    try:
        await shutdown_pipeline_automation()
        logger.info("Pipeline automation system shut down")
    except Exception as e:
        logger.error("Failed to shutdown pipeline automation", error=str(e))
    
    await system_monitor.stop_monitoring()
    logger.info("System monitoring stopped")
    
    await degradation_manager.stop_health_monitoring()
    logger.info("Health monitoring stopped")
    
    await redis_manager.disconnect()
    logger.info("Redis disconnected")
    
    logger.info("Shutting down AI Career Recommender API")


def create_application() -> FastAPI:
    """Create and configure FastAPI application"""
    app = FastAPI(
        title=settings.PROJECT_NAME,
        description="""
        # AI-Powered Career & Learning Path Recommender API
        
        A comprehensive platform that analyzes user profiles including resume, skills, coding profiles 
        (GitHub, LeetCode), and social profiles (LinkedIn) to provide personalized career trajectory 
        recommendations, learning paths, and project suggestions.
        
        ## Features
        
        ### 🔐 Authentication
        - JWT-based authentication with refresh tokens
        - User registration and login
        - Secure session management
        
        ### 👤 Profile Management
        - Multi-source profile creation (resume, GitHub, LeetCode, LinkedIn)
        - Intelligent skill extraction and merging
        - Profile analytics and completeness scoring
        - Data validation and conflict resolution
        
        ### 🎯 Career Recommendations
        - AI-powered career trajectory suggestions
        - Job matching with compatibility scoring
        - Skill gap analysis and prioritization
        - Market demand integration
        
        ### 📚 Learning Paths
        - Personalized learning path generation
        - Multi-platform resource integration (Coursera, Udemy, edX)
        - Project-based learning recommendations
        - Progress tracking and milestones
        
        ### 📊 Analytics & Reporting
        - Professional skill radar charts
        - Interactive career roadmap visualizations
        - Comprehensive progress tracking
        - PDF report generation
        
        ### 📈 Market Analysis
        - Real-time job market trend analysis
        - Salary prediction and forecasting
        - Emerging skills detection
        - Geographic market insights
        
        ### 🔍 Advanced Features
        - Semantic job search and matching
        - Vector-based similarity calculations
        - Comprehensive filtering and pagination
        - Data export in multiple formats (JSON, CSV, PDF)
        
        ## API Endpoints
        
        The API is organized into several main categories:
        
        - **Authentication**: User registration, login, and token management
        - **Profiles**: Profile creation, management, and analytics
        - **Recommendations**: Career and learning path recommendations
        - **Career Trajectory**: Career path analysis and planning
        - **Learning Paths**: Personalized learning recommendations
        - **Job Market**: Market analysis and job matching
        - **Analytics**: Comprehensive reporting and visualizations
        - **Comprehensive API**: Advanced endpoints with filtering and pagination
        
        ## Getting Started
        
        1. Register a new user account using `/api/v1/auth/register`
        2. Login to get access tokens using `/api/v1/auth/login`
        3. Create your profile using `/api/v1/profiles/` or `/api/v1/profiles/with-resume`
        4. Get personalized recommendations using various recommendation endpoints
        5. Analyze your career trajectory using `/api/v1/career-trajectory/trajectories`
        6. Generate learning paths using `/api/v1/learning-paths/generate`
        
        ## Authentication
        
        Most endpoints require authentication. Include the JWT token in the Authorization header:
        ```
        Authorization: Bearer <your_jwt_token>
        ```
        """,
        version=settings.VERSION,
        openapi_url=f"{settings.API_V1_STR}/openapi.json",
        docs_url=f"{settings.API_V1_STR}/docs",
        redoc_url=f"{settings.API_V1_STR}/redoc",
        lifespan=lifespan,
        contact={
            "name": "AI Career Recommender API",
            "email": "support@aicareerrecommender.com",
        },
        license_info={
            "name": "MIT License",
            "url": "https://opensource.org/licenses/MIT",
        },
        openapi_tags=[
            {
                "name": "authentication",
                "description": "User authentication and authorization operations"
            },
            {
                "name": "profiles",
                "description": "User profile management and data integration"
            },
            {
                "name": "recommendations",
                "description": "AI-powered career and learning recommendations"
            },
            {
                "name": "career-trajectory",
                "description": "Career path analysis and trajectory planning"
            },
            {
                "name": "learning-paths",
                "description": "Personalized learning path generation and management"
            },
            {
                "name": "job-market",
                "description": "Job market analysis and trend insights"
            },
            {
                "name": "analytics",
                "description": "Comprehensive analytics and reporting"
            },
            {
                "name": "comprehensive-api",
                "description": "Advanced API endpoints with filtering, pagination, and export capabilities"
            },
            {
                "name": "vector-search",
                "description": "Semantic search and similarity operations"
            },
            {
                "name": "external-profiles",
                "description": "External platform integration (GitHub, LeetCode, LinkedIn)"
            },
            {
                "name": "performance",
                "description": "Performance monitoring and system optimization"
            }
        ]
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.ALLOWED_HOSTS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Add custom middleware (order matters - first added is outermost)
    app.add_middleware(ErrorHandlerMiddleware)
    app.add_middleware(LoggingMiddleware)
    app.add_middleware(MetricsMiddleware)
    app.add_middleware(RateLimitMiddleware)

    # Include API router
    app.include_router(api_router, prefix=settings.API_V1_STR)

    # Add Prometheus metrics endpoint
    metrics_app = make_asgi_app()
    app.mount("/metrics", metrics_app)

    return app


app = create_application()


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "version": settings.VERSION}


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
        log_config=None,  # Use our custom logging
    )