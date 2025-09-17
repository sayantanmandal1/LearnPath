"""
Minimal API v1 router for authentication testing
"""
from fastapi import APIRouter

from app.api.v1.endpoints import auth, health, career_analysis

api_router = APIRouter()

# Include only essential endpoint routers
api_router.include_router(auth.router, prefix="/auth", tags=["authentication"])
api_router.include_router(health.router, prefix="/health", tags=["health"])
api_router.include_router(career_analysis.router, prefix="/career-analysis", tags=["career-analysis"])