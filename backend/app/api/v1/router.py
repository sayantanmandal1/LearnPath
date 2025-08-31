"""
Main API v1 router
"""
from fastapi import APIRouter

from app.api.v1.endpoints import auth, external_profiles, profiles

api_router = APIRouter()

# Include endpoint routers
api_router.include_router(auth.router, prefix="/auth", tags=["authentication"])
api_router.include_router(external_profiles.router, prefix="/external-profiles", tags=["external-profiles"])
api_router.include_router(profiles.router, prefix="/profiles", tags=["profiles"])