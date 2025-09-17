"""
Minimal FastAPI application for authentication testing
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.api.v1.endpoints.auth import router as auth_router

# Create FastAPI app
app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    debug=settings.DEBUG,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_HOSTS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include authentication routes
app.include_router(auth_router, prefix="/api/v1/auth", tags=["authentication"])

@app.get("/")
async def root():
    return {"message": "AI Career Recommender API", "version": settings.VERSION}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "ai-career-recommender"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)