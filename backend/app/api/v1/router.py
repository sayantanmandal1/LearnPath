"""
Main API v1 router
"""
from fastapi import APIRouter

from app.api.v1.endpoints import (
    auth, external_profiles, profiles, career_trajectory, 
    learning_paths, analytics, job_market, vector_search, comprehensive_api, performance, performance_optimization, health,
    privacy, security, pipeline_automation, career_analysis, market_insights, dashboard, resume,
    ai_analysis, real_time_jobs, data_sync, supabase_webhooks, platform_accounts, career_guidance, error_handling,
    workflow_integration, job_applications, job_recommendation_integration, system_monitoring
)
from app.api.v1.endpoints import recommendations_fixed as recommendations

api_router = APIRouter()

# Include endpoint routers
api_router.include_router(auth.router, prefix="/auth", tags=["authentication"])
api_router.include_router(external_profiles.router, prefix="/external-profiles", tags=["external-profiles"])
api_router.include_router(profiles.router, prefix="/profiles", tags=["profiles"])
api_router.include_router(recommendations.router, prefix="/recommendations", tags=["recommendations"])
api_router.include_router(career_trajectory.router, prefix="/career-trajectory", tags=["career-trajectory"])
api_router.include_router(learning_paths.router, prefix="/learning-paths", tags=["learning-paths"])
api_router.include_router(job_market.router, prefix="/job-market", tags=["job-market"])
api_router.include_router(analytics.router, prefix="/analytics", tags=["analytics"])
api_router.include_router(career_analysis.router, prefix="/career-analysis", tags=["career-analysis"])
api_router.include_router(vector_search.router, tags=["vector-search"])
api_router.include_router(comprehensive_api.router, prefix="/comprehensive", tags=["comprehensive-api"])
api_router.include_router(performance.router, prefix="/performance", tags=["performance"])
api_router.include_router(performance_optimization.router, prefix="/performance-optimization", tags=["performance-optimization"])
api_router.include_router(health.router, prefix="/health", tags=["health"])
api_router.include_router(privacy.router, prefix="/privacy", tags=["privacy"])
api_router.include_router(security.router, prefix="/security", tags=["security"])
api_router.include_router(pipeline_automation.router, prefix="/pipeline", tags=["pipeline-automation"])
api_router.include_router(market_insights.router, prefix="/market-insights", tags=["market-insights"])
api_router.include_router(dashboard.router, prefix="/dashboard", tags=["dashboard"])
api_router.include_router(resume.router, prefix="/resume", tags=["resume"])
api_router.include_router(ai_analysis.router, prefix="/ai-analysis", tags=["ai-analysis"])
api_router.include_router(real_time_jobs.router, prefix="/real-time-jobs", tags=["real-time-jobs"])
api_router.include_router(data_sync.router, prefix="/data-sync", tags=["data-sync"])
api_router.include_router(supabase_webhooks.router, tags=["supabase-webhooks"])
api_router.include_router(platform_accounts.router, prefix="/platforms", tags=["platform-accounts"])
api_router.include_router(career_guidance.router, prefix="/career-guidance", tags=["career-guidance"])
api_router.include_router(error_handling.router, prefix="/error-handling", tags=["error-handling"])
api_router.include_router(workflow_integration.router, prefix="/workflow", tags=["workflow-integration"])
api_router.include_router(job_applications.router, prefix="/job-applications", tags=["job-applications"])
api_router.include_router(job_recommendation_integration.router, prefix="/job-recommendations", tags=["job-recommendation-integration"])
api_router.include_router(system_monitoring.router, prefix="/monitoring", tags=["system-monitoring"])