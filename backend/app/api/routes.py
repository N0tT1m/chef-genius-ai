from fastapi import APIRouter
from app.api.endpoints import recipes, substitutions, meal_plans, vision, health, performance, drift_monitoring

api_router = APIRouter()

api_router.include_router(recipes.router, prefix="/recipes", tags=["recipes"])
api_router.include_router(substitutions.router, prefix="/substitutions", tags=["substitutions"])
api_router.include_router(meal_plans.router, prefix="/meal-plans", tags=["meal-plans"])
api_router.include_router(vision.router, prefix="/vision", tags=["vision"])

# Monitoring and observability
api_router.include_router(health.router, tags=["health"])
api_router.include_router(performance.router, prefix="/performance", tags=["performance"])
api_router.include_router(drift_monitoring.router, prefix="/monitoring", tags=["drift-monitoring"])