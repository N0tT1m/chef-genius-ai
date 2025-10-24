from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import logging
from typing import Dict, Any

from app.api.routes import api_router
from app.core.config import settings
from app.core.database import engine, Base
from app.services.drift_monitoring import initialize_drift_monitoring

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting ChefGenius API")
    Base.metadata.create_all(bind=engine)
    
    # Initialize drift monitoring service
    try:
        await initialize_drift_monitoring()
        logger.info("✅ Drift monitoring initialized")
    except Exception as e:
        logger.warning(f"⚠️  Failed to initialize drift monitoring: {e}")
    
    yield
    logger.info("Shutting down ChefGenius API")

app = FastAPI(
    title="ChefGenius API",
    description="AI-powered cooking assistant API",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_HOSTS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api/v1")

@app.get("/")
async def root() -> Dict[str, Any]:
    return {
        "message": "Welcome to ChefGenius API",
        "version": "1.0.0",
        "status": "active"
    }

@app.get("/health")
async def health_check() -> Dict[str, str]:
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )