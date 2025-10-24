"""
Performance API endpoints for Chef Genius

Provides API access to performance monitoring, optimization controls,
and system statistics.
"""

from fastapi import APIRouter, HTTPException, Depends, Query
from typing import Dict, Any, Optional
import logging
from app.services.performance_optimizer import PerformanceOptimizer, MemoryManager
from app.core.dependencies import get_current_user

router = APIRouter()
logger = logging.getLogger(__name__)

# Global performance optimizer instance
performance_optimizer = PerformanceOptimizer()

@router.get("/stats")
async def get_performance_stats(current_user = Depends(get_current_user)):
    """Get comprehensive system performance statistics."""
    try:
        stats = performance_optimizer.get_system_stats()
        return {
            "status": "success",
            "performance_stats": stats
        }
    except Exception as e:
        logger.error(f"Failed to get performance stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/memory")
async def get_memory_stats(current_user = Depends(get_current_user)):
    """Get detailed memory usage statistics."""
    try:
        memory_stats = MemoryManager.get_memory_stats()
        return {
            "status": "success",
            "memory": memory_stats
        }
    except Exception as e:
        logger.error(f"Failed to get memory stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/cleanup")
async def trigger_cleanup(
    force: bool = Query(default=False, description="Force cleanup even if not needed"),
    current_user = Depends(get_current_user)
):
    """Trigger manual resource cleanup."""
    try:
        if force or performance_optimizer.should_auto_cleanup():
            await performance_optimizer.cleanup_resources()
            message = "Resource cleanup completed"
        else:
            message = "Cleanup not needed at this time"
        
        # Get updated stats after cleanup
        stats = performance_optimizer.get_system_stats()
        
        return {
            "status": "success",
            "message": message,
            "memory_stats": stats["memory"]
        }
    except Exception as e:
        logger.error(f"Cleanup failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/cache/stats")
async def get_cache_stats(current_user = Depends(get_current_user)):
    """Get cache performance statistics."""
    try:
        cache_stats = performance_optimizer.cache.stats()
        return {
            "status": "success",
            "cache": cache_stats
        }
    except Exception as e:
        logger.error(f"Failed to get cache stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/cache/clear")
async def clear_cache(current_user = Depends(get_current_user)):
    """Clear all cached data."""
    try:
        performance_optimizer.cache.clear()
        return {
            "status": "success",
            "message": "Cache cleared successfully"
        }
    except Exception as e:
        logger.error(f"Failed to clear cache: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/metrics/{operation}")
async def get_operation_metrics(
    operation: str,
    current_user = Depends(get_current_user)
):
    """Get performance metrics for a specific operation."""
    try:
        metrics = performance_optimizer.performance_monitor.get_stats(operation)
        
        if not metrics:
            raise HTTPException(
                status_code=404, 
                detail=f"No metrics found for operation: {operation}"
            )
        
        return {
            "status": "success",
            "operation": operation,
            "metrics": metrics
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get operation metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/metrics")
async def get_all_metrics(current_user = Depends(get_current_user)):
    """Get performance metrics for all operations."""
    try:
        all_metrics = performance_optimizer.performance_monitor.get_all_stats()
        return {
            "status": "success",
            "metrics": all_metrics
        }
    except Exception as e:
        logger.error(f"Failed to get all metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/config/auto-cleanup")
async def configure_auto_cleanup(
    enabled: bool = Query(description="Enable/disable automatic cleanup"),
    interval: Optional[int] = Query(default=None, description="Cleanup interval in seconds"),
    current_user = Depends(get_current_user)
):
    """Configure automatic cleanup settings."""
    try:
        performance_optimizer.auto_cleanup_enabled = enabled
        
        if interval is not None:
            if interval < 60:  # Minimum 1 minute
                raise HTTPException(
                    status_code=400, 
                    detail="Cleanup interval must be at least 60 seconds"
                )
            performance_optimizer.cleanup_interval = interval
        
        return {
            "status": "success",
            "message": "Auto-cleanup configuration updated",
            "config": {
                "enabled": performance_optimizer.auto_cleanup_enabled,
                "interval": performance_optimizer.cleanup_interval
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to configure auto-cleanup: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/config")
async def get_performance_config(current_user = Depends(get_current_user)):
    """Get current performance configuration."""
    try:
        config = {
            "auto_cleanup": {
                "enabled": performance_optimizer.auto_cleanup_enabled,
                "interval": performance_optimizer.cleanup_interval,
                "last_cleanup": performance_optimizer.last_cleanup
            },
            "cache": {
                "max_size": performance_optimizer.cache.max_size,
                "default_ttl": performance_optimizer.cache.default_ttl
            },
            "batch_processor": {
                "batch_size": performance_optimizer.batch_processor.batch_size,
                "max_wait_time": performance_optimizer.batch_processor.max_wait_time
            }
        }
        
        return {
            "status": "success",
            "config": config
        }
    except Exception as e:
        logger.error(f"Failed to get performance config: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def performance_health_check():
    """Get performance health status."""
    try:
        memory_stats = MemoryManager.get_memory_stats()
        cache_stats = performance_optimizer.cache.stats()
        
        # Determine health status
        health_status = "healthy"
        warnings = []
        
        # Check memory usage
        if memory_stats.get("ram_percent", 0) > 90:
            health_status = "warning"
            warnings.append("High RAM usage")
        
        if memory_stats.get("gpu_allocated_gb", 0) > 20:  # For RTX 4090
            health_status = "warning"
            warnings.append("High GPU memory usage")
        
        # Check cache utilization
        if cache_stats.get("utilization", 0) > 0.95:
            warnings.append("Cache nearly full")
        
        return {
            "status": "success",
            "health": {
                "status": health_status,
                "warnings": warnings,
                "memory": memory_stats,
                "cache_utilization": cache_stats.get("utilization", 0)
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))