"""
Health check endpoints for Chef Genius MCP system
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any
import logging

from app.services.health_monitor import health_monitor, get_health_status
from app.services.drift_monitoring import is_drift_monitoring_available, get_drift_monitoring_stats

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """
    Basic health check endpoint.
    Returns simple status for load balancer checks.
    """
    try:
        # Quick service check
        service_health = await health_monitor.check_all_services()
        
        # Check if critical services are running
        critical_services = ["backend", "weaviate"]
        all_critical_healthy = True
        
        for service in critical_services:
            if service in service_health:
                if service_health[service].status.value != "healthy":
                    all_critical_healthy = False
                    break
        
        if all_critical_healthy:
            return {
                "status": "healthy",
                "timestamp": service_health.get("backend", {}).last_check.isoformat() if service_health.get("backend") else None,
                "services": len(service_health),
                "version": "1.0.0"
            }
        else:
            return {
                "status": "degraded",
                "message": "Some critical services are not healthy",
                "services": len(service_health)
            }
            
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Health check failed")

@router.get("/health/detailed")
async def detailed_health_check() -> Dict[str, Any]:
    """
    Detailed health check with full system status.
    """
    try:
        health_status = await get_health_status()
        
        # Add drift monitoring status
        drift_status = {
            "drift_monitoring_available": is_drift_monitoring_available(),
            "drift_monitoring_active": False,
            "drift_stats": None
        }
        
        if is_drift_monitoring_available():
            try:
                drift_stats = get_drift_monitoring_stats()
                drift_status.update({
                    "drift_monitoring_active": drift_stats.get("monitoring_status") == "active",
                    "drift_stats": {
                        "features_monitored": drift_stats.get("summary", {}).get("total_features_monitored", 0),
                        "drift_rate": drift_stats.get("summary", {}).get("drift_rate", 0.0),
                        "alerts_24h": drift_stats.get("summary", {}).get("alerts_last_24h", 0),
                        "critical_alerts": drift_stats.get("summary", {}).get("critical_alerts", 0)
                    }
                })
            except Exception as e:
                logger.warning(f"Failed to get drift monitoring stats: {e}")
                drift_status["error"] = str(e)
        
        health_status["drift_monitoring"] = drift_status
        return health_status
        
    except Exception as e:
        logger.error(f"Detailed health check failed: {e}")
        raise HTTPException(status_code=500, detail="Detailed health check failed")

@router.get("/health/services")
async def services_health() -> Dict[str, Any]:
    """
    Health status of all MCP services.
    """
    try:
        service_health = await health_monitor.check_all_services()
        
        return {
            "services": {
                name: {
                    "status": health.status.value,
                    "response_time": health.response_time,
                    "last_check": health.last_check.isoformat() if health.last_check else None,
                    "error": health.error_message
                }
                for name, health in service_health.items()
            },
            "summary": {
                "total": len(service_health),
                "healthy": sum(1 for h in service_health.values() if h.status.value == "healthy"),
                "degraded": sum(1 for h in service_health.values() if h.status.value == "degraded"),
                "unhealthy": sum(1 for h in service_health.values() if h.status.value == "unhealthy")
            }
        }
        
    except Exception as e:
        logger.error(f"Services health check failed: {e}")
        raise HTTPException(status_code=500, detail="Services health check failed")

@router.get("/health/metrics")
async def system_metrics() -> Dict[str, Any]:
    """
    Current system resource metrics.
    """
    try:
        metrics = health_monitor.get_system_metrics()
        
        return {
            "cpu_percent": metrics.cpu_percent,
            "memory_percent": metrics.memory_percent,
            "disk_percent": metrics.disk_percent,
            "network_io": metrics.network_io,
            "process_count": metrics.process_count,
            "uptime_seconds": metrics.uptime_seconds,
            "timestamp": "now"
        }
        
    except Exception as e:
        logger.error(f"System metrics check failed: {e}")
        raise HTTPException(status_code=500, detail="System metrics check failed")

@router.get("/health/alerts")
async def get_alerts() -> Dict[str, Any]:
    """
    Current system alerts and warnings.
    """
    try:
        health_status = await get_health_status()
        alerts = health_status.get("alerts", [])
        
        return {
            "alerts": alerts,
            "count": len(alerts),
            "severity_breakdown": {
                "critical": sum(1 for a in alerts if a.get("severity") == "critical"),
                "warning": sum(1 for a in alerts if a.get("severity") == "warning"),
                "info": sum(1 for a in alerts if a.get("severity") == "info")
            }
        }
        
    except Exception as e:
        logger.error(f"Alerts check failed: {e}")
        raise HTTPException(status_code=500, detail="Alerts check failed")

@router.get("/health/mcp")
async def mcp_status() -> Dict[str, Any]:
    """
    Status of MCP servers specifically.
    """
    try:
        service_health = await health_monitor.check_all_services()
        
        mcp_services = {
            name: health for name, health in service_health.items()
            if name in ["recipe-server", "knowledge-server", "tool-server"]
        }
        
        return {
            "mcp_services": {
                name: {
                    "status": health.status.value,
                    "response_time": health.response_time,
                    "metadata": health.metadata,
                    "last_check": health.last_check.isoformat() if health.last_check else None
                }
                for name, health in mcp_services.items()
            },
            "mcp_system_healthy": all(
                h.status.value == "healthy" for h in mcp_services.values()
            ),
            "count": len(mcp_services)
        }
        
    except Exception as e:
        logger.error(f"MCP status check failed: {e}")
        raise HTTPException(status_code=500, detail="MCP status check failed")

@router.get("/health/rag")
async def rag_status() -> Dict[str, Any]:
    """
    Status of RAG system components.
    """
    try:
        service_health = await health_monitor.check_all_services()
        
        rag_components = {
            "weaviate": service_health.get("weaviate"),
            "knowledge-server": service_health.get("knowledge-server")
        }
        
        rag_healthy = all(
            comp and comp.status.value == "healthy" 
            for comp in rag_components.values()
        )
        
        return {
            "rag_components": {
                name: {
                    "status": comp.status.value if comp else "unknown",
                    "response_time": comp.response_time if comp else None,
                    "error": comp.error_message if comp else None
                }
                for name, comp in rag_components.items()
            },
            "rag_system_healthy": rag_healthy,
            "vector_db_status": rag_components["weaviate"].status.value if rag_components["weaviate"] else "unknown"
        }
        
    except Exception as e:
        logger.error(f"RAG status check failed: {e}")
        raise HTTPException(status_code=500, detail="RAG status check failed")

@router.post("/health/refresh")
async def refresh_health_check() -> Dict[str, Any]:
    """
    Force refresh of all health checks.
    """
    try:
        # Force a fresh health check
        service_health = await health_monitor.check_all_services()
        metrics = health_monitor.get_system_metrics()
        
        return {
            "message": "Health checks refreshed",
            "services_checked": len(service_health),
            "timestamp": "now",
            "overall_healthy": all(
                h.status.value in ["healthy", "degraded"] 
                for h in service_health.values()
            )
        }
        
    except Exception as e:
        logger.error(f"Health refresh failed: {e}")
        raise HTTPException(status_code=500, detail="Health refresh failed")