"""
API endpoints for model drift monitoring and detection
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime, timedelta
from pydantic import BaseModel, Field

from app.services.drift_monitoring import (
    drift_monitor, 
    is_drift_monitoring_available,
    get_drift_monitoring_stats,
    DriftAlert,
    DriftMetrics
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/drift", tags=["drift-monitoring"])


class SetReferenceDataRequest(BaseModel):
    feature_name: str = Field(..., description="Name of the feature to monitor")
    data: List[float] = Field(..., description="Reference data values")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Optional metadata")


class SetCategoricalReferenceRequest(BaseModel):
    feature_name: str = Field(..., description="Name of the categorical feature")
    categories: List[str] = Field(..., description="Reference category values")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Optional metadata")


class DetectDriftRequest(BaseModel):
    feature_name: str = Field(..., description="Feature to check for drift")
    current_data: List[float] = Field(..., description="Current data to compare")
    methods: Optional[List[str]] = Field(
        default=None, 
        description="Drift detection methods to use"
    )


class RecipeDriftRequest(BaseModel):
    recipes: List[Dict[str, Any]] = Field(..., description="Generated recipes to analyze")


class TrainingBaselineRequest(BaseModel):
    training_recipes: List[Dict[str, Any]] = Field(..., description="Training recipes for baseline")


@router.get("/status")
async def get_drift_monitoring_status() -> Dict[str, Any]:
    """Get drift monitoring system status"""
    return {
        "drift_monitoring_available": is_drift_monitoring_available(),
        "monitoring_active": drift_monitor.monitoring_active,
        "rust_engine_available": is_drift_monitoring_available(),
        "features_monitored": len(drift_monitor.reference_manager.list_features()),
        "total_alerts": len(drift_monitor.alerts),
        "last_check": datetime.utcnow().isoformat()
    }


@router.get("/metrics")
async def get_drift_metrics() -> Dict[str, Any]:
    """Get current drift monitoring metrics"""
    if not is_drift_monitoring_available():
        raise HTTPException(
            status_code=503, 
            detail="Drift monitoring not available - Rust engine required"
        )
    
    metrics = drift_monitor.get_current_metrics()
    return metrics.to_dict()


@router.get("/report")
async def get_drift_report() -> Dict[str, Any]:
    """Get comprehensive drift monitoring report"""
    if not is_drift_monitoring_available():
        raise HTTPException(
            status_code=503, 
            detail="Drift monitoring not available - Rust engine required"
        )
    
    return drift_monitor.get_drift_report()


@router.post("/reference/numerical")
async def set_numerical_reference_data(request: SetReferenceDataRequest) -> Dict[str, str]:
    """Set reference data for numerical feature drift detection"""
    if not is_drift_monitoring_available():
        raise HTTPException(
            status_code=503, 
            detail="Drift monitoring not available - Rust engine required"
        )
    
    if len(request.data) < 10:
        raise HTTPException(
            status_code=400,
            detail="Reference data must contain at least 10 samples"
        )
    
    try:
        drift_monitor.reference_manager.set_reference_data(
            request.feature_name,
            request.data,
            request.metadata
        )
        
        # Also set in Rust engine if available
        if drift_monitor.drift_detector:
            drift_monitor.drift_detector.set_reference_data(
                request.feature_name,
                request.data,
                request.metadata
            )
        
        logger.info(f"Reference data set for {request.feature_name}: {len(request.data)} samples")
        
        return {
            "status": "success",
            "message": f"Reference data set for feature '{request.feature_name}'"
        }
        
    except Exception as e:
        logger.error(f"Failed to set reference data: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to set reference data: {str(e)}")


@router.post("/reference/categorical")
async def set_categorical_reference_data(request: SetCategoricalReferenceRequest) -> Dict[str, str]:
    """Set reference data for categorical feature drift detection"""
    if not is_drift_monitoring_available():
        raise HTTPException(
            status_code=503, 
            detail="Drift monitoring not available - Rust engine required"
        )
    
    if len(request.categories) < 5:
        raise HTTPException(
            status_code=400,
            detail="Categorical reference data must contain at least 5 samples"
        )
    
    try:
        drift_monitor.reference_manager.set_reference_data(
            request.feature_name,
            request.categories,
            request.metadata
        )
        
        # Also set in Rust engine if available
        if drift_monitor.drift_detector:
            drift_monitor.drift_detector.set_categorical_reference(
                request.feature_name,
                request.categories,
                request.metadata
            )
        
        logger.info(f"Categorical reference data set for {request.feature_name}: {len(request.categories)} categories")
        
        return {
            "status": "success",
            "message": f"Categorical reference data set for feature '{request.feature_name}'"
        }
        
    except Exception as e:
        logger.error(f"Failed to set categorical reference data: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to set categorical reference data: {str(e)}")


@router.post("/baseline/training")
async def set_training_baseline(
    request: TrainingBaselineRequest,
    background_tasks: BackgroundTasks
) -> Dict[str, str]:
    """Set training data as baseline for drift detection"""
    if not is_drift_monitoring_available():
        raise HTTPException(
            status_code=503, 
            detail="Drift monitoring not available - Rust engine required"
        )
    
    if len(request.training_recipes) < 100:
        raise HTTPException(
            status_code=400,
            detail="Training baseline requires at least 100 recipes for statistical validity"
        )
    
    try:
        # Set baseline in background task for performance
        background_tasks.add_task(
            drift_monitor.set_recipe_training_baseline,
            request.training_recipes
        )
        
        return {
            "status": "success",
            "message": f"Training baseline being set from {len(request.training_recipes)} recipes"
        }
        
    except Exception as e:
        logger.error(f"Failed to set training baseline: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to set training baseline: {str(e)}")


@router.post("/detect")
async def detect_drift(request: DetectDriftRequest) -> Dict[str, Any]:
    """Detect drift for a specific feature"""
    if not is_drift_monitoring_available():
        raise HTTPException(
            status_code=503, 
            detail="Drift monitoring not available - Rust engine required"
        )
    
    if not drift_monitor.drift_detector:
        raise HTTPException(
            status_code=503,
            detail="Rust drift detector not initialized"
        )
    
    if len(request.current_data) < 10:
        raise HTTPException(
            status_code=400,
            detail="Current data must contain at least 10 samples for reliable drift detection"
        )
    
    try:
        # Detect drift using Rust engine
        drift_results = drift_monitor.drift_detector.detect_drift(
            request.feature_name,
            request.current_data,
            request.methods
        )
        
        # Convert to API response format
        results = []
        for result in drift_results:
            results.append({
                "method": result.method,
                "drift_score": result.drift_score,
                "p_value": result.p_value,
                "is_drift": result.is_drift,
                "severity": result.severity,
                "threshold": result.threshold,
                "timestamp": result.timestamp,
                "sample_size": result.sample_size,
                "reference_size": result.reference_size
            })
        
        return {
            "feature_name": request.feature_name,
            "drift_detected": any(r["is_drift"] for r in results),
            "results": results,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Drift detection failed: {e}")
        raise HTTPException(status_code=500, detail=f"Drift detection failed: {str(e)}")


@router.post("/detect/recipes")
async def detect_recipe_drift(request: RecipeDriftRequest) -> Dict[str, Any]:
    """Detect drift in recipe generation outputs"""
    if not is_drift_monitoring_available():
        raise HTTPException(
            status_code=503, 
            detail="Drift monitoring not available - Rust engine required"
        )
    
    if len(request.recipes) < 10:
        raise HTTPException(
            status_code=400,
            detail="Recipe drift detection requires at least 10 recipes"
        )
    
    try:
        # Detect recipe drift
        alerts = await drift_monitor.detect_recipe_generation_drift(request.recipes)
        
        # Convert alerts to API response format
        alert_data = []
        for alert in alerts:
            alert_data.append({
                "feature_name": alert.feature_name,
                "drift_method": alert.drift_method,
                "drift_score": alert.drift_score,
                "severity": alert.severity,
                "is_drift": alert.severity in ['medium', 'high', 'critical'],
                "timestamp": alert.timestamp.isoformat(),
                "p_value": alert.p_value,
                "threshold": alert.threshold,
                "message": alert.message,
                "metadata": alert.metadata
            })
        
        # Summary statistics
        drift_detected = len([a for a in alerts if a.severity in ['medium', 'high', 'critical']]) > 0
        critical_issues = len([a for a in alerts if a.severity == 'critical'])
        
        return {
            "drift_detected": drift_detected,
            "total_alerts": len(alerts),
            "critical_issues": critical_issues,
            "recipes_analyzed": len(request.recipes),
            "alerts": alert_data,
            "timestamp": datetime.utcnow().isoformat(),
            "summary": {
                "features_checked": len(set(a.feature_name for a in alerts)),
                "max_severity": max([a.severity for a in alerts], default="none"),
                "recommendation": self._get_drift_recommendation(alerts)
            }
        }
        
    except Exception as e:
        logger.error(f"Recipe drift detection failed: {e}")
        raise HTTPException(status_code=500, detail=f"Recipe drift detection failed: {str(e)}")


@router.get("/alerts")
async def get_drift_alerts(
    hours: int = 24,
    feature_name: Optional[str] = None,
    severity: Optional[str] = None,
    limit: int = 100
) -> Dict[str, Any]:
    """Get drift alerts from the specified time period"""
    if not is_drift_monitoring_available():
        raise HTTPException(
            status_code=503, 
            detail="Drift monitoring not available - Rust engine required"
        )
    
    try:
        # Get alerts from history
        alerts = drift_monitor.get_drift_history(feature_name, hours)
        
        # Filter by severity if specified
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        
        # Limit results
        alerts = alerts[:limit]
        
        # Convert to API format
        alert_data = [alert.to_dict() for alert in alerts]
        
        return {
            "alerts": alert_data,
            "total_alerts": len(alert_data),
            "time_range_hours": hours,
            "feature_filter": feature_name,
            "severity_filter": severity,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get drift alerts: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get drift alerts: {str(e)}")


@router.get("/features")
async def get_monitored_features() -> Dict[str, Any]:
    """Get list of features being monitored for drift"""
    if not is_drift_monitoring_available():
        raise HTTPException(
            status_code=503, 
            detail="Drift monitoring not available - Rust engine required"
        )
    
    try:
        features = drift_monitor.reference_manager.list_features()
        stats = drift_monitor.reference_manager.get_stats()
        
        return {
            "monitored_features": features,
            "total_features": len(features),
            "feature_details": stats['features'],
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get monitored features: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get monitored features: {str(e)}")


@router.post("/monitoring/start")
async def start_drift_monitoring() -> Dict[str, str]:
    """Start drift monitoring service"""
    if not is_drift_monitoring_available():
        raise HTTPException(
            status_code=503, 
            detail="Drift monitoring not available - Rust engine required"
        )
    
    try:
        await drift_monitor.start_monitoring()
        return {
            "status": "success",
            "message": "Drift monitoring started"
        }
    except Exception as e:
        logger.error(f"Failed to start drift monitoring: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start monitoring: {str(e)}")


@router.post("/monitoring/stop")
async def stop_drift_monitoring() -> Dict[str, str]:
    """Stop drift monitoring service"""
    if not is_drift_monitoring_available():
        raise HTTPException(
            status_code=503, 
            detail="Drift monitoring not available - Rust engine required"
        )
    
    try:
        await drift_monitor.stop_monitoring()
        return {
            "status": "success",
            "message": "Drift monitoring stopped"
        }
    except Exception as e:
        logger.error(f"Failed to stop drift monitoring: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to stop monitoring: {str(e)}")


@router.delete("/alerts/clear")
async def clear_drift_alerts(
    feature_name: Optional[str] = None,
    older_than_hours: Optional[int] = None
) -> Dict[str, str]:
    """Clear drift alerts"""
    if not is_drift_monitoring_available():
        raise HTTPException(
            status_code=503, 
            detail="Drift monitoring not available - Rust engine required"
        )
    
    try:
        if older_than_hours:
            # Clear alerts older than specified hours
            cutoff_time = datetime.utcnow() - timedelta(hours=older_than_hours)
            original_count = len(drift_monitor.alerts)
            drift_monitor.alerts = [
                alert for alert in drift_monitor.alerts 
                if alert.timestamp > cutoff_time
            ]
            cleared_count = original_count - len(drift_monitor.alerts)
        else:
            # Clear all alerts (or for specific feature)
            if feature_name:
                original_count = len(drift_monitor.alerts)
                drift_monitor.alerts = [
                    alert for alert in drift_monitor.alerts 
                    if alert.feature_name != feature_name
                ]
                cleared_count = original_count - len(drift_monitor.alerts)
            else:
                cleared_count = len(drift_monitor.alerts)
                drift_monitor.alerts.clear()
        
        # Also clear from Rust engine if available
        if drift_monitor.drift_detector:
            drift_monitor.drift_detector.clear_history(feature_name)
        
        return {
            "status": "success",
            "message": f"Cleared {cleared_count} drift alerts"
        }
        
    except Exception as e:
        logger.error(f"Failed to clear drift alerts: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to clear alerts: {str(e)}")


@router.get("/health")
async def drift_monitoring_health_check() -> Dict[str, Any]:
    """Health check for drift monitoring system"""
    health = {
        "drift_monitoring_available": is_drift_monitoring_available(),
        "rust_engine_available": is_drift_monitoring_available(),
        "monitoring_active": drift_monitor.monitoring_active,
        "overall_status": "unhealthy"
    }
    
    if is_drift_monitoring_available():
        try:
            # Test basic functionality
            features = drift_monitor.reference_manager.list_features()
            metrics = drift_monitor.get_current_metrics()
            
            health.update({
                "features_monitored": len(features),
                "total_alerts": len(drift_monitor.alerts),
                "last_check": metrics.last_check.isoformat(),
                "drift_rate": metrics.drift_rate,
                "overall_status": "healthy"
            })
            
        except Exception as e:
            health.update({
                "overall_status": "degraded",
                "error": str(e)
            })
    
    return health


def _get_drift_recommendation(alerts: List[DriftAlert]) -> str:
    """Get recommendation based on drift alerts"""
    if not alerts:
        return "No drift detected - model performance is stable"
    
    critical_count = len([a for a in alerts if a.severity == 'critical'])
    high_count = len([a for a in alerts if a.severity == 'high'])
    
    if critical_count > 0:
        return f"CRITICAL: {critical_count} critical drift issues detected. Consider retraining model immediately."
    elif high_count > 2:
        return f"HIGH: {high_count} high-severity drift issues. Monitor closely and consider retraining."
    elif high_count > 0:
        return f"MEDIUM: {high_count} high-severity drift detected. Investigate data distribution changes."
    else:
        return "LOW: Minor drift detected. Continue monitoring - no immediate action required."