"""
Model Drift Monitoring Service

Provides comprehensive model drift detection and monitoring capabilities
for the Chef Genius AI system, integrating with the high-performance
Rust drift detection engine.
"""

import logging
from typing import List, Dict, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
import asyncio
import json
from dataclasses import dataclass, asdict
from contextlib import asynccontextmanager
import numpy as np
from collections import defaultdict, deque
import statistics

from app.core.config import settings
from app.models.recipe import RecipeCreate, NutritionInfo

logger = logging.getLogger(__name__)

# Try to import the Rust drift detection engine
try:
    import chef_genius_core as cgc
    RUST_DRIFT_AVAILABLE = True
    logger.info("✅ Rust drift detection engine loaded successfully")
except ImportError as e:
    RUST_DRIFT_AVAILABLE = False
    logger.warning(f"⚠️  Rust drift detection not available: {e}")
    logger.warning("Using Python fallback implementations")


@dataclass
class DriftAlert:
    """Drift detection alert"""
    feature_name: str
    drift_method: str
    drift_score: float
    severity: str  # "low", "medium", "high", "critical"
    timestamp: datetime
    p_value: Optional[float] = None
    threshold: Optional[float] = None
    message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


@dataclass
class DriftMetrics:
    """Drift monitoring metrics"""
    total_features_monitored: int
    features_with_drift: int
    drift_rate: float
    alerts_last_24h: int
    critical_alerts: int
    last_check: datetime
    monitoring_status: str
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['last_check'] = self.last_check.isoformat()
        return data


class ReferenceDataManager:
    """Manages reference datasets for drift detection"""
    
    def __init__(self):
        self.reference_data: Dict[str, Dict[str, Any]] = {}
        self.last_updated: Dict[str, datetime] = {}
        
    def set_reference_data(
        self, 
        feature_name: str, 
        data: Union[List[float], List[str]], 
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Set reference data for a feature"""
        self.reference_data[feature_name] = {
            'data': data,
            'metadata': metadata or {},
            'sample_size': len(data),
            'data_type': 'numerical' if isinstance(data[0], (int, float)) else 'categorical'
        }
        self.last_updated[feature_name] = datetime.utcnow()
        logger.info(f"Reference data set for {feature_name}: {len(data)} samples")
    
    def get_reference_data(self, feature_name: str) -> Optional[Dict[str, Any]]:
        """Get reference data for a feature"""
        return self.reference_data.get(feature_name)
    
    def list_features(self) -> List[str]:
        """List all monitored features"""
        return list(self.reference_data.keys())
    
    def get_stats(self) -> Dict[str, Any]:
        """Get reference data statistics"""
        return {
            'total_features': len(self.reference_data),
            'features': {
                name: {
                    'sample_size': data['sample_size'],
                    'data_type': data['data_type'],
                    'last_updated': self.last_updated.get(name, datetime.min).isoformat()
                }
                for name, data in self.reference_data.items()
            }
        }


class ModelDriftMonitor:
    """High-level model drift monitoring service"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._default_config()
        self.reference_manager = ReferenceDataManager()
        self.drift_detector = None
        self.recipe_monitor = None
        self.alerts: deque = deque(maxlen=1000)  # Keep last 1000 alerts
        self.metrics_history: deque = deque(maxlen=24 * 7)  # Keep 7 days of hourly metrics
        self.monitoring_active = False
        
        # Initialize Rust engines if available
        if RUST_DRIFT_AVAILABLE:
            self._initialize_rust_engines()
    
    def _default_config(self) -> Dict[str, Any]:
        """Default drift monitoring configuration"""
        return {
            'ks_threshold': 0.05,
            'psi_threshold': 0.2,
            'js_threshold': 0.1,
            'chi2_threshold': 0.05,
            'wasserstein_threshold': 0.3,
            'min_sample_size': 50,
            'check_interval_hours': 1,
            'alert_cooldown_hours': 6,
            'auto_update_reference': False,
            'enable_notifications': True,
            'notification_channels': ['discord', 'logging'],
            'critical_threshold_multiplier': 2.0,
        }
    
    def _initialize_rust_engines(self) -> None:
        """Initialize Rust drift detection engines"""
        try:
            # Create drift configuration
            rust_config = cgc.DriftConfig()
            rust_config.ks_threshold = self.config['ks_threshold']
            rust_config.psi_threshold = self.config['psi_threshold']
            rust_config.js_threshold = self.config['js_threshold']
            rust_config.chi2_threshold = self.config['chi2_threshold']
            rust_config.wasserstein_threshold = self.config['wasserstein_threshold']
            rust_config.min_sample_size = self.config['min_sample_size']
            
            # Initialize engines
            self.drift_detector = cgc.PyDriftDetector(rust_config)
            self.recipe_monitor = cgc.PyRecipeDriftMonitor(rust_config)
            
            logger.info("🚀 Rust drift detection engines initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize Rust drift engines: {e}")
            self.drift_detector = None
            self.recipe_monitor = None
    
    async def start_monitoring(self) -> None:
        """Start drift monitoring"""
        if not RUST_DRIFT_AVAILABLE:
            logger.warning("Cannot start monitoring - Rust drift detection not available")
            return
            
        self.monitoring_active = True
        logger.info("🔍 Model drift monitoring started")
        
        # Start background monitoring task
        asyncio.create_task(self._monitoring_loop())
    
    async def stop_monitoring(self) -> None:
        """Stop drift monitoring"""
        self.monitoring_active = False
        logger.info("⏹️  Model drift monitoring stopped")
    
    async def _monitoring_loop(self) -> None:
        """Background monitoring loop"""
        check_interval = timedelta(hours=self.config['check_interval_hours'])
        
        while self.monitoring_active:
            try:
                await self._perform_drift_check()
                await asyncio.sleep(check_interval.total_seconds())
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retrying
    
    async def _perform_drift_check(self) -> None:
        """Perform scheduled drift check"""
        logger.info("Performing scheduled drift check...")
        
        # This would be called with actual model outputs/features
        # For now, we'll just update metrics
        metrics = self.get_current_metrics()
        self.metrics_history.append(metrics)
        
        logger.info(f"Drift check completed. Monitoring {metrics.total_features_monitored} features")
    
    def set_recipe_training_baseline(
        self, 
        training_recipes: List[Dict[str, Any]]
    ) -> None:
        """Set baseline data from training recipes"""
        logger.info(f"Setting recipe training baseline from {len(training_recipes)} recipes")
        
        # Extract features from training recipes
        recipe_lengths = []
        ingredient_counts = []
        instruction_counts = []
        cooking_times = []
        cuisines = []
        difficulties = []
        
        for recipe in training_recipes:
            # Title length
            if 'title' in recipe:
                recipe_lengths.append(len(str(recipe['title'])))
            
            # Ingredient count
            if 'ingredients' in recipe:
                ingredient_counts.append(len(recipe['ingredients']))
            
            # Instruction count
            if 'instructions' in recipe:
                instruction_counts.append(len(recipe['instructions']))
            
            # Cooking time (convert to minutes)
            if 'cooking_time' in recipe:
                cooking_time_mins = self._parse_cooking_time(recipe['cooking_time'])
                if cooking_time_mins is not None:
                    cooking_times.append(cooking_time_mins)
            
            # Categorical features
            if 'cuisine_type' in recipe and recipe['cuisine_type']:
                cuisines.append(str(recipe['cuisine_type']))
            
            if 'difficulty' in recipe and recipe['difficulty']:
                difficulties.append(str(recipe['difficulty']))
        
        # Set reference data
        if recipe_lengths:
            self.reference_manager.set_reference_data('recipe_title_length', recipe_lengths)
            if self.drift_detector:
                self.drift_detector.set_reference_data('recipe_title_length', recipe_lengths)
        
        if ingredient_counts:
            self.reference_manager.set_reference_data('ingredient_count', ingredient_counts)
            if self.drift_detector:
                self.drift_detector.set_reference_data('ingredient_count', ingredient_counts)
        
        if instruction_counts:
            self.reference_manager.set_reference_data('instruction_count', instruction_counts)
            if self.drift_detector:
                self.drift_detector.set_reference_data('instruction_count', instruction_counts)
        
        if cooking_times:
            self.reference_manager.set_reference_data('cooking_time_minutes', cooking_times)
            if self.drift_detector:
                self.drift_detector.set_reference_data('cooking_time_minutes', cooking_times)
        
        if cuisines:
            self.reference_manager.set_reference_data('cuisine_type', cuisines)
            if self.drift_detector:
                self.drift_detector.set_categorical_reference('cuisine_type', cuisines)
        
        if difficulties:
            self.reference_manager.set_reference_data('difficulty', difficulties)
            if self.drift_detector:
                self.drift_detector.set_categorical_reference('difficulty', difficulties)
        
        logger.info("Training baseline set for recipe drift detection")
    
    async def detect_recipe_generation_drift(
        self, 
        generated_recipes: List[Dict[str, Any]]
    ) -> List[DriftAlert]:
        """Detect drift in recipe generation outputs"""
        if not RUST_DRIFT_AVAILABLE or not self.recipe_monitor:
            return await self._fallback_recipe_drift_detection(generated_recipes)
        
        try:
            # Use Rust engine for high-performance detection
            rust_results = self.recipe_monitor.detect_recipe_drift(generated_recipes)
            
            # Convert Rust results to alerts
            alerts = []
            for result in rust_results:
                alert = DriftAlert(
                    feature_name=result.feature_name,
                    drift_method=result.method,
                    drift_score=result.drift_score,
                    severity=result.severity,
                    timestamp=datetime.fromtimestamp(result.timestamp),
                    p_value=result.p_value,
                    threshold=result.threshold,
                    message=f"Drift detected in {result.feature_name} using {result.method}",
                    metadata={
                        'sample_size': result.sample_size,
                        'reference_size': result.reference_size,
                        'is_drift': result.is_drift
                    }
                )
                alerts.append(alert)
                
                # Store alert
                self.alerts.append(alert)
            
            # Send notifications for critical alerts
            critical_alerts = [a for a in alerts if a.severity == 'critical']
            if critical_alerts:
                await self._send_drift_notifications(critical_alerts)
            
            logger.info(f"Detected {len(alerts)} drift signals from {len(generated_recipes)} recipes")
            return alerts
            
        except Exception as e:
            logger.error(f"Rust drift detection failed: {e}")
            return await self._fallback_recipe_drift_detection(generated_recipes)
    
    async def _fallback_recipe_drift_detection(
        self, 
        generated_recipes: List[Dict[str, Any]]
    ) -> List[DriftAlert]:
        """Fallback Python implementation for drift detection"""
        alerts = []
        
        # Extract current features
        current_features = self._extract_recipe_features(generated_recipes)
        
        # Check each feature against reference data
        for feature_name, current_data in current_features.items():
            reference_data = self.reference_manager.get_reference_data(feature_name)
            if not reference_data:
                continue
            
            # Simple statistical comparison (placeholder)
            reference_values = reference_data['data']
            
            if isinstance(current_data[0], (int, float)) and isinstance(reference_values[0], (int, float)):
                # Numerical comparison using means and standard deviations
                current_mean = statistics.mean(current_data)
                ref_mean = statistics.mean(reference_values)
                
                current_std = statistics.stdev(current_data) if len(current_data) > 1 else 0
                ref_std = statistics.stdev(reference_values) if len(reference_values) > 1 else 0
                
                # Simple z-score based drift detection
                if ref_std > 0:
                    z_score = abs(current_mean - ref_mean) / ref_std
                    is_drift = z_score > 2.0  # 2 standard deviations
                    severity = 'high' if z_score > 3.0 else 'medium' if z_score > 2.0 else 'low'
                    
                    if is_drift:
                        alert = DriftAlert(
                            feature_name=feature_name,
                            drift_method='z_score',
                            drift_score=z_score,
                            severity=severity,
                            timestamp=datetime.utcnow(),
                            threshold=2.0,
                            message=f"Statistical drift detected in {feature_name}",
                            metadata={
                                'current_mean': current_mean,
                                'reference_mean': ref_mean,
                                'z_score': z_score
                            }
                        )
                        alerts.append(alert)
        
        return alerts
    
    def _extract_recipe_features(self, recipes: List[Dict[str, Any]]) -> Dict[str, List]:
        """Extract numerical features from recipes"""
        features = defaultdict(list)
        
        for recipe in recipes:
            if 'title' in recipe:
                features['recipe_title_length'].append(len(str(recipe['title'])))
            
            if 'ingredients' in recipe:
                features['ingredient_count'].append(len(recipe['ingredients']))
            
            if 'instructions' in recipe:
                features['instruction_count'].append(len(recipe['instructions']))
            
            if 'cooking_time' in recipe:
                cooking_time_mins = self._parse_cooking_time(recipe['cooking_time'])
                if cooking_time_mins is not None:
                    features['cooking_time_minutes'].append(cooking_time_mins)
        
        return dict(features)
    
    def _parse_cooking_time(self, time_str: str) -> Optional[float]:
        """Parse cooking time string to minutes"""
        if not time_str:
            return None
        
        try:
            time_str = str(time_str).lower()
            
            # Extract numbers
            import re
            numbers = re.findall(r'[\d.]+', time_str)
            if not numbers:
                return None
            
            value = float(numbers[0])
            
            if 'hour' in time_str:
                return value * 60
            elif 'min' in time_str:
                return value
            else:
                # Assume minutes if no unit
                return value
                
        except (ValueError, IndexError):
            return None
    
    async def _send_drift_notifications(self, alerts: List[DriftAlert]) -> None:
        """Send notifications for drift alerts"""
        if not self.config.get('enable_notifications', True):
            return
        
        channels = self.config.get('notification_channels', ['logging'])
        
        for alert in alerts:
            message = f"🚨 Critical Model Drift Alert\n"
            message += f"Feature: {alert.feature_name}\n"
            message += f"Method: {alert.drift_method}\n"
            message += f"Score: {alert.drift_score:.3f}\n"
            message += f"Severity: {alert.severity}\n"
            message += f"Time: {alert.timestamp.isoformat()}"
            
            for channel in channels:
                if channel == 'logging':
                    logger.critical(message)
                elif channel == 'discord':
                    # Would integrate with Discord webhook
                    logger.info(f"Would send Discord notification: {message}")
                # Add other notification channels as needed
    
    def get_current_metrics(self) -> DriftMetrics:
        """Get current drift monitoring metrics"""
        now = datetime.utcnow()
        last_24h = now - timedelta(hours=24)
        
        # Count alerts in last 24 hours
        recent_alerts = [a for a in self.alerts if a.timestamp > last_24h]
        critical_alerts = [a for a in recent_alerts if a.severity == 'critical']
        
        # Count features with drift
        features_with_drift = len(set(a.feature_name for a in recent_alerts))
        total_features = len(self.reference_manager.list_features())
        
        drift_rate = features_with_drift / total_features if total_features > 0 else 0.0
        
        return DriftMetrics(
            total_features_monitored=total_features,
            features_with_drift=features_with_drift,
            drift_rate=drift_rate,
            alerts_last_24h=len(recent_alerts),
            critical_alerts=len(critical_alerts),
            last_check=now,
            monitoring_status='active' if self.monitoring_active else 'inactive'
        )
    
    def get_drift_history(
        self, 
        feature_name: Optional[str] = None,
        hours: int = 24
    ) -> List[DriftAlert]:
        """Get drift alert history"""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        filtered_alerts = [
            alert for alert in self.alerts 
            if alert.timestamp > cutoff_time
        ]
        
        if feature_name:
            filtered_alerts = [
                alert for alert in filtered_alerts 
                if alert.feature_name == feature_name
            ]
        
        return sorted(filtered_alerts, key=lambda x: x.timestamp, reverse=True)
    
    def get_drift_report(self) -> Dict[str, Any]:
        """Generate comprehensive drift monitoring report"""
        metrics = self.get_current_metrics()
        recent_alerts = self.get_drift_history(hours=24)
        
        # Group alerts by feature
        alerts_by_feature = defaultdict(list)
        for alert in recent_alerts:
            alerts_by_feature[alert.feature_name].append(alert)
        
        # Calculate per-feature statistics
        feature_stats = {}
        for feature_name in self.reference_manager.list_features():
            feature_alerts = alerts_by_feature.get(feature_name, [])
            feature_stats[feature_name] = {
                'alerts_24h': len(feature_alerts),
                'max_severity': max([a.severity for a in feature_alerts], default='none'),
                'last_alert': feature_alerts[0].timestamp.isoformat() if feature_alerts else None,
                'drift_detected': len(feature_alerts) > 0
            }
        
        return {
            'summary': metrics.to_dict(),
            'reference_data': self.reference_manager.get_stats(),
            'recent_alerts': [alert.to_dict() for alert in recent_alerts[:10]],  # Last 10 alerts
            'feature_statistics': feature_stats,
            'configuration': self.config,
            'rust_available': RUST_DRIFT_AVAILABLE,
            'monitoring_status': 'active' if self.monitoring_active else 'inactive'
        }


# Global drift monitoring service instance
drift_monitor = ModelDriftMonitor()


async def initialize_drift_monitoring() -> None:
    """Initialize the drift monitoring service"""
    await drift_monitor.start_monitoring()
    logger.info("🔍 Drift monitoring service initialized")


def is_drift_monitoring_available() -> bool:
    """Check if drift monitoring is available"""
    return RUST_DRIFT_AVAILABLE


def get_drift_monitoring_stats() -> Dict[str, Any]:
    """Get drift monitoring statistics"""
    return drift_monitor.get_drift_report()