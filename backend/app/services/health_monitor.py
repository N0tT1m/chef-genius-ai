"""
Health monitoring and system diagnostics for Chef Genius MCP system
"""

import asyncio
import logging
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import aiohttp
import psutil
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class HealthStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"

@dataclass
class ServiceHealth:
    name: str
    status: HealthStatus
    response_time: Optional[float] = None
    last_check: Optional[datetime] = None
    error_message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class SystemMetrics:
    cpu_percent: float
    memory_percent: float
    disk_percent: float
    network_io: Dict[str, int]
    process_count: int
    uptime_seconds: float

class HealthMonitor:
    """
    Comprehensive health monitoring for Chef Genius MCP system.
    
    Monitors:
    - MCP server health and performance
    - Vector database connectivity
    - System resources
    - Application metrics
    """
    
    def __init__(self):
        """Initialize the health monitor."""
        self.services = {
            "weaviate": "http://localhost:8080/v1/meta",
            "recipe-server": "http://localhost:8001/health",
            "knowledge-server": "http://localhost:8002/health", 
            "tool-server": "http://localhost:8003/health",
            "backend": "http://localhost:8000/health",
            "redis": "http://localhost:6379/ping"
        }
        
        self.health_history = {}
        self.metrics_history = []
        self.alert_thresholds = {
            "response_time": 5.0,  # seconds
            "cpu_percent": 80.0,
            "memory_percent": 85.0,
            "disk_percent": 90.0,
            "error_rate": 5.0  # percent
        }
        
        self.start_time = time.time()
        
    async def check_all_services(self) -> Dict[str, ServiceHealth]:
        """Check health of all registered services."""
        health_results = {}
        
        # Check all services concurrently
        tasks = []
        for service_name, endpoint in self.services.items():
            task = self._check_service_health(service_name, endpoint)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        for i, result in enumerate(results):
            service_name = list(self.services.keys())[i]
            
            if isinstance(result, Exception):
                health_results[service_name] = ServiceHealth(
                    name=service_name,
                    status=HealthStatus.UNHEALTHY,
                    error_message=str(result),
                    last_check=datetime.now()
                )
            else:
                health_results[service_name] = result
        
        # Store in history
        self.health_history[datetime.now()] = health_results
        
        # Clean old history (keep last hour)
        self._cleanup_history()
        
        return health_results
    
    async def _check_service_health(self, service_name: str, endpoint: str) -> ServiceHealth:
        """Check health of a single service."""
        start_time = time.time()
        
        try:
            timeout = aiohttp.ClientTimeout(total=10.0)
            
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(endpoint) as response:
                    response_time = time.time() - start_time
                    
                    if response.status == 200:
                        # Try to get additional metadata
                        metadata = None
                        try:
                            if response.content_type == 'application/json':
                                metadata = await response.json()
                        except:
                            pass
                        
                        return ServiceHealth(
                            name=service_name,
                            status=HealthStatus.HEALTHY,
                            response_time=response_time,
                            last_check=datetime.now(),
                            metadata=metadata
                        )
                    else:
                        return ServiceHealth(
                            name=service_name,
                            status=HealthStatus.DEGRADED,
                            response_time=response_time,
                            last_check=datetime.now(),
                            error_message=f"HTTP {response.status}"
                        )
                        
        except asyncio.TimeoutError:
            return ServiceHealth(
                name=service_name,
                status=HealthStatus.UNHEALTHY,
                response_time=10.0,
                last_check=datetime.now(),
                error_message="Timeout"
            )
        except Exception as e:
            return ServiceHealth(
                name=service_name,
                status=HealthStatus.UNHEALTHY,
                last_check=datetime.now(),
                error_message=str(e)
            )
    
    def get_system_metrics(self) -> SystemMetrics:
        """Get current system resource metrics."""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            
            # Network I/O
            network = psutil.net_io_counters()
            network_io = {
                "bytes_sent": network.bytes_sent,
                "bytes_recv": network.bytes_recv,
                "packets_sent": network.packets_sent,
                "packets_recv": network.packets_recv
            }
            
            # Process count
            process_count = len(psutil.pids())
            
            # Uptime
            uptime_seconds = time.time() - self.start_time
            
            metrics = SystemMetrics(
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                disk_percent=disk_percent,
                network_io=network_io,
                process_count=process_count,
                uptime_seconds=uptime_seconds
            )
            
            # Store in history
            self.metrics_history.append({
                "timestamp": datetime.now(),
                "metrics": metrics
            })
            
            # Keep only last hour of metrics
            cutoff_time = datetime.now() - timedelta(hours=1)
            self.metrics_history = [
                entry for entry in self.metrics_history 
                if entry["timestamp"] > cutoff_time
            ]
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to get system metrics: {e}")
            return SystemMetrics(
                cpu_percent=0.0,
                memory_percent=0.0,
                disk_percent=0.0,
                network_io={},
                process_count=0,
                uptime_seconds=0.0
            )
    
    async def get_comprehensive_health(self) -> Dict[str, Any]:
        """Get comprehensive health report."""
        # Get service health
        service_health = await self.check_all_services()
        
        # Get system metrics
        system_metrics = self.get_system_metrics()
        
        # Calculate overall status
        overall_status = self._calculate_overall_status(service_health, system_metrics)
        
        # Check for alerts
        alerts = self._check_alerts(service_health, system_metrics)
        
        # Performance summary
        performance_summary = self._get_performance_summary()
        
        return {
            "overall_status": overall_status.value,
            "timestamp": datetime.now().isoformat(),
            "services": {name: asdict(health) for name, health in service_health.items()},
            "system_metrics": asdict(system_metrics),
            "alerts": alerts,
            "performance_summary": performance_summary,
            "uptime": {
                "seconds": system_metrics.uptime_seconds,
                "human_readable": self._format_uptime(system_metrics.uptime_seconds)
            }
        }
    
    def _calculate_overall_status(self, service_health: Dict[str, ServiceHealth], 
                                system_metrics: SystemMetrics) -> HealthStatus:
        """Calculate overall system health status."""
        # Check if any critical services are down
        critical_services = ["backend", "weaviate"]
        for service_name in critical_services:
            if service_name in service_health:
                if service_health[service_name].status == HealthStatus.UNHEALTHY:
                    return HealthStatus.UNHEALTHY
        
        # Check system resources
        if (system_metrics.cpu_percent > self.alert_thresholds["cpu_percent"] or
            system_metrics.memory_percent > self.alert_thresholds["memory_percent"] or
            system_metrics.disk_percent > self.alert_thresholds["disk_percent"]):
            return HealthStatus.DEGRADED
        
        # Check if any services are degraded
        for health in service_health.values():
            if health.status == HealthStatus.DEGRADED:
                return HealthStatus.DEGRADED
            elif health.status == HealthStatus.UNHEALTHY:
                return HealthStatus.DEGRADED
        
        return HealthStatus.HEALTHY
    
    def _check_alerts(self, service_health: Dict[str, ServiceHealth], 
                     system_metrics: SystemMetrics) -> List[Dict[str, Any]]:
        """Check for alert conditions."""
        alerts = []
        
        # Service alerts
        for service_name, health in service_health.items():
            if health.status == HealthStatus.UNHEALTHY:
                alerts.append({
                    "type": "service_down",
                    "severity": "critical",
                    "message": f"Service {service_name} is unhealthy: {health.error_message}",
                    "service": service_name,
                    "timestamp": datetime.now().isoformat()
                })
            elif (health.response_time and 
                  health.response_time > self.alert_thresholds["response_time"]):
                alerts.append({
                    "type": "slow_response",
                    "severity": "warning",
                    "message": f"Service {service_name} is responding slowly: {health.response_time:.2f}s",
                    "service": service_name,
                    "timestamp": datetime.now().isoformat()
                })
        
        # System resource alerts
        if system_metrics.cpu_percent > self.alert_thresholds["cpu_percent"]:
            alerts.append({
                "type": "high_cpu",
                "severity": "warning",
                "message": f"High CPU usage: {system_metrics.cpu_percent:.1f}%",
                "timestamp": datetime.now().isoformat()
            })
        
        if system_metrics.memory_percent > self.alert_thresholds["memory_percent"]:
            alerts.append({
                "type": "high_memory",
                "severity": "warning", 
                "message": f"High memory usage: {system_metrics.memory_percent:.1f}%",
                "timestamp": datetime.now().isoformat()
            })
        
        if system_metrics.disk_percent > self.alert_thresholds["disk_percent"]:
            alerts.append({
                "type": "high_disk",
                "severity": "critical",
                "message": f"High disk usage: {system_metrics.disk_percent:.1f}%",
                "timestamp": datetime.now().isoformat()
            })
        
        return alerts
    
    def _get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary from recent metrics."""
        if not self.metrics_history:
            return {}
        
        recent_metrics = [entry["metrics"] for entry in self.metrics_history[-10:]]
        
        if not recent_metrics:
            return {}
        
        avg_cpu = sum(m.cpu_percent for m in recent_metrics) / len(recent_metrics)
        avg_memory = sum(m.memory_percent for m in recent_metrics) / len(recent_metrics)
        
        # Calculate response time trends
        response_times = []
        recent_health = list(self.health_history.values())[-5:] if self.health_history else []
        
        for health_check in recent_health:
            for service_health in health_check.values():
                if service_health.response_time:
                    response_times.append(service_health.response_time)
        
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0
        
        return {
            "avg_cpu_percent": round(avg_cpu, 2),
            "avg_memory_percent": round(avg_memory, 2),
            "avg_response_time": round(avg_response_time, 3),
            "sample_count": len(recent_metrics),
            "trend": self._calculate_trend(recent_metrics)
        }
    
    def _calculate_trend(self, metrics: List[SystemMetrics]) -> str:
        """Calculate resource usage trend."""
        if len(metrics) < 2:
            return "stable"
        
        cpu_trend = metrics[-1].cpu_percent - metrics[0].cpu_percent
        memory_trend = metrics[-1].memory_percent - metrics[0].memory_percent
        
        if cpu_trend > 10 or memory_trend > 10:
            return "increasing"
        elif cpu_trend < -10 or memory_trend < -10:
            return "decreasing"
        else:
            return "stable"
    
    def _format_uptime(self, seconds: float) -> str:
        """Format uptime in human readable format."""
        days = int(seconds // 86400)
        hours = int((seconds % 86400) // 3600)
        minutes = int((seconds % 3600) // 60)
        
        if days > 0:
            return f"{days}d {hours}h {minutes}m"
        elif hours > 0:
            return f"{hours}h {minutes}m"
        else:
            return f"{minutes}m"
    
    def _cleanup_history(self):
        """Clean up old health history entries."""
        cutoff_time = datetime.now() - timedelta(hours=1)
        self.health_history = {
            timestamp: health for timestamp, health in self.health_history.items()
            if timestamp > cutoff_time
        }
    
    async def run_continuous_monitoring(self, interval: int = 30):
        """Run continuous health monitoring."""
        logger.info(f"Starting continuous health monitoring (interval: {interval}s)")
        
        while True:
            try:
                await self.check_all_services()
                self.get_system_metrics()
                await asyncio.sleep(interval)
            except Exception as e:
                logger.error(f"Health monitoring error: {e}")
                await asyncio.sleep(interval)

# Global health monitor instance
health_monitor = HealthMonitor()

async def get_health_status() -> Dict[str, Any]:
    """Get current health status (for API endpoints)."""
    return await health_monitor.get_comprehensive_health()