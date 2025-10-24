use crate::config::Config;
use crate::error::{ApiError, Result};
use crate::models::*;
use axum::extract::ws::{Message, WebSocket};
use futures_util::{SinkExt, StreamExt};
use serde_json::json;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use sysinfo::{CpuExt, System, SystemExt};
use tokio::sync::{broadcast, RwLock};
use tokio::time::{interval, Interval};
use tracing::{debug, error, info, warn};
use uuid::Uuid;

pub struct MonitoringService {
    config: Arc<Config>,
    system_info: Arc<RwLock<System>>,
    active_connections: Arc<RwLock<HashMap<Uuid, WebSocketConnection>>>,
    metrics_broadcaster: broadcast::Sender<SystemMetrics>,
    training_updates: broadcast::Sender<TrainingUpdate>,
}

struct WebSocketConnection {
    connection_id: Uuid,
    job_id: Option<Uuid>,
    connected_at: chrono::DateTime<chrono::Utc>,
    last_ping: chrono::DateTime<chrono::Utc>,
}

impl MonitoringService {
    pub async fn new(config: &Config) -> Result<Self> {
        info!("Initializing Monitoring Service");
        
        let (metrics_broadcaster, _) = broadcast::channel(1000);
        let (training_updates, _) = broadcast::channel(1000);
        
        let service = MonitoringService {
            config: Arc::new(config.clone()),
            system_info: Arc::new(RwLock::new(System::new_all())),
            active_connections: Arc::new(RwLock::new(HashMap::new())),
            metrics_broadcaster,
            training_updates,
        };
        
        // Start background monitoring tasks
        service.start_system_monitoring().await;
        service.start_connection_cleanup().await;
        
        Ok(service)
    }
    
    async fn start_system_monitoring(&self) {
        let system_info = self.system_info.clone();
        let metrics_broadcaster = self.metrics_broadcaster.clone();
        let interval_seconds = self.config.monitoring.system_metrics_interval_seconds;
        
        tokio::spawn(async move {
            let mut interval = interval(Duration::from_secs(interval_seconds));
            
            loop {
                interval.tick().await;
                
                let metrics = {
                    let mut system = system_info.write().await;
                    system.refresh_all();
                    Self::collect_system_metrics(&system).await
                };
                
                if let Ok(metrics) = metrics {
                    let _ = metrics_broadcaster.send(metrics);
                }
            }
        });
        
        info!("System monitoring started (interval: {}s)", interval_seconds);
    }
    
    async fn start_connection_cleanup(&self) {
        let active_connections = self.active_connections.clone();
        
        tokio::spawn(async move {
            let mut interval = interval(Duration::from_secs(60)); // Cleanup every minute
            
            loop {
                interval.tick().await;
                
                let mut connections = active_connections.write().await;
                let now = chrono::Utc::now();
                
                // Remove stale connections (no ping in last 5 minutes)
                connections.retain(|_, conn| {
                    now.signed_duration_since(conn.last_ping).num_minutes() < 5
                });
            }
        });
        
        debug!("Connection cleanup task started");
    }
    
    pub async fn handle_training_stream(&self, mut socket: WebSocket, job_id: Uuid) {
        let connection_id = Uuid::new_v4();
        
        info!("New WebSocket connection for training job {}: {}", job_id, connection_id);
        
        // Register connection
        {
            let mut connections = self.active_connections.write().await;
            connections.insert(connection_id, WebSocketConnection {
                connection_id,
                job_id: Some(job_id),
                connected_at: chrono::Utc::now(),
                last_ping: chrono::Utc::now(),
            });
        }
        
        // Subscribe to training updates
        let mut training_receiver = self.training_updates.subscribe();
        let mut metrics_receiver = self.metrics_broadcaster.subscribe();
        
        // Send initial connection acknowledgment
        let ack_message = json!({
            "type": "connection_established",
            "connection_id": connection_id,
            "job_id": job_id,
            "timestamp": chrono::Utc::now()
        });
        
        if socket.send(Message::Text(ack_message.to_string())).await.is_err() {
            warn!("Failed to send acknowledgment to WebSocket connection {}", connection_id);
            return;
        }
        
        // Handle WebSocket communication
        loop {
            tokio::select! {
                // Handle incoming WebSocket messages
                msg = socket.recv() => {
                    match msg {
                        Some(Ok(Message::Text(text))) => {
                            if let Err(e) = self.handle_websocket_message(&text, connection_id).await {
                                error!("Error handling WebSocket message: {}", e);
                            }
                        }
                        Some(Ok(Message::Ping(data))) => {
                            if socket.send(Message::Pong(data)).await.is_err() {
                                break;
                            }
                            self.update_connection_ping(connection_id).await;
                        }
                        Some(Ok(Message::Pong(_))) => {
                            self.update_connection_ping(connection_id).await;
                        }
                        Some(Ok(Message::Close(_))) => {
                            info!("WebSocket connection {} closed by client", connection_id);
                            break;
                        }
                        Some(Err(e)) => {
                            error!("WebSocket error for connection {}: {}", connection_id, e);
                            break;
                        }
                        None => {
                            info!("WebSocket connection {} ended", connection_id);
                            break;
                        }
                        _ => {}
                    }
                }
                
                // Forward training updates for this specific job
                update = training_receiver.recv() => {
                    if let Ok(update) = update {
                        if update.job_id == job_id {
                            let message = json!({
                                "type": "training_update",
                                "data": update
                            });
                            
                            if socket.send(Message::Text(message.to_string())).await.is_err() {
                                break;
                            }
                        }
                    }
                }
                
                // Forward system metrics
                metrics = metrics_receiver.recv() => {
                    if let Ok(metrics) = metrics {
                        let message = json!({
                            "type": "system_metrics",
                            "data": metrics
                        });
                        
                        if socket.send(Message::Text(message.to_string())).await.is_err() {
                            break;
                        }
                    }
                }
                
                // Send periodic ping
                _ = tokio::time::sleep(Duration::from_secs(30)) => {
                    if socket.send(Message::Ping(vec![])).await.is_err() {
                        break;
                    }
                }
            }
        }
        
        // Clean up connection
        {
            let mut connections = self.active_connections.write().await;
            connections.remove(&connection_id);
        }
        
        info!("WebSocket connection {} closed", connection_id);
    }
    
    async fn handle_websocket_message(&self, message: &str, connection_id: Uuid) -> Result<()> {
        let parsed: serde_json::Value = serde_json::from_str(message)
            .map_err(|e| ApiError::InvalidRequest(format!("Invalid JSON: {}", e)))?;
        
        match parsed.get("type").and_then(|t| t.as_str()) {
            Some("ping") => {
                self.update_connection_ping(connection_id).await;
            }
            Some("subscribe_metrics") => {
                // Client wants to subscribe to additional metrics
                debug!("Connection {} subscribed to additional metrics", connection_id);
            }
            Some("request_status") => {
                // Client requesting current status
                debug!("Connection {} requested status update", connection_id);
            }
            _ => {
                warn!("Unknown message type from connection {}: {}", connection_id, message);
            }
        }
        
        Ok(())
    }
    
    async fn update_connection_ping(&self, connection_id: Uuid) {
        let mut connections = self.active_connections.write().await;
        if let Some(conn) = connections.get_mut(&connection_id) {
            conn.last_ping = chrono::Utc::now();
        }
    }
    
    pub async fn get_system_metrics(&self) -> Result<SystemMetrics> {
        let mut system = self.system_info.write().await;
        system.refresh_all();
        Self::collect_system_metrics(&system).await
    }
    
    async fn collect_system_metrics(system: &System) -> Result<SystemMetrics> {
        // CPU metrics
        let cpu_percent = system.global_cpu_info().cpu_usage();
        
        // Memory metrics
        let total_memory = system.total_memory();
        let used_memory = system.used_memory();
        let memory_percent = (used_memory as f32 / total_memory as f32) * 100.0;
        let memory_available_gb = (total_memory - used_memory) as f32 / 1024.0 / 1024.0 / 1024.0;
        
        // GPU metrics (simplified - would use proper GPU monitoring)
        let (gpu_utilization, gpu_memory_percent, gpu_memory_allocated_gb, gpu_temperature) = 
            Self::get_gpu_metrics().await;
        
        // Disk metrics
        let disk_usage_percent = Self::get_disk_usage_percent();
        
        // Network metrics
        let network_io = Self::get_network_metrics(system);
        
        // Active connections
        let active_connections = 0; // Would be populated from actual connection count
        
        Ok(SystemMetrics {
            timestamp: chrono::Utc::now(),
            cpu_percent,
            memory_percent,
            memory_available_gb,
            gpu_utilization,
            gpu_memory_percent,
            gpu_memory_allocated_gb,
            gpu_temperature,
            disk_usage_percent,
            network_io,
            active_connections,
        })
    }
    
    async fn get_gpu_metrics() -> (Option<f32>, Option<f32>, Option<f32>, Option<f32>) {
        // This would integrate with CUDA/NVML for real GPU metrics
        // For now, return mock data
        (
            Some(75.0),  // GPU utilization
            Some(60.0),  // GPU memory percent
            Some(14.4),  // GPU memory allocated GB (60% of 24GB)
            Some(68.0),  // GPU temperature
        )
    }
    
    fn get_disk_usage_percent() -> f32 {
        // Simplified disk usage - would scan actual filesystem
        65.0
    }
    
    fn get_network_metrics(system: &System) -> Option<NetworkIO> {
        // Network I/O metrics
        let networks = system.networks();
        let mut total_bytes_sent = 0;
        let mut total_bytes_recv = 0;
        let mut total_packets_sent = 0;
        let mut total_packets_recv = 0;
        
        for (_, network) in networks {
            total_bytes_sent += network.total_transmitted();
            total_bytes_recv += network.total_received();
            total_packets_sent += network.total_packets_transmitted();
            total_packets_recv += network.total_packets_received();
        }
        
        Some(NetworkIO {
            bytes_sent: total_bytes_sent,
            bytes_recv: total_bytes_recv,
            packets_sent: total_packets_sent,
            packets_recv: total_packets_recv,
        })
    }
    
    pub async fn send_training_update(&self, update: TrainingUpdate) -> Result<()> {
        self.training_updates.send(update)
            .map_err(|e| ApiError::InternalError(format!("Failed to send training update: {}", e)))?;
        Ok(())
    }
    
    pub async fn get_connection_stats(&self) -> ConnectionStats {
        let connections = self.active_connections.read().await;
        
        let total_connections = connections.len();
        let training_connections = connections.values()
            .filter(|conn| conn.job_id.is_some())
            .count();
        
        let oldest_connection = connections.values()
            .map(|conn| conn.connected_at)
            .min();
        
        ConnectionStats {
            total_connections: total_connections as u32,
            training_connections: training_connections as u32,
            monitoring_connections: (total_connections - training_connections) as u32,
            oldest_connection_age_seconds: oldest_connection.map(|oldest| {
                chrono::Utc::now().signed_duration_since(oldest).num_seconds() as u64
            }),
        }
    }
    
    pub async fn broadcast_system_alert(&self, alert: SystemAlert) -> Result<()> {
        let alert_message = json!({
            "type": "system_alert",
            "data": alert,
            "timestamp": chrono::Utc::now()
        });
        
        let connections = self.active_connections.read().await;
        let message_text = alert_message.to_string();
        
        // In a real implementation, you'd send this to all active WebSocket connections
        // For now, we'll just log it
        match alert.severity.as_str() {
            "critical" => error!("System Alert [CRITICAL]: {}", alert.message),
            "warning" => warn!("System Alert [WARNING]: {}", alert.message),
            "info" => info!("System Alert [INFO]: {}", alert.message),
            _ => debug!("System Alert: {}", alert.message),
        }
        
        Ok(())
    }
    
    pub async fn start_performance_monitoring(&self) -> Result<()> {
        let metrics_broadcaster = self.metrics_broadcaster.clone();
        let threshold_config = PerformanceThresholds::default();
        
        tokio::spawn(async move {
            let mut interval = interval(Duration::from_secs(5));
            let mut last_metrics: Option<SystemMetrics> = None;
            
            loop {
                interval.tick().await;
                
                // Get current metrics (this would come from the main monitoring loop)
                // For now, we'll create sample metrics
                if let Ok(mut receiver) = metrics_broadcaster.subscribe() {
                    if let Ok(current_metrics) = receiver.recv().await {
                        // Check for performance issues
                        Self::check_performance_thresholds(&current_metrics, &threshold_config, &last_metrics).await;
                        last_metrics = Some(current_metrics);
                    }
                }
            }
        });
        
        info!("Performance monitoring started");
        Ok(())
    }
    
    async fn check_performance_thresholds(
        current: &SystemMetrics,
        thresholds: &PerformanceThresholds,
        previous: &Option<SystemMetrics>,
    ) {
        // Check CPU usage
        if current.cpu_percent > thresholds.cpu_critical_percent {
            warn!("Critical CPU usage: {:.1}%", current.cpu_percent);
        }
        
        // Check memory usage
        if current.memory_percent > thresholds.memory_critical_percent {
            warn!("Critical memory usage: {:.1}%", current.memory_percent);
        }
        
        // Check GPU usage
        if let Some(gpu_util) = current.gpu_utilization {
            if gpu_util > thresholds.gpu_critical_percent {
                warn!("Critical GPU utilization: {:.1}%", gpu_util);
            }
        }
        
        // Check GPU memory
        if let Some(gpu_mem) = current.gpu_memory_percent {
            if gpu_mem > thresholds.gpu_memory_critical_percent {
                warn!("Critical GPU memory usage: {:.1}%", gpu_mem);
            }
        }
        
        // Check GPU temperature
        if let Some(gpu_temp) = current.gpu_temperature {
            if gpu_temp > thresholds.gpu_temperature_critical {
                error!("Critical GPU temperature: {:.1}°C", gpu_temp);
            }
        }
        
        // Check for performance degradation
        if let Some(prev) = previous {
            if let (Some(current_gpu), Some(prev_gpu)) = (current.gpu_utilization, prev.gpu_utilization) {
                if current_gpu < prev_gpu * 0.5 && prev_gpu > 50.0 {
                    warn!("GPU utilization dropped significantly: {:.1}% -> {:.1}%", prev_gpu, current_gpu);
                }
            }
        }
    }
    
    pub fn subscribe_to_metrics(&self) -> broadcast::Receiver<SystemMetrics> {
        self.metrics_broadcaster.subscribe()
    }
    
    pub fn subscribe_to_training_updates(&self) -> broadcast::Receiver<TrainingUpdate> {
        self.training_updates.subscribe()
    }
}

#[derive(Debug, Clone)]
pub struct ConnectionStats {
    pub total_connections: u32,
    pub training_connections: u32,
    pub monitoring_connections: u32,
    pub oldest_connection_age_seconds: Option<u64>,
}

#[derive(Debug, Clone)]
pub struct SystemAlert {
    pub id: Uuid,
    pub severity: String, // "critical", "warning", "info"
    pub message: String,
    pub component: String, // "gpu", "cpu", "memory", "disk", "network"
    pub threshold_value: Option<f32>,
    pub current_value: Option<f32>,
    pub recommendation: Option<String>,
}

#[derive(Debug, Clone)]
pub struct PerformanceThresholds {
    pub cpu_warning_percent: f32,
    pub cpu_critical_percent: f32,
    pub memory_warning_percent: f32,
    pub memory_critical_percent: f32,
    pub gpu_warning_percent: f32,
    pub gpu_critical_percent: f32,
    pub gpu_memory_warning_percent: f32,
    pub gpu_memory_critical_percent: f32,
    pub gpu_temperature_warning: f32,
    pub gpu_temperature_critical: f32,
    pub disk_warning_percent: f32,
    pub disk_critical_percent: f32,
}

impl Default for PerformanceThresholds {
    fn default() -> Self {
        Self {
            cpu_warning_percent: 80.0,
            cpu_critical_percent: 95.0,
            memory_warning_percent: 85.0,
            memory_critical_percent: 95.0,
            gpu_warning_percent: 90.0,
            gpu_critical_percent: 98.0,
            gpu_memory_warning_percent: 90.0,
            gpu_memory_critical_percent: 95.0,
            gpu_temperature_warning: 80.0,
            gpu_temperature_critical: 90.0,
            disk_warning_percent: 85.0,
            disk_critical_percent: 95.0,
        }
    }
}

// WebSocket message types for client communication
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct WebSocketMessage {
    pub message_type: String,
    pub data: serde_json::Value,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct ClientSubscription {
    pub metrics: bool,
    pub training_updates: bool,
    pub system_alerts: bool,
    pub specific_job_id: Option<Uuid>,
}

impl MonitoringService {
    pub async fn handle_system_monitoring_websocket(&self, mut socket: WebSocket) {
        let connection_id = Uuid::new_v4();
        
        info!("New system monitoring WebSocket connection: {}", connection_id);
        
        // Register connection
        {
            let mut connections = self.active_connections.write().await;
            connections.insert(connection_id, WebSocketConnection {
                connection_id,
                job_id: None, // This is a general monitoring connection
                connected_at: chrono::Utc::now(),
                last_ping: chrono::Utc::now(),
            });
        }
        
        let mut metrics_receiver = self.metrics_broadcaster.subscribe();
        let mut training_receiver = self.training_updates.subscribe();
        
        // Send initial connection acknowledgment
        let ack_message = json!({
            "type": "connection_established",
            "connection_id": connection_id,
            "timestamp": chrono::Utc::now(),
            "available_subscriptions": ["metrics", "training_updates", "system_alerts"]
        });
        
        if socket.send(Message::Text(ack_message.to_string())).await.is_err() {
            return;
        }
        
        let mut subscription = ClientSubscription {
            metrics: true,
            training_updates: false,
            system_alerts: true,
            specific_job_id: None,
        };
        
        loop {
            tokio::select! {
                // Handle incoming WebSocket messages
                msg = socket.recv() => {
                    match msg {
                        Some(Ok(Message::Text(text))) => {
                            if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(&text) {
                                if let Some("subscribe") = parsed.get("type").and_then(|t| t.as_str()) {
                                    if let Ok(new_subscription) = serde_json::from_value::<ClientSubscription>(parsed["data"].clone()) {
                                        subscription = new_subscription;
                                        debug!("Updated subscription for connection {}: {:?}", connection_id, subscription);
                                    }
                                }
                            }
                        }
                        Some(Ok(Message::Ping(data))) => {
                            if socket.send(Message::Pong(data)).await.is_err() {
                                break;
                            }
                        }
                        Some(Ok(Message::Close(_))) => break,
                        Some(Err(_)) => break,
                        None => break,
                        _ => {}
                    }
                }
                
                // Send system metrics if subscribed
                metrics = metrics_receiver.recv(), if subscription.metrics => {
                    if let Ok(metrics) = metrics {
                        let message = json!({
                            "type": "system_metrics",
                            "data": metrics,
                            "timestamp": chrono::Utc::now()
                        });
                        
                        if socket.send(Message::Text(message.to_string())).await.is_err() {
                            break;
                        }
                    }
                }
                
                // Send training updates if subscribed
                update = training_receiver.recv(), if subscription.training_updates => {
                    if let Ok(update) = update {
                        // Filter by specific job if requested
                        if subscription.specific_job_id.is_none() || subscription.specific_job_id == Some(update.job_id) {
                            let message = json!({
                                "type": "training_update",
                                "data": update,
                                "timestamp": chrono::Utc::now()
                            });
                            
                            if socket.send(Message::Text(message.to_string())).await.is_err() {
                                break;
                            }
                        }
                    }
                }
                
                // Send periodic heartbeat
                _ = tokio::time::sleep(Duration::from_secs(30)) => {
                    let heartbeat = json!({
                        "type": "heartbeat",
                        "timestamp": chrono::Utc::now(),
                        "connection_id": connection_id
                    });
                    
                    if socket.send(Message::Text(heartbeat.to_string())).await.is_err() {
                        break;
                    }
                }
            }
        }
        
        // Clean up connection
        {
            let mut connections = self.active_connections.write().await;
            connections.remove(&connection_id);
        }
        
        info!("System monitoring WebSocket connection {} closed", connection_id);
    }
}