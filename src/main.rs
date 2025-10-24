use axum::{
    extract::{Path, Query, State, WebSocketUpgrade},
    http::StatusCode,
    response::{Json, Response},
    routing::{delete, get, post},
    Router,
};
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, sync::Arc};
use tokio::net::TcpListener;
use tower_http::{cors::CorsLayer, trace::TraceLayer};
use tracing::{info, warn};
use uuid::Uuid;

mod config;
mod models;
mod inference;
mod training;
mod monitoring;
mod tensorrt;
mod error;

use config::Config;
use models::*;
use inference::InferenceEngine;
use training::TrainingManager;
use monitoring::MonitoringService;
use error::{ApiError, Result};

#[derive(Clone)]
pub struct AppState {
    pub inference_engine: Arc<InferenceEngine>,
    pub training_manager: Arc<TrainingManager>,
    pub monitoring: Arc<MonitoringService>,
    pub config: Arc<Config>,
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter("chef_genius_api=debug,tower_http=debug")
        .json()
        .init();

    info!("Starting Chef Genius AI API Server with TensorRT acceleration");

    // Load configuration
    let config = Arc::new(Config::from_env()?);
    
    // Initialize CUDA and TensorRT
    tensorrt::init_cuda()?;
    
    // Initialize services
    let inference_engine = Arc::new(InferenceEngine::new(&config).await?);
    let training_manager = Arc::new(TrainingManager::new(&config).await?);
    let monitoring = Arc::new(MonitoringService::new(&config).await?);

    let app_state = AppState {
        inference_engine,
        training_manager,
        monitoring,
        config: config.clone(),
    };

    // Build the application routes
    let app = Router::new()
        // Model Inference
        .route("/api/v1/inference/generate-recipe", post(generate_recipe))
        .route("/api/v1/inference/batch", post(batch_inference))
        
        // Model Management
        .route("/api/v1/models", get(list_models))
        .route("/api/v1/models/:model_id", get(get_model))
        .route("/api/v1/models/:model_id", delete(delete_model))
        
        // Training Management
        .route("/api/v1/training/start", post(start_training))
        .route("/api/v1/training/:job_id/status", get(get_training_status))
        .route("/api/v1/training/:job_id/cancel", post(cancel_training))
        
        // Model Conversion
        .route("/api/v1/conversion/pytorch-to-tensorrt", post(convert_pytorch_to_tensorrt))
        .route("/api/v1/conversion/onnx-to-tensorrt", post(convert_onnx_to_tensorrt))
        
        // Real-time Monitoring
        .route("/api/v1/training/:job_id/stream", get(training_stream))
        .route("/api/v1/system/metrics", get(system_metrics))
        .route("/api/v1/monitoring/stream", get(system_monitoring_stream))
        
        // Health Check
        .route("/health", get(health_check))
        .route("/metrics", get(prometheus_metrics))
        
        .layer(CorsLayer::permissive())
        .layer(TraceLayer::new_for_http())
        .with_state(app_state);

    let listener = TcpListener::bind(&format!("0.0.0.0:{}", config.server.port)).await?;
    info!("Server listening on port {}", config.server.port);
    
    axum::serve(listener, app).await?;
    
    Ok(())
}

// Inference Endpoints
async fn generate_recipe(
    State(state): State<AppState>,
    Json(request): Json<InferenceRequest>,
) -> Result<Json<InferenceResponse>> {
    let response = state.inference_engine.generate_recipe(request).await?;
    Ok(Json(response))
}

async fn batch_inference(
    State(state): State<AppState>,
    Json(requests): Json<Vec<InferenceRequest>>,
) -> Result<Json<Vec<InferenceResponse>>> {
    let responses = state.inference_engine.batch_inference(requests).await?;
    Ok(Json(responses))
}

// Model Management Endpoints
async fn list_models(State(state): State<AppState>) -> Result<Json<Vec<ModelInfo>>> {
    let models = state.training_manager.list_models().await?;
    Ok(Json(models))
}

async fn get_model(
    State(state): State<AppState>,
    Path(model_id): Path<Uuid>,
) -> Result<Json<ModelInfo>> {
    let model = state.training_manager.get_model(model_id).await?;
    Ok(Json(model))
}

async fn delete_model(
    State(state): State<AppState>,
    Path(model_id): Path<Uuid>,
) -> Result<StatusCode> {
    state.training_manager.delete_model(model_id).await?;
    Ok(StatusCode::NO_CONTENT)
}

// Training Endpoints
async fn start_training(
    State(state): State<AppState>,
    Json(request): Json<TrainingRequest>,
) -> Result<Json<TrainingResponse>> {
    let response = state.training_manager.start_training(request).await?;
    Ok(Json(response))
}

async fn get_training_status(
    State(state): State<AppState>,
    Path(job_id): Path<Uuid>,
) -> Result<Json<TrainingStatus>> {
    let status = state.training_manager.get_training_status(job_id).await?;
    Ok(Json(status))
}

async fn cancel_training(
    State(state): State<AppState>,
    Path(job_id): Path<Uuid>,
) -> Result<StatusCode> {
    state.training_manager.cancel_training(job_id).await?;
    Ok(StatusCode::OK)
}

// Conversion Endpoints
async fn convert_pytorch_to_tensorrt(
    State(state): State<AppState>,
    Json(request): Json<ConversionRequest>,
) -> Result<Json<ConversionResponse>> {
    let response = state.training_manager.convert_pytorch_to_tensorrt(request).await?;
    Ok(Json(response))
}

async fn convert_onnx_to_tensorrt(
    State(state): State<AppState>,
    Json(request): Json<ConversionRequest>,
) -> Result<Json<ConversionResponse>> {
    let response = state.training_manager.convert_onnx_to_tensorrt(request).await?;
    Ok(Json(response))
}

// WebSocket for real-time training updates
async fn training_stream(
    ws: WebSocketUpgrade,
    State(state): State<AppState>,
    Path(job_id): Path<Uuid>,
) -> Response {
    ws.on_upgrade(move |socket| state.monitoring.handle_training_stream(socket, job_id))
}

// WebSocket for general system monitoring
async fn system_monitoring_stream(
    ws: WebSocketUpgrade,
    State(state): State<AppState>,
) -> Response {
    ws.on_upgrade(move |socket| state.monitoring.handle_system_monitoring_websocket(socket))
}

// System metrics endpoint
async fn system_metrics(State(state): State<AppState>) -> Result<Json<SystemMetrics>> {
    let metrics = state.monitoring.get_system_metrics().await?;
    Ok(Json(metrics))
}

// Health check
async fn health_check() -> Result<Json<serde_json::Value>> {
    Ok(Json(serde_json::json!({
        "status": "healthy",
        "timestamp": chrono::Utc::now(),
        "version": env!("CARGO_PKG_VERSION")
    })))
}

// Prometheus metrics
async fn prometheus_metrics() -> String {
    // Return Prometheus-formatted metrics
    String::from("# HELP chef_genius_requests_total Total requests\n# TYPE chef_genius_requests_total counter\nchef_genius_requests_total 42\n")
}