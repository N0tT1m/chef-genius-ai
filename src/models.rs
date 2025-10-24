use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

// Request/Response Models for API

#[derive(Debug, Serialize, Deserialize)]
pub struct InferenceRequest {
    pub model_path: Option<String>,
    pub model_id: Option<Uuid>,
    pub ingredients: Vec<String>,
    pub max_length: Option<u32>,
    pub temperature: Option<f32>,
    pub top_k: Option<u32>,
    pub top_p: Option<f32>,
    pub num_return_sequences: Option<u32>,
    pub cuisine_style: Option<String>,
    pub dietary_restrictions: Option<Vec<String>>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct InferenceResponse {
    pub recipe: Recipe,
    pub confidence: f32,
    pub generation_time_ms: u64,
    pub model_version: String,
    pub model_id: Uuid,
    pub alternatives: Option<Vec<Recipe>>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Recipe {
    pub title: String,
    pub ingredients: Vec<String>,
    pub instructions: Vec<String>,
    pub cooking_time: Option<String>,
    pub prep_time: Option<String>,
    pub servings: Option<u32>,
    pub difficulty: Option<String>,
    pub cuisine_type: Option<String>,
    pub dietary_tags: Option<Vec<String>>,
    pub nutrition_info: Option<NutritionInfo>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct NutritionInfo {
    pub calories: Option<u32>,
    pub protein_g: Option<f32>,
    pub carbs_g: Option<f32>,
    pub fat_g: Option<f32>,
    pub fiber_g: Option<f32>,
}

// Training Models

#[derive(Debug, Serialize, Deserialize)]
pub struct TrainingRequest {
    pub pretrained_model: String,
    pub epochs: Option<u32>,
    pub batch_size: Option<u32>,
    pub output_dir: String,
    pub dataset_path: Option<String>,
    pub learning_rate: Option<f32>,
    pub weight_decay: Option<f32>,
    pub gradient_accumulation_steps: Option<u32>,
    pub warmup_steps: Option<u32>,
    pub max_grad_norm: Option<f32>,
    pub alert_phone: Option<String>,
    pub discord_webhook: Option<String>,
    pub wandb_project: Option<String>,
    pub use_wandb: Option<bool>,
    pub hardware_config: Option<HardwareConfig>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TrainingResponse {
    pub job_id: Uuid,
    pub status: String,
    pub message: String,
    pub model_path: Option<String>,
    pub estimated_duration_hours: Option<f32>,
    pub dataset_info: Option<DatasetInfo>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct DatasetInfo {
    pub total_samples: u64,
    pub num_datasets: u32,
    pub dataset_names: Vec<String>,
    pub total_size_mb: u64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TrainingStatus {
    pub job_id: Uuid,
    pub status: String, // "pending", "running", "completed", "failed", "cancelled"
    pub progress: f32,
    pub current_epoch: u32,
    pub total_epochs: u32,
    pub current_loss: Option<f32>,
    pub learning_rate: Option<f32>,
    pub samples_per_sec: Option<f32>,
    pub time_remaining_seconds: Option<u64>,
    pub started_at: Option<DateTime<Utc>>,
    pub completed_at: Option<DateTime<Utc>>,
    pub error_message: Option<String>,
    pub system_metrics: Option<SystemMetrics>,
    pub training_metrics: Option<TrainingMetrics>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TrainingMetrics {
    pub final_loss: Option<f32>,
    pub best_loss: Option<f32>,
    pub total_steps: u64,
    pub samples_processed: u64,
    pub gpu_utilization_avg: Option<f32>,
    pub memory_usage_avg: Option<f32>,
    pub throughput_samples_per_sec: Option<f32>,
}

// Model Conversion

#[derive(Debug, Serialize, Deserialize)]
pub struct ConversionRequest {
    pub source_model_path: String,
    pub output_format: String, // "tensorrt", "onnx"
    pub optimization_level: Option<String>, // "O0", "O1", "O2", "O3"
    pub batch_size: Option<u32>,
    pub max_sequence_length: Option<u32>,
    pub precision: Option<String>, // "fp32", "fp16", "int8"
    pub workspace_size_mb: Option<u32>,
    pub dynamic_shapes: Option<bool>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ConversionResponse {
    pub job_id: Uuid,
    pub status: String,
    pub converted_path: Option<String>,
    pub file_size: Option<u64>,
    pub metadata: Option<ModelMetadata>,
    pub conversion_time_seconds: Option<u64>,
    pub performance_improvement: Option<f32>, // Speed improvement factor
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ModelMetadata {
    pub input_shape: Vec<i64>,
    pub output_shape: Vec<i64>,
    pub parameters: u64,
    pub model_size_mb: f32,
    pub precision: String,
    pub supported_batch_sizes: Vec<u32>,
    pub max_sequence_length: u32,
    pub optimization_level: String,
}

// Model Management

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ModelInfo {
    pub model_id: Uuid,
    pub name: String,
    pub version: String,
    pub format: String, // "pytorch", "onnx", "tensorrt"
    pub status: String, // "training", "ready", "converting", "error"
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub metrics: Option<TrainingMetrics>,
    pub file_path: String,
    pub file_size: u64,
    pub description: Option<String>,
    pub tags: Option<Vec<String>>,
    pub hardware_requirements: Option<HardwareRequirements>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct HardwareRequirements {
    pub min_gpu_memory_gb: u32,
    pub min_ram_gb: u32,
    pub cuda_compute_capability: String,
    pub recommended_batch_size: u32,
}

// System Monitoring

#[derive(Debug, Serialize, Deserialize)]
pub struct SystemMetrics {
    pub timestamp: DateTime<Utc>,
    pub cpu_percent: f32,
    pub memory_percent: f32,
    pub memory_available_gb: f32,
    pub gpu_utilization: Option<f32>,
    pub gpu_memory_percent: Option<f32>,
    pub gpu_memory_allocated_gb: Option<f32>,
    pub gpu_temperature: Option<f32>,
    pub disk_usage_percent: f32,
    pub network_io: Option<NetworkIO>,
    pub active_connections: u32,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct NetworkIO {
    pub bytes_sent: u64,
    pub bytes_recv: u64,
    pub packets_sent: u64,
    pub packets_recv: u64,
}

// Hardware Configuration

#[derive(Debug, Serialize, Deserialize)]
pub struct HardwareConfig {
    pub cpu_threads: Option<u32>,
    pub batch_size: Option<u32>,
    pub gradient_accumulation_steps: Option<u32>,
    pub use_bfloat16: Option<bool>,
    pub enable_gradient_checkpointing: Option<bool>,
    pub cuda_memory_fraction: Option<f32>,
    pub tensorrt_workspace_size_mb: Option<u32>,
    pub tensorrt_precision: Option<String>, // "fp32", "fp16", "int8"
}

// WebSocket Messages

#[derive(Debug, Serialize, Deserialize)]
pub struct TrainingUpdate {
    pub job_id: Uuid,
    pub update_type: String, // "progress", "error", "completed", "system_metrics"
    pub timestamp: DateTime<Utc>,
    pub data: serde_json::Value,
}

// Batch Processing

#[derive(Debug, Serialize, Deserialize)]
pub struct BatchInferenceRequest {
    pub requests: Vec<InferenceRequest>,
    pub batch_size: Option<u32>,
    pub max_concurrent: Option<u32>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct BatchInferenceResponse {
    pub responses: Vec<InferenceResponse>,
    pub batch_id: Uuid,
    pub total_time_ms: u64,
    pub successful_count: u32,
    pub failed_count: u32,
    pub errors: Option<Vec<String>>,
}

// Pagination

#[derive(Debug, Serialize, Deserialize)]
pub struct PaginationQuery {
    pub page: Option<u32>,
    pub limit: Option<u32>,
    pub sort_by: Option<String>,
    pub sort_order: Option<String>, // "asc", "desc"
}

#[derive(Debug, Serialize, Deserialize)]
pub struct PaginatedResponse<T> {
    pub data: Vec<T>,
    pub total: u64,
    pub page: u32,
    pub limit: u32,
    pub total_pages: u32,
}