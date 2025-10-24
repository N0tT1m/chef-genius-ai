use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde::{Deserialize, Serialize};
use std::fmt;
use thiserror::Error;

pub type Result<T> = std::result::Result<T, ApiError>;

#[derive(Debug, Error)]
pub enum ApiError {
    #[error("Model not found: {0}")]
    ModelNotFound(String),

    #[error("Model loading error: {0}")]
    ModelLoadError(String),

    #[error("Inference error: {0}")]
    InferenceError(String),

    #[error("Tokenization error: {0}")]
    TokenizationError(String),

    #[error("Training error: {0}")]
    TrainingError(String),

    #[error("Conversion error: {0}")]
    ConversionError(String),

    #[error("Database error: {0}")]
    DatabaseError(#[from] sqlx::Error),

    #[error("Redis error: {0}")]
    RedisError(#[from] redis::RedisError),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),

    #[error("Configuration error: {0}")]
    ConfigError(String),

    #[error("Authentication error: {0}")]
    AuthError(String),

    #[error("Rate limit exceeded")]
    RateLimitExceeded,

    #[error("Invalid request: {0}")]
    InvalidRequest(String),

    #[error("Resource not found: {0}")]
    ResourceNotFound(String),

    #[error("Internal server error: {0}")]
    InternalError(String),

    #[error("Service unavailable: {0}")]
    ServiceUnavailable(String),

    #[error("Timeout error: {0}")]
    TimeoutError(String),

    #[error("Hardware error: {0}")]
    HardwareError(String),

    #[error("CUDA error: {0}")]
    CudaError(String),

    #[error("TensorRT error: {0}")]
    TensorRTError(String),

    #[error("Unknown error: {0}")]
    Unknown(String),
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ErrorResponse {
    pub error: String,
    pub code: u16,
    pub message: String,
    pub details: Option<serde_json::Value>,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub trace_id: Option<String>,
}

impl ApiError {
    pub fn status_code(&self) -> StatusCode {
        match self {
            ApiError::ModelNotFound(_) | ApiError::ResourceNotFound(_) => StatusCode::NOT_FOUND,
            ApiError::InvalidRequest(_) => StatusCode::BAD_REQUEST,
            ApiError::AuthError(_) => StatusCode::UNAUTHORIZED,
            ApiError::RateLimitExceeded => StatusCode::TOO_MANY_REQUESTS,
            ApiError::ServiceUnavailable(_) => StatusCode::SERVICE_UNAVAILABLE,
            ApiError::TimeoutError(_) => StatusCode::REQUEST_TIMEOUT,
            ApiError::DatabaseError(_) 
            | ApiError::RedisError(_) 
            | ApiError::IoError(_) 
            | ApiError::SerializationError(_)
            | ApiError::ConfigError(_)
            | ApiError::ModelLoadError(_)
            | ApiError::InferenceError(_)
            | ApiError::TokenizationError(_)
            | ApiError::TrainingError(_)
            | ApiError::ConversionError(_)
            | ApiError::InternalError(_)
            | ApiError::HardwareError(_)
            | ApiError::CudaError(_)
            | ApiError::TensorRTError(_)
            | ApiError::Unknown(_) => StatusCode::INTERNAL_SERVER_ERROR,
        }
    }

    pub fn error_code(&self) -> &'static str {
        match self {
            ApiError::ModelNotFound(_) => "MODEL_NOT_FOUND",
            ApiError::ModelLoadError(_) => "MODEL_LOAD_ERROR",
            ApiError::InferenceError(_) => "INFERENCE_ERROR",
            ApiError::TokenizationError(_) => "TOKENIZATION_ERROR",
            ApiError::TrainingError(_) => "TRAINING_ERROR",
            ApiError::ConversionError(_) => "CONVERSION_ERROR",
            ApiError::DatabaseError(_) => "DATABASE_ERROR",
            ApiError::RedisError(_) => "REDIS_ERROR",
            ApiError::IoError(_) => "IO_ERROR",
            ApiError::SerializationError(_) => "SERIALIZATION_ERROR",
            ApiError::ConfigError(_) => "CONFIG_ERROR",
            ApiError::AuthError(_) => "AUTH_ERROR",
            ApiError::RateLimitExceeded => "RATE_LIMIT_EXCEEDED",
            ApiError::InvalidRequest(_) => "INVALID_REQUEST",
            ApiError::ResourceNotFound(_) => "RESOURCE_NOT_FOUND",
            ApiError::InternalError(_) => "INTERNAL_ERROR",
            ApiError::ServiceUnavailable(_) => "SERVICE_UNAVAILABLE",
            ApiError::TimeoutError(_) => "TIMEOUT_ERROR",
            ApiError::HardwareError(_) => "HARDWARE_ERROR",
            ApiError::CudaError(_) => "CUDA_ERROR",
            ApiError::TensorRTError(_) => "TENSORRT_ERROR",
            ApiError::Unknown(_) => "UNKNOWN_ERROR",
        }
    }

    pub fn with_details(mut self, details: serde_json::Value) -> Self {
        // In a real implementation, you might want to store details in the error
        // For now, we'll just return the error as-is
        self
    }

    pub fn from_anyhow(err: anyhow::Error) -> Self {
        ApiError::InternalError(err.to_string())
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let status = self.status_code();
        let error_response = ErrorResponse {
            error: self.error_code().to_string(),
            code: status.as_u16(),
            message: self.to_string(),
            details: None,
            timestamp: chrono::Utc::now(),
            trace_id: None, // Would be populated by tracing middleware
        };

        tracing::error!(
            error = %self,
            status_code = %status,
            error_code = self.error_code(),
            "API error occurred"
        );

        (status, Json(error_response)).into_response()
    }
}

impl From<anyhow::Error> for ApiError {
    fn from(err: anyhow::Error) -> Self {
        ApiError::InternalError(err.to_string())
    }
}

impl From<tokio::time::error::Elapsed> for ApiError {
    fn from(err: tokio::time::error::Elapsed) -> Self {
        ApiError::TimeoutError(err.to_string())
    }
}

impl From<uuid::Error> for ApiError {
    fn from(err: uuid::Error) -> Self {
        ApiError::InvalidRequest(format!("Invalid UUID: {}", err))
    }
}

// Helper macros for common error patterns
#[macro_export]
macro_rules! model_not_found {
    ($id:expr) => {
        ApiError::ModelNotFound(format!("Model with ID {} not found", $id))
    };
}

#[macro_export]
macro_rules! invalid_request {
    ($msg:expr) => {
        ApiError::InvalidRequest($msg.to_string())
    };
    ($fmt:expr, $($arg:tt)*) => {
        ApiError::InvalidRequest(format!($fmt, $($arg)*))
    };
}

#[macro_export]
macro_rules! internal_error {
    ($msg:expr) => {
        ApiError::InternalError($msg.to_string())
    };
    ($fmt:expr, $($arg:tt)*) => {
        ApiError::InternalError(format!($fmt, $($arg)*))
    };
}

// Custom error types for specific domains

#[derive(Debug, Error)]
pub enum ModelError {
    #[error("Model file not found: {path}")]
    FileNotFound { path: String },

    #[error("Invalid model format: expected {expected}, got {actual}")]
    InvalidFormat { expected: String, actual: String },

    #[error("Model loading timeout after {seconds} seconds")]
    LoadTimeout { seconds: u64 },

    #[error("Insufficient GPU memory: required {required_mb}MB, available {available_mb}MB")]
    InsufficientMemory { required_mb: u64, available_mb: u64 },

    #[error("Model version mismatch: API expects {api_version}, model is {model_version}")]
    VersionMismatch { api_version: String, model_version: String },
}

#[derive(Debug, Error)]
pub enum InferenceError {
    #[error("Input validation failed: {reason}")]
    InvalidInput { reason: String },

    #[error("Sequence too long: {length} exceeds maximum {max_length}")]
    SequenceTooLong { length: usize, max_length: usize },

    #[error("Batch size {batch_size} exceeds maximum {max_batch_size}")]
    BatchSizeExceeded { batch_size: usize, max_batch_size: usize },

    #[error("Generation failed after {attempts} attempts")]
    GenerationFailed { attempts: u32 },

    #[error("Model not ready: {status}")]
    ModelNotReady { status: String },
}

#[derive(Debug, Error)]
pub enum TrainingError {
    #[error("Training job {job_id} not found")]
    JobNotFound { job_id: uuid::Uuid },

    #[error("Training job {job_id} already running")]
    JobAlreadyRunning { job_id: uuid::Uuid },

    #[error("Dataset not found: {path}")]
    DatasetNotFound { path: String },

    #[error("Insufficient disk space: required {required_gb}GB, available {available_gb}GB")]
    InsufficientDiskSpace { required_gb: u64, available_gb: u64 },

    #[error("Training failed at epoch {epoch}: {reason}")]
    TrainingFailed { epoch: u32, reason: String },

    #[error("Checkpoint corrupted: {path}")]
    CorruptedCheckpoint { path: String },
}

// Convert domain-specific errors to API errors
impl From<ModelError> for ApiError {
    fn from(err: ModelError) -> Self {
        match err {
            ModelError::FileNotFound { .. } => ApiError::ModelNotFound(err.to_string()),
            ModelError::LoadTimeout { .. } => ApiError::TimeoutError(err.to_string()),
            ModelError::InsufficientMemory { .. } => ApiError::HardwareError(err.to_string()),
            _ => ApiError::ModelLoadError(err.to_string()),
        }
    }
}

impl From<InferenceError> for ApiError {
    fn from(err: InferenceError) -> Self {
        match err {
            InferenceError::InvalidInput { .. } => ApiError::InvalidRequest(err.to_string()),
            InferenceError::ModelNotReady { .. } => ApiError::ServiceUnavailable(err.to_string()),
            _ => ApiError::InferenceError(err.to_string()),
        }
    }
}

impl From<TrainingError> for ApiError {
    fn from(err: TrainingError) -> Self {
        match err {
            TrainingError::JobNotFound { .. } => ApiError::ResourceNotFound(err.to_string()),
            TrainingError::DatasetNotFound { .. } => ApiError::ResourceNotFound(err.to_string()),
            TrainingError::InsufficientDiskSpace { .. } => ApiError::HardwareError(err.to_string()),
            _ => ApiError::TrainingError(err.to_string()),
        }
    }
}

// Result type aliases for convenience
pub type ModelResult<T> = std::result::Result<T, ModelError>;
pub type InferenceResult<T> = std::result::Result<T, InferenceError>;
pub type TrainingResult<T> = std::result::Result<T, TrainingError>;

// Error context helpers
pub trait ErrorContext<T> {
    fn with_context(self, context: &str) -> Result<T>;
    fn with_model_context(self, model_id: uuid::Uuid) -> Result<T>;
    fn with_training_context(self, job_id: uuid::Uuid) -> Result<T>;
}

impl<T, E> ErrorContext<T> for std::result::Result<T, E>
where
    E: Into<ApiError>,
{
    fn with_context(self, context: &str) -> Result<T> {
        self.map_err(|e| {
            let api_error = e.into();
            ApiError::InternalError(format!("{}: {}", context, api_error))
        })
    }

    fn with_model_context(self, model_id: uuid::Uuid) -> Result<T> {
        self.with_context(&format!("Model {}", model_id))
    }

    fn with_training_context(self, job_id: uuid::Uuid) -> Result<T> {
        self.with_context(&format!("Training job {}", job_id))
    }
}