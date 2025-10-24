# Chef Genius AI Model API Specification

## Overview

This document describes the API specification for the Chef Genius AI model pipeline, which processes recipe generation models from PyTorch training through ONNX conversion to production PyTorch deployment.

## Model Architecture

- **Model Type**: Sequence-to-Sequence Transformer (AutoModelForSeq2SeqLM)
- **Task**: Recipe Generation from Ingredients
- **Framework**: PyTorch -> ONNX -> Production PyTorch
- **Hardware Optimization**: Ryzen 3900X CPU + RTX 4090 GPU

## Go API Endpoints

### 1. Model Training Service

```go
type TrainingRequest struct {
    PretrainedModel  string            `json:"pretrained_model" binding:"required"`
    Epochs          int               `json:"epochs" default:"3"`
    BatchSize       int               `json:"batch_size" default:"8"`
    OutputDir       string            `json:"output_dir" binding:"required"`
    AlertPhone      string            `json:"alert_phone,omitempty"`
    DiscordWebhook  string            `json:"discord_webhook,omitempty"`
    WandbProject    string            `json:"wandb_project" default:"chef-genius-optimized"`
    UseWandb        bool              `json:"use_wandb" default:"true"`
}

type TrainingResponse struct {
    JobID           string            `json:"job_id"`
    Status          string            `json:"status"`
    Message         string            `json:"message"`
    ModelPath       string            `json:"model_path,omitempty"`
}
```

**POST /api/v1/training/start**
- Starts model training with optimized pipeline
- Returns job ID for tracking

### 2. Model Conversion Service

```go
type ConversionRequest struct {
    SourceModelPath string            `json:"source_model_path" binding:"required"`
    OutputFormat    string            `json:"output_format" binding:"required"` // "onnx" or "torch"
    OptimizationLevel string          `json:"optimization_level" default:"O2"`
    BatchSize       int               `json:"batch_size" default:"1"`
}

type ConversionResponse struct {
    JobID           string            `json:"job_id"`
    Status          string            `json:"status"`
    ConvertedPath   string            `json:"converted_path,omitempty"`
    FileSize        int64             `json:"file_size,omitempty"`
    Metadata        ModelMetadata     `json:"metadata,omitempty"`
}

type ModelMetadata struct {
    InputShape      []int             `json:"input_shape"`
    OutputShape     []int             `json:"output_shape"`
    Parameters      int64             `json:"parameters"`
    ModelSize       string            `json:"model_size"`
    Precision       string            `json:"precision"`
}
```

**POST /api/v1/conversion/torch-to-onnx**
- Converts PyTorch model to ONNX format

**POST /api/v1/conversion/onnx-to-torch**
- Converts ONNX model back to PyTorch for production

### 3. Model Inference Service

```go
type InferenceRequest struct {
    ModelPath       string            `json:"model_path" binding:"required"`
    Ingredients     []string          `json:"ingredients" binding:"required"`
    MaxLength       int               `json:"max_length" default:"512"`
    Temperature     float32           `json:"temperature" default:"0.8"`
    TopK            int               `json:"top_k" default:"50"`
    TopP            float32           `json:"top_p" default:"0.9"`
}

type InferenceResponse struct {
    Recipe          Recipe            `json:"recipe"`
    Confidence      float32           `json:"confidence"`
    GenerationTime  int64             `json:"generation_time_ms"`
    ModelVersion    string            `json:"model_version"`
}

type Recipe struct {
    Title           string            `json:"title"`
    Ingredients     []string          `json:"ingredients"`
    Instructions    []string          `json:"instructions"`
    CookingTime     string            `json:"cooking_time,omitempty"`
    Servings        int               `json:"servings,omitempty"`
    Difficulty      string            `json:"difficulty,omitempty"`
}
```

**POST /api/v1/inference/generate-recipe**
- Generates recipe from ingredients using trained model

### 4. Model Management Service

```go
type ModelInfo struct {
    ModelID         string            `json:"model_id"`
    Name            string            `json:"name"`
    Version         string            `json:"version"`
    Format          string            `json:"format"` // "pytorch", "onnx"
    Status          string            `json:"status"` // "training", "ready", "converting", "error"
    CreatedAt       time.Time         `json:"created_at"`
    UpdatedAt       time.Time         `json:"updated_at"`
    Metrics         TrainingMetrics   `json:"metrics,omitempty"`
    FilePath        string            `json:"file_path"`
    FileSize        int64             `json:"file_size"`
}

type TrainingMetrics struct {
    FinalLoss       float32           `json:"final_loss"`
    TotalEpochs     int               `json:"total_epochs"`
    TotalHours      float32           `json:"total_hours"`
    TotalSteps      int               `json:"total_steps"`
    GPUUtilization  float32           `json:"gpu_utilization"`
    MemoryUsage     float32           `json:"memory_usage"`
}
```

**GET /api/v1/models**
- Lists all available models

**GET /api/v1/models/{model_id}**
- Gets specific model information

**DELETE /api/v1/models/{model_id}**
- Deletes a model

### 5. Training Monitoring Service

```go
type TrainingStatus struct {
    JobID           string            `json:"job_id"`
    Status          string            `json:"status"`
    Progress        float32           `json:"progress"`
    CurrentEpoch    int               `json:"current_epoch"`
    TotalEpochs     int               `json:"total_epochs"`
    CurrentLoss     float32           `json:"current_loss"`
    LearningRate    float32           `json:"learning_rate"`
    SamplesPerSec   float32           `json:"samples_per_sec"`
    TimeRemaining   int64             `json:"time_remaining_seconds"`
    SystemMetrics   SystemMetrics     `json:"system_metrics"`
}

type SystemMetrics struct {
    CPUPercent      float32           `json:"cpu_percent"`
    MemoryPercent   float32           `json:"memory_percent"`
    GPUUtilization  float32           `json:"gpu_utilization"`
    GPUMemoryPercent float32          `json:"gpu_memory_percent"`
    GPUTemperature  float32           `json:"gpu_temperature"`
}
```

**GET /api/v1/training/{job_id}/status**
- Gets training job status and metrics

**WebSocket /api/v1/training/{job_id}/stream**
- Real-time training progress updates

## Model Pipeline Flow

### 1. Training Phase (PyTorch)
```
Input: Raw Recipe Datasets -> 
Data Processing -> 
Unified DataLoader -> 
Optimized Training (Ryzen 3900X + RTX 4090) -> 
Trained PyTorch Model
```

### 2. Conversion Phase (PyTorch -> ONNX)
```
Trained PyTorch Model -> 
ONNX Export -> 
Optimization -> 
ONNX Model (Optimized)
```

### 3. Production Deployment (ONNX -> PyTorch)
```
ONNX Model -> 
PyTorch Loading -> 
Production Inference Service
```

## Configuration

### Hardware Optimization Settings
```go
type HardwareConfig struct {
    CPUThreads              int     `json:"cpu_threads" default:"24"`
    BatchSize               int     `json:"batch_size" default:"8"`
    GradientAccumulationSteps int   `json:"gradient_accumulation_steps" default:"2"`
    UseBFloat16             bool    `json:"use_bfloat16" default:"true"`
    EnableGradientCheckpointing bool `json:"enable_gradient_checkpointing" default:"true"`
    CUDAMemoryConfig        string  `json:"cuda_memory_config"`
}
```

### Monitoring Configuration
```go
type MonitoringConfig struct {
    EnableWandB             bool    `json:"enable_wandb" default:"true"`
    WandBProject            string  `json:"wandb_project"`
    DiscordWebhook          string  `json:"discord_webhook,omitempty"`
    AlertPhone              string  `json:"alert_phone,omitempty"`
    SystemMonitoringInterval int    `json:"system_monitoring_interval" default:"10"`
}
```

## Error Handling

### Error Response Format
```go
type ErrorResponse struct {
    Error           string            `json:"error"`
    Code            int               `json:"code"`
    Message         string            `json:"message"`
    Details         map[string]interface{} `json:"details,omitempty"`
    Timestamp       time.Time         `json:"timestamp"`
}
```

### Common Error Codes
- `400`: Invalid request parameters
- `404`: Model not found
- `409`: Training already in progress
- `500`: Internal server error
- `503`: Service temporarily unavailable
- `507`: Insufficient storage space

## Performance Metrics

### Training Performance
- **Target Speed**: 50+ samples/second
- **Memory Usage**: <95% GPU memory
- **CPU Utilization**: 80-90% (Ryzen 3900X)
- **GPU Utilization**: 90-95% (RTX 4090)

### Inference Performance
- **Latency**: <500ms per recipe generation
- **Throughput**: 100+ requests/second
- **Memory**: <2GB RAM per model instance

## Security Considerations

- API key authentication required for all endpoints
- Rate limiting: 100 requests/minute per API key
- Model files stored with encryption at rest
- Training data anonymized and secured
- Audit logging for all model operations

## Dependencies

### Go Packages Required
```go
import (
    "github.com/gin-gonic/gin"
    "github.com/gorilla/websocket"
    "gorm.io/gorm"
    "github.com/redis/go-redis/v9"
    "github.com/prometheus/client_golang/prometheus"
)
```

### External Services
- **W&B**: Training monitoring and logging
- **Discord**: Real-time training notifications
- **SMS Service**: Critical alert notifications
- **Redis**: Job queue and caching
- **PostgreSQL**: Model metadata storage