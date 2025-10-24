use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::env;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub server: ServerConfig,
    pub database: DatabaseConfig,
    pub redis: RedisConfig,
    pub inference: InferenceConfig,
    pub training: TrainingConfig,
    pub monitoring: MonitoringConfig,
    pub tensorrt: TensorRTConfig,
    pub security: SecurityConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
    pub workers: usize,
    pub max_connections: u32,
    pub request_timeout_seconds: u64,
    pub enable_cors: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatabaseConfig {
    pub url: String,
    pub max_connections: u32,
    pub min_connections: u32,
    pub connection_timeout_seconds: u64,
    pub idle_timeout_seconds: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RedisConfig {
    pub url: String,
    pub max_connections: u32,
    pub connection_timeout_seconds: u64,
    pub command_timeout_seconds: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceConfig {
    pub models_directory: String,
    pub max_concurrent_requests: Option<usize>,
    pub default_batch_size: u32,
    pub max_sequence_length: u32,
    pub cache_size: usize,
    pub warmup_requests: u32,
    pub enable_batching: bool,
    pub batch_timeout_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    pub output_directory: String,
    pub checkpoint_directory: String,
    pub max_concurrent_jobs: u32,
    pub default_epochs: u32,
    pub default_batch_size: u32,
    pub default_learning_rate: f32,
    pub enable_wandb: bool,
    pub wandb_project: String,
    pub enable_discord_notifications: bool,
    pub discord_webhook_url: Option<String>,
    pub enable_sms_alerts: bool,
    pub alert_phone_number: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoringConfig {
    pub enable_prometheus: bool,
    pub prometheus_port: u16,
    pub metrics_interval_seconds: u64,
    pub log_level: String,
    pub enable_tracing: bool,
    pub jaeger_endpoint: Option<String>,
    pub system_metrics_interval_seconds: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorRTConfig {
    pub workspace_size_mb: u32,
    pub max_batch_size: u32,
    pub precision: String, // "fp32", "fp16", "int8"
    pub enable_dynamic_shapes: bool,
    pub optimization_level: String, // "O0", "O1", "O2", "O3"
    pub cache_directory: String,
    pub enable_profiling: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityConfig {
    pub enable_api_keys: bool,
    pub api_key_header: String,
    pub rate_limit_requests_per_minute: u32,
    pub enable_https: bool,
    pub tls_cert_path: Option<String>,
    pub tls_key_path: Option<String>,
    pub enable_cors: bool,
    pub allowed_origins: Vec<String>,
}

impl Config {
    pub fn from_env() -> Result<Self> {
        Ok(Config {
            server: ServerConfig {
                host: env::var("SERVER_HOST").unwrap_or_else(|_| "0.0.0.0".to_string()),
                port: env::var("SERVER_PORT")
                    .unwrap_or_else(|_| "8080".to_string())
                    .parse()
                    .unwrap_or(8080),
                workers: env::var("SERVER_WORKERS")
                    .unwrap_or_else(|_| num_cpus::get().to_string())
                    .parse()
                    .unwrap_or(num_cpus::get()),
                max_connections: env::var("SERVER_MAX_CONNECTIONS")
                    .unwrap_or_else(|_| "1000".to_string())
                    .parse()
                    .unwrap_or(1000),
                request_timeout_seconds: env::var("SERVER_REQUEST_TIMEOUT")
                    .unwrap_or_else(|_| "30".to_string())
                    .parse()
                    .unwrap_or(30),
                enable_cors: env::var("SERVER_ENABLE_CORS")
                    .unwrap_or_else(|_| "true".to_string())
                    .parse()
                    .unwrap_or(true),
            },
            
            database: DatabaseConfig {
                url: env::var("DATABASE_URL")
                    .unwrap_or_else(|_| "postgresql://localhost:5432/chef_genius".to_string()),
                max_connections: env::var("DATABASE_MAX_CONNECTIONS")
                    .unwrap_or_else(|_| "20".to_string())
                    .parse()
                    .unwrap_or(20),
                min_connections: env::var("DATABASE_MIN_CONNECTIONS")
                    .unwrap_or_else(|_| "5".to_string())
                    .parse()
                    .unwrap_or(5),
                connection_timeout_seconds: env::var("DATABASE_CONNECTION_TIMEOUT")
                    .unwrap_or_else(|_| "10".to_string())
                    .parse()
                    .unwrap_or(10),
                idle_timeout_seconds: env::var("DATABASE_IDLE_TIMEOUT")
                    .unwrap_or_else(|_| "600".to_string())
                    .parse()
                    .unwrap_or(600),
            },
            
            redis: RedisConfig {
                url: env::var("REDIS_URL")
                    .unwrap_or_else(|_| "redis://localhost:6379".to_string()),
                max_connections: env::var("REDIS_MAX_CONNECTIONS")
                    .unwrap_or_else(|_| "10".to_string())
                    .parse()
                    .unwrap_or(10),
                connection_timeout_seconds: env::var("REDIS_CONNECTION_TIMEOUT")
                    .unwrap_or_else(|_| "5".to_string())
                    .parse()
                    .unwrap_or(5),
                command_timeout_seconds: env::var("REDIS_COMMAND_TIMEOUT")
                    .unwrap_or_else(|_| "5".to_string())
                    .parse()
                    .unwrap_or(5),
            },
            
            inference: InferenceConfig {
                models_directory: env::var("MODELS_DIRECTORY")
                    .unwrap_or_else(|_| "./models".to_string()),
                max_concurrent_requests: env::var("INFERENCE_MAX_CONCURRENT")
                    .ok()
                    .and_then(|s| s.parse().ok()),
                default_batch_size: env::var("INFERENCE_DEFAULT_BATCH_SIZE")
                    .unwrap_or_else(|_| "1".to_string())
                    .parse()
                    .unwrap_or(1),
                max_sequence_length: env::var("INFERENCE_MAX_SEQUENCE_LENGTH")
                    .unwrap_or_else(|_| "512".to_string())
                    .parse()
                    .unwrap_or(512),
                cache_size: env::var("INFERENCE_CACHE_SIZE")
                    .unwrap_or_else(|_| "1000".to_string())
                    .parse()
                    .unwrap_or(1000),
                warmup_requests: env::var("INFERENCE_WARMUP_REQUESTS")
                    .unwrap_or_else(|_| "5".to_string())
                    .parse()
                    .unwrap_or(5),
                enable_batching: env::var("INFERENCE_ENABLE_BATCHING")
                    .unwrap_or_else(|_| "true".to_string())
                    .parse()
                    .unwrap_or(true),
                batch_timeout_ms: env::var("INFERENCE_BATCH_TIMEOUT_MS")
                    .unwrap_or_else(|_| "100".to_string())
                    .parse()
                    .unwrap_or(100),
            },
            
            training: TrainingConfig {
                output_directory: env::var("TRAINING_OUTPUT_DIR")
                    .unwrap_or_else(|_| "./trained_models".to_string()),
                checkpoint_directory: env::var("TRAINING_CHECKPOINT_DIR")
                    .unwrap_or_else(|_| "./checkpoints".to_string()),
                max_concurrent_jobs: env::var("TRAINING_MAX_CONCURRENT_JOBS")
                    .unwrap_or_else(|_| "2".to_string())
                    .parse()
                    .unwrap_or(2),
                default_epochs: env::var("TRAINING_DEFAULT_EPOCHS")
                    .unwrap_or_else(|_| "3".to_string())
                    .parse()
                    .unwrap_or(3),
                default_batch_size: env::var("TRAINING_DEFAULT_BATCH_SIZE")
                    .unwrap_or_else(|_| "8".to_string())
                    .parse()
                    .unwrap_or(8),
                default_learning_rate: env::var("TRAINING_DEFAULT_LR")
                    .unwrap_or_else(|_| "5e-5".to_string())
                    .parse()
                    .unwrap_or(5e-5),
                enable_wandb: env::var("TRAINING_ENABLE_WANDB")
                    .unwrap_or_else(|_| "true".to_string())
                    .parse()
                    .unwrap_or(true),
                wandb_project: env::var("WANDB_PROJECT")
                    .unwrap_or_else(|_| "chef-genius-api".to_string()),
                enable_discord_notifications: env::var("TRAINING_ENABLE_DISCORD")
                    .unwrap_or_else(|_| "false".to_string())
                    .parse()
                    .unwrap_or(false),
                discord_webhook_url: env::var("DISCORD_WEBHOOK_URL").ok(),
                enable_sms_alerts: env::var("TRAINING_ENABLE_SMS")
                    .unwrap_or_else(|_| "false".to_string())
                    .parse()
                    .unwrap_or(false),
                alert_phone_number: env::var("ALERT_PHONE_NUMBER").ok(),
            },
            
            monitoring: MonitoringConfig {
                enable_prometheus: env::var("MONITORING_ENABLE_PROMETHEUS")
                    .unwrap_or_else(|_| "true".to_string())
                    .parse()
                    .unwrap_or(true),
                prometheus_port: env::var("PROMETHEUS_PORT")
                    .unwrap_or_else(|_| "9090".to_string())
                    .parse()
                    .unwrap_or(9090),
                metrics_interval_seconds: env::var("MONITORING_METRICS_INTERVAL")
                    .unwrap_or_else(|_| "10".to_string())
                    .parse()
                    .unwrap_or(10),
                log_level: env::var("LOG_LEVEL")
                    .unwrap_or_else(|_| "info".to_string()),
                enable_tracing: env::var("MONITORING_ENABLE_TRACING")
                    .unwrap_or_else(|_| "false".to_string())
                    .parse()
                    .unwrap_or(false),
                jaeger_endpoint: env::var("JAEGER_ENDPOINT").ok(),
                system_metrics_interval_seconds: env::var("MONITORING_SYSTEM_METRICS_INTERVAL")
                    .unwrap_or_else(|_| "5".to_string())
                    .parse()
                    .unwrap_or(5),
            },
            
            tensorrt: TensorRTConfig {
                workspace_size_mb: env::var("TENSORRT_WORKSPACE_SIZE_MB")
                    .unwrap_or_else(|_| "1024".to_string())
                    .parse()
                    .unwrap_or(1024),
                max_batch_size: env::var("TENSORRT_MAX_BATCH_SIZE")
                    .unwrap_or_else(|_| "8".to_string())
                    .parse()
                    .unwrap_or(8),
                precision: env::var("TENSORRT_PRECISION")
                    .unwrap_or_else(|_| "fp16".to_string()),
                enable_dynamic_shapes: env::var("TENSORRT_ENABLE_DYNAMIC_SHAPES")
                    .unwrap_or_else(|_| "true".to_string())
                    .parse()
                    .unwrap_or(true),
                optimization_level: env::var("TENSORRT_OPTIMIZATION_LEVEL")
                    .unwrap_or_else(|_| "O2".to_string()),
                cache_directory: env::var("TENSORRT_CACHE_DIR")
                    .unwrap_or_else(|_| "./tensorrt_cache".to_string()),
                enable_profiling: env::var("TENSORRT_ENABLE_PROFILING")
                    .unwrap_or_else(|_| "false".to_string())
                    .parse()
                    .unwrap_or(false),
            },
            
            security: SecurityConfig {
                enable_api_keys: env::var("SECURITY_ENABLE_API_KEYS")
                    .unwrap_or_else(|_| "false".to_string())
                    .parse()
                    .unwrap_or(false),
                api_key_header: env::var("SECURITY_API_KEY_HEADER")
                    .unwrap_or_else(|_| "X-API-Key".to_string()),
                rate_limit_requests_per_minute: env::var("SECURITY_RATE_LIMIT_RPM")
                    .unwrap_or_else(|_| "100".to_string())
                    .parse()
                    .unwrap_or(100),
                enable_https: env::var("SECURITY_ENABLE_HTTPS")
                    .unwrap_or_else(|_| "false".to_string())
                    .parse()
                    .unwrap_or(false),
                tls_cert_path: env::var("SECURITY_TLS_CERT_PATH").ok(),
                tls_key_path: env::var("SECURITY_TLS_KEY_PATH").ok(),
                enable_cors: env::var("SECURITY_ENABLE_CORS")
                    .unwrap_or_else(|_| "true".to_string())
                    .parse()
                    .unwrap_or(true),
                allowed_origins: env::var("SECURITY_ALLOWED_ORIGINS")
                    .unwrap_or_else(|_| "*".to_string())
                    .split(',')
                    .map(|s| s.trim().to_string())
                    .collect(),
            },
        })
    }
    
    pub fn validate(&self) -> Result<()> {
        // Validate configuration values
        if self.server.port == 0 {
            return Err(anyhow::anyhow!("Server port cannot be 0"));
        }
        
        if self.server.workers == 0 {
            return Err(anyhow::anyhow!("Server workers cannot be 0"));
        }
        
        if self.database.max_connections < self.database.min_connections {
            return Err(anyhow::anyhow!("Database max_connections must be >= min_connections"));
        }
        
        if !matches!(self.tensorrt.precision.as_str(), "fp32" | "fp16" | "int8") {
            return Err(anyhow::anyhow!("Invalid TensorRT precision: {}", self.tensorrt.precision));
        }
        
        if !matches!(self.tensorrt.optimization_level.as_str(), "O0" | "O1" | "O2" | "O3") {
            return Err(anyhow::anyhow!("Invalid TensorRT optimization level: {}", self.tensorrt.optimization_level));
        }
        
        Ok(())
    }
    
    pub fn print_summary(&self) {
        println!("🚀 Chef Genius AI API Configuration:");
        println!("   Server: {}:{}", self.server.host, self.server.port);
        println!("   Workers: {}", self.server.workers);
        println!("   Models: {}", self.inference.models_directory);
        println!("   TensorRT: {} precision, workspace {}MB", 
                 self.tensorrt.precision, self.tensorrt.workspace_size_mb);
        println!("   Database: {}", mask_url(&self.database.url));
        println!("   Redis: {}", mask_url(&self.redis.url));
        
        if self.training.enable_wandb {
            println!("   W&B: enabled ({})", self.training.wandb_project);
        }
        
        if self.training.enable_discord_notifications {
            println!("   Discord: enabled");
        }
        
        if self.monitoring.enable_prometheus {
            println!("   Prometheus: enabled (port {})", self.monitoring.prometheus_port);
        }
    }
}

fn mask_url(url: &str) -> String {
    if let Ok(parsed) = url::Url::parse(url) {
        if parsed.password().is_some() {
            let mut masked = parsed.clone();
            masked.set_password(Some("***")).ok();
            return masked.to_string();
        }
    }
    url.to_string()
}

impl Default for Config {
    fn default() -> Self {
        Self::from_env().unwrap_or_else(|_| Config {
            server: ServerConfig {
                host: "0.0.0.0".to_string(),
                port: 8080,
                workers: num_cpus::get(),
                max_connections: 1000,
                request_timeout_seconds: 30,
                enable_cors: true,
            },
            database: DatabaseConfig {
                url: "postgresql://localhost:5432/chef_genius".to_string(),
                max_connections: 20,
                min_connections: 5,
                connection_timeout_seconds: 10,
                idle_timeout_seconds: 600,
            },
            redis: RedisConfig {
                url: "redis://localhost:6379".to_string(),
                max_connections: 10,
                connection_timeout_seconds: 5,
                command_timeout_seconds: 5,
            },
            inference: InferenceConfig {
                models_directory: "./models".to_string(),
                max_concurrent_requests: Some(10),
                default_batch_size: 1,
                max_sequence_length: 512,
                cache_size: 1000,
                warmup_requests: 5,
                enable_batching: true,
                batch_timeout_ms: 100,
            },
            training: TrainingConfig {
                output_directory: "./trained_models".to_string(),
                checkpoint_directory: "./checkpoints".to_string(),
                max_concurrent_jobs: 2,
                default_epochs: 3,
                default_batch_size: 8,
                default_learning_rate: 5e-5,
                enable_wandb: true,
                wandb_project: "chef-genius-api".to_string(),
                enable_discord_notifications: false,
                discord_webhook_url: None,
                enable_sms_alerts: false,
                alert_phone_number: None,
            },
            monitoring: MonitoringConfig {
                enable_prometheus: true,
                prometheus_port: 9090,
                metrics_interval_seconds: 10,
                log_level: "info".to_string(),
                enable_tracing: false,
                jaeger_endpoint: None,
                system_metrics_interval_seconds: 5,
            },
            tensorrt: TensorRTConfig {
                workspace_size_mb: 1024,
                max_batch_size: 8,
                precision: "fp16".to_string(),
                enable_dynamic_shapes: true,
                optimization_level: "O2".to_string(),
                cache_directory: "./tensorrt_cache".to_string(),
                enable_profiling: false,
            },
            security: SecurityConfig {
                enable_api_keys: false,
                api_key_header: "X-API-Key".to_string(),
                rate_limit_requests_per_minute: 100,
                enable_https: false,
                tls_cert_path: None,
                tls_key_path: None,
                enable_cors: true,
                allowed_origins: vec!["*".to_string()],
            },
        })
    }
}