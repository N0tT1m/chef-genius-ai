use crate::config::Config;
use crate::error::{ApiError, Result};
use crate::models::*;
use crate::tensorrt::{TensorRTEngine, benchmark_tensorrt_engine};
use anyhow::anyhow;
use dashmap::DashMap;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};
use uuid::Uuid;

pub struct InferenceEngine {
    models: Arc<DashMap<Uuid, LoadedModel>>,
    tokenizer_cache: Arc<DashMap<String, Arc<dyn Tokenizer>>>,
    config: Arc<Config>,
    performance_stats: Arc<RwLock<PerformanceStats>>,
}

struct LoadedModel {
    model_id: Uuid,
    engine: TensorRTEngine,
    tokenizer: Arc<dyn Tokenizer>,
    metadata: ModelMetadata,
    load_time: std::time::SystemTime,
    inference_count: std::sync::atomic::AtomicU64,
    total_inference_time: std::sync::atomic::AtomicU64,
}

#[derive(Debug, Default)]
struct PerformanceStats {
    total_requests: u64,
    successful_requests: u64,
    failed_requests: u64,
    avg_latency_ms: f64,
    throughput_per_sec: f64,
    models_loaded: u32,
}

// Simplified tokenizer trait
pub trait Tokenizer: Send + Sync {
    fn encode(&self, text: &str) -> Result<Vec<i32>>;
    fn decode(&self, tokens: &[i32]) -> Result<String>;
    fn get_vocab_size(&self) -> usize;
    fn get_pad_token_id(&self) -> i32;
    fn get_eos_token_id(&self) -> i32;
}

// Simple tokenizer implementation (in practice, you'd use a real tokenizer)
pub struct SimpleTokenizer {
    vocab_size: usize,
    pad_token_id: i32,
    eos_token_id: i32,
}

impl SimpleTokenizer {
    pub fn new() -> Self {
        Self {
            vocab_size: 50257, // GPT-2 vocab size
            pad_token_id: 50256,
            eos_token_id: 50256,
        }
    }
}

impl Tokenizer for SimpleTokenizer {
    fn encode(&self, text: &str) -> Result<Vec<i32>> {
        // Simplified encoding - in practice use proper tokenizer
        let words: Vec<&str> = text.split_whitespace().collect();
        let mut tokens = Vec::new();
        
        for word in words {
            // Simple hash-based token assignment (not real tokenization)
            let token_id = (word.chars().map(|c| c as u32).sum::<u32>() % (self.vocab_size as u32 - 100)) as i32;
            tokens.push(token_id);
        }
        
        Ok(tokens)
    }
    
    fn decode(&self, tokens: &[i32]) -> Result<String> {
        // Simplified decoding
        Ok(format!("decoded_text_{}", tokens.len()))
    }
    
    fn get_vocab_size(&self) -> usize {
        self.vocab_size
    }
    
    fn get_pad_token_id(&self) -> i32 {
        self.pad_token_id
    }
    
    fn get_eos_token_id(&self) -> i32 {
        self.eos_token_id
    }
}

impl InferenceEngine {
    pub async fn new(config: &Config) -> Result<Self> {
        info!("Initializing TensorRT Inference Engine");
        
        let engine = Self {
            models: Arc::new(DashMap::new()),
            tokenizer_cache: Arc::new(DashMap::new()),
            config: Arc::new(config.clone()),
            performance_stats: Arc::new(RwLock::new(PerformanceStats::default())),
        };
        
        // Load default models at startup
        engine.load_default_models().await?;
        
        Ok(engine)
    }
    
    async fn load_default_models(&self) -> Result<()> {
        // Check for models in the models directory
        let models_dir = std::path::Path::new("models");
        if !models_dir.exists() {
            warn!("Models directory not found, skipping default model loading");
            return Ok(());
        }
        
        // Look for .trt files (TensorRT engines)
        for entry in std::fs::read_dir(models_dir)? {
            let entry = entry?;
            let path = entry.path();
            
            if path.extension().and_then(|s| s.to_str()) == Some("trt") {
                match self.load_model_from_path(path.to_string_lossy().to_string()).await {
                    Ok(model_id) => {
                        info!("Loaded default model: {} -> {}", path.display(), model_id);
                    }
                    Err(e) => {
                        warn!("Failed to load model {}: {}", path.display(), e);
                    }
                }
            }
        }
        
        Ok(())
    }
    
    pub async fn load_model_from_path(&self, model_path: String) -> Result<Uuid> {
        let model_id = Uuid::new_v4();
        
        info!("Loading TensorRT model: {} -> {}", model_path, model_id);
        
        // Create TensorRT engine
        let mut engine = TensorRTEngine::new()
            .map_err(|e| ApiError::ModelLoadError(format!("Failed to create TensorRT engine: {}", e)))?;
        
        // Load engine from file
        engine.load_from_file(&model_path)
            .map_err(|e| ApiError::ModelLoadError(format!("Failed to load engine: {}", e)))?;
        
        // Create tokenizer (simplified - in practice load from model config)
        let tokenizer = Arc::new(SimpleTokenizer::new()) as Arc<dyn Tokenizer>;
        
        // Create metadata
        let metadata = ModelMetadata {
            input_shape: engine.get_input_shape(0).unwrap_or(&vec![1, 512]).clone(),
            output_shape: engine.get_output_shape(0).unwrap_or(&vec![1, 512, 50257]).clone(),
            parameters: 124_000_000, // Example parameter count
            model_size_mb: std::fs::metadata(&model_path)?.len() as f32 / 1024.0 / 1024.0,
            precision: "fp16".to_string(),
            supported_batch_sizes: vec![1, 2, 4, 8],
            max_sequence_length: 512,
            optimization_level: "O2".to_string(),
        };
        
        let loaded_model = LoadedModel {
            model_id,
            engine,
            tokenizer,
            metadata,
            load_time: std::time::SystemTime::now(),
            inference_count: std::sync::atomic::AtomicU64::new(0),
            total_inference_time: std::sync::atomic::AtomicU64::new(0),
        };
        
        self.models.insert(model_id, loaded_model);
        
        // Update performance stats
        {
            let mut stats = self.performance_stats.write().await;
            stats.models_loaded += 1;
        }
        
        info!("Model loaded successfully: {}", model_id);
        Ok(model_id)
    }
    
    pub async fn generate_recipe(&self, request: InferenceRequest) -> Result<InferenceResponse> {
        let start_time = Instant::now();
        
        // Update stats
        {
            let mut stats = self.performance_stats.write().await;
            stats.total_requests += 1;
        }
        
        // Determine which model to use
        let model_id = if let Some(id) = request.model_id {
            id
        } else {
            // Use first available model
            self.models.iter()
                .next()
                .ok_or_else(|| ApiError::ModelNotFound("No models loaded".to_string()))?
                .key().clone()
        };
        
        let model = self.models.get(&model_id)
            .ok_or_else(|| ApiError::ModelNotFound(format!("Model {} not found", model_id)))?;
        
        // Prepare input text
        let input_text = self.format_ingredients_to_prompt(&request.ingredients, &request);
        
        // Tokenize input
        let input_tokens = model.tokenizer.encode(&input_text)
            .map_err(|e| ApiError::TokenizationError(e.to_string()))?;
        
        // Pad/truncate to model's expected length
        let max_length = request.max_length.unwrap_or(512) as usize;
        let mut padded_tokens = input_tokens;
        if padded_tokens.len() > max_length {
            padded_tokens.truncate(max_length);
        } else {
            let pad_token = model.tokenizer.get_pad_token_id();
            while padded_tokens.len() < max_length {
                padded_tokens.push(pad_token);
            }
        }
        
        // Run inference
        let output_logits = model.engine.infer(&padded_tokens)
            .map_err(|e| ApiError::InferenceError(format!("TensorRT inference failed: {}", e)))?;
        
        // Post-process outputs
        let generated_tokens = self.postprocess_logits(
            &output_logits,
            &request,
            &*model.tokenizer,
        )?;
        
        // Decode generated text
        let generated_text = model.tokenizer.decode(&generated_tokens)
            .map_err(|e| ApiError::TokenizationError(e.to_string()))?;
        
        // Parse generated text into recipe
        let recipe = self.parse_generated_text_to_recipe(&generated_text, &request.ingredients)?;
        
        let generation_time = start_time.elapsed();
        let generation_time_ms = generation_time.as_millis() as u64;
        
        // Update model statistics
        model.inference_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        model.total_inference_time.fetch_add(generation_time_ms, std::sync::atomic::Ordering::Relaxed);
        
        // Update global stats
        {
            let mut stats = self.performance_stats.write().await;
            stats.successful_requests += 1;
            stats.avg_latency_ms = (stats.avg_latency_ms * (stats.successful_requests - 1) as f64 + generation_time_ms as f64) / stats.successful_requests as f64;
        }
        
        info!("Recipe generated in {}ms for model {}", generation_time_ms, model_id);
        debug!("Generated recipe: {}", recipe.title);
        
        Ok(InferenceResponse {
            recipe,
            confidence: 0.85, // Would calculate actual confidence from logits
            generation_time_ms,
            model_version: "1.0.0".to_string(),
            model_id,
            alternatives: None,
        })
    }
    
    pub async fn batch_inference(&self, requests: Vec<InferenceRequest>) -> Result<Vec<InferenceResponse>> {
        let batch_size = requests.len();
        info!("Processing batch inference with {} requests", batch_size);
        
        // Process requests in parallel with concurrency limit
        let max_concurrent = self.config.inference.max_concurrent_requests.unwrap_or(10);
        let semaphore = Arc::new(tokio::sync::Semaphore::new(max_concurrent));
        
        let mut tasks = Vec::new();
        
        for request in requests {
            let engine = Arc::new(self);
            let permit = semaphore.clone();
            
            let task = tokio::spawn(async move {
                let _permit = permit.acquire().await.unwrap();
                engine.generate_recipe(request).await
            });
            
            tasks.push(task);
        }
        
        // Wait for all tasks to complete
        let mut responses = Vec::new();
        for task in tasks {
            match task.await {
                Ok(Ok(response)) => responses.push(response),
                Ok(Err(e)) => {
                    error!("Batch inference task failed: {}", e);
                    return Err(e);
                }
                Err(e) => {
                    error!("Batch inference task panicked: {}", e);
                    return Err(ApiError::InferenceError("Task panicked".to_string()));
                }
            }
        }
        
        info!("Batch inference completed: {} responses", responses.len());
        Ok(responses)
    }
    
    fn format_ingredients_to_prompt(&self, ingredients: &[String], request: &InferenceRequest) -> String {
        let mut prompt = String::from("Generate a recipe using these ingredients:\n");
        
        for ingredient in ingredients {
            prompt.push_str(&format!("- {}\n", ingredient));
        }
        
        if let Some(cuisine) = &request.cuisine_style {
            prompt.push_str(&format!("\nCuisine style: {}\n", cuisine));
        }
        
        if let Some(restrictions) = &request.dietary_restrictions {
            if !restrictions.is_empty() {
                prompt.push_str(&format!("\nDietary restrictions: {}\n", restrictions.join(", ")));
            }
        }
        
        prompt.push_str("\nRecipe:\n");
        prompt
    }
    
    fn postprocess_logits(&self, logits: &[f32], request: &InferenceRequest, tokenizer: &dyn Tokenizer) -> Result<Vec<i32>> {
        let temperature = request.temperature.unwrap_or(0.8);
        let top_k = request.top_k.unwrap_or(50) as usize;
        let top_p = request.top_p.unwrap_or(0.9);
        
        // Simplified sampling - in practice you'd implement proper nucleus sampling
        let mut tokens = Vec::new();
        let vocab_size = tokenizer.get_vocab_size();
        let seq_len = logits.len() / vocab_size;
        
        for i in 0..seq_len.min(100) { // Generate up to 100 tokens
            let start_idx = i * vocab_size;
            let end_idx = start_idx + vocab_size;
            let token_logits = &logits[start_idx..end_idx];
            
            // Apply temperature
            let scaled_logits: Vec<f32> = token_logits.iter()
                .map(|&x| x / temperature)
                .collect();
            
            // Find top-k tokens
            let mut indexed_logits: Vec<(usize, f32)> = scaled_logits.iter()
                .enumerate()
                .map(|(idx, &val)| (idx, val))
                .collect();
            
            indexed_logits.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            indexed_logits.truncate(top_k);
            
            // Simple sampling - just take the highest probability token
            let selected_token = indexed_logits[0].0 as i32;
            tokens.push(selected_token);
            
            // Stop at EOS token
            if selected_token == tokenizer.get_eos_token_id() {
                break;
            }
        }
        
        Ok(tokens)
    }
    
    fn parse_generated_text_to_recipe(&self, text: &str, ingredients: &[String]) -> Result<Recipe> {
        // Simplified recipe parsing - in practice you'd use more sophisticated NLP
        let title = if text.len() > 20 {
            format!("Recipe with {}", ingredients.join(", "))
        } else {
            "Generated Recipe".to_string()
        };
        
        // Mock recipe structure
        let recipe = Recipe {
            title,
            ingredients: ingredients.to_vec(),
            instructions: vec![
                "Prepare all ingredients".to_string(),
                "Combine ingredients according to recipe".to_string(),
                "Cook as directed".to_string(),
                "Serve and enjoy".to_string(),
            ],
            cooking_time: Some("30 minutes".to_string()),
            prep_time: Some("15 minutes".to_string()),
            servings: Some(4),
            difficulty: Some("Medium".to_string()),
            cuisine_type: None,
            dietary_tags: None,
            nutrition_info: None,
        };
        
        Ok(recipe)
    }
    
    pub async fn get_performance_stats(&self) -> PerformanceStats {
        *self.performance_stats.read().await
    }
    
    pub async fn benchmark_model(&self, model_id: Uuid, iterations: u32) -> Result<BenchmarkResults> {
        let model = self.models.get(&model_id)
            .ok_or_else(|| ApiError::ModelNotFound(format!("Model {} not found", model_id)))?;
        
        info!("Benchmarking model {} with {} iterations", model_id, iterations);
        
        let results = benchmark_tensorrt_engine(&model.engine, iterations)
            .map_err(|e| ApiError::InferenceError(format!("Benchmark failed: {}", e)))?;
        
        info!("Benchmark completed: avg={}ms, throughput={}req/s", 
              results.avg_latency_ms, results.throughput_per_sec);
        
        Ok(results)
    }
    
    pub fn list_loaded_models(&self) -> Vec<(Uuid, ModelMetadata)> {
        self.models.iter()
            .map(|entry| (*entry.key(), entry.value().metadata.clone()))
            .collect()
    }
    
    pub async fn unload_model(&self, model_id: Uuid) -> Result<()> {
        if self.models.remove(&model_id).is_some() {
            let mut stats = self.performance_stats.write().await;
            stats.models_loaded = stats.models_loaded.saturating_sub(1);
            info!("Model {} unloaded", model_id);
            Ok(())
        } else {
            Err(ApiError::ModelNotFound(format!("Model {} not found", model_id)))
        }
    }
}

#[derive(Debug, Clone)]
pub struct BenchmarkResults {
    pub avg_latency_ms: f64,
    pub min_latency_ms: f64,
    pub max_latency_ms: f64,
    pub throughput_per_sec: f64,
    pub total_iterations: u32,
}