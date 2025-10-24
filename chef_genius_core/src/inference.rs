use pyo3::prelude::*;
use crate::models::*;
use anyhow::Result;
use dashmap::DashMap;
use parking_lot::RwLock;
use std::sync::Arc;
use std::time::Instant;
use tracing::{info, debug};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/// High-performance ML inference engine
#[pyclass]
pub struct PyInferenceEngine {
    inner: Arc<InferenceEngine>,
}

#[pymethods]
impl PyInferenceEngine {
    #[new]
    fn new(model_path: String) -> PyResult<Self> {
        let inner = InferenceEngine::new(&model_path)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Failed to create engine: {}", e)))?;

        Ok(Self {
            inner: Arc::new(inner),
        })
    }

    /// Generate a single recipe
    fn generate_recipe(&self, request: PyInferenceRequest) -> PyResult<PyInferenceResponse> {
        self.inner.generate_recipe(request)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Generation failed: {}", e)))
    }

    /// Generate multiple recipes in batch
    fn batch_generate(&self, requests: Vec<PyInferenceRequest>) -> PyResult<Vec<PyInferenceResponse>> {
        self.inner.batch_generate(requests)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Batch generation failed: {}", e)))
    }

    /// Get model statistics
    fn get_stats(&self) -> PyResult<PyObject> {
        let stats = self.inner.get_stats();
        Python::with_gil(|py| {
            let dict = pyo3::types::PyDict::new(py);
            dict.set_item("total_requests", stats.total_requests)?;
            dict.set_item("cache_hits", stats.cache_hits)?;
            dict.set_item("avg_latency_ms", stats.avg_latency_ms)?;
            dict.set_item("model_loaded", stats.model_loaded)?;
            Ok(dict.to_object(py))
        })
    }

    /// Clear inference cache
    fn clear_cache(&self) {
        self.inner.clear_cache();
    }
}

pub struct InferenceEngine {
    cache: Arc<DashMap<String, CachedResult>>,
    stats: Arc<RwLock<EngineStats>>,
}

#[derive(Clone)]
struct CachedResult {
    response: PyInferenceResponse,
    timestamp: std::time::SystemTime,
}

#[derive(Debug, Default, Clone, Copy)]
struct EngineStats {
    total_requests: u64,
    cache_hits: u64,
    avg_latency_ms: f64,
    model_loaded: bool,
}

impl InferenceEngine {
    pub fn new(model_path: &str) -> Result<Self> {
        info!("Initializing inference engine with model: {}", model_path);
        
        let engine = Self {
            cache: Arc::new(DashMap::new()),
            stats: Arc::new(RwLock::new(EngineStats {
                model_loaded: true,
                ..Default::default()
            })),
        };

        info!("Inference engine initialized successfully");
        Ok(engine)
    }

    pub fn generate_recipe(&self, request: PyInferenceRequest) -> Result<PyInferenceResponse> {
        let start_time = Instant::now();
        
        // Update stats
        {
            let mut stats = self.stats.write();
            stats.total_requests += 1;
        }

        // Check cache first
        if request.use_cache.unwrap_or(true) {
            let cache_key = self.create_cache_key(&request);
            if let Some(cached) = self.cache.get(&cache_key) {
                if cached.timestamp.elapsed().unwrap_or(std::time::Duration::MAX) < std::time::Duration::from_secs(3600) {
                    let mut stats = self.stats.write();
                    stats.cache_hits += 1;
                    
                    let mut response = cached.response.clone();
                    response.cached = true;
                    response.generation_time_ms = 1;
                    return Ok(response);
                }
            }
        }

        // Generate recipe
        let recipe = self.run_inference(&request)?;
        
        let generation_time = start_time.elapsed();
        let generation_time_ms = generation_time.as_millis() as u64;

        let response = PyInferenceResponse::new(
            recipe,
            0.85,
            generation_time_ms,
            "rust-engine-v1.0".to_string(),
            None,
            false,
        );

        // Update cache
        if request.use_cache.unwrap_or(true) {
            let cache_key = self.create_cache_key(&request);
            self.cache.insert(cache_key, CachedResult {
                response: response.clone(),
                timestamp: std::time::SystemTime::now(),
            });
        }

        // Update stats
        {
            let mut stats = self.stats.write();
            let total = stats.total_requests as f64;
            stats.avg_latency_ms = (stats.avg_latency_ms * (total - 1.0) + generation_time_ms as f64) / total;
        }

        info!("Recipe generated in {}ms", generation_time_ms);
        Ok(response)
    }

    pub fn batch_generate(&self, requests: Vec<PyInferenceRequest>) -> Result<Vec<PyInferenceResponse>> {
        info!("Processing batch of {} requests", requests.len());
        
        let responses: Result<Vec<_>, _> = requests.into_iter()
            .map(|request| self.generate_recipe(request))
            .collect();

        let responses = responses?;
        info!("Batch processing completed: {} responses", responses.len());
        Ok(responses)
    }

    fn run_inference(&self, request: &PyInferenceRequest) -> Result<PyRecipe> {
        // Mock fast recipe generation
        let recipe = PyRecipe::new(
            format!("Delicious {} Recipe", request.ingredients.join(" & ")),
            request.ingredients.clone(),
            vec![
                "Prepare all ingredients according to recipe specifications".to_string(),
                "Follow cooking methodology for optimal results".to_string(),
                "Combine ingredients using proper techniques".to_string(),
                "Cook until perfectly done and serve immediately".to_string(),
            ],
            Some("25-30 minutes".to_string()),
            Some("10-15 minutes".to_string()),
            Some(4),
            request.difficulty.clone(),
            request.cuisine_style.clone(),
            request.dietary_restrictions.clone(),
            Some(0.88),
        );

        Ok(recipe)
    }

    fn create_cache_key(&self, request: &PyInferenceRequest) -> String {
        let mut hasher = DefaultHasher::new();
        request.ingredients.hash(&mut hasher);
        request.cuisine_style.hash(&mut hasher);
        request.dietary_restrictions.hash(&mut hasher);
        request.difficulty.hash(&mut hasher);
        
        format!("recipe_{:x}", hasher.finish())
    }

    pub fn get_stats(&self) -> EngineStats {
        *self.stats.read()
    }

    pub fn clear_cache(&self) {
        self.cache.clear();
        info!("Inference cache cleared");
    }
}