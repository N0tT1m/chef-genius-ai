use pyo3::prelude::*;
use crate::models::*;
use anyhow::Result;
use std::sync::Arc;
use std::collections::HashMap;
use dashmap::DashMap;
use parking_lot::RwLock;
use ndarray::{Array1, Array2};
use std::time::Instant;

pub struct VectorSearchEngine {
    embeddings: Arc<RwLock<Array2<f32>>>,
    recipes: Arc<RwLock<Vec<PyRecipe>>>,
    index_map: Arc<DashMap<String, usize>>,
    embedding_cache: Arc<DashMap<String, Array1<f32>>>,
    stats: Arc<RwLock<SearchStats>>,
}

#[derive(Debug, Default)]
struct SearchStats {
    total_searches: u64,
    total_embeddings_computed: u64,
    cache_hits: u64,
    avg_search_time_ms: f64,
}

#[pyclass]
pub struct PyVectorSearchEngine {
    engine: Arc<RwLock<VectorSearchEngine>>,
}

#[pymethods]
impl PyVectorSearchEngine {
    #[new]
    fn new(embedding_dim: Option<usize>) -> PyResult<Self> {
        let dim = embedding_dim.unwrap_or(384); // Default sentence-transformer dimension
        let engine = VectorSearchEngine::new_internal(dim)?;
        Ok(Self {
            engine: Arc::new(RwLock::new(engine)),
        })
    }

    /// Add recipes to the search index
    fn add_recipes(&self, recipes: Vec<PyRecipe>) -> PyResult<()> {
        let mut engine = self.engine.write();
        engine.add_recipes_internal(recipes)
    }

    /// Search for similar recipes
    fn search(&self, py: Python, query: String, top_k: Option<usize>, filters: Option<HashMap<String, String>>) -> PyResult<Vec<PySearchResult>> {
        let engine = self.engine.clone();
        let top_k = top_k.unwrap_or(10);
        
        py.allow_threads(|| {
            let engine = engine.read();
            engine.search_internal(query, top_k, filters)
        })
    }

    /// Search by ingredients
    fn search_by_ingredients(&self, py: Python, ingredients: Vec<String>, top_k: Option<usize>) -> PyResult<Vec<PySearchResult>> {
        let query = format!("Recipe with ingredients: {}", ingredients.join(", "));
        self.search(py, query, top_k, None)
    }

    /// Get embedding for text
    fn get_embedding(&self, py: Python, text: String) -> PyResult<Vec<f32>> {
        let engine = self.engine.clone();
        
        py.allow_threads(|| {
            let engine = engine.read();
            let embedding = engine.compute_embedding(&text)?;
            Ok(embedding.to_vec())
        })
    }

    /// Get search statistics
    fn get_stats(&self) -> PyResult<PyObject> {
        let engine = self.engine.read();
        let stats = engine.stats.read();
        
        Python::with_gil(|py| {
            let dict = pyo3::types::PyDict::new(py);
            dict.set_item("total_searches", stats.total_searches)?;
            dict.set_item("total_embeddings_computed", stats.total_embeddings_computed)?;
            dict.set_item("cache_hits", stats.cache_hits)?;
            dict.set_item("cache_hit_rate", 
                if stats.total_embeddings_computed > 0 { 
                    stats.cache_hits as f64 / stats.total_embeddings_computed as f64 
                } else { 0.0 })?;
            dict.set_item("avg_search_time_ms", stats.avg_search_time_ms)?;
            dict.set_item("indexed_recipes", engine.recipes.read().len())?;
            Ok(dict.to_object(py))
        })
    }

    /// Clear the search index
    fn clear_index(&self) -> PyResult<()> {
        let mut engine = self.engine.write();
        engine.clear_internal();
        Ok(())
    }

    /// Save index to file
    fn save_index(&self, file_path: String) -> PyResult<()> {
        let engine = self.engine.read();
        engine.save_index_internal(&file_path)
    }

    /// Load index from file
    fn load_index(&self, file_path: String) -> PyResult<()> {
        let mut engine = self.engine.write();
        engine.load_index_internal(&file_path)
    }
}

impl VectorSearchEngine {
    fn new_internal(embedding_dim: usize) -> PyResult<Self> {
        Ok(Self {
            embeddings: Arc::new(RwLock::new(Array2::zeros((0, embedding_dim)))),
            recipes: Arc::new(RwLock::new(Vec::new())),
            index_map: Arc::new(DashMap::new()),
            embedding_cache: Arc::new(DashMap::new()),
            stats: Arc::new(RwLock::new(SearchStats::default())),
        })
    }

    fn add_recipes_internal(&mut self, recipes: Vec<PyRecipe>) -> PyResult<()> {
        let start_time = Instant::now();
        
        for (i, recipe) in recipes.iter().enumerate() {
            // Create recipe text for embedding
            let recipe_text = format!(
                "{} {} {}",
                recipe.title,
                recipe.ingredients.join(" "),
                recipe.instructions.join(" ")
            );
            
            // Compute embedding
            let embedding = self.compute_embedding(&recipe_text)?;
            
            // Add to index
            let current_len = self.recipes.read().len();
            let recipe_id = format!("recipe_{}", current_len + i);
            self.index_map.insert(recipe_id, current_len + i);
            
            // Extend embeddings matrix
            {
                let mut embeddings = self.embeddings.write();
                let new_embedding = embedding.view().insert_axis(ndarray::Axis(0));
                
                if embeddings.nrows() == 0 {
                    *embeddings = new_embedding.to_owned();
                } else {
                    *embeddings = ndarray::concatenate![ndarray::Axis(0), *embeddings, new_embedding];
                }
            }
        }
        
        // Add recipes to storage
        {
            let mut recipe_storage = self.recipes.write();
            recipe_storage.extend(recipes);
        }
        
        let processing_time = start_time.elapsed().as_millis() as f64;
        
        // Update stats
        {
            let mut stats = self.stats.write();
            stats.total_embeddings_computed += recipes.len() as u64;
        }
        
        Ok(())
    }

    fn search_internal(
        &self, 
        query: String, 
        top_k: usize, 
        _filters: Option<HashMap<String, String>>
    ) -> PyResult<Vec<PySearchResult>> {
        let start_time = Instant::now();
        
        // Update stats
        {
            let mut stats = self.stats.write();
            stats.total_searches += 1;
        }

        // Compute query embedding
        let query_embedding = self.compute_embedding(&query)?;
        
        // Get all embeddings
        let embeddings = self.embeddings.read();
        let recipes = self.recipes.read();
        
        if embeddings.nrows() == 0 {
            return Ok(Vec::new());
        }
        
        // Compute similarities
        let mut similarities = Vec::new();
        for (i, recipe_embedding) in embeddings.outer_iter().enumerate() {
            let similarity = cosine_similarity(&query_embedding, &recipe_embedding.to_owned());
            similarities.push((i, similarity));
        }
        
        // Sort by similarity (descending)
        similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        
        // Take top-k results
        let mut results = Vec::new();
        for (recipe_idx, similarity) in similarities.into_iter().take(top_k) {
            if let Some(recipe) = recipes.get(recipe_idx) {
                let search_result = PySearchResult::new(
                    recipe.clone(),
                    similarity,
                    1.0 - similarity, // Distance = 1 - similarity for cosine
                    None, // No metadata for now
                );
                results.push(search_result);
            }
        }
        
        let search_time = start_time.elapsed().as_millis() as f64;
        
        // Update average search time
        {
            let mut stats = self.stats.write();
            stats.avg_search_time_ms = (stats.avg_search_time_ms * (stats.total_searches - 1) as f64 + search_time) / stats.total_searches as f64;
        }
        
        Ok(results)
    }

    fn compute_embedding(&self, text: &str) -> PyResult<Array1<f32>> {
        // Check cache first
        if let Some(cached_embedding) = self.embedding_cache.get(text) {
            let mut stats = self.stats.write();
            stats.cache_hits += 1;
            return Ok(cached_embedding.clone());
        }
        
        // For now, use a simple mock embedding based on text features
        // In production, this would use a real sentence transformer model
        let embedding = self.mock_embedding(text);
        
        // Cache the result
        self.embedding_cache.insert(text.to_string(), embedding.clone());
        
        {
            let mut stats = self.stats.write();
            stats.total_embeddings_computed += 1;
        }
        
        Ok(embedding)
    }

    fn mock_embedding(&self, text: &str) -> Array1<f32> {
        // Simple mock embedding based on text characteristics
        let words: Vec<&str> = text.split_whitespace().collect();
        let mut features = vec![0.0f32; 384]; // Mock 384-dimensional embedding
        
        // Feature 1: Text length
        features[0] = (text.len() as f32 / 1000.0).min(1.0);
        
        // Feature 2: Number of words
        features[1] = (words.len() as f32 / 100.0).min(1.0);
        
        // Feature 3-10: Common cooking terms
        let cooking_terms = ["cook", "bake", "fry", "boil", "steam", "grill", "roast", "sauté"];
        for (i, term) in cooking_terms.iter().enumerate() {
            features[2 + i] = if text.to_lowercase().contains(term) { 1.0 } else { 0.0 };
        }
        
        // Feature 11-50: Ingredient categories
        let ingredient_categories = [
            "chicken", "beef", "pork", "fish", "vegetables", "pasta", "rice", "bread",
            "cheese", "milk", "eggs", "flour", "sugar", "salt", "pepper", "garlic",
            "onion", "tomato", "potato", "carrot", "spinach", "mushroom", "pepper",
            "oil", "butter", "cream", "wine", "herbs", "spices", "lemon", "lime",
            "apple", "orange", "strawberry", "chocolate", "vanilla", "cinnamon",
            "ginger", "basil", "oregano"
        ];
        
        for (i, category) in ingredient_categories.iter().enumerate() {
            if i + 10 < features.len() {
                features[10 + i] = if text.to_lowercase().contains(category) { 1.0 } else { 0.0 };
            }
        }
        
        // Fill remaining features with hash-based values for uniqueness
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        for i in 50..features.len() {
            let mut hasher = DefaultHasher::new();
            format!("{}_{}", text, i).hash(&mut hasher);
            features[i] = (hasher.finish() % 1000) as f32 / 1000.0;
        }
        
        // Normalize the vector
        let norm: f32 = features.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for feature in features.iter_mut() {
                *feature /= norm;
            }
        }
        
        Array1::from(features)
    }

    fn clear_internal(&mut self) {
        let embedding_dim = self.embeddings.read().ncols();
        *self.embeddings.write() = Array2::zeros((0, embedding_dim));
        self.recipes.write().clear();
        self.index_map.clear();
        self.embedding_cache.clear();
    }

    fn save_index_internal(&self, file_path: &str) -> PyResult<()> {
        use std::fs::File;
        use std::io::Write;
        
        let recipes = self.recipes.read();
        let serialized = serde_json::to_string(&*recipes)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
        
        let mut file = File::create(file_path)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
        
        file.write_all(serialized.as_bytes())
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
        
        Ok(())
    }

    fn load_index_internal(&mut self, file_path: &str) -> PyResult<()> {
        use std::fs;
        
        let content = fs::read_to_string(file_path)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
        
        let recipes: Vec<PyRecipe> = serde_json::from_str(&content)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        
        self.add_recipes_internal(recipes)?;
        Ok(())
    }
}

fn cosine_similarity(a: &Array1<f32>, b: &Array1<f32>) -> f32 {
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    
    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot_product / (norm_a * norm_b)
    }
}