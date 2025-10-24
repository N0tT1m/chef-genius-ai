use pyo3::prelude::*;
use serde_json::Value;
use std::collections::HashMap;
use rayon::prelude::*;

/// High-performance simple tokenizer for recipe monitoring
/// Matches training data format exactly, no enterprise tokens
#[pyclass]
pub struct PySimpleTokenizer {
    vocab: HashMap<String, u32>,
    vocab_size: usize,
    max_length: usize,
}

#[pymethods]
impl PySimpleTokenizer {
    #[new]
    pub fn new(vocab_file: Option<String>, max_length: Option<usize>) -> PyResult<Self> {
        let vocab = if let Some(file) = vocab_file {
            Self::load_vocab(&file)?
        } else {
            Self::create_basic_vocab()
        };
        
        Ok(PySimpleTokenizer {
            vocab_size: vocab.len(),
            vocab,
            max_length: max_length.unwrap_or(512),
        })
    }
    
    /// Fast tokenization matching training format
    pub fn encode(&self, text: &str) -> PyResult<Vec<u32>> {
        let tokens = self.tokenize_fast(text);
        let mut token_ids = Vec::with_capacity(tokens.len().min(self.max_length));
        
        for token in tokens.into_iter().take(self.max_length) {
            if let Some(&id) = self.vocab.get(&token) {
                token_ids.push(id);
            } else {
                token_ids.push(self.vocab[&"<UNK>".to_string()]);
            }
        }
        
        Ok(token_ids)
    }
    
    /// Parallel batch encoding for monitoring
    pub fn encode_batch(&self, texts: Vec<String>) -> PyResult<Vec<Vec<u32>>> {
        let results: Vec<Vec<u32>> = texts
            .par_iter()
            .map(|text| self.encode(text).unwrap_or_default())
            .collect();
        
        Ok(results)
    }
    
    /// Format prompt to match training data
    pub fn format_monitoring_prompt(&self, request: &str) -> PyResult<String> {
        // Match the exact training format: simple text prompts
        let formatted = if request.contains("create") || request.contains("generate") {
            request.to_string()
        } else {
            format!("Create a {}", request.to_lowercase())
        };
        
        // Append the training format structure
        Ok(format!("{}\n\nIngredients:", formatted))
    }
    
    /// Create B2B-style prompt but in simple format (no special tokens)
    pub fn format_b2b_simple(&self, request: &str, business_context: Option<String>) -> PyResult<String> {
        let context = business_context.unwrap_or_else(|| "restaurant".to_string());
        
        let formatted = if request.contains(&context) {
            request.to_string()
        } else {
            format!("{} for {}", request, context)
        };
        
        Ok(format!("Create a {}\n\nIngredients:", formatted.to_lowercase()))
    }
    
    /// Check if prompt exceeds token limit
    pub fn check_token_limit(&self, text: &str) -> PyResult<(usize, bool)> {
        let tokens = self.encode(text)?;
        let within_limit = tokens.len() <= self.max_length;
        Ok((tokens.len(), within_limit))
    }
    
    /// Get vocab size
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }
    
    /// Get max length
    pub fn max_length(&self) -> usize {
        self.max_length
    }
}

impl PySimpleTokenizer {
    fn tokenize_fast(&self, text: &str) -> Vec<String> {
        // Simple whitespace + punctuation tokenization
        // Much faster than complex tokenizers for monitoring
        text.to_lowercase()
            .split_whitespace()
            .map(|word| {
                word.chars()
                    .map(|c| if c.is_alphanumeric() { c } else { ' ' })
                    .collect::<String>()
                    .split_whitespace()
                    .map(|s| s.to_string())
                    .collect::<Vec<_>>()
            })
            .flatten()
            .filter(|token| !token.is_empty())
            .collect()
    }
    
    fn load_vocab(file_path: &str) -> PyResult<HashMap<String, u32>> {
        let content = std::fs::read_to_string(file_path)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Failed to read vocab: {}", e)))?;
        
        let vocab_json: Value = serde_json::from_str(&content)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Invalid JSON: {}", e)))?;
        
        let mut vocab = HashMap::new();
        if let Value::Object(obj) = vocab_json {
            for (token, id) in obj {
                if let Value::Number(num) = id {
                    if let Some(id_u64) = num.as_u64() {
                        vocab.insert(token, id_u64 as u32);
                    }
                }
            }
        }
        
        Ok(vocab)
    }
    
    fn create_basic_vocab() -> HashMap<String, u32> {
        let mut vocab = HashMap::new();
        
        // Special tokens
        vocab.insert("<PAD>".to_string(), 0);
        vocab.insert("<UNK>".to_string(), 1);
        vocab.insert("<BOS>".to_string(), 2);
        vocab.insert("<EOS>".to_string(), 3);
        
        // Common recipe words (most frequent from cooking)
        let common_words = vec![
            // Basic cooking terms
            "the", "and", "a", "to", "in", "of", "for", "with", "on", "at",
            "cook", "add", "mix", "heat", "bake", "fry", "boil", "simmer", "serve",
            "ingredients", "recipe", "minutes", "cups", "tablespoons", "teaspoons",
            
            // Common ingredients
            "chicken", "beef", "pork", "fish", "eggs", "milk", "butter", "oil",
            "salt", "pepper", "garlic", "onion", "tomato", "cheese", "flour",
            "sugar", "water", "rice", "pasta", "bread", "potato", "carrot",
            
            // Cooking methods
            "roast", "grill", "steam", "saute", "brown", "season", "chop", "slice",
            "dice", "mince", "blend", "whisk", "stir", "combine", "prepare",
            
            // Measurements
            "cup", "tablespoon", "teaspoon", "pound", "ounce", "gram", "liter",
            "inch", "degrees", "fahrenheit", "celsius", "medium", "large", "small",
            
            // Time and temperature
            "hot", "cold", "warm", "cool", "until", "about", "approximately",
            "hour", "minute", "second", "low", "high", "medium",
        ];
        
        for (idx, word) in common_words.iter().enumerate() {
            vocab.insert(word.to_string(), (idx + 4) as u32);
        }
        
        vocab
    }
}

/// Fast recipe prompt processor for monitoring
#[pyclass]
pub struct PyRecipePromptProcessor {
    tokenizer: PySimpleTokenizer,
    b2b_contexts: Vec<String>,
}

#[pymethods]
impl PyRecipePromptProcessor {
    #[new]
    pub fn new() -> PyResult<Self> {
        Ok(PyRecipePromptProcessor {
            tokenizer: PySimpleTokenizer::new(None, Some(512))?,
            b2b_contexts: vec![
                "restaurant".to_string(),
                "catering".to_string(), 
                "meal kit".to_string(),
                "food service".to_string(),
                "commercial kitchen".to_string(),
            ],
        })
    }
    
    /// Process multiple prompts for monitoring (parallel)
    pub fn process_monitoring_batch(&self, prompts: Vec<String>) -> PyResult<Vec<MonitoringResult>> {
        let results: Vec<MonitoringResult> = prompts
            .par_iter()
            .map(|prompt| {
                let formatted = self.tokenizer.format_monitoring_prompt(prompt).unwrap_or_default();
                let (token_count, within_limit) = self.tokenizer.check_token_limit(&formatted).unwrap_or((0, false));
                
                MonitoringResult {
                    original_prompt: prompt.clone(),
                    formatted_prompt: formatted,
                    token_count,
                    within_limit,
                    is_b2b: self.detect_b2b_context(prompt),
                }
            })
            .collect();
        
        Ok(results)
    }
    
    /// Smart B2B detection and formatting
    pub fn process_b2b_prompt(&self, prompt: &str) -> PyResult<MonitoringResult> {
        let context = self.detect_b2b_context(prompt);
        
        let formatted = if context {
            // Extract business context and format simply
            let business_type = self.extract_business_type(prompt);
            self.tokenizer.format_b2b_simple(prompt, Some(business_type))?
        } else {
            self.tokenizer.format_monitoring_prompt(prompt)?
        };
        
        let (token_count, within_limit) = self.tokenizer.check_token_limit(&formatted)?;
        
        Ok(MonitoringResult {
            original_prompt: prompt.to_string(),
            formatted_prompt: formatted,
            token_count,
            within_limit,
            is_b2b: context,
        })
    }
}

impl PyRecipePromptProcessor {
    fn detect_b2b_context(&self, prompt: &str) -> bool {
        let prompt_lower = prompt.to_lowercase();
        
        // B2B indicators
        let b2b_keywords = [
            "restaurant", "catering", "commercial", "business", "food service",
            "covers", "servings", "volume", "cost", "budget", "staff", "kitchen",
            "meal kit", "institutional", "school", "hospital", "corporate"
        ];
        
        b2b_keywords.iter().any(|keyword| prompt_lower.contains(keyword))
    }
    
    fn extract_business_type(&self, prompt: &str) -> String {
        let prompt_lower = prompt.to_lowercase();
        
        if prompt_lower.contains("restaurant") || prompt_lower.contains("dining") {
            "restaurant service".to_string()
        } else if prompt_lower.contains("catering") {
            "catering service".to_string()
        } else if prompt_lower.contains("meal kit") {
            "meal kit service".to_string()
        } else if prompt_lower.contains("school") || prompt_lower.contains("institutional") {
            "institutional food service".to_string()
        } else {
            "commercial food service".to_string()
        }
    }
}

#[pyclass]
#[derive(Clone)]
pub struct MonitoringResult {
    #[pyo3(get)]
    pub original_prompt: String,
    #[pyo3(get)]
    pub formatted_prompt: String,
    #[pyo3(get)]
    pub token_count: usize,
    #[pyo3(get)]
    pub within_limit: bool,
    #[pyo3(get)]
    pub is_b2b: bool,
}

#[pymethods]
impl MonitoringResult {
    fn __repr__(&self) -> String {
        format!(
            "MonitoringResult(tokens={}, within_limit={}, is_b2b={})",
            self.token_count, self.within_limit, self.is_b2b
        )
    }
}