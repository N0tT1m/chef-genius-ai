use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::fs::File;
use std::io::Read;
use std::sync::Arc;
use crossbeam_channel::{bounded, Receiver, Sender};
use parking_lot::RwLock;
use rand::seq::SliceRandom;
use rand::thread_rng;
use serde::{Deserialize, Serialize};
use rayon::prelude::*;
use simd_json;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct RecipeData {
    input: Box<str>,
    output: Box<str>,
    quality_score: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct LegacyRecipeData {
    instruction: String,
    ingredients: String,
    title: String,
}

#[pyclass]
struct FastDataLoader {
    data: Arc<RwLock<Vec<RecipeData>>>,
    batch_size: usize,
    shuffle: bool,
    current_epoch: usize,
    indices: Arc<RwLock<Vec<usize>>>,
    position: Arc<RwLock<usize>>,
    buffer_sender: Option<Sender<Vec<RecipeData>>>,
    buffer_receiver: Option<Receiver<Vec<RecipeData>>>,
}

#[pymethods]
impl FastDataLoader {
    #[new]
    #[pyo3(signature = (data_path, batch_size, shuffle, buffer_size=None))]
    fn new(
        data_path: String,
        batch_size: usize,
        shuffle: bool,
        buffer_size: Option<usize>,
    ) -> PyResult<Self> {
        let data = Self::load_data(&data_path)?;
        let len = data.len();
        let mut indices: Vec<usize> = (0..len).collect();
        
        if shuffle {
            indices.shuffle(&mut thread_rng());
        }
        
        let buffer_size = buffer_size.unwrap_or(16);
        let (sender, receiver) = bounded(buffer_size);
        
        let mut loader = FastDataLoader {
            data: Arc::new(RwLock::new(data)),
            batch_size,
            shuffle,
            current_epoch: 0,
            indices: Arc::new(RwLock::new(indices)),
            position: Arc::new(RwLock::new(0)),
            buffer_sender: Some(sender),
            buffer_receiver: Some(receiver),
        };
        
        // Start background preloading
        loader.start_preloading();
        
        Ok(loader)
    }
    
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }
    
    fn __next__(&mut self) -> Option<PyObject> {
        // Try to get from preloaded buffer first
        if let Some(receiver) = &self.buffer_receiver {
            if let Ok(batch) = receiver.try_recv() {
                return Some(self.batch_to_python(batch));
            }
        }
        
        // Fallback to immediate loading
        let batch = self.get_next_batch()?;
        Some(self.batch_to_python(batch))
    }
    
    fn get_batch(&mut self) -> Option<PyObject> {
        let batch = self.get_next_batch()?;
        Some(self.batch_to_python(batch))
    }
    
    fn reset(&mut self) {
        *self.position.write() = 0;
        self.current_epoch += 1;
        
        if self.shuffle {
            self.indices.write().shuffle(&mut thread_rng());
        }
        
        // Restart preloading
        self.start_preloading();
    }
    
    fn __len__(&self) -> usize {
        let data_len = self.data.read().len();
        (data_len + self.batch_size - 1) / self.batch_size
    }
    
    fn len(&self) -> usize {
        let data_len = self.data.read().len();
        (data_len + self.batch_size - 1) / self.batch_size
    }
    
    fn get_stats(&self) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            let dict = pyo3::types::PyDict::new_bound(py);
            dict.set_item("total_samples", self.data.read().len())?;
            dict.set_item("batch_size", self.batch_size)?;
            dict.set_item("current_epoch", self.current_epoch)?;
            dict.set_item("position", *self.position.read())?;
            Ok(dict.to_object(py))
        })
    }
}

impl FastDataLoader {
    fn load_data(path: &str) -> PyResult<Vec<RecipeData>> {
        let mut data = Vec::new();
        
        // Handle JSONL, JSON, and CSV formats
        if path.ends_with(".jsonl") || path.ends_with(".json") {
            let mut file = File::open(path)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Failed to open file: {}", e)))?;
            
            // Memory-mapped file for ultra-fast access
            let mut file_bytes = Vec::new();
            file.read_to_end(&mut file_bytes)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
            
            // Pre-allocate with estimated capacity
            let estimated_lines = file_bytes.iter().filter(|&&b| b == b'\n').count();
            data.reserve(estimated_lines);
            
            // Parse in parallel chunks for maximum throughput
            let chunk_size = std::cmp::max(8192, estimated_lines / (rayon::current_num_threads() * 4));
            
            // Split into line byte ranges first
            let mut line_starts = vec![0];
            for (i, &byte) in file_bytes.iter().enumerate() {
                if byte == b'\n' {
                    line_starts.push(i + 1);
                }
            }
            
            let results: Vec<Vec<RecipeData>> = line_starts
                .par_chunks(chunk_size)
                .filter_map(|chunk_starts| {
                    if chunk_starts.is_empty() {
                        return None;
                    }
                    
                    let mut chunk_data = Vec::with_capacity(chunk_starts.len());
                    
                    for window in chunk_starts.windows(2) {
                        let start = window[0];
                        let end = window[1].saturating_sub(1);
                        
                        if start >= end || end > file_bytes.len() {
                            continue;
                        }
                        
                        let line_bytes = &file_bytes[start..end];
                        if line_bytes.is_empty() || line_bytes.iter().all(|&b| b.is_ascii_whitespace()) {
                            continue;
                        }
                        
                        // Parse with simd_json (zero-copy when possible)
                        let mut mutable_bytes = line_bytes.to_vec();
                        
                        // Try modern format
                        if let Ok(recipe) = simd_json::from_slice::<RecipeData>(&mut mutable_bytes) {
                            chunk_data.push(recipe);
                            continue;
                        }
                        
                        // Reset for legacy format
                        mutable_bytes = line_bytes.to_vec();
                        if let Ok(legacy) = simd_json::from_slice::<LegacyRecipeData>(&mut mutable_bytes) {
                            let input = format!("Generate a recipe using these ingredients: {}", legacy.ingredients);
                            let output = format!("Recipe: {}\n\nIngredients: {}\n\nInstructions: {}", 
                                               legacy.title, legacy.ingredients, legacy.instruction);
                            
                            chunk_data.push(RecipeData {
                                input: input.into_boxed_str(),
                                output: output.into_boxed_str(),
                                quality_score: 0.8,
                            });
                        }
                    }
                    
                    Some(chunk_data)
                })
                .collect();
            
            // Flatten with minimal allocations
            let total_capacity: usize = results.iter().map(|v| v.len()).sum();
            data.reserve(total_capacity);
            
            for chunk_data in results {
                data.extend(chunk_data);
            }
        } else if path.ends_with(".csv") {
            let file = File::open(path)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Failed to open file: {}", e)))?;
            
            // CSV parsing - assume format: title,ingredients,instruction
            let mut rdr = csv::Reader::from_reader(file);
            for result in rdr.records() {
                if let Ok(record) = result {
                    if record.len() >= 3 {
                        data.push(RecipeData {
                            input: format!("Generate a recipe using these ingredients: {}", &record[1]).into_boxed_str(),
                            output: format!("Recipe: {}\n\nIngredients: {}\n\nInstructions: {}", 
                                          &record[0], &record[1], &record[2]).into_boxed_str(),
                            quality_score: 0.7,
                        });
                    }
                }
            }
        }
        
        if data.is_empty() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("No data loaded from file"));
        }
        
        println!("Loaded {} recipes from {}", data.len(), path);
        Ok(data)
    }
    
    fn get_next_batch(&mut self) -> Option<Vec<RecipeData>> {
        let data = self.data.read();
        let indices = self.indices.read();
        let mut position = self.position.write();
        
        if *position >= data.len() {
            return None;
        }
        
        let end_pos = std::cmp::min(*position + self.batch_size, data.len());
        let batch: Vec<RecipeData> = (*position..end_pos)
            .filter_map(|i| indices.get(i).and_then(|&idx| data.get(idx)))
            .cloned()
            .collect();
        
        *position = end_pos;
        
        if batch.is_empty() {
            None
        } else {
            Some(batch)
        }
    }
    
    fn start_preloading(&mut self) {
        if let Some(sender) = self.buffer_sender.clone() {
            let data = Arc::clone(&self.data);
            let indices = Arc::clone(&self.indices);
            let position = Arc::clone(&self.position);
            let batch_size = self.batch_size;
            
            std::thread::spawn(move || {
                loop {
                    let batch = {
                        let data = data.read();
                        let indices = indices.read();
                        let mut pos = position.write();
                        
                        if *pos >= data.len() {
                            break;
                        }
                        
                        let end_pos = std::cmp::min(*pos + batch_size, data.len());
                        let batch: Vec<RecipeData> = (*pos..end_pos)
                            .filter_map(|i| indices.get(i).and_then(|&idx| data.get(idx)))
                            .cloned()
                            .collect();
                        
                        *pos = end_pos;
                        batch
                    };
                    
                    if batch.is_empty() {
                        break;
                    }
                    
                    if sender.send(batch).is_err() {
                        break;
                    }
                }
            });
        }
    }
    
    fn batch_to_python(&self, batch: Vec<RecipeData>) -> PyObject {
        Python::with_gil(|py| {
            let dict = pyo3::types::PyDict::new_bound(py);
            
            let inputs: Vec<&str> = batch.iter().map(|r| r.input.as_ref()).collect();
            let outputs: Vec<&str> = batch.iter().map(|r| r.output.as_ref()).collect();
            let quality_scores: Vec<f64> = batch.iter().map(|r| r.quality_score).collect();
            
            dict.set_item("input_ids", pyo3::types::PyList::new_bound(py, &inputs)).unwrap();
            dict.set_item("outputs", pyo3::types::PyList::new_bound(py, &outputs)).unwrap();
            dict.set_item("quality_scores", pyo3::types::PyList::new_bound(py, &quality_scores)).unwrap();
            
            dict.to_object(py)
        })
    }
}

#[pyfunction]
#[pyo3(signature = (data_path, batch_size, shuffle=None, buffer_size=None))]
fn create_fast_dataloader(
    data_path: String,
    batch_size: usize,
    shuffle: Option<bool>,
    buffer_size: Option<usize>,
) -> PyResult<FastDataLoader> {
    FastDataLoader::new(
        data_path,
        batch_size,
        shuffle.unwrap_or(true),
        buffer_size,
    )
}

#[pyfunction]
fn benchmark_loading(data_path: String, num_batches: usize, batch_size: usize) -> PyResult<f64> {
    let start = std::time::Instant::now();
    let mut loader = FastDataLoader::new(data_path, batch_size, false, Some(32))?;
    
    for _ in 0..num_batches {
        if loader.get_next_batch().is_none() {
            break;
        }
    }
    
    let duration = start.elapsed();
    Ok(duration.as_secs_f64())
}

#[pymodule]
fn chef_genius_dataloader(m: &pyo3::Bound<'_, pyo3::types::PyModule>) -> PyResult<()> {
    m.add_class::<FastDataLoader>()?;
    m.add_function(wrap_pyfunction!(create_fast_dataloader, m)?)?;
    m.add_function(wrap_pyfunction!(benchmark_loading, m)?)?;
    Ok(())
}