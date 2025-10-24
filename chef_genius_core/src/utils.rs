use pyo3::prelude::*;
use std::fs;
use std::path::Path;

/// Get available GPU memory in bytes
pub fn get_available_gpu_memory() -> Option<u64> {
    // Try to read from CUDA API if available
    // For now, return a reasonable estimate
    if let Ok(_) = candle_core::Device::cuda_if_available(0) {
        // Estimate based on common GPU memory sizes
        Some(8 * 1024 * 1024 * 1024) // 8GB
    } else {
        None
    }
}

/// Get number of CPU cores
pub fn get_cpu_count() -> usize {
    num_cpus::get()
}

/// Check if CUDA is available
pub fn is_cuda_available() -> bool {
    candle_core::Device::cuda_if_available(0).is_ok()
}

/// Get optimal batch size for current hardware
pub fn get_optimal_batch_size() -> usize {
    let available_memory = get_available_gpu_memory().unwrap_or(4 * 1024 * 1024 * 1024); // 4GB default
    let cpu_count = get_cpu_count();
    
    if is_cuda_available() {
        // GPU-based calculation
        (available_memory / (512 * 1024 * 1024)).clamp(1, 32) as usize
    } else {
        // CPU-based calculation
        (cpu_count * 2).clamp(1, 16)
    }
}

/// Create directory if it doesn't exist
pub fn ensure_directory_exists(path: &str) -> PyResult<()> {
    if !Path::new(path).exists() {
        fs::create_dir_all(path)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
    }
    Ok(())
}

/// Get file size in bytes
pub fn get_file_size(path: &str) -> PyResult<u64> {
    let metadata = fs::metadata(path)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
    Ok(metadata.len())
}

/// Format bytes to human readable string
pub fn format_bytes(bytes: u64) -> String {
    const UNITS: &[&str] = &["B", "KB", "MB", "GB", "TB"];
    
    if bytes == 0 {
        return "0 B".to_string();
    }
    
    let base = 1024f64;
    let exp = (bytes as f64).log(base).floor() as usize;
    let exp = exp.min(UNITS.len() - 1);
    
    let value = bytes as f64 / base.powi(exp as i32);
    
    if value >= 100.0 {
        format!("{:.0} {}", value, UNITS[exp])
    } else if value >= 10.0 {
        format!("{:.1} {}", value, UNITS[exp])
    } else {
        format!("{:.2} {}", value, UNITS[exp])
    }
}

/// Simple hash function for strings
pub fn simple_hash(text: &str) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    
    let mut hasher = DefaultHasher::new();
    text.hash(&mut hasher);
    hasher.finish()
}

/// Normalize text for better matching
pub fn normalize_text(text: &str) -> String {
    text.to_lowercase()
        .chars()
        .filter(|c| c.is_alphanumeric() || c.is_whitespace())
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<&str>>()
        .join(" ")
}

/// Truncate text to specified length
pub fn truncate_text(text: &str, max_length: usize) -> String {
    if text.len() <= max_length {
        text.to_string()
    } else {
        format!("{}...", &text[..max_length.saturating_sub(3)])
    }
}

/// Get system information
#[pyfunction]
pub fn get_system_info() -> PyResult<PyObject> {
    Python::with_gil(|py| {
        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("cpu_count", get_cpu_count())?;
        dict.set_item("cuda_available", is_cuda_available())?;
        dict.set_item("optimal_batch_size", get_optimal_batch_size())?;
        
        if let Some(gpu_memory) = get_available_gpu_memory() {
            dict.set_item("gpu_memory_bytes", gpu_memory)?;
            dict.set_item("gpu_memory_formatted", format_bytes(gpu_memory))?;
        }
        
        Ok(dict.to_object(py))
    })
}