use pyo3::prelude::*;

mod dataset_transformer;

use dataset_transformer::PyDatasetTransformer;

/// High-performance Chef Genius core library
/// Provides fast dataset transformation for B2B recipe conversion
#[pymodule]
fn chef_genius_core(_py: Python, m: &PyModule) -> PyResult<()> {
    // Add version info
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    
    // Dataset transformer for B2B conversion
    m.add_class::<PyDatasetTransformer>()?;
    
    // Utility functions
    m.add_function(wrap_pyfunction!(get_system_info, m)?)?;
    
    Ok(())
}


/// Get system performance information
#[pyfunction]
fn get_system_info() -> PyResult<PyObject> {
    Python::with_gil(|py| {
        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("cpu_count", num_cpus::get()).unwrap();
        dict.set_item("version", env!("CARGO_PKG_VERSION")).unwrap();
        Ok(dict.to_object(py))
    })
}