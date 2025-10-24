use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use ndarray::{Array1, Array2, Axis};
use statrs::distribution::{ContinuousCDF, Normal};
use statrs::statistics::{Statistics, OrderStatistics};
use chrono::{DateTime, Utc};
use dashmap::DashMap;
use std::sync::Arc;

/// Statistical drift detection methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DriftMethod {
    KolmogorovSmirnov,
    PopulationStabilityIndex,
    JensenShannonDivergence,
    ChiSquareTest,
    WassersteinDistance,
}

/// Drift detection result with severity levels
#[derive(Debug, Clone, Serialize, Deserialize)]
#[pyclass]
pub struct DriftResult {
    #[pyo3(get)]
    pub method: String,
    #[pyo3(get)]
    pub drift_score: f64,
    #[pyo3(get)]
    pub p_value: Option<f64>,
    #[pyo3(get)]
    pub is_drift: bool,
    #[pyo3(get)]
    pub severity: String, // "low", "medium", "high", "critical"
    #[pyo3(get)]
    pub threshold: f64,
    #[pyo3(get)]
    pub feature_name: String,
    #[pyo3(get)]
    pub timestamp: f64,
    #[pyo3(get)]
    pub sample_size: usize,
    #[pyo3(get)]
    pub reference_size: usize,
}

#[pymethods]
impl DriftResult {
    #[new]
    pub fn new(
        method: String,
        drift_score: f64,
        p_value: Option<f64>,
        is_drift: bool,
        severity: String,
        threshold: f64,
        feature_name: String,
        timestamp: f64,
        sample_size: usize,
        reference_size: usize,
    ) -> Self {
        Self {
            method,
            drift_score,
            p_value,
            is_drift,
            severity,
            threshold,
            feature_name,
            timestamp,
            sample_size,
            reference_size,
        }
    }

    pub fn to_dict(&self) -> PyResult<HashMap<String, PyObject>> {
        Python::with_gil(|py| {
            let mut dict = HashMap::new();
            dict.insert("method".to_string(), self.method.to_object(py));
            dict.insert("drift_score".to_string(), self.drift_score.to_object(py));
            dict.insert("p_value".to_string(), self.p_value.to_object(py));
            dict.insert("is_drift".to_string(), self.is_drift.to_object(py));
            dict.insert("severity".to_string(), self.severity.to_object(py));
            dict.insert("threshold".to_string(), self.threshold.to_object(py));
            dict.insert("feature_name".to_string(), self.feature_name.to_object(py));
            dict.insert("timestamp".to_string(), self.timestamp.to_object(py));
            dict.insert("sample_size".to_string(), self.sample_size.to_object(py));
            dict.insert("reference_size".to_string(), self.reference_size.to_object(py));
            Ok(dict)
        })
    }
}

/// Reference dataset for drift comparison
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReferenceDataset {
    pub features: HashMap<String, Vec<f64>>,
    pub categorical_features: HashMap<String, Vec<String>>,
    pub timestamp: DateTime<Utc>,
    pub sample_count: usize,
    pub metadata: HashMap<String, String>,
}

/// Configuration for drift detection
#[derive(Debug, Clone, Serialize, Deserialize)]
#[pyclass]
pub struct DriftConfig {
    #[pyo3(get, set)]
    pub ks_threshold: f64,
    #[pyo3(get, set)]
    pub psi_threshold: f64,
    #[pyo3(get, set)]
    pub js_threshold: f64,
    #[pyo3(get, set)]
    pub chi2_threshold: f64,
    #[pyo3(get, set)]
    pub wasserstein_threshold: f64,
    #[pyo3(get, set)]
    pub min_sample_size: usize,
    #[pyo3(get, set)]
    pub enable_auto_update: bool,
    #[pyo3(get, set)]
    pub detection_window_hours: u64,
}

#[pymethods]
impl DriftConfig {
    #[new]
    pub fn new() -> Self {
        Self {
            ks_threshold: 0.05,      // 5% significance level
            psi_threshold: 0.2,       // PSI > 0.2 indicates significant drift
            js_threshold: 0.1,        // JS divergence threshold
            chi2_threshold: 0.05,     // Chi-square p-value threshold
            wasserstein_threshold: 0.3, // Wasserstein distance threshold
            min_sample_size: 100,     // Minimum samples for reliable drift detection
            enable_auto_update: false, // Don't auto-update reference by default
            detection_window_hours: 24, // Check drift over 24-hour windows
        }
    }
}

impl Default for DriftConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// High-performance drift detection engine
#[pyclass]
pub struct PyDriftDetector {
    reference_data: Arc<DashMap<String, ReferenceDataset>>,
    config: DriftConfig,
    drift_history: Arc<DashMap<String, Vec<DriftResult>>>,
    last_update: DateTime<Utc>,
}

#[pymethods]
impl PyDriftDetector {
    #[new]
    pub fn new(config: Option<DriftConfig>) -> Self {
        Self {
            reference_data: Arc::new(DashMap::new()),
            config: config.unwrap_or_default(),
            drift_history: Arc::new(DashMap::new()),
            last_update: Utc::now(),
        }
    }

    /// Set reference dataset for a feature
    pub fn set_reference_data(
        &mut self,
        feature_name: String,
        data: Vec<f64>,
        metadata: Option<HashMap<String, String>>,
    ) -> PyResult<()> {
        let mut features = HashMap::new();
        features.insert(feature_name.clone(), data.clone());
        
        let reference = ReferenceDataset {
            features,
            categorical_features: HashMap::new(),
            timestamp: Utc::now(),
            sample_count: data.len(),
            metadata: metadata.unwrap_or_default(),
        };
        
        self.reference_data.insert(feature_name, reference);
        Ok(())
    }

    /// Set categorical reference data
    pub fn set_categorical_reference(
        &mut self,
        feature_name: String,
        categories: Vec<String>,
        metadata: Option<HashMap<String, String>>,
    ) -> PyResult<()> {
        let mut categorical_features = HashMap::new();
        categorical_features.insert(feature_name.clone(), categories.clone());
        
        let reference = ReferenceDataset {
            features: HashMap::new(),
            categorical_features,
            timestamp: Utc::now(),
            sample_count: categories.len(),
            metadata: metadata.unwrap_or_default(),
        };
        
        self.reference_data.insert(feature_name, reference);
        Ok(())
    }

    /// Detect drift using multiple methods
    pub fn detect_drift(
        &self,
        feature_name: String,
        current_data: Vec<f64>,
        methods: Option<Vec<String>>,
    ) -> PyResult<Vec<DriftResult>> {
        let methods = methods.unwrap_or_else(|| vec![
            "kolmogorov_smirnov".to_string(),
            "population_stability_index".to_string(),
            "jensen_shannon_divergence".to_string(),
        ]);

        if current_data.len() < self.config.min_sample_size {
            return Err(pyo3::exceptions::PyValueError::new_err(
                format!("Sample size {} below minimum {}", current_data.len(), self.config.min_sample_size)
            ));
        }

        let reference = self.reference_data.get(&feature_name)
            .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err(
                format!("No reference data found for feature: {}", feature_name)
            ))?;

        let reference_values = reference.features.get(&feature_name)
            .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err(
                format!("Feature {} not found in reference data", feature_name)
            ))?;

        let mut results = Vec::new();
        let timestamp = Utc::now().timestamp() as f64;

        for method in methods {
            let result = match method.as_str() {
                "kolmogorov_smirnov" => self.kolmogorov_smirnov_test(
                    &feature_name, reference_values, &current_data, timestamp
                )?,
                "population_stability_index" => self.population_stability_index(
                    &feature_name, reference_values, &current_data, timestamp
                )?,
                "jensen_shannon_divergence" => self.jensen_shannon_divergence(
                    &feature_name, reference_values, &current_data, timestamp
                )?,
                "wasserstein_distance" => self.wasserstein_distance(
                    &feature_name, reference_values, &current_data, timestamp
                )?,
                _ => return Err(pyo3::exceptions::PyValueError::new_err(
                    format!("Unknown drift detection method: {}", method)
                )),
            };
            results.push(result);
        }

        // Store results in history
        self.drift_history.entry(feature_name.clone())
            .or_insert_with(Vec::new)
            .extend(results.clone());

        Ok(results)
    }

    /// Detect categorical drift
    pub fn detect_categorical_drift(
        &self,
        feature_name: String,
        current_categories: Vec<String>,
    ) -> PyResult<DriftResult> {
        let reference = self.reference_data.get(&feature_name)
            .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err(
                format!("No reference data found for feature: {}", feature_name)
            ))?;

        let reference_categories = reference.categorical_features.get(&feature_name)
            .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err(
                format!("Categorical feature {} not found in reference data", feature_name)
            ))?;

        self.chi_square_test(&feature_name, reference_categories, &current_categories)
    }

    /// Get drift detection statistics
    pub fn get_drift_stats(&self) -> PyResult<HashMap<String, PyObject>> {
        Python::with_gil(|py| {
            let mut stats = HashMap::new();
            
            let total_features = self.reference_data.len();
            let mut features_with_drift = 0;
            let mut total_detections = 0;

            for entry in self.drift_history.iter() {
                let (feature_name, history) = entry.pair();
                total_detections += history.len();
                
                if history.iter().any(|r| r.is_drift) {
                    features_with_drift += 1;
                }
            }

            stats.insert("total_features".to_string(), total_features.to_object(py));
            stats.insert("features_with_drift".to_string(), features_with_drift.to_object(py));
            stats.insert("total_detections".to_string(), total_detections.to_object(py));
            stats.insert("drift_rate".to_string(), 
                if total_features > 0 { 
                    (features_with_drift as f64 / total_features as f64).to_object(py) 
                } else { 
                    0.0.to_object(py) 
                }
            );
            stats.insert("last_update".to_string(), self.last_update.timestamp().to_object(py));

            Ok(stats)
        })
    }

    /// Get drift history for a feature
    pub fn get_drift_history(&self, feature_name: String) -> PyResult<Vec<DriftResult>> {
        Ok(self.drift_history.get(&feature_name)
            .map(|h| h.clone())
            .unwrap_or_default())
    }

    /// Clear drift history
    pub fn clear_history(&mut self, feature_name: Option<String>) -> PyResult<()> {
        match feature_name {
            Some(name) => {
                self.drift_history.remove(&name);
            }
            None => {
                self.drift_history.clear();
            }
        }
        Ok(())
    }

    /// Update configuration
    pub fn update_config(&mut self, config: DriftConfig) -> PyResult<()> {
        self.config = config;
        Ok(())
    }
}

impl PyDriftDetector {
    /// Kolmogorov-Smirnov test for distribution drift
    fn kolmogorov_smirnov_test(
        &self,
        feature_name: &str,
        reference: &[f64],
        current: &[f64],
        timestamp: f64,
    ) -> PyResult<DriftResult> {
        let mut ref_sorted = reference.to_vec();
        let mut cur_sorted = current.to_vec();
        ref_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        cur_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let n1 = reference.len() as f64;
        let n2 = current.len() as f64;
        
        // Calculate empirical CDFs and find maximum difference
        let mut max_diff = 0.0;
        let all_values: Vec<f64> = [reference, current].concat();
        let mut unique_values = all_values.clone();
        unique_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        unique_values.dedup();

        for &value in &unique_values {
            let cdf1 = ref_sorted.iter().filter(|&&x| x <= value).count() as f64 / n1;
            let cdf2 = cur_sorted.iter().filter(|&&x| x <= value).count() as f64 / n2;
            max_diff = max_diff.max((cdf1 - cdf2).abs());
        }

        // Calculate p-value (approximation)
        let sqrt_term = ((n1 + n2) / (n1 * n2)).sqrt();
        let lambda = max_diff / sqrt_term;
        let p_value = 2.0 * (-2.0 * lambda * lambda).exp(); // Simplified approximation

        let is_drift = p_value < self.config.ks_threshold;
        let severity = self.calculate_severity(max_diff, &[0.1, 0.2, 0.3, 0.5]);

        Ok(DriftResult::new(
            "kolmogorov_smirnov".to_string(),
            max_diff,
            Some(p_value),
            is_drift,
            severity,
            self.config.ks_threshold,
            feature_name.to_string(),
            timestamp,
            current.len(),
            reference.len(),
        ))
    }

    /// Population Stability Index calculation
    fn population_stability_index(
        &self,
        feature_name: &str,
        reference: &[f64],
        current: &[f64],
        timestamp: f64,
    ) -> PyResult<DriftResult> {
        // Create bins based on reference data quantiles
        let mut ref_sorted = reference.to_vec();
        ref_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let num_bins = 10;
        let mut bin_edges = Vec::new();
        for i in 0..=num_bins {
            let quantile = i as f64 / num_bins as f64;
            let index = ((ref_sorted.len() - 1) as f64 * quantile) as usize;
            bin_edges.push(ref_sorted[index.min(ref_sorted.len() - 1)]);
        }

        // Calculate bin counts for both distributions
        let ref_counts = self.calculate_bin_counts(reference, &bin_edges);
        let cur_counts = self.calculate_bin_counts(current, &bin_edges);

        // Calculate PSI
        let mut psi = 0.0;
        for i in 0..num_bins {
            let expected = ref_counts[i] / reference.len() as f64;
            let actual = cur_counts[i] / current.len() as f64;
            
            if expected > 0.0 && actual > 0.0 {
                psi += (actual - expected) * (actual / expected).ln();
            }
        }

        let is_drift = psi > self.config.psi_threshold;
        let severity = self.calculate_severity(psi, &[0.1, 0.25, 0.5, 1.0]);

        Ok(DriftResult::new(
            "population_stability_index".to_string(),
            psi,
            None,
            is_drift,
            severity,
            self.config.psi_threshold,
            feature_name.to_string(),
            timestamp,
            current.len(),
            reference.len(),
        ))
    }

    /// Jensen-Shannon divergence calculation
    fn jensen_shannon_divergence(
        &self,
        feature_name: &str,
        reference: &[f64],
        current: &[f64],
        timestamp: f64,
    ) -> PyResult<DriftResult> {
        // Create histograms
        let num_bins = 20;
        let min_val = reference.iter().chain(current.iter()).fold(f64::INFINITY, |a, &b| a.min(b));
        let max_val = reference.iter().chain(current.iter()).fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        
        let bin_width = (max_val - min_val) / num_bins as f64;
        let mut bin_edges = Vec::new();
        for i in 0..=num_bins {
            bin_edges.push(min_val + i as f64 * bin_width);
        }

        let ref_hist = self.calculate_histogram(reference, &bin_edges);
        let cur_hist = self.calculate_histogram(current, &bin_edges);

        // Normalize to probability distributions
        let ref_sum: f64 = ref_hist.iter().sum();
        let cur_sum: f64 = cur_hist.iter().sum();
        
        let ref_prob: Vec<f64> = ref_hist.iter().map(|&x| x / ref_sum + 1e-10).collect();
        let cur_prob: Vec<f64> = cur_hist.iter().map(|&x| x / cur_sum + 1e-10).collect();

        // Calculate JS divergence
        let mut js_div = 0.0;
        for i in 0..num_bins {
            let m = (ref_prob[i] + cur_prob[i]) / 2.0;
            js_div += 0.5 * ref_prob[i] * (ref_prob[i] / m).ln();
            js_div += 0.5 * cur_prob[i] * (cur_prob[i] / m).ln();
        }

        let is_drift = js_div > self.config.js_threshold;
        let severity = self.calculate_severity(js_div, &[0.05, 0.1, 0.2, 0.5]);

        Ok(DriftResult::new(
            "jensen_shannon_divergence".to_string(),
            js_div,
            None,
            is_drift,
            severity,
            self.config.js_threshold,
            feature_name.to_string(),
            timestamp,
            current.len(),
            reference.len(),
        ))
    }

    /// Wasserstein (Earth Mover's) distance
    fn wasserstein_distance(
        &self,
        feature_name: &str,
        reference: &[f64],
        current: &[f64],
        timestamp: f64,
    ) -> PyResult<DriftResult> {
        let mut ref_sorted = reference.to_vec();
        let mut cur_sorted = current.to_vec();
        ref_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        cur_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // Simple 1-D Wasserstein distance calculation
        let n1 = reference.len();
        let n2 = current.len();
        
        let mut distance = 0.0;
        let mut i = 0;
        let mut j = 0;
        let mut mass1 = 0.0;
        let mut mass2 = 0.0;

        while i < n1 && j < n2 {
            let next_mass1 = (i + 1) as f64 / n1 as f64;
            let next_mass2 = (j + 1) as f64 / n2 as f64;
            
            if next_mass1 < next_mass2 {
                distance += (mass1 - mass2).abs() * (ref_sorted[i] - 
                    if j > 0 { cur_sorted[j-1] } else { cur_sorted[0] }).abs();
                mass1 = next_mass1;
                i += 1;
            } else {
                distance += (mass1 - mass2).abs() * (cur_sorted[j] - 
                    if i > 0 { ref_sorted[i-1] } else { ref_sorted[0] }).abs();
                mass2 = next_mass2;
                j += 1;
            }
        }

        let is_drift = distance > self.config.wasserstein_threshold;
        let severity = self.calculate_severity(distance, &[0.1, 0.3, 0.6, 1.0]);

        Ok(DriftResult::new(
            "wasserstein_distance".to_string(),
            distance,
            None,
            is_drift,
            severity,
            self.config.wasserstein_threshold,
            feature_name.to_string(),
            timestamp,
            current.len(),
            reference.len(),
        ))
    }

    /// Chi-square test for categorical features
    fn chi_square_test(
        &self,
        feature_name: &str,
        reference: &[String],
        current: &[String],
    ) -> PyResult<DriftResult> {
        // Get all unique categories
        let mut all_categories: Vec<String> = reference.iter().chain(current.iter()).cloned().collect();
        all_categories.sort();
        all_categories.dedup();

        // Count occurrences in each dataset
        let mut ref_counts = HashMap::new();
        let mut cur_counts = HashMap::new();
        
        for cat in &all_categories {
            ref_counts.insert(cat.clone(), reference.iter().filter(|&x| x == cat).count());
            cur_counts.insert(cat.clone(), current.iter().filter(|&x| x == cat).count());
        }

        // Calculate chi-square statistic
        let mut chi2 = 0.0;
        let n1 = reference.len() as f64;
        let n2 = current.len() as f64;
        
        for cat in &all_categories {
            let observed1 = *ref_counts.get(cat).unwrap() as f64;
            let observed2 = *cur_counts.get(cat).unwrap() as f64;
            let total = observed1 + observed2;
            
            if total > 0.0 {
                let expected1 = total * n1 / (n1 + n2);
                let expected2 = total * n2 / (n1 + n2);
                
                if expected1 > 0.0 {
                    chi2 += (observed1 - expected1).powi(2) / expected1;
                }
                if expected2 > 0.0 {
                    chi2 += (observed2 - expected2).powi(2) / expected2;
                }
            }
        }

        // Approximate p-value (simplified)
        let degrees_of_freedom = all_categories.len() - 1;
        let p_value = if degrees_of_freedom > 0 {
            // Very rough approximation - in practice would use proper chi-square CDF
            (-chi2 / 2.0).exp()
        } else {
            1.0
        };

        let is_drift = p_value < self.config.chi2_threshold;
        let severity = self.calculate_severity(chi2, &[3.84, 7.88, 15.51, 25.0]); // Chi-square critical values

        Ok(DriftResult::new(
            "chi_square_test".to_string(),
            chi2,
            Some(p_value),
            is_drift,
            severity,
            self.config.chi2_threshold,
            feature_name.to_string(),
            Utc::now().timestamp() as f64,
            current.len(),
            reference.len(),
        ))
    }

    /// Helper function to calculate bin counts
    fn calculate_bin_counts(&self, data: &[f64], bin_edges: &[f64]) -> Vec<f64> {
        let mut counts = vec![0.0; bin_edges.len() - 1];
        
        for &value in data {
            for i in 0..bin_edges.len() - 1 {
                if value >= bin_edges[i] && value < bin_edges[i + 1] {
                    counts[i] += 1.0;
                    break;
                } else if i == bin_edges.len() - 2 && value >= bin_edges[i + 1] {
                    counts[i] += 1.0;
                    break;
                }
            }
        }
        
        counts
    }

    /// Helper function to calculate histogram
    fn calculate_histogram(&self, data: &[f64], bin_edges: &[f64]) -> Vec<f64> {
        self.calculate_bin_counts(data, bin_edges)
    }

    /// Calculate severity level based on thresholds
    fn calculate_severity(&self, score: f64, thresholds: &[f64]) -> String {
        if score < thresholds[0] {
            "low".to_string()
        } else if score < thresholds[1] {
            "medium".to_string()
        } else if score < thresholds[2] {
            "high".to_string()
        } else {
            "critical".to_string()
        }
    }
}

/// Recipe-specific drift detection utilities
#[pyclass]
pub struct PyRecipeDriftMonitor {
    detector: PyDriftDetector,
    ingredient_vocab: HashMap<String, usize>,
    cuisine_types: Vec<String>,
}

#[pymethods]
impl PyRecipeDriftMonitor {
    #[new]
    pub fn new(config: Option<DriftConfig>) -> Self {
        Self {
            detector: PyDriftDetector::new(config),
            ingredient_vocab: HashMap::new(),
            cuisine_types: Vec::new(),
        }
    }

    /// Initialize with training data vocabulary
    pub fn initialize_vocabulary(
        &mut self,
        ingredients: Vec<String>,
        cuisines: Vec<String>,
    ) -> PyResult<()> {
        // Build ingredient vocabulary
        for (i, ingredient) in ingredients.iter().enumerate() {
            self.ingredient_vocab.insert(ingredient.clone(), i);
        }
        
        self.cuisine_types = cuisines;
        Ok(())
    }

    /// Monitor recipe generation drift
    pub fn detect_recipe_drift(
        &self,
        generated_recipes: Vec<HashMap<String, PyObject>>,
    ) -> PyResult<Vec<DriftResult>> {
        Python::with_gil(|py| {
            let mut results = Vec::new();
            
            // Extract features from recipes
            let mut recipe_lengths = Vec::new();
            let mut ingredient_counts = Vec::new();
            let mut instruction_counts = Vec::new();
            let mut cooking_times = Vec::new();
            
            for recipe in &generated_recipes {
                // Recipe length (character count)
                if let Some(title) = recipe.get("title") {
                    if let Ok(title_str) = title.extract::<String>(py) {
                        recipe_lengths.push(title_str.len() as f64);
                    }
                }
                
                // Ingredient count
                if let Some(ingredients) = recipe.get("ingredients") {
                    if let Ok(ing_list) = ingredients.extract::<Vec<String>>(py) {
                        ingredient_counts.push(ing_list.len() as f64);
                    }
                }
                
                // Instruction count
                if let Some(instructions) = recipe.get("instructions") {
                    if let Ok(inst_list) = instructions.extract::<Vec<String>>(py) {
                        instruction_counts.push(inst_list.len() as f64);
                    }
                }
                
                // Cooking time (if parseable)
                if let Some(cooking_time) = recipe.get("cooking_time") {
                    if let Ok(time_str) = cooking_time.extract::<String>(py) {
                        if let Ok(time_mins) = self.parse_cooking_time(&time_str) {
                            cooking_times.push(time_mins);
                        }
                    }
                }
            }
            
            // Detect drift for each feature
            if !recipe_lengths.is_empty() {
                if let Ok(drift_results) = self.detector.detect_drift(
                    "recipe_title_length".to_string(),
                    recipe_lengths,
                    None,
                ) {
                    results.extend(drift_results);
                }
            }
            
            if !ingredient_counts.is_empty() {
                if let Ok(drift_results) = self.detector.detect_drift(
                    "ingredient_count".to_string(),
                    ingredient_counts,
                    None,
                ) {
                    results.extend(drift_results);
                }
            }
            
            if !instruction_counts.is_empty() {
                if let Ok(drift_results) = self.detector.detect_drift(
                    "instruction_count".to_string(),
                    instruction_counts,
                    None,
                ) {
                    results.extend(drift_results);
                }
            }
            
            if !cooking_times.is_empty() {
                if let Ok(drift_results) = self.detector.detect_drift(
                    "cooking_time_minutes".to_string(),
                    cooking_times,
                    None,
                ) {
                    results.extend(drift_results);
                }
            }
            
            Ok(results)
        })
    }

    /// Get comprehensive drift report
    pub fn get_drift_report(&self) -> PyResult<HashMap<String, PyObject>> {
        self.detector.get_drift_stats()
    }
}

impl PyRecipeDriftMonitor {
    /// Parse cooking time string to minutes
    fn parse_cooking_time(&self, time_str: &str) -> Result<f64, Box<dyn std::error::Error>> {
        let time_str = time_str.to_lowercase();
        
        if time_str.contains("hour") {
            let hours: f64 = time_str.chars()
                .filter(|c| c.is_numeric() || *c == '.')
                .collect::<String>()
                .parse()?;
            Ok(hours * 60.0)
        } else if time_str.contains("min") {
            let minutes: f64 = time_str.chars()
                .filter(|c| c.is_numeric() || *c == '.')
                .collect::<String>()
                .parse()?;
            Ok(minutes)
        } else {
            // Try to parse as number (assume minutes)
            let number: f64 = time_str.chars()
                .filter(|c| c.is_numeric() || *c == '.')
                .collect::<String>()
                .parse()?;
            Ok(number)
        }
    }
}