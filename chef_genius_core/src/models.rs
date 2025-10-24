use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Recipe data structure
#[pyclass]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PyRecipe {
    #[pyo3(get, set)]
    pub title: String,
    #[pyo3(get, set)]
    pub ingredients: Vec<String>,
    #[pyo3(get, set)]
    pub instructions: Vec<String>,
    #[pyo3(get, set)]
    pub cooking_time: Option<String>,
    #[pyo3(get, set)]
    pub prep_time: Option<String>,
    #[pyo3(get, set)]
    pub servings: Option<i32>,
    #[pyo3(get, set)]
    pub difficulty: Option<String>,
    #[pyo3(get, set)]
    pub cuisine_type: Option<String>,
    #[pyo3(get, set)]
    pub dietary_tags: Option<Vec<String>>,
    #[pyo3(get, set)]
    pub confidence: Option<f32>,
}

#[pymethods]
impl PyRecipe {
    #[new]
    fn new(
        title: String,
        ingredients: Vec<String>,
        instructions: Vec<String>,
        cooking_time: Option<String>,
        prep_time: Option<String>,
        servings: Option<i32>,
        difficulty: Option<String>,
        cuisine_type: Option<String>,
        dietary_tags: Option<Vec<String>>,
        confidence: Option<f32>,
    ) -> Self {
        Self {
            title,
            ingredients,
            instructions,
            cooking_time,
            prep_time,
            servings,
            difficulty,
            cuisine_type,
            dietary_tags,
            confidence,
        }
    }
    
    fn __str__(&self) -> String {
        format!("Recipe: {} ({} ingredients)", self.title, self.ingredients.len())
    }
    
    fn to_dict(&self) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            let dict = pyo3::types::PyDict::new(py);
            dict.set_item("title", &self.title)?;
            dict.set_item("ingredients", &self.ingredients)?;
            dict.set_item("instructions", &self.instructions)?;
            dict.set_item("cooking_time", &self.cooking_time)?;
            dict.set_item("prep_time", &self.prep_time)?;
            dict.set_item("servings", &self.servings)?;
            dict.set_item("difficulty", &self.difficulty)?;
            dict.set_item("cuisine_type", &self.cuisine_type)?;
            dict.set_item("dietary_tags", &self.dietary_tags)?;
            dict.set_item("confidence", &self.confidence)?;
            Ok(dict.to_object(py))
        })
    }
}

/// Inference request
#[pyclass]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PyInferenceRequest {
    #[pyo3(get, set)]
    pub ingredients: Vec<String>,
    #[pyo3(get, set)]
    pub max_length: Option<i32>,
    #[pyo3(get, set)]
    pub temperature: Option<f32>,
    #[pyo3(get, set)]
    pub top_k: Option<i32>,
    #[pyo3(get, set)]
    pub top_p: Option<f32>,
    #[pyo3(get, set)]
    pub cuisine_style: Option<String>,
    #[pyo3(get, set)]
    pub dietary_restrictions: Option<Vec<String>>,
    #[pyo3(get, set)]
    pub cooking_time: Option<String>,
    #[pyo3(get, set)]
    pub difficulty: Option<String>,
    #[pyo3(get, set)]
    pub use_cache: Option<bool>,
}

#[pymethods]
impl PyInferenceRequest {
    #[new]
    fn new(
        ingredients: Vec<String>,
        max_length: Option<i32>,
        temperature: Option<f32>,
        top_k: Option<i32>,
        top_p: Option<f32>,
        cuisine_style: Option<String>,
        dietary_restrictions: Option<Vec<String>>,
        cooking_time: Option<String>,
        difficulty: Option<String>,
        use_cache: Option<bool>,
    ) -> Self {
        Self {
            ingredients,
            max_length,
            temperature,
            top_k,
            top_p,
            cuisine_style,
            dietary_restrictions,
            cooking_time,
            difficulty,
            use_cache,
        }
    }
}

/// Inference response
#[pyclass]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PyInferenceResponse {
    #[pyo3(get, set)]
    pub recipe: PyRecipe,
    #[pyo3(get, set)]
    pub confidence: f32,
    #[pyo3(get, set)]
    pub generation_time_ms: u64,
    #[pyo3(get, set)]
    pub model_version: String,
    #[pyo3(get, set)]
    pub alternatives: Option<Vec<PyRecipe>>,
    #[pyo3(get, set)]
    pub cached: bool,
}

#[pymethods]
impl PyInferenceResponse {
    #[new]
    fn new(
        recipe: PyRecipe,
        confidence: f32,
        generation_time_ms: u64,
        model_version: String,
        alternatives: Option<Vec<PyRecipe>>,
        cached: bool,
    ) -> Self {
        Self {
            recipe,
            confidence,
            generation_time_ms,
            model_version,
            alternatives,
            cached,
        }
    }
}

/// Vector search result
#[pyclass]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PySearchResult {
    #[pyo3(get, set)]
    pub recipe: PyRecipe,
    #[pyo3(get, set)]
    pub score: f32,
    #[pyo3(get, set)]
    pub distance: f32,
    #[pyo3(get, set)]
    pub metadata: Option<HashMap<String, String>>,
}

#[pymethods]
impl PySearchResult {
    #[new]
    fn new(
        recipe: PyRecipe,
        score: f32,
        distance: f32,
        metadata: Option<HashMap<String, String>>,
    ) -> Self {
        Self {
            recipe,
            score,
            distance,
            metadata,
        }
    }
}

/// Nutrition information
#[pyclass]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PyNutritionInfo {
    #[pyo3(get, set)]
    pub calories: Option<f32>,
    #[pyo3(get, set)]
    pub protein_g: Option<f32>,
    #[pyo3(get, set)]
    pub carbs_g: Option<f32>,
    #[pyo3(get, set)]
    pub fat_g: Option<f32>,
    #[pyo3(get, set)]
    pub fiber_g: Option<f32>,
    #[pyo3(get, set)]
    pub sugar_g: Option<f32>,
    #[pyo3(get, set)]
    pub sodium_mg: Option<f32>,
    #[pyo3(get, set)]
    pub vitamins: Option<HashMap<String, f32>>,
    #[pyo3(get, set)]
    pub minerals: Option<HashMap<String, f32>>,
    #[pyo3(get, set)]
    pub allergens: Option<Vec<String>>,
    #[pyo3(get, set)]
    pub health_score: Option<f32>,
}

#[pymethods]
impl PyNutritionInfo {
    #[new]
    fn new(
        calories: Option<f32>,
        protein_g: Option<f32>,
        carbs_g: Option<f32>,
        fat_g: Option<f32>,
        fiber_g: Option<f32>,
        sugar_g: Option<f32>,
        sodium_mg: Option<f32>,
        vitamins: Option<HashMap<String, f32>>,
        minerals: Option<HashMap<String, f32>>,
        allergens: Option<Vec<String>>,
        health_score: Option<f32>,
    ) -> Self {
        Self {
            calories,
            protein_g,
            carbs_g,
            fat_g,
            fiber_g,
            sugar_g,
            sodium_mg,
            vitamins,
            minerals,
            allergens,
            health_score,
        }
    }
    
    fn to_dict(&self) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            let dict = pyo3::types::PyDict::new(py);
            dict.set_item("calories", &self.calories)?;
            dict.set_item("protein_g", &self.protein_g)?;
            dict.set_item("carbs_g", &self.carbs_g)?;
            dict.set_item("fat_g", &self.fat_g)?;
            dict.set_item("fiber_g", &self.fiber_g)?;
            dict.set_item("sugar_g", &self.sugar_g)?;
            dict.set_item("sodium_mg", &self.sodium_mg)?;
            dict.set_item("vitamins", &self.vitamins)?;
            dict.set_item("minerals", &self.minerals)?;
            dict.set_item("allergens", &self.allergens)?;
            dict.set_item("health_score", &self.health_score)?;
            Ok(dict.to_object(py))
        })
    }
}