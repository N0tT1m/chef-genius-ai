use pyo3::prelude::*;
use crate::models::*;
use std::collections::HashMap;
use std::sync::Arc;
use parking_lot::RwLock;

pub struct NutritionAnalyzer {
    nutrition_db: Arc<RwLock<HashMap<String, NutritionData>>>,
    allergen_db: Arc<RwLock<HashMap<String, Vec<String>>>>,
    stats: Arc<RwLock<NutritionStats>>,
}

#[derive(Debug, Clone)]
struct NutritionData {
    calories_per_100g: f32,
    protein_g: f32,
    carbs_g: f32,
    fat_g: f32,
    fiber_g: f32,
    sugar_g: f32,
    sodium_mg: f32,
    vitamins: HashMap<String, f32>,
    minerals: HashMap<String, f32>,
}

#[derive(Debug, Default)]
struct NutritionStats {
    analyses_performed: u64,
    avg_processing_time_ms: f64,
}

#[pyclass]
pub struct PyNutritionAnalyzer {
    analyzer: Arc<RwLock<NutritionAnalyzer>>,
}

#[pymethods]
impl PyNutritionAnalyzer {
    #[new]
    fn new() -> PyResult<Self> {
        let analyzer = NutritionAnalyzer::new_internal()?;
        Ok(Self {
            analyzer: Arc::new(RwLock::new(analyzer)),
        })
    }

    /// Analyze nutrition for a recipe
    fn analyze_recipe(&self, py: Python, recipe: PyRecipe) -> PyResult<PyNutritionInfo> {
        let analyzer = self.analyzer.clone();
        
        py.allow_threads(|| {
            let mut analyzer = analyzer.write();
            analyzer.analyze_recipe_internal(&recipe)
        })
    }

    /// Analyze nutrition for individual ingredients
    fn analyze_ingredients(&self, py: Python, ingredients: Vec<String>, quantities: Option<Vec<f32>>) -> PyResult<Vec<PyNutritionInfo>> {
        let analyzer = self.analyzer.clone();
        
        py.allow_threads(|| {
            let analyzer = analyzer.read();
            analyzer.analyze_ingredients_internal(ingredients, quantities)
        })
    }

    /// Calculate health score for a recipe
    fn calculate_health_score(&self, py: Python, nutrition_info: PyNutritionInfo) -> PyResult<f32> {
        let analyzer = self.analyzer.clone();
        
        py.allow_threads(|| {
            let analyzer = analyzer.read();
            analyzer.calculate_health_score_internal(&nutrition_info)
        })
    }

    /// Get allergen information for ingredients
    fn get_allergens(&self, py: Python, ingredients: Vec<String>) -> PyResult<Vec<String>> {
        let analyzer = self.analyzer.clone();
        
        py.allow_threads(|| {
            let analyzer = analyzer.read();
            analyzer.get_allergens_internal(ingredients)
        })
    }

    /// Get nutrition statistics
    fn get_stats(&self) -> PyResult<PyObject> {
        let analyzer = self.analyzer.read();
        let stats = analyzer.stats.read();
        
        Python::with_gil(|py| {
            let dict = pyo3::types::PyDict::new(py);
            dict.set_item("analyses_performed", stats.analyses_performed)?;
            dict.set_item("avg_processing_time_ms", stats.avg_processing_time_ms)?;
            Ok(dict.to_object(py))
        })
    }
}

impl NutritionAnalyzer {
    fn new_internal() -> PyResult<Self> {
        let mut analyzer = Self {
            nutrition_db: Arc::new(RwLock::new(HashMap::new())),
            allergen_db: Arc::new(RwLock::new(HashMap::new())),
            stats: Arc::new(RwLock::new(NutritionStats::default())),
        };
        
        analyzer.initialize_nutrition_db();
        analyzer.initialize_allergen_db();
        
        Ok(analyzer)
    }

    fn initialize_nutrition_db(&mut self) {
        let mut db = self.nutrition_db.write();
        
        // Add common ingredients with detailed nutrition data
        let nutrition_data = vec![
            ("chicken breast", 165.0, 31.0, 0.0, 3.6, 0.0, 0.0, 74.0),
            ("rice", 130.0, 2.7, 28.0, 0.3, 0.4, 0.1, 5.0),
            ("broccoli", 34.0, 2.8, 7.0, 0.4, 2.6, 1.5, 33.0),
            ("salmon", 208.0, 22.0, 0.0, 12.0, 0.0, 0.0, 59.0),
            ("eggs", 155.0, 13.0, 1.1, 11.0, 0.0, 1.1, 124.0),
            ("olive oil", 884.0, 0.0, 0.0, 100.0, 0.0, 0.0, 2.0),
            ("onion", 40.0, 1.1, 9.3, 0.1, 1.7, 4.2, 4.0),
            ("garlic", 149.0, 6.4, 33.0, 0.5, 2.1, 1.0, 17.0),
            ("tomato", 18.0, 0.9, 3.9, 0.2, 1.2, 2.6, 5.0),
            ("pasta", 131.0, 5.0, 25.0, 1.1, 1.8, 0.6, 6.0),
            ("spinach", 23.0, 2.9, 3.6, 0.4, 2.2, 0.4, 79.0),
            ("carrots", 41.0, 0.9, 9.6, 0.2, 2.8, 4.7, 69.0),
            ("potatoes", 77.0, 2.0, 17.0, 0.1, 2.2, 0.8, 6.0),
            ("beef", 250.0, 26.0, 0.0, 15.0, 0.0, 0.0, 72.0),
            ("cheese", 113.0, 7.0, 1.0, 9.0, 0.0, 0.5, 621.0),
        ];

        for (name, calories, protein, carbs, fat, fiber, sugar, sodium) in nutrition_data {
            let mut vitamins = HashMap::new();
            let mut minerals = HashMap::new();
            
            // Add some sample vitamin/mineral data
            match name {
                "broccoli" => {
                    vitamins.insert("C".to_string(), 89.2);
                    vitamins.insert("K".to_string(), 101.6);
                    minerals.insert("folate".to_string(), 63.0);
                },
                "spinach" => {
                    vitamins.insert("K".to_string(), 483.0);
                    vitamins.insert("A".to_string(), 469.0);
                    minerals.insert("iron".to_string(), 2.7);
                },
                "carrots" => {
                    vitamins.insert("A".to_string(), 835.0);
                    vitamins.insert("K".to_string(), 13.2);
                },
                _ => {}
            }

            let nutrition = NutritionData {
                calories_per_100g: calories,
                protein_g: protein,
                carbs_g: carbs,
                fat_g: fat,
                fiber_g: fiber,
                sugar_g: sugar,
                sodium_mg: sodium,
                vitamins,
                minerals,
            };
            
            db.insert(name.to_string(), nutrition);
        }
    }

    fn initialize_allergen_db(&mut self) {
        let mut db = self.allergen_db.write();
        
        let allergen_data = vec![
            ("eggs", vec!["eggs"]),
            ("milk", vec!["dairy"]),
            ("cheese", vec!["dairy"]),
            ("butter", vec!["dairy"]),
            ("cream", vec!["dairy"]),
            ("wheat flour", vec!["gluten", "wheat"]),
            ("pasta", vec!["gluten", "wheat"]),
            ("bread", vec!["gluten", "wheat"]),
            ("almonds", vec!["nuts", "tree nuts"]),
            ("walnuts", vec!["nuts", "tree nuts"]),
            ("peanuts", vec!["peanuts"]),
            ("salmon", vec!["fish"]),
            ("tuna", vec!["fish"]),
            ("shrimp", vec!["shellfish", "crustaceans"]),
            ("crab", vec!["shellfish", "crustaceans"]),
            ("soy sauce", vec!["soy"]),
            ("tofu", vec!["soy"]),
        ];

        for (ingredient, allergens) in allergen_data {
            db.insert(ingredient.to_string(), allergens.into_iter().map(|s| s.to_string()).collect());
        }
    }

    fn analyze_recipe_internal(&mut self, recipe: &PyRecipe) -> PyResult<PyNutritionInfo> {
        let start_time = std::time::Instant::now();
        
        let nutrition_db = self.nutrition_db.read();
        let allergen_db = self.allergen_db.read();
        
        let mut total_calories = 0.0;
        let mut total_protein = 0.0;
        let mut total_carbs = 0.0;
        let mut total_fat = 0.0;
        let mut total_fiber = 0.0;
        let mut total_sugar = 0.0;
        let mut total_sodium = 0.0;
        let mut all_vitamins: HashMap<String, f32> = HashMap::new();
        let mut all_minerals: HashMap<String, f32> = HashMap::new();
        let mut all_allergens = Vec::new();

        // Analyze each ingredient
        for ingredient in &recipe.ingredients {
            let ingredient_key = self.find_ingredient_key(&ingredient, &nutrition_db);
            
            if let Some(nutrition_data) = nutrition_db.get(&ingredient_key) {
                // Assume 100g serving size if not specified
                let serving_size = 100.0; // This would be parsed from ingredient text in real implementation
                let multiplier = serving_size / 100.0;
                
                total_calories += nutrition_data.calories_per_100g * multiplier;
                total_protein += nutrition_data.protein_g * multiplier;
                total_carbs += nutrition_data.carbs_g * multiplier;
                total_fat += nutrition_data.fat_g * multiplier;
                total_fiber += nutrition_data.fiber_g * multiplier;
                total_sugar += nutrition_data.sugar_g * multiplier;
                total_sodium += nutrition_data.sodium_mg * multiplier;
                
                // Aggregate vitamins and minerals
                for (vitamin, amount) in &nutrition_data.vitamins {
                    *all_vitamins.entry(vitamin.clone()).or_insert(0.0) += amount * multiplier;
                }
                
                for (mineral, amount) in &nutrition_data.minerals {
                    *all_minerals.entry(mineral.clone()).or_insert(0.0) += amount * multiplier;
                }
            }
            
            // Check for allergens
            if let Some(allergens) = allergen_db.get(&ingredient_key) {
                for allergen in allergens {
                    if !all_allergens.contains(allergen) {
                        all_allergens.push(allergen.clone());
                    }
                }
            }
        }

        // Calculate health score
        let health_score = self.calculate_health_score_from_values(
            total_calories, total_protein, total_carbs, total_fat, 
            total_fiber, total_sugar, total_sodium
        );

        // Adjust for serving size
        let servings = recipe.servings.unwrap_or(4) as f32;
        
        let nutrition_info = PyNutritionInfo::new(
            Some(total_calories / servings),
            Some(total_protein / servings),
            Some(total_carbs / servings),
            Some(total_fat / servings),
            Some(total_fiber / servings),
            Some(total_sugar / servings),
            Some(total_sodium / servings),
            if all_vitamins.is_empty() { None } else { Some(all_vitamins) },
            if all_minerals.is_empty() { None } else { Some(all_minerals) },
            if all_allergens.is_empty() { None } else { Some(all_allergens) },
            Some(health_score),
        );

        // Update stats
        {
            let mut stats = self.stats.write();
            stats.analyses_performed += 1;
            let processing_time = start_time.elapsed().as_millis() as f64;
            stats.avg_processing_time_ms = (stats.avg_processing_time_ms * (stats.analyses_performed - 1) as f64 + processing_time) / stats.analyses_performed as f64;
        }

        Ok(nutrition_info)
    }

    fn analyze_ingredients_internal(&self, ingredients: Vec<String>, quantities: Option<Vec<f32>>) -> PyResult<Vec<PyNutritionInfo>> {
        let nutrition_db = self.nutrition_db.read();
        let mut results = Vec::new();

        for (i, ingredient) in ingredients.iter().enumerate() {
            let ingredient_key = self.find_ingredient_key(ingredient, &nutrition_db);
            let quantity = quantities.as_ref().and_then(|q| q.get(i)).unwrap_or(&100.0);
            
            if let Some(nutrition_data) = nutrition_db.get(&ingredient_key) {
                let multiplier = quantity / 100.0;
                
                let nutrition_info = PyNutritionInfo::new(
                    Some(nutrition_data.calories_per_100g * multiplier),
                    Some(nutrition_data.protein_g * multiplier),
                    Some(nutrition_data.carbs_g * multiplier),
                    Some(nutrition_data.fat_g * multiplier),
                    Some(nutrition_data.fiber_g * multiplier),
                    Some(nutrition_data.sugar_g * multiplier),
                    Some(nutrition_data.sodium_mg * multiplier),
                    if nutrition_data.vitamins.is_empty() { None } else { 
                        Some(nutrition_data.vitamins.iter()
                            .map(|(k, v)| (k.clone(), v * multiplier))
                            .collect()) 
                    },
                    if nutrition_data.minerals.is_empty() { None } else { 
                        Some(nutrition_data.minerals.iter()
                            .map(|(k, v)| (k.clone(), v * multiplier))
                            .collect()) 
                    },
                    None, // Allergens handled separately
                    None, // Health score calculated separately
                );
                
                results.push(nutrition_info);
            } else {
                // Return empty nutrition info for unknown ingredients
                results.push(PyNutritionInfo::new(
                    None, None, None, None, None, None, None, None, None, None, None
                ));
            }
        }

        Ok(results)
    }

    fn calculate_health_score_internal(&self, nutrition_info: &PyNutritionInfo) -> PyResult<f32> {
        let calories = nutrition_info.calories.unwrap_or(0.0);
        let protein = nutrition_info.protein_g.unwrap_or(0.0);
        let carbs = nutrition_info.carbs_g.unwrap_or(0.0);
        let fat = nutrition_info.fat_g.unwrap_or(0.0);
        let fiber = nutrition_info.fiber_g.unwrap_or(0.0);
        let sugar = nutrition_info.sugar_g.unwrap_or(0.0);
        let sodium = nutrition_info.sodium_mg.unwrap_or(0.0);

        let score = self.calculate_health_score_from_values(
            calories, protein, carbs, fat, fiber, sugar, sodium
        );

        Ok(score)
    }

    fn calculate_health_score_from_values(&self, calories: f32, protein: f32, carbs: f32, fat: f32, fiber: f32, sugar: f32, sodium: f32) -> f32 {
        let mut score = 50.0; // Base score
        
        // Protein bonus (good)
        if protein > 10.0 {
            score += 15.0;
        } else if protein > 5.0 {
            score += 10.0;
        }
        
        // Fiber bonus (good)
        if fiber > 5.0 {
            score += 15.0;
        } else if fiber > 2.0 {
            score += 10.0;
        }
        
        // Sugar penalty (bad)
        if sugar > 20.0 {
            score -= 20.0;
        } else if sugar > 10.0 {
            score -= 10.0;
        }
        
        // Sodium penalty (bad)
        if sodium > 1000.0 {
            score -= 20.0;
        } else if sodium > 500.0 {
            score -= 10.0;
        }
        
        // Calorie balance
        if calories > 800.0 {
            score -= 15.0;
        } else if calories < 200.0 {
            score -= 10.0;
        }
        
        // Fat balance
        let fat_percentage = (fat * 9.0 / calories) * 100.0;
        if fat_percentage > 40.0 {
            score -= 10.0;
        } else if fat_percentage < 20.0 && fat_percentage > 10.0 {
            score += 5.0;
        }

        score.clamp(0.0, 100.0)
    }

    fn get_allergens_internal(&self, ingredients: Vec<String>) -> PyResult<Vec<String>> {
        let allergen_db = self.allergen_db.read();
        let mut all_allergens = Vec::new();

        for ingredient in ingredients {
            let ingredient_key = self.find_ingredient_key(&ingredient, &HashMap::new());
            
            if let Some(allergens) = allergen_db.get(&ingredient_key) {
                for allergen in allergens {
                    if !all_allergens.contains(allergen) {
                        all_allergens.push(allergen.clone());
                    }
                }
            }
        }

        Ok(all_allergens)
    }

    fn find_ingredient_key(&self, ingredient: &str, nutrition_db: &HashMap<String, NutritionData>) -> String {
        let ingredient_lower = ingredient.to_lowercase();
        
        // Try exact match first
        if nutrition_db.contains_key(&ingredient_lower) {
            return ingredient_lower;
        }
        
        // Try partial matches
        for key in nutrition_db.keys() {
            if ingredient_lower.contains(key) || key.contains(&ingredient_lower) {
                return key.clone();
            }
        }
        
        // Try word-by-word matching
        let ingredient_words: Vec<&str> = ingredient_lower.split_whitespace().collect();
        for key in nutrition_db.keys() {
            let key_words: Vec<&str> = key.split_whitespace().collect();
            for ingredient_word in &ingredient_words {
                for key_word in &key_words {
                    if ingredient_word == key_word && ingredient_word.len() > 3 {
                        return key.clone();
                    }
                }
            }
        }
        
        // Default to the original ingredient if no match found
        ingredient_lower
    }
}